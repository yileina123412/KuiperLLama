import argparse
import struct
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

# 核心目标：将 FP16 模型量化为 4-bit 权重，通过 SmoothQuant 降低量化误差，最终输出可被 C++ 直接读取的二进制文件。

"""
实现了SmoothQuant + RTN的4bit量化
KuiperLLama SmoothQuant + RTN Pro (V3 - Segmented Layout)
1. 激活值观察：捕捉每层输入的 max(abs(X))。
2. 等效折叠：将 1/s 乘入 RMSNorm，将 s 乘入 Weight（数学等价）。
3. 顺序量化：对处理后的 Weight 进行 Naive RTN 4-bit 量化。
4. 分段导出：按权重类型（WQ, WK...）分块写出，贴合 C++ 现有加载结构。
"""

BLOCK_TYPE_SQ4 = 1  
# 按指定对齐数（16B）填充空字节，保证二进制文件地址对齐（提升 C++ 读取效率） 让数据地址是16的整数倍，方便C++端使用SIMD指令批量加载和处理数据
# 保证：下一段要写入的数据，起始地址一定是 16 的整数倍！
def _write_align_padding(f, align: int = 16):
    """Pad file with zeros so that f.tell() becomes a multiple of `align`."""
    assert align in (4, 8, 16, 32, 64)
    # 计算文件指针位置和需要填充的字节数
    cur = f.tell() # 获取当前文件指针位置
    pad = (-cur) % align
    if pad:
        f.write(b"\x00" * pad)
        print(f"  [Padding] 填充了 {pad} 字节以实现 {align}B 对齐")

def _write_block_descriptor(f, b_type, rows, cols, q_size, s_size, z_size, group_size):

    """写入32字节描述符，帮助 C++ 动态解析块大小"""
    # 字段顺序：[类型, 行, 列, Q大小, S大小, Z大小, GroupSize(128), 预留(0)]
    desc = struct.pack("<8i", b_type, rows, cols, q_size, s_size, z_size, group_size, 0)
    f.write(desc)

# --- 激活值捕获 ---
act_scales = {} # 字典，存储每个Linear层输入激活值的绝对值最大值
# 前向钩子函数：捕获每个 Linear 层输入激活值的绝对值最大值（用于 SmoothQuant 计算缩放因子）
def get_act_scales(name, module, input, output):
    inp = input[0].detach().float().abs()  # # 40: 取输入张量→脱离计算图→转float→取绝对值（避免梯度干扰，统一精度）
    scales = inp.max(dim=0)[0].max(dim=0)[0]  # 41: 计算激活值的绝对值最大值（降维取max，适配LLM的输入形状）
    if name not in act_scales:
        act_scales[name] = scales
    else:
        act_scales[name] = torch.max(act_scales[name], scales)
# 核心量化函数：实现分组 RTN 4-bit 量化，输出量化后权重（uint8）、缩放因子（scales）、零点（zeros）
def quantize_rtn_naive(tensor, group_size=128):
    """顺序 RTN 量化，产生 100% 物理顺序的数据"""
    rows, cols = tensor.shape
    tensor = tensor.to(torch.float32)
    reshaped_w = tensor.view(rows, -1, group_size)
    min_val = reshaped_w.min(dim=-1, keepdim=True)[0] # 每组最小值
    max_val = reshaped_w.max(dim=-1, keepdim=True)[0]
    
    scales = (max_val - min_val) / 15.0
    scales = torch.clamp(scales, min=1e-6)
    zeros = torch.round(-min_val / scales)
    zeros = torch.clamp(zeros, 0, 15)
    
    q_weight = torch.round(reshaped_w / scales + zeros)
    q_weight = torch.clamp(q_weight, 0, 15).to(torch.uint8)
    
    return q_weight.view(rows, cols), scales.view(rows, -1), zeros.view(rows, -1).to(torch.uint8)
# 装饰器，作用是让被装饰函数内的所有张量运算不构建计算图、不追踪梯度（等价于在函数体外包一层 with torch.no_grad():）。
@torch.no_grad()
# 执行smoothquant折叠：根据捕获的激活值计算缩放因子 s，将其应用于权重和 RMSNorm，实现数学等价的平滑量化准备
def apply_smooth_quant(model, alpha=0.5):
    """执行 SmoothQuant 折叠，将 1/s 吸收进 RMSNorm"""
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        # --- Attention 输入平滑 (Q, K, V 共享输入) ---
        q_name = f"model.layers.{i}.self_attn.q_proj"
        k_name = f"model.layers.{i}.self_attn.k_proj"
        v_name = f"model.layers.{i}.self_attn.v_proj"
        
        act_max = torch.max(torch.stack([act_scales[q_name], act_scales[k_name], act_scales[v_name]]), dim=0)[0]
        weight_max = torch.max(torch.stack([
            layer.self_attn.q_proj.weight.abs().max(dim=0)[0],
            layer.self_attn.k_proj.weight.abs().max(dim=0)[0],
            layer.self_attn.v_proj.weight.abs().max(dim=0)[0]
        ]), dim=0)[0]
        
        s = act_max.pow(alpha) / weight_max.pow(1 - alpha)
        s = s.clamp(min=1e-5)
        
        layer.self_attn.q_proj.weight.mul_(s)
        layer.self_attn.k_proj.weight.mul_(s)
        layer.self_attn.v_proj.weight.mul_(s)
        layer.input_layernorm.weight.div_(s)
        
        # --- MLP 输入平滑 (Gate, Up 共享输入) ---
        gate_name = f"model.layers.{i}.mlp.gate_proj"
        up_name = f"model.layers.{i}.mlp.up_proj"
        
        act_max_mlp = torch.max(torch.stack([act_scales[gate_name], act_scales[up_name]]), dim=0)[0]
        weight_max_mlp = torch.max(torch.stack([
            layer.mlp.gate_proj.weight.abs().max(dim=0)[0],
            layer.mlp.up_proj.weight.abs().max(dim=0)[0]
        ]), dim=0)[0]
        
        s_mlp = act_max_mlp.pow(alpha) / weight_max_mlp.pow(1 - alpha)
        s_mlp = s_mlp.clamp(min=1e-5)
        
        layer.mlp.gate_proj.weight.mul_(s_mlp)
        layer.mlp.up_proj.weight.mul_(s_mlp)
        layer.post_attention_layernorm.weight.div_(s_mlp)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_bin", required=True)

    parser.add_argument("--no_smoothquant", action="store_true",
                        help="Disable SmoothQuant folding; do pure RTN4.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant alpha (only used if not --no_smoothquant).")
    
    parser.add_argument("--calib_path", type=str, default="",
                        help="A local text file; one sample per line.")
    parser.add_argument("--calib_num", type=int, default=512,
                        help="How many calibration samples to use.")
    
    parser.add_argument("--group_size", type=int, default=64, choices=[32, 64, 128])
    
    args = parser.parse_args()

    # 1. 加载模型和配套的tokenizer，准备进行量化和导出
    print(" 正在加载 FP16 模型...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 2. 收集激活值 给所有linear层注册前向钩子，捕获输入激活值的绝对值最大值，为 SmoothQuant 计算缩放因子做准备
    hooks = []
    for name, m in model.named_modules():
        # 仅对 Linear 层注册钩子，捕获其输入激活值的绝对值最大值（用于后续 SmoothQuant 计算缩放因子）
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(partial(get_act_scales, name)))
    


    print(" 运行校准集...")

    calib_texts = []
    if args.calib_path:
        with open(args.calib_path, "r", encoding="utf-8") as rf:
            for line in rf:
                t = line.strip()
                if t:
                    calib_texts.append(t)
        if len(calib_texts) > args.calib_num:
            calib_texts = calib_texts[: args.calib_num]
        print(f"  使用外部校准集: {args.calib_path}, 条数={len(calib_texts)}")

    if not calib_texts:
        calib_texts = [
            "The Llama architecture uses RMSNorm and rotary positional embeddings.",
            "SmoothQuant migrates quantization difficulty from activations to weights.",
            "Quantizing a model to 4-bit significantly reduces VRAM usage.",
            "Transformers are sensitive to quantization in the output projection.",
        ] * 128
        calib_texts = calib_texts[: args.calib_num]
        print(f"  使用内置校准集(兜底), 条数={len(calib_texts)}")

    with torch.no_grad():
        for text in calib_texts:
            model(**tokenizer(text, return_tensors="pt"))

    # 3. 平滑折叠
    # print(" 执行数学等效折叠 (SmoothQuant)...")
    # apply_smooth_quant(model)

    if args.no_smoothquant:
        print(" 关闭 SmoothQuant：只做 RTN4（用于对照实验）")
    else:
        print(f" 执行数学等效折叠 (SmoothQuant), alpha={args.alpha} ...")
        apply_smooth_quant(model, alpha=args.alpha)

    # 4. 分段导出
    print(f" 正在导出分段排布权重至 {args.output_bin}...")
    with open(args.output_bin, "wb") as f:
        cfg = model.config
        layers = model.model.layers
        
        # A. 写入全局 Header (兼容 ModelConfig)
        f.write(struct.pack("<7i", cfg.hidden_size, cfg.intermediate_size, len(layers), 
                            cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), 
                            -abs(cfg.vocab_size), 2048))
        # f.write(struct.pack("<i", 128)) # group_size
        f.write(struct.pack("<i", args.group_size))

        def export_layer_module(module, name):
            """内部量化并写入带描述符的块"""
            q_int, scales, z_int = quantize_rtn_naive(module.weight.data, group_size=args.group_size)
            
            # 对拍校验
            z_exp = z_int.repeat_interleave(args.group_size, dim=1)[:, :q_int.shape[1]]
            s_exp = scales.repeat_interleave(args.group_size, dim=1)[:, :q_int.shape[1]]
            W_dequant = (q_int.float() - z_exp) * s_exp
            err = (module.weight.data.float() - W_dequant).abs().max().item()
            print(f"  {name:15} | 误差: {err:.6f}")

            # 数据打包
            q_flat = q_int.flatten().numpy()
            q_bytes = ((q_flat[1::2] << 4) | (q_flat[0::2])).astype(np.uint8).tobytes()
            z_flat = z_int.flatten().numpy()
            if z_flat.size % 2 != 0: z_flat = np.append(z_flat, 0)
            z_bytes = ((z_flat[1::2] << 4) | (z_flat[0::2])).astype(np.uint8).tobytes()
            s_bytes = scales.flatten().numpy().astype(np.float32).tobytes()

            # 写描述符 + 数据
            _write_block_descriptor(f, BLOCK_TYPE_SQ4, q_int.shape[0], q_int.shape[1],
                        len(q_bytes), len(s_bytes), len(z_bytes), args.group_size)
            f.write(q_bytes)
            f.write(s_bytes)
            f.write(z_bytes)

        # B. 【关键】按类型分段导出，贴合 C++ 循环结构
        print("写入分段线性层...")
        module_map = [
            ("WQ", [l.self_attn.q_proj for l in layers]),
            ("WK", [l.self_attn.k_proj for l in layers]),
            ("WV", [l.self_attn.v_proj for l in layers]),
            ("WO", [l.self_attn.o_proj for l in layers]),
            ("W1", [l.mlp.gate_proj for l in layers]),
            ("W2", [l.mlp.down_proj for l in layers]),
            ("W3", [l.mlp.up_proj for l in layers]),
        ]

        for label, modules in module_map:
            for i, m in enumerate(modules):
                export_layer_module(m, f"{label}_{i}")

        # C. 导出 lm_head
        export_layer_module(model.lm_head, "lm_head")

        # --- 在此插入对齐填充 ---
        print(" 正在插入对齐填充...")
        _write_align_padding(f, align=16)

        # D. 导出 Embedding 和 Norms (保持 FP32)
        print("写入常数项 (Embedding & Norms)...")
        _write_tensor = lambda t: f.write(t.detach().float().cpu().numpy().tobytes())
        _write_tensor(model.model.embed_tokens.weight)
        for l in layers: _write_tensor(l.input_layernorm.weight)
        for l in layers: _write_tensor(l.post_attention_layernorm.weight)
        _write_tensor(model.model.norm.weight)

    print(f"\n [V3 成功] 文件已生成！")

if __name__ == "__main__":
    main()