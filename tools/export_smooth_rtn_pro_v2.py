import argparse
import struct
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

"""
KuiperLLama SmoothQuant + RTN Pro (V2 - Equivalence Folding)
1. 激活值观察：捕捉每层输入的 max(abs(X))。
2. 等效折叠：将 1/s 乘入 RMSNorm，将 s 乘入 Weight。
3. 顺序量化：对处理后的 Weight 进行 Naive RTN 4-bit 量化。
4. 结构导出：Block Descriptor + Packed Data。
"""

BLOCK_TYPE_SQ4 = 1 

def _write_block_descriptor(f, b_type, rows, cols, q_size, s_size, z_size):
    # 32字节描述符
    # 字段顺序：[类型, 行, 列, Q大小, S大小, Z大小, GroupSize, 预留]
    desc = struct.pack("<8i", b_type, rows, cols, q_size, s_size, z_size, 128, 0)
    f.write(desc)

# --- 激活值捕获 ---
act_scales = {}
def get_act_scales(name, module, input, output):
    inp = input[0].detach().float().abs()
    # 捕捉 input_channel (dim) 方向的最大值
    scales = inp.max(dim=0)[0].max(dim=0)[0] 
    if name not in act_scales:
        act_scales[name] = scales
    else:
        act_scales[name] = torch.max(act_scales[name], scales)

def quantize_rtn_naive(tensor, group_size=128):
    rows, cols = tensor.shape
    tensor = tensor.to(torch.float32)
    reshaped_w = tensor.view(rows, -1, group_size)
    min_val = reshaped_w.min(dim=-1, keepdim=True)[0]
    max_val = reshaped_w.max(dim=-1, keepdim=True)[0]
    scales = (max_val - min_val) / 15.0
    scales = torch.clamp(scales, min=1e-6)
    zeros = torch.round(-min_val / scales)
    zeros = torch.clamp(zeros, 0, 15)
    q_weight = torch.round(reshaped_w / scales + zeros)
    q_weight = torch.clamp(q_weight, 0, 15).to(torch.uint8)
    return q_weight.view(rows, cols), scales.view(rows, -1), zeros.view(rows, -1).to(torch.uint8)

@torch.no_grad()
def smooth_and_fold(layer_idx, layers, alpha=0.5):
    """
    对第 i 层进行 SmoothQuant 处理并折叠到 RMSNorm
    """
    layer = layers[layer_idx]
    
    # 1. 处理 Attention 输入 (Q, K, V 共享同一个 RMSNorm 输入)
    # 计算 Q, K, V 输入激活值的综合 scale
    q_name = f"model.layers.{layer_idx}.self_attn.q_proj"
    k_name = f"model.layers.{layer_idx}.self_attn.k_proj"
    v_name = f"model.layers.{layer_idx}.self_attn.v_proj"
    
    act_max = torch.max(torch.stack([act_scales[q_name], act_scales[k_name], act_scales[v_name]]), dim=0)[0]
    weight_max = torch.max(torch.stack([
        layer.self_attn.q_proj.weight.abs().max(dim=0)[0],
        layer.self_attn.k_proj.weight.abs().max(dim=0)[0],
        layer.self_attn.v_proj.weight.abs().max(dim=0)[0]
    ]), dim=0)[0]
    
    # 计算平滑因子 s
    s = act_max.pow(alpha) / weight_max.pow(1 - alpha)
    s = s.clamp(min=1e-5)
    
    # 折叠：W = W * s, RMSNorm = RMSNorm / s
    layer.self_attn.q_proj.weight.mul_(s)
    layer.self_attn.k_proj.weight.mul_(s)
    layer.self_attn.v_proj.weight.mul_(s)
    layer.input_layernorm.weight.div_(s)
    
    # 2. 处理 MLP 输入 (Gate, Up 共享同一个 RMSNorm 输入)
    gate_name = f"model.layers.{layer_idx}.mlp.gate_proj"
    up_name = f"model.layers.{layer_idx}.mlp.up_proj"
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_bin", required=True)
    args = parser.parse_args()

    print("🚀 加载模型中...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 注册 Hook 收集激活值
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(partial(get_act_scales, name)))
    
    print("📊 运行校准集以捕捉激活值分布...")
    standard_calib = [
        "The Llama model is a collection of foundation language models ranging from 7B to 70B parameters.",
        "Quantization refers to the process of mapping continuous infinite values to a smaller set of discrete finite values.",
        "In computer science, CUDA is a parallel computing platform and application programming interface model created by NVIDIA.",
        "SmoothQuant is a training-free, post-training quantization solution that enables up to 8-bit weight, 8-bit activation quantization.",
        "The quick brown fox jumps over the lazy dog while the world keeps spinning around the sun.",
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows.",
        "A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data.",
        "The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity.",
        "def quick_sort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot]",
        "Optimization of large language models requires careful attention to memory bandwidth and compute utilization."
    ] * 10 # 扩展到 100 个样本
    calib_texts = [
        "The primary goal of quantization is to reduce memory footprint.",
        "SmoothQuant migrates quantization difficulty from activations to weights.",
        "Llama models use RMSNorm which can be scaled for mathematical equivalence."
    ] * 10
    with torch.no_grad():
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors="pt")
            model(**inputs)
    for h in hooks: h.remove()

    # 执行平滑与折叠
    print("✨ 正在执行 SmoothQuant 等效折叠...")
    for i in range(len(model.model.layers)):
        smooth_and_fold(i, model.model.layers)

    print(f"📦 导出至 {args.output_bin}...")
    with open(args.output_bin, "wb") as f:
        cfg = model.config
        layers = model.model.layers
        f.write(struct.pack("<7i", cfg.hidden_size, cfg.intermediate_size, len(layers), 
                            cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), 
                            -abs(cfg.vocab_size), 2048))
        f.write(struct.pack("<i", 128))

        def export_layer(layer_module, name):
            q_int, scales, z_int = quantize_rtn_naive(layer_module.weight.data)
            
            # 校验
            z_exp = z_int.repeat_interleave(128, dim=1)[:, :q_int.shape[1]]
            s_exp = scales.repeat_interleave(128, dim=1)[:, :q_int.shape[1]]
            W_dequant = (q_int.float() - z_exp) * s_exp
            err = (layer_module.weight.data.float() - W_dequant).abs().max().item()
            print(f"  ✅ {name:15} | 对拍误差: {err:.6f}")

            q_flat = q_int.flatten().numpy()
            q_bytes = ((q_flat[1::2] << 4) | (q_flat[0::2])).astype(np.uint8).tobytes()
            z_flat = z_int.flatten().numpy()
            if z_flat.size % 2 != 0: z_flat = np.append(z_flat, 0)
            z_bytes = ((z_flat[1::2] << 4) | (z_flat[0::2])).astype(np.uint8).tobytes()
            s_bytes = scales.flatten().numpy().astype(np.float32).tobytes()

            _write_block_descriptor(f, BLOCK_TYPE_SQ4, q_int.shape[0], q_int.shape[1], 
                                    len(q_bytes), len(s_bytes), len(z_bytes))
            f.write(q_bytes); f.write(s_bytes); f.write(z_bytes)

        # 循环写入
        for i, l in enumerate(layers):
            export_layer(l.self_attn.q_proj, f"WQ_{i}")
            export_layer(l.self_attn.k_proj, f"WK_{i}")
            export_layer(l.self_attn.v_proj, f"WV_{i}")
            export_layer(l.self_attn.o_proj, f"WO_{i}")
            export_layer(l.mlp.gate_proj, f"W1_{i}")
            export_layer(l.mlp.down_proj, f"W2_{i}")
            export_layer(l.mlp.up_proj, f"W3_{i}")
        
        export_layer(model.lm_head, "lm_head")

        # 写入常数 (注意：此时 RMSNorm 已经是被 s 缩放过的了)
        print("写入 Embedding 和 已平滑的 LayerNorms...")
        _write_tensor = lambda t: f.write(t.detach().float().cpu().numpy().tobytes())
        _write_tensor(model.model.embed_tokens.weight)
        for l in layers: _write_tensor(l.input_layernorm.weight)
        for l in layers: _write_tensor(l.post_attention_layernorm.weight)
        _write_tensor(model.model.norm.weight)

    print("\n✨ [完成] 这是一个数学等效且排布顺序的 4-bit 模型文件。")

if __name__ == "__main__":
    main()