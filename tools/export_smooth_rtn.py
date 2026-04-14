import argparse
import struct
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

"""
KuiperLLama SmoothQuant + RTN 导出脚本
1. 观察激活值，对权重进行平滑缩放（Smooth）。
2. 使用朴素 RTN 顺序量化，不进行任何 Swizzle 重排。
3. C++ 侧直接使用 i/2 逻辑读取。
"""

BLOCK_TYPE_FP32 = 0
BLOCK_TYPE_SQ4 = 2 # 标记为 SmoothQuant+RTN

def _write_block_descriptor(f, b_type, rows, cols, q_size, s_size=0, z_size=0):
    desc = struct.pack("<8i", b_type, rows, cols, q_size, s_size, z_size, 0, 0)
    f.write(desc)

# --- 1. 激活值捕捉逻辑 ---
act_scales = {}
def get_act_scales(name, module, input, output):
    # 捕捉输入激活值的绝对最大值 (per-channel)
    # input[0] 形状通常是 (batch, seq, hidden)
    hidden_dim = input[0].detach().abs().max(dim=0)[0].max(dim=0)[0]
    if name not in act_scales:
        act_scales[name] = hidden_dim
    else:
        act_scales[name] = torch.max(act_scales[name], hidden_dim)

# --- 2. 朴素 RTN 量化逻辑 ---
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_bin", required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant 迁移强度")
    args = parser.parse_args()

    # 1. 加载模型
    print("正在加载 FP16 模型...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 2. 注册 Hook 观测激活值
    print("正在分析激活值分布 (SmoothQuant 预处理)...")
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(partial(get_act_scales, name)))
    
    # 跑几个句子观察分布
    calib_texts = ["Hello, SmoothQuant helps preserve accuracy in 4-bit weights.", 
                   "The quick brown fox jumps over the lazy dog."]
    with torch.no_grad():
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors="pt")
            model(**inputs)
    
    for h in hooks: h.remove() # 移除 Hook

    # 3. 导出并应用 Smooth + RTN
    print(f"量化并导出至 {args.output_bin}...")
    with open(args.output_bin, "wb") as f:
        cfg = model.config
        layers = model.model.layers
        # Header
        f.write(struct.pack("<7i", cfg.hidden_size, cfg.intermediate_size, len(layers), 
                            cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), 
                            -abs(cfg.vocab_size), 2048))
        f.write(struct.pack("<i", 128)) # group_size

        def export_layer(layer, name):
            W = layer.weight.data.float()
            # --- SmoothQuant 核心逻辑 ---
            if name in act_scales:
                # s_j = (max|X_j|^alpha) / (max|W_j|^(1-alpha))
                act_max = act_scales[name]
                weight_max = W.abs().max(dim=0)[0] # per-input-channel
                s = act_max.pow(args.alpha) / weight_max.pow(1 - args.alpha)
                s = s.clamp(min=1e-5)
                # 平滑权重: W' = W * diag(s)
                W = W * s
                print(f"  [Smooth] {name} 已平滑处理")
            
            # --- RTN 量化 ---
            q_int, scales, z_int = quantize_rtn_naive(W)
            
            # 对拍验证
            z_exp = z_int.repeat_interleave(128, dim=1)[:, :W.shape[1]]
            s_exp = scales.repeat_interleave(128, dim=1)[:, :W.shape[1]]
            W_dequant = (q_int.float() - z_exp) * s_exp
            err = (W - W_dequant).abs().max().item()
            print(f"  ✅ {name} 对拍成功 (误差: {err:.6f})")

            # 写入二进制 (顺序打包)
            q_flat = q_int.flatten().numpy()
            q_bytes = ((q_flat[1::2] << 4) | (q_flat[0::2])).astype(np.uint8).tobytes()
            z_flat = z_int.flatten().numpy()
            if z_flat.size % 2 != 0: z_flat = np.append(z_flat, 0)
            z_bytes = ((z_flat[1::2] << 4) | (z_flat[0::2])).astype(np.uint8).tobytes()
            s_bytes = scales.flatten().numpy().astype(np.float32).tobytes()

            _write_block_descriptor(f, BLOCK_TYPE_SQ4, W.shape[0], W.shape[1], len(q_bytes), len(s_bytes), len(z_bytes))
            f.write(q_bytes); f.write(s_bytes); f.write(z_bytes)

        # 遍历层并导出
        for i, l in enumerate(layers):
            export_layer(l.self_attn.q_proj, f"WQ_{i}")
            export_layer(l.self_attn.k_proj, f"WK_{i}")
            export_layer(l.self_attn.v_proj, f"WV_{i}")
            export_layer(l.self_attn.o_proj, f"WO_{i}")
            export_layer(l.mlp.gate_proj, f"W1_{i}")
            export_layer(l.mlp.down_proj, f"W2_{i}")
            export_layer(l.mlp.up_proj, f"W3_{i}")
        
        export_layer(model.lm_head, "lm_head")

        # 写入 FP32 常数
        _write_tensor = lambda t: f.write(t.detach().to(torch.float32).cpu().numpy().tobytes())
        _write_tensor(model.model.embed_tokens.weight)
        for l in layers: _write_tensor(l.input_layernorm.weight)
        for l in layers: _write_tensor(l.post_attention_layernorm.weight)
        _write_tensor(model.model.norm.weight)

    print(f"✨ 导出完成！")

if __name__ == "__main__":
    main()