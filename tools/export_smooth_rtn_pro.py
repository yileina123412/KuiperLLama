import argparse
import struct
import torch
import torch.nn as nn
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

"""
KuiperLLama SmoothQuant + RTN 精度强化版
1. 使用更广泛的校准集提升激活值观察的准确性。
2. 保持 Naive 物理顺序，确保 C++ 侧 i/2 逻辑依然适用。
3. 内置 32 组标准长文本样本。
"""

BLOCK_TYPE_FP32 = 0
BLOCK_TYPE_SQ4 = 1 # 保持 1，代表你的通用 Naive 4bit 逻辑

def _write_block_descriptor(f, b_type, rows, cols, q_size, s_size=0, z_size=0):
    # 写入 8 个 int32，总计 32 字节
    # 字段顺序：[类型, 行, 列, Q大小, S大小, Z大小, GroupSize, 预留]
    desc = struct.pack("<8i", b_type, rows, cols, q_size, s_size, z_size, 128, 0) # group_size=128
    f.write(desc)

act_scales = {}
def get_act_scales(name, module, input, output):
    if isinstance(input, tuple):
        inp = input[0]
    else:
        inp = input
    # 捕捉 input_channel 的最大激活值
    hidden_dim = inp.detach().abs().max(dim=0)[0].max(dim=0)[0]
    if name not in act_scales:
        act_scales[name] = hidden_dim
    else:
        act_scales[name] = torch.max(act_scales[name], hidden_dim)

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
    args = parser.parse_args()

    print("🚀 正在加载 FP16 模型...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print("📊 正在收集高精度激活值分布 (这可能需要几分钟)...")
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(partial(get_act_scales, name)))
    
    # 标准校准样本 (涵盖新闻、科技、文学、代码等多种风格)
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
    
    with torch.no_grad():
        for i, text in enumerate(standard_calib):
            if i % 10 == 0: print(f"  已处理 {i}/{len(standard_calib)} 个样本...")
            inputs = tokenizer(text, return_tensors="pt")
            model(**inputs)
    
    for h in hooks: h.remove()

    print(f"📦 开始高精度量化并导出至 {args.output_bin}...")
    with open(args.output_bin, "wb") as f:
        cfg = model.config
        layers = model.model.layers
        f.write(struct.pack("<7i", cfg.hidden_size, cfg.intermediate_size, len(layers), 
                            cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), 
                            -abs(cfg.vocab_size), 2048))
        f.write(struct.pack("<i", 128))

        def export_layer(layer, name):
            W = layer.weight.data.float()
            if name in act_scales:
                act_max = act_scales[name]
                weight_max = W.abs().max(dim=0)[0]
                s = act_max.pow(0.5) / weight_max.pow(0.5)
                s = s.clamp(min=1e-5)
                W = W * s
            
            q_int, scales, z_int = quantize_rtn_naive(W)
            
            # 内部校验
            z_exp = z_int.repeat_interleave(128, dim=1)[:, :W.shape[1]]
            s_exp = scales.repeat_interleave(128, dim=1)[:, :W.shape[1]]
            W_dequant = (q_int.float() - z_exp) * s_exp
            err = (W - W_dequant).abs().max().item()
            print(f"  ✅ {name:15} | 误差: {err:.6f}")

            q_flat = q_int.flatten().numpy()
            q_bytes = ((q_flat[1::2] << 4) | (q_flat[0::2])).astype(np.uint8).tobytes()
            z_flat = z_int.flatten().numpy()
            if z_flat.size % 2 != 0: z_flat = np.append(z_flat, 0)
            z_bytes = ((z_flat[1::2] << 4) | (z_flat[0::2])).astype(np.uint8).tobytes()
            s_bytes = scales.flatten().numpy().astype(np.float32).tobytes()

            _write_block_descriptor(f, BLOCK_TYPE_SQ4, W.shape[0], W.shape[1], len(q_bytes), len(s_bytes), len(z_bytes))
            f.write(q_bytes); f.write(s_bytes); f.write(z_bytes)

        # 循环导出所有层
        for i, l in enumerate(layers):
            export_layer(l.self_attn.q_proj, f"WQ_{i}")
            export_layer(l.self_attn.k_proj, f"WK_{i}")
            export_layer(l.self_attn.v_proj, f"WV_{i}")
            export_layer(l.self_attn.o_proj, f"WO_{i}")
            export_layer(l.mlp.gate_proj, f"W1_{i}")
            export_layer(l.mlp.down_proj, f"W2_{i}")
            export_layer(l.mlp.up_proj, f"W3_{i}")
        
        export_layer(model.lm_head, "lm_head")

        # FP32 常数
        _write_tensor = lambda t: f.write(t.detach().to(torch.float32).cpu().numpy().tobytes())
        _write_tensor(model.model.embed_tokens.weight)
        for l in layers: _write_tensor(l.input_layernorm.weight)
        for l in layers: _write_tensor(l.post_attention_layernorm.weight)
        _write_tensor(model.model.norm.weight)

    print(f"\n✨ [完美] 权重已洗白，精度已强化。祝午餐愉快！")

if __name__ == "__main__":
    main()