import argparse
import struct
import torch
import numpy as np
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from awq.utils.packing_utils import dequantize_gemm

"""
KuiperLLama 终极对齐导出脚本 (V3 - Final - Fixed)
修正点：
1. 修复 NotImplementedError：直接传递字符串列表作为 calib_data。
2. 修复 torch.cat 报错：确保样本长度足够，不让 AutoAWQ 过滤成空集。
3. 保持逆置换逻辑与 C++ 坐标系对齐。
"""

BLOCK_TYPE_FP32 = 0
BLOCK_TYPE_AWQ4_NAIVE = 1

def _write_block_descriptor(f, b_type, rows, cols, q_size, s_size=0, z_size=0):
    desc = struct.pack("<8i", b_type, rows, cols, q_size, s_size, z_size, 0, 0)
    f.write(desc)

def unpack_and_reorder_awq(layer):
    """撤销 AWQ GEMM 的重排逻辑"""
    qweight = layer.qweight.cpu()
    rows, cols = layer.out_features, layer.in_features
    # AWQ 的物理重排顺序：nibble i 对应第 order[i] 行
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    
    q_int = torch.zeros((cols, rows), dtype=torch.uint8)
    for i, row_offset in enumerate(order):
        q_int[:, row_offset::8] = (qweight >> (i * 4)) & 0xF
    
    qzeros = layer.qzeros.cpu()
    z_int = torch.zeros((qzeros.shape[0], rows), dtype=torch.uint8)
    for i, row_offset in enumerate(order):
        z_int[:, row_offset::8] = (qzeros >> (i * 4)) & 0xF
        
    return q_int.t().contiguous(), z_int.t().contiguous()

def _write_linear_layer(f, layer, name):
    if hasattr(layer, "qweight"):
        rows, cols = layer.out_features, layer.in_features
        # 核心修复：AutoAWQ GEMM 层的属性名为 group_size 而非 q_group_size
        group_size = getattr(layer, "group_size", 128)
        
        q_int, z_int = unpack_and_reorder_awq(layer)
        scales = layer.scales.cpu()
        
        with torch.no_grad():
            ref_w = dequantize_gemm(layer.qweight, layer.qzeros, layer.scales, 4, group_size).cpu()
            # 这里的 group_size 已经对齐
            z_exp = z_int.repeat_interleave(group_size, dim=1)[:, :cols]
            s_exp = scales.t().repeat_interleave(group_size, dim=1)[:, :cols]
            my_w = (q_int.float() - z_exp.float()) * s_exp.float()
            
            error = (ref_w - my_w).abs().max().item()
            if error > 1e-2:
                print(f"❌ 层 {name} 对拍失败！误差: {error:.6f}")
                raise ValueError(f"层 {name} 还原逻辑有误。")
            else:
                print(f"✅ 层 {name} 对拍成功！误差: {error:.6f}")

        q_flat = q_int.flatten().numpy()
        q_bytes = ((q_flat[1::2] << 4) | (q_flat[0::2])).astype(np.uint8).tobytes()
        z_flat = z_int.flatten().numpy()
        z_bytes = ((z_flat[1::2] << 4) | (z_flat[0::2])).astype(np.uint8).tobytes()
        s_data = scales.t().detach().contiguous().to(torch.float32).numpy().tobytes()

        _write_block_descriptor(f, BLOCK_TYPE_AWQ4_NAIVE, rows, cols, len(q_bytes), len(s_data), len(z_bytes))
        f.write(q_bytes)
        f.write(s_data)
        f.write(z_bytes)
    else:
        w = layer.weight.detach().contiguous().to(torch.float32).cpu().numpy()
        print(f"📦 层 {name} (FP32) | 写入中...")
        _write_block_descriptor(f, BLOCK_TYPE_FP32, w.shape[0], w.shape[1], w.nbytes)
        f.write(w.tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_bin", required=True)
    args = parser.parse_args()

    # 定义临时存档路径，防止重复量化
    temp_quant_dir = "/home/furina/code_learnning/cpp/cuda/KuiperLLama/models/llama2_7b_awq_temp"

    if os.path.exists(temp_quant_dir):
        print(f"✨ 检测到量化存档点，正在直接加载，跳过 16 分钟搜索...")
        model = AutoAWQForCausalLM.from_quantized(
            args.model_path, 
            quant_path=temp_quant_dir, 
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        print("🚀 未发现存档，开始加载模型...")
        model = AutoAWQForCausalLM.from_pretrained(args.model_path, device_map="cpu", low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        calib_data = [
            "The capital of France is Paris, a city known for its culture.",
            "GPU acceleration with CUDA allows for rapid processing.",
            "Data structures and algorithms form the foundation of computing."
        ] * 16 

        print("正在执行 AWQ 量化搜索 (这次我会帮你存下来)...")
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
        
        print(f"💾 搜索完成！正在创建存档点至 {temp_quant_dir}，下次出错无需重跑...")
        model.save_quantized(temp_quant_dir)

    print(f"开始导出权重至 {args.output_bin}...")
    llama = model.model
    with open(args.output_bin, "wb") as f:
        # 1. 写入 Header
        cfg = llama.config
        layers = llama.model.layers
        f.write(struct.pack("<7i", cfg.hidden_size, cfg.intermediate_size, len(layers), 
                            cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), 
                            -abs(cfg.vocab_size), 2048))
        f.write(struct.pack("<i", 128)) # group_size

        # 2. 导出层
        module_map = [
            ("WQ", [l.self_attn.q_proj for l in layers]),
            ("WK", [l.self_attn.k_proj for l in layers]),
            ("WV", [l.self_attn.v_proj for l in layers]),
            ("WO", [l.self_attn.o_proj for l in layers]),
            ("W1", [l.mlp.gate_proj for l in layers]),
            ("W2", [l.mlp.down_proj for l in layers]),
            ("W3", [l.mlp.up_proj for l in layers]),
        ]

        for name_base, projs in module_map:
            for i, p in enumerate(projs):
                _write_linear_layer(f, p, f"{name_base}_{i}")

        _write_linear_layer(f, llama.lm_head, "lm_head")

        print("写入 Embedding 和 Norms...")
        _write_tensor = lambda t: f.write(t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes())
        _write_tensor(llama.model.embed_tokens.weight)
        for l in layers: _write_tensor(l.input_layernorm.weight)
        for l in layers: _write_tensor(l.post_attention_layernorm.weight)
        _write_tensor(llama.model.norm.weight)

    print(f"\n✨ [成功] 导出完成：{args.output_bin}")

if __name__ == "__main__":
    main()