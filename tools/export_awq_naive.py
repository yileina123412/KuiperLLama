import argparse
import struct
import torch
import numpy as np
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

"""
KuiperLLama Naive AWQ4 顺序导出脚本
特点：
1. 依然使用 AWQ 算法寻找最优 scale 和 zero。
2. 导出时撤销 GEMM 重排，改为 Naive 顺序排布：
   一个字节 = 2个权重，低4位是 W[i], 高4位是 W[i+1]。
3. C++ 侧不再需要复杂的位偏移公式，直接 index/2 即可。
"""

BLOCK_TYPE_FP32 = 0
BLOCK_TYPE_AWQ4_NAIVE = 1 # 标记为 Naive 格式

def _write_i32(f, v: int) -> None:
    f.write(struct.pack("<i", int(v)))

def _write_tensor_as(f, tensor: torch.Tensor, dtype: torch.dtype) -> int:
    t = tensor.detach().contiguous().to(dtype=dtype).cpu()
    data = t.numpy().tobytes()
    f.write(data)
    return len(data)

def _write_block_descriptor(f, b_type, rows, cols, q_size, s_size=0, z_size=0):
    desc = struct.pack("<8i", b_type, rows, cols, q_size, s_size, z_size, 0, 0)
    f.write(desc)

def unpack_awq_to_naive_integers(layer):
    """
    将 AWQ GEMM 格式的 qweight 还原为原始的 (rows, cols) 整数矩阵 (0-15)
    """
    qweight = layer.qweight # (in_features, out_features // 8)
    rows = layer.out_features
    cols = layer.in_features
    
    # 1. 提取原始的 4-bit 整数
    # AWQ GEMM 存储是 (cols, rows/8)，每个 int32 有 8 个权重
    # 我们先在 CPU 上把它解包成平铺的整数
    q_integers = torch.zeros((cols, rows), dtype=torch.uint8)
    
    qweight_cpu = qweight.cpu()
    for i in range(8):
        # 提取第 i 个 nibble
        # AWQ 的顺序通常是简单的位移，我们把它还原到对应的行
        part = (qweight_cpu >> (i * 4)) & 0xF
        # 注意：AWQ 的 qweight 维度是 (in_features, out_features // 8)
        # 还原回 (cols, rows)
        q_integers[:, i::8] = part.to(torch.uint8)
    
    # 转置回 (rows, cols) 符合线性层逻辑
    return q_integers.t().contiguous()

def unpack_awq_zeros_to_naive(layer):
    """
    将打包的 qzeros 还原为平铺的整数 (每组一个 zero)
    """
    # qzeros 形状通常是 (in_features // group_size, out_features // 8)
    qzeros = layer.qzeros.cpu()
    rows = layer.out_features
    cols_groups = qzeros.shape[0]
    
    z_integers = torch.zeros((cols_groups, rows), dtype=torch.uint8)
    for i in range(8):
        part = (qzeros >> (i * 4)) & 0xF
        z_integers[:, i::8] = part.to(torch.uint8)
        
    return z_integers.t().contiguous() # (rows, cols_groups)

def _write_linear_block_naive(f, layer, name="layer"):
    if hasattr(layer, "qweight"):
        rows = layer.out_features
        cols = layer.in_features
        
        # --- 1. 获取顺序的整数权重 (0-15) ---
        q_int = unpack_awq_to_naive_integers(layer) # (rows, cols)
        
        # --- 2. 打包为 Naive 字节流 (1字节=2个权重) ---
        # 格式：byte = (W1 << 4) | W0
        q_flat = q_int.flatten().numpy()
        # 确保数据是偶数，方便两两打包
        if q_flat.size % 2 != 0:
            q_flat = np.append(q_flat, 0)
            
        low_nibbles = q_flat[0::2] & 0x0F
        high_nibbles = (q_flat[1::2] & 0x0F) << 4
        q_packed_bytes = (low_nibbles | high_nibbles).astype(np.uint8).tobytes()
        
        # --- 3. 获取 Scales (FP32) ---
        s_data = layer.scales.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
        
        # --- 4. 获取顺序的 Zeros (1字节=2个zero) ---
        z_int = unpack_awq_zeros_to_naive(layer) # (rows, n_groups)
        z_flat = z_int.flatten().numpy()
        if z_flat.size % 2 != 0:
            z_flat = np.append(z_flat, 0)
        z_packed_bytes = ((z_flat[1::2] << 4) | (z_flat[0::2])).astype(np.uint8).tobytes()

        print(f"  导出 [Naive AWQ4] {name}: {rows}x{cols} | Q_Bytes:{len(q_packed_bytes)} S_Bytes:{len(s_data)} Z_Bytes:{len(z_packed_bytes)}")
        
        _write_block_descriptor(f, BLOCK_TYPE_AWQ4_NAIVE, rows, cols, len(q_packed_bytes), len(s_data), len(z_packed_bytes))
        f.write(q_packed_bytes)
        f.write(s_data)
        f.write(z_packed_bytes)
    else:
        # FP32 兼容逻辑 (lm_head)
        weight = layer.weight.detach().contiguous().to(torch.float32).cpu()
        w_data = weight.numpy().tobytes()
        rows, cols = weight.shape
        print(f"  导出 [FP32] {name}: {rows}x{cols} | Size:{len(w_data)}")
        _write_block_descriptor(f, BLOCK_TYPE_FP32, rows, cols, len(w_data))
        f.write(w_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_bin", type=str, required=True)
    parser.add_argument("--group_size", type=int, default=128)
    args = parser.parse_args()

    print("加载并量化模型 (CPU)...")
    model = AutoAWQForCausalLM.from_pretrained(args.model_path, device_map="cpu", low_cpu_mem_usage=True, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 校准数据
    calib_data = ["Hello, KuiperLLama is here.", "Coding is fun!"] * 16
    quant_config = {"zero_point": True, "q_group_size": args.group_size, "w_bit": 4, "version": "GEMM"}
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    llama_for_causal = model.model
    llama_model = llama_for_causal.model
    hf_cfg = llama_for_causal.config
    layers = list(llama_model.layers)

    # 强制限制 seq_len 防止显存爆炸
    seq_len = 2048
    vocab_size_flag = -abs(int(hf_cfg.vocab_size))

    with open(args.output_bin, "wb") as f:
        # 1. Header
        f.write(struct.pack("<7i", hf_cfg.hidden_size, hf_cfg.intermediate_size, len(layers), 
                            hf_cfg.num_attention_heads, getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads), 
                            vocab_size_flag, seq_len))
        _write_i32(f, args.group_size)

        # 2. 权重块 (线性层)
        module_map = [
            ("WQ", [l.self_attn.q_proj for l in layers]),
            ("WK", [l.self_attn.k_proj for l in layers]),
            ("WV", [l.self_attn.v_proj for l in layers]),
            ("WO", [l.self_attn.o_proj for l in layers]),
            ("W1", [l.mlp.gate_proj for l in layers]),
            ("W2", [l.mlp.down_proj for l in layers]),
            ("W3", [l.mlp.up_proj for l in layers]),
        ]

        for name, proj_list in module_map:
            for i, proj in enumerate(proj_list):
                _write_linear_block_naive(f, proj, name=f"{name}_{i}")

        _write_linear_block_naive(f, llama_for_causal.lm_head, name="lm_head")

        # 3. FP32 数组 (Embedding/Norms)
        print("写入 FP32 层...")
        _write_tensor_as(f, llama_model.embed_tokens.weight, torch.float32)
        for l in layers: _write_tensor_as(f, l.input_layernorm.weight, torch.float32)
        for l in layers: _write_tensor_as(f, l.post_attention_layernorm.weight, torch.float32)
        _write_tensor_as(f, llama_model.norm.weight, torch.float32)

    print(f"导出完成：{args.output_bin}")

if __name__ == "__main__":
    main()