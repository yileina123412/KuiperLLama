import argparse
import struct
import torch
import numpy as np
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

"""
KuiperLLama Naive AWQ4 V2 导出脚本
修正点：
1. 撤销了 AWQ GEMM 专有的 [0, 4, 1, 5, 2, 6, 3, 7] 重排。
2. 统一了 qweight/scales/qzeros 的坐标系为 (Rows, Cols)。
3. 强制 1 字节 = 2 个权重的 Naive 打包。
"""

BLOCK_TYPE_FP32 = 0
BLOCK_TYPE_AWQ4_NAIVE = 1

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

def unpack_awq_to_naive(layer):
    """撤销 AWQ 的顺序置换，还原为正常的 (Rows, Cols) 矩阵"""
    qweight = layer.qweight.cpu()
    rows = layer.out_features
    cols = layer.in_features
    # AWQ GEMM 顺序映射表
    reverse_order = [0, 4, 1, 5, 2, 6, 3, 7]
    
    # 逻辑：AutoAWQ 存储是 (cols, rows/8)
    # 每一个 int32 的 8 个 nibble 对应 rows 方向的 8 个不同位置
    q_int = torch.zeros((cols, rows), dtype=torch.uint8)
    for i, start_row in enumerate(reverse_order):
        # 提取第 i 个 nibble，填入 rows 方向对应的位置
        q_int[:, start_row::8] = (qweight >> (i * 4)) & 0xF
    
    return q_int.t().contiguous() # 返回 (Rows, Cols)

def unpack_zeros_to_naive(layer):
    """撤销 qzeros 的顺序置换"""
    qzeros = layer.qzeros.cpu()
    rows = layer.out_features
    cols_g = qzeros.shape[0]
    reverse_order = [0, 4, 1, 5, 2, 6, 3, 7]
    
    z_int = torch.zeros((cols_g, rows), dtype=torch.uint8)
    for i, start_row in enumerate(reverse_order):
        z_int[:, start_row::8] = (qzeros >> (i * 4)) & 0xF
        
    return z_int.t().contiguous() # 返回 (Rows, Groups)

def _write_linear_block_naive(f, layer, name="layer"):
    if hasattr(layer, "qweight"):
        rows, cols = layer.out_features, layer.in_features
        
        # 1. 权重打包 (1字节=2个权重)
        q_int = unpack_awq_to_naive(layer).flatten().numpy()
        low = q_int[0::2] & 0x0F
        high = (q_int[1::2] & 0x0F) << 4
        q_bytes = (low | high).astype(np.uint8).tobytes()
        
        # 2. Scales (转置为 Rows 方向优先，方便 C++ 寻址)
        # 原本是 (Groups, Rows)，转为 (Rows, Groups)
        s_data = layer.scales.t().detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
        
        # 3. Zeros 打包 (1字节=2个zero)
        z_int = unpack_zeros_to_naive(layer).flatten().numpy()
        z_low = z_int[0::2] & 0x0F
        z_high = (z_int[1::2] & 0x0F) << 4
        z_bytes = (z_low | z_high).astype(np.uint8).tobytes()

        print(f"  [AWQ4-NAIVE] {name}: {rows}x{cols} | Q:{len(q_bytes)} S:{len(s_data)} Z:{len(z_bytes)}")
        _write_block_descriptor(f, BLOCK_TYPE_AWQ4_NAIVE, rows, cols, len(q_bytes), len(s_data), len(z_bytes))
        f.write(q_bytes)
        f.write(s_data)
        f.write(z_bytes)
    else:
        # FP32 兼容逻辑
        weight = layer.weight.detach().contiguous().to(torch.float32).cpu()
        w_data = weight.numpy().tobytes()
        r, c = weight.shape
        print(f"  [FP32] {name}: {r}x{c} | Size:{len(w_data)}")
        _write_block_descriptor(f, BLOCK_TYPE_FP32, r, c, len(w_data))
        f.write(w_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HF模型路径")
    parser.add_argument("--output_bin", type=str, required=True, help="输出.bin路径")
    parser.add_argument("--group_size", type=int, default=128)
    args = parser.parse_args()

    # 步骤：加载 -> 量化 -> 导出
    print("正在加载并进行 AWQ 4-bit 量化...")
    model = AutoAWQForCausalLM.from_pretrained(args.model_path, device_map="cpu", low_cpu_mem_usage=True, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    calib_data = ["Hello, KuiperLLama!", "The weight is now ordered."] * 16
    model.quantize(tokenizer, quant_config={"zero_point":True, "q_group_size":args.group_size, "w_bit":4, "version":"GEMM"}, calib_data=calib_data)

    llama = model.model
    with open(args.output_bin, "wb") as f:
        # 1. 全局头部 (维度, 层数, 头数, 词表负标志, 限制后的SeqLen)
        cfg = llama.config
        f.write(struct.pack("<7i", cfg.hidden_size, cfg.intermediate_size, len(llama.model.layers), 
                            cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), 
                            -abs(cfg.vocab_size), 2048)) # 强限 2048
        _write_i32(f, args.group_size)

        # 2. 顺序写入层
        layers = llama.model.layers
        module_map = [("WQ", [l.self_attn.q_proj for l in layers]), ("WK", [l.self_attn.k_proj for l in layers]),
                      ("WV", [l.self_attn.v_proj for l in layers]), ("WO", [l.self_attn.o_proj for l in layers]),
                      ("W1", [l.mlp.gate_proj for l in layers]), ("W2", [l.mlp.down_proj for l in layers]), ("W3", [l.mlp.up_proj for l in layers])]
        
        for name, projs in module_map:
            for i, p in enumerate(projs): _write_linear_block_naive(f, p, f"{name}_{i}")

        _write_linear_block_naive(f, llama.lm_head, "lm_head")

        print("写入 Embedding 和 Norms...")
        _write_tensor_as(f, llama.model.embed_tokens.weight, torch.float32)
        for l in layers: _write_tensor_as(f, l.input_layernorm.weight, torch.float32)
        for l in layers: _write_tensor_as(f, l.post_attention_layernorm.weight, torch.float32)
        _write_tensor_as(f, llama.model.norm.weight, torch.float32)

    print(f"\n[大功告成] 文件已导出: {args.output_bin}")

if __name__ == "__main__":
    main()