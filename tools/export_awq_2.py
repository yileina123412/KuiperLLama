import argparse
import struct
import torch
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

"""
KuiperLLama AWQ4 增强版导出脚本

二进制布局 (Binary Layout):
1. Global Header (28 bytes): 7个 int32 (dim, hidden_dim, layers, heads, kv_heads, vocab, seq_len)
2. Group Size (4 bytes): 1个 int32
3. Weights Region:
   每个线性层块都包含一个 [Block Descriptor (32 bytes)] + [实际数据数据]
   顺序如下：
   - 32层 WQ, 32层 WK, 32层 WV, 32层 WO
   - 32层 W1(gate), 32层 W2(down), 32层 W3(up)
   - lm_head
   - Embedding (FP32)
   - 32层 AttentionNorm, 32层 FFNNorm, FinalNorm (FP32)
"""

# 定义块类型，方便 C++ 识别
BLOCK_TYPE_FP32 = 0
BLOCK_TYPE_AWQ4 = 1

def _write_i32(f, v: int) -> None:
    f.write(struct.pack("<i", int(v)))

def _write_tensor_as(f, tensor: torch.Tensor, dtype: torch.dtype) -> int:
    """写入张量数据并返回字节数"""
    t = tensor.detach().contiguous().to(dtype=dtype).cpu()
    data = t.numpy().tobytes()
    f.write(data)
    return len(data)

def _write_block_descriptor(f, b_type, rows, cols, q_size, s_size=0, z_size=0):
    """写入8个int32的描述符，总计32字节"""
    # [类型, 行, 列, q字节数, s字节数, z字节数, 预留, 预留]
    desc = struct.pack("<8i", b_type, rows, cols, q_size, s_size, z_size, 0, 0)
    f.write(desc)

def _write_linear_block(f, layer, name="layer"):
    """
    自描述线性层写入：
    如果是AWQ量化层，写入3段数据；如果是普通FP32层，写入1段。
    """
    if hasattr(layer, "qweight"):
        # 获取打包后的原始字节流
        q_data = layer.qweight.detach().contiguous().cpu().numpy().tobytes()
        s_data = layer.scales.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
        z_data = layer.qzeros.detach().contiguous().cpu().numpy().tobytes()
        
        # 逻辑维度（用于 C++ 初始化算子形状）
        rows = getattr(layer, "out_features", 0)
        cols = getattr(layer, "in_features", 0)

        print(f"  导出 [AWQ4] {name}: {rows}x{cols} | Q:{len(q_data)} S:{len(s_data)} Z:{len(z_data)}")
        _write_block_descriptor(f, BLOCK_TYPE_AWQ4, rows, cols, len(q_data), len(s_data), len(z_data))
        f.write(q_data)
        f.write(s_data)
        f.write(z_data)
    else:
        # 兼容 FP32 权重（如未量化的 lm_head）
        weight = layer.weight.detach().contiguous().to(torch.float32).cpu()
        w_data = weight.numpy().tobytes()
        rows, cols = weight.shape
        print(f"  导出 [FP32] {name}: {rows}x{cols} | Size:{len(w_data)}")
        _write_block_descriptor(f, BLOCK_TYPE_FP32, rows, cols, len(w_data))
        f.write(w_data)

def main():
    parser = argparse.ArgumentParser(description="Export AWQ 4-bit with Block Descriptors")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_bin", type=str, required=True)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--n_utils", type=int, default=32)
    args = parser.parse_args()

    # 1. 加载模型（强制 CPU 避开驱动兼容性问题）
    print("正在加载模型并进行 AWQ 4-bit 校准搜索 (CPU 模式)...")
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, device_map="cpu", low_cpu_mem_usage=True, dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 2. 准备本地校准数据（绕过 HuggingFace 网络连接）
    calib_data = ["Hello, I am KuiperLLama.", "The quick brown fox jumps over the lazy dog."] * 16
    
    # 3. 执行量化
    quant_config = {"zero_point": True, "q_group_size": args.group_size, "w_bit": 4, "version": "GEMM"}
    model.quantize(
        tokenizer, 
        quant_config=quant_config, 
        calib_data=calib_data,
        max_calib_samples=len(calib_data), 
        max_calib_seq_len=args.seqlen
    )

    # 4. 获取内核模型
    llama_for_causal = model.model # LlamaForCausalLM
    llama_model = llama_for_causal.model # LlamaModel
    hf_cfg = llama_for_causal.config
    layers = list(llama_model.layers)

    # 5. 准备 Header 参数
    dim = int(hf_cfg.hidden_size)
    hidden_dim = int(hf_cfg.intermediate_size)
    vocab_size_flag = -abs(int(hf_cfg.vocab_size)) # 负数标志
    seq_len = int(hf_cfg.max_position_embeddings)

    print(f"量化完成，开始写入带有描述符的二进制文件...")

    with open(args.output_bin, "wb") as f:
        # --- 0. 全局头部 ---
        f.write(struct.pack("<7i", dim, hidden_dim, len(layers), hf_cfg.num_attention_heads, 
                            getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads), 
                            vocab_size_flag, seq_len))
        _write_i32(f, args.group_size)

        # --- 1. 线性层模块 (模块化分组) ---
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
                _write_linear_block(f, proj, name=f"{name}_{i}")

        # --- 2. 输出头 (lm_head) ---
        _write_linear_block(f, llama_for_causal.lm_head, name="lm_head")

        # --- 3. 非线性层 (保持 FP32，暂时不加描述符，因为形状固定) ---
        print("写入 Embedding 和 Norms (FP32)...")
        _write_tensor_as(f, llama_model.embed_tokens.weight, torch.float32)
        for i, l in enumerate(layers):
            _write_tensor_as(f, l.input_layernorm.weight, torch.float32)
        for i, l in enumerate(layers):
            _write_tensor_as(f, l.post_attention_layernorm.weight, torch.float32)
        _write_tensor_as(f, llama_model.norm.weight, torch.float32)

    print(f"\n[成功] 文件已导出至: {args.output_bin}")
    print(f"提示：C++端读取线性层时，请先读取32字节的描述符(8个int32)，再根据里面的q_size, s_size, z_size进行内存分配和读取。")

if __name__ == "__main__":
    main()