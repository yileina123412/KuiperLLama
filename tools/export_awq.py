import argparse
import struct

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


"""\
KuiperLLama AWQ4 导出说明（仅导出；C++ 端需要配套实现 AWQ4 MatMul 读取/计算）：

目标：尽可能贴合 KuiperLLama 现有量化 `.bin` 的加载流程：
  - C++ 端 `Model::read_model_file()` 会先读 `ModelConfig`（7 个 int32），
    然后在 `is_quant_model_==true` 时再读一个 int32 `group_size_`。
  - 随后使用 mmap，并把 `weight_data` 指向 (header + group_size) 后的权重区域。

因此，本脚本写出的二进制布局为：

  [ModelConfig(7*i32)]
  [group_size(i32)]
  ======= weights region (byte stream) =======
  1) 所有层的 WQ (按 layer 0..L-1)
  2) 所有层的 WK
  3) 所有层的 WV
  4) 所有层的 WO
  5) 所有层的 W1 (gate_proj)
  6) 所有层的 W2 (down_proj)
  7) 所有层的 W3 (up_proj)
  8) WCLS / lm_head
  9) Embedding (fp32)
 10) Attention RMSNorm per-layer (fp32)
 11) FFN RMSNorm per-layer (fp32)
 12) Final RMSNorm (fp32)

其中每个 AWQ 线性层按顺序写入 3 个张量：
  - qweight: torch.int32 (AutoAWQ GEMM packed format)
  - scales : 写成 torch.float32（为了后续 C++ 端实现更省心；也更贴合现有 int8 scale 读取方式）
  - qzeros : torch.int32

注意：
  - 这里不尝试把 qweight 解包成 int8（避免误解 AutoAWQ 的 packing 细节）。
    C++ 端应实现与 AutoAWQ(GEMM) 对齐的解包/矩阵乘。
  - 为了避开 KuiperLLama 现有 `create_param_quant_layers()` 在 shared-weight 分支的
    pos/weight_ptr 重叠问题，这里强制把 vocab_size 写成负数（等价于“非共享 head”标志）。
"""


def _write_tensor_raw(f, tensor: torch.Tensor) -> None:
    """Write tensor bytes in its current dtype/layout."""
    t = tensor.detach().contiguous().cpu()
    f.write(t.numpy().tobytes())


def _write_tensor_as(f, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    """Convert tensor to dtype on CPU then write bytes."""
    t = tensor.detach().contiguous().to(dtype=dtype).cpu()
    f.write(t.numpy().tobytes())


def _write_i32(f, v: int) -> None:
    f.write(struct.pack("<i", int(v)))


def _write_model_config_header(
    f,
    *,
    dim: int,
    hidden_dim: int,
    layer_num: int,
    head_num: int,
    kv_head_num: int,
    vocab_size: int,
    seq_len: int,
) -> None:
    # Must match `model::ModelConfig` memory layout for non-QWEN3 builds:
    # int32 dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len
    f.write(struct.pack("<7i", dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len))


def _write_awq_linear(f, wq_linear_layer) -> None:
    """Write one AutoAWQ WQLinear in (qweight, scales(fp32), qzeros) order."""
    _write_tensor_raw(f, wq_linear_layer.qweight)  # int32 packed
    _write_tensor_as(f, wq_linear_layer.scales, torch.float32)
    _write_tensor_raw(f, wq_linear_layer.qzeros)   # int32 packed


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AWQ 4-bit model to KuiperLLama-style .bin")
    parser.add_argument("--model_path", type=str, required=True, help="HF model directory")
    parser.add_argument("--output_bin", type=str, required=True, help="Output .bin path")
    parser.add_argument("--group_size", type=int, default=128, help="AWQ q_group_size")
    parser.add_argument("--seqlen", type=int, default=256, help="Calibration seqlen")
    parser.add_argument("--n_utils", type=int, default=32, help="AWQ calibration utility batches")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    print("正在加载模型并进行 AWQ 4-bit 校准搜索...（可能需要较长时间）")

    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        device_map="cpu",
        # device_map=args.device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    quant_config = {
        "zero_point": True,
        "q_group_size": int(args.group_size),
        "w_bit": 4,
        "version": "GEMM",
    }
    # 在调用 quantize 之前，手动准备一小段文本作为校准数据
    calib_data = [
        "Hello, how are you today?",
        "The capital of France is Paris.",
        "Llama is a great large language model.",
        "Artificial intelligence is changing the world."
    ] * 8  # 简单凑够 32 条数据
    # model.quantize(tokenizer, quant_config=quant_config, max_calib_samples=int(args.n_utils), max_calib_seq_len=int(args.seqlen))
    model.quantize(
        tokenizer, 
        quant_config=quant_config, 
        calib_data=calib_data,  # 传入本地数据，这样它就不会去联网下载了
        max_calib_samples=len(calib_data), 
        max_calib_seq_len=int(args.seqlen)
    )
    llama_model = model.model
    hf_cfg = llama_model.config

    dim = int(getattr(hf_cfg, "hidden_size"))
    hidden_dim = int(getattr(hf_cfg, "intermediate_size"))
    layer_num = int(getattr(hf_cfg, "num_hidden_layers"))
    head_num = int(getattr(hf_cfg, "num_attention_heads"))
    kv_head_num = int(getattr(hf_cfg, "num_key_value_heads", head_num))
    vocab_size = int(getattr(hf_cfg, "vocab_size"))
    seq_len = int(getattr(hf_cfg, "max_position_embeddings", 0))
    if seq_len <= 0:
        raise ValueError("Cannot determine seq_len from model config (max_position_embeddings missing/<=0)")

    # KuiperLLama 内部用 vocab_size 正负来表示是否共享 classifier 权重。
    # 量化分支下 shared-weight 的 pos 推进存在潜在重叠风险，因此这里强制写成负数。
    vocab_size_flag = -abs(vocab_size)

    print("量化完成，开始按 KuiperLLama(quant) 权重顺序导出...")
    print(f"  dim={dim}, hidden_dim={hidden_dim}, layers={layer_num}, heads={head_num}, kv_heads={kv_head_num}, vocab={vocab_size}, seq_len={seq_len}")

    # layers = list(llama_model.layers)
    layers = list(llama_model.model.layers)
    if len(layers) != layer_num:
        print(f"[WARN] config num_hidden_layers={layer_num} but model.layers={len(layers)}")

    with open(args.output_bin, "wb") as f:
        # 0) header
        _write_model_config_header(
            f,
            dim=dim,
            hidden_dim=hidden_dim,
            layer_num=len(layers),
            head_num=head_num,
            kv_head_num=kv_head_num,
            vocab_size=vocab_size_flag,
            seq_len=seq_len,
        )
        _write_i32(f, int(args.group_size))

        # 1) WQ (q_proj)
        for i, layer in enumerate(layers):
            print(f"导出 WQ layer {i}")
            _write_awq_linear(f, layer.self_attn.q_proj)
        # 2) WK (k_proj)
        for i, layer in enumerate(layers):
            print(f"导出 WK layer {i}")
            _write_awq_linear(f, layer.self_attn.k_proj)
        # 3) WV (v_proj)
        for i, layer in enumerate(layers):
            print(f"导出 WV layer {i}")
            _write_awq_linear(f, layer.self_attn.v_proj)
        # 4) WO (o_proj)
        for i, layer in enumerate(layers):
            print(f"导出 WO layer {i}")
            _write_awq_linear(f, layer.self_attn.o_proj)

        # 5) W1 (gate_proj)
        for i, layer in enumerate(layers):
            print(f"导出 W1(gate) layer {i}")
            _write_awq_linear(f, layer.mlp.gate_proj)
        # 6) W2 (down_proj)
        for i, layer in enumerate(layers):
            print(f"导出 W2(down) layer {i}")
            _write_awq_linear(f, layer.mlp.down_proj)
        # 7) W3 (up_proj)
        for i, layer in enumerate(layers):
            print(f"导出 W3(up) layer {i}")
            _write_awq_linear(f, layer.mlp.up_proj)

        # 8) WCLS / lm_head
        print("导出 WCLS(lm_head)...")
        target_head = model.model.lm_head
        # AutoAWQ 会把 Linear 替换成 WQLinear，lm_head 可能仍然是 Linear（取决于版本/配置）。
        if hasattr(target_head, "qweight"):
            _write_awq_linear(f, target_head)
            # _write_awq_linear(f, model.lm_head)
        else:
            print("  lm_head 未量化，按 FP32 格式导出")
            _write_tensor_as(f, target_head.weight, torch.float32)

        # 9) Embedding (fp32)
        print("导出 Embedding(fp32)...")
        # _write_tensor_as(f, llama_model.embed_tokens.weight, torch.float32)
        _write_tensor_as(f, llama_model.model.embed_tokens.weight, torch.float32)

        # 10) Attention RMSNorm per layer (fp32)
        print("导出 Attention RMSNorm(fp32)...")
        for layer in layers:
            # _write_tensor_as(f, layer.input_layernorm.weight, torch.float32)
            # _write_tensor_as(f, llama_model.model.norm.weight, torch.float32)
            _write_tensor_as(f, layer.input_layernorm.weight, torch.float32)

        # 11) FFN RMSNorm per layer (fp32)
        print("导出 FFN RMSNorm(fp32)...")
        for layer in layers:
            _write_tensor_as(f, layer.post_attention_layernorm.weight, torch.float32)

        # 12) Final RMSNorm (fp32)
        print("导出 Final RMSNorm(fp32)...")
        _write_tensor_as(f, llama_model.model.norm.weight, torch.float32)

    print(f"\n完成：AWQ 4-bit KuiperLLama bin 已导出至: {args.output_bin}")


if __name__ == "__main__":
    main()