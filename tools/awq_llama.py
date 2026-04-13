import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

# 1. 配置路径
model_path = "/home/furina/code_learnning/cpp/cuda/KuiperLLama/models/Llama-2-7b-chat-hf"  # 你的原始模型路径
save_path = "/home/furina/code_learnning/cpp/cuda/KuiperLLama/models/Llama-2-7b-chat-awq"              # 量化后临时保存路径

# 2. 量化配置
quant_config = {
    "zero_point": True,    # AWQ 通常使用非对称量化
    "q_group_size": 128,   # 每 128 个权重共用一个 scale
    "w_bit": 4,            # 4-bit 量化
    "version": "GEMM"      # 导出格式
}

print("--- 开始加载模型并搜索最优系数 ---")
# 加载模型和分词器
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 执行量化（这一步会自动跑校准集，寻找最优 Scale）
model.quantize(tokenizer, quant_config=quant_config)

# 3. 提取并观察系数
print("\n--- 提取每一层的量化数据 ---")

# 我们可以遍历模型，看看第一个 Transformer Block 的 Q 投影层
# 注意：AutoAWQ 会将 Linear 层替换为 WQLinear
layer0 = model.model.layers[0].self_attn.q_proj

print(f"Layer 0 Q_Proj 权重形状 (打包后): {layer0.qweight.shape}")
print(f"Layer 0 Q_Proj Scale 形状: {layer0.scales.shape}")
if hasattr(layer0, 'qzeros'):
    print(f"Layer 0 Q_Proj Zero-points 形状: {layer0.qzeros.shape}")

# 4. 导出逻辑建议
# 你可以参考之前的 export_llama.py 逻辑，在这里遍历 model.model.layers
# 将 qweight.cpu().numpy() 和 scales.cpu().numpy() 按照顺序写入二进制文件