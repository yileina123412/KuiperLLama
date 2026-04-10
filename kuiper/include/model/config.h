#ifndef KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
namespace model {
struct ModelConfig {
  int32_t dim = 0;         // 主要维度  决定了每个token的向量表示大小
  int32_t hidden_dim = 0;  // 前馈网络中的中间层维度
  // 模型结构
  int32_t layer_num = 0;    // Transformer层数 如Llama2-7B有32层，Llama2-70B有80层
  int32_t head_num = 0;     // 注意力头的总数
  int32_t kv_head_num = 0;  // Key-Value缓存的头数(用于GQA - Grouped Query Attention)
  int32_t vocab_size = 0;   // 词汇表大小
  int32_t seq_len = 0;      // 最大序列长度
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;
#endif
};
// 内部计算用的详细配置
struct TransformerConfig {
  int32_t kv_dim_ = 0;      // K/V 向量拼起来的总维度
  int32_t kv_mul_ = 0;      //
  int32_t head_size_ = 0;   // 每个 head 的维度（每个头的通道数）
  int32_t vocab_size_ = 0;  // 词表大小（用于 embedding 和 lm head 输出维度）。

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;
#endif
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
