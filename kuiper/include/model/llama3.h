#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include <unordered_set>
#include "base/buffer.h"
#include "model.h"
#include "model/quant_cursor.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"

// 根据model读取的权重，其内部构建了llama模型的结构，其知道每一层拿多少，把权重信息分给每一层

// 实现了llama 2/3语言模型的推理引擎

// 定义模型的结构和接口
namespace model {
// 结构体  存储了LLAMA模型的所有层
struct LLama2Layers {
  // 基础运算层
  std::shared_ptr<op::Layer> add_layer_;     // 向量加法 參差连接
  std::shared_ptr<op::Layer> rope_layer_;    // 旋转位置编码
  std::shared_ptr<op::Layer> swiglu_layer_;  // SwiGLU激活函数
  std::shared_ptr<op::Layer> mha_layer_;     // 多投注意力
  // 注意力机制的权重矩阵
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;
  // 前馈网络的权重矩阵
  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;  // 分类输出层

  std::shared_ptr<op::Layer> embedding_layer_;  // 词嵌入输出层

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};
// 主控制器 负责加载模型权重，管理内存以及执行推理流程
class LLama2Model : public Model {
 public:
  explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model,
                       QuantFormat quant_format = QuantFormat::kInt8Q8);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  base::Status prefill(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  // ===== Speculative decode helpers =====
  struct BatchVerifyResult {
    bool all_accepted = true;
    int32_t accepted_prefix_len = 0;
    int32_t mismatch_large_token = -1;  // only valid when all_accepted=false
  };

  int32_t decode_one_greedy(int32_t pos, int32_t input_token);
  std::vector<int32_t> draft_block_greedy(int32_t start_pos, int32_t last_token, int32_t k);
  BatchVerifyResult verify_draft_batch_block(int32_t start_pos, int32_t last_token,
                                             const std::vector<int32_t>& draft_tokens);

  // 兼容旧调用
  BatchVerifyResult verify_draft_batch(int32_t start_pos, int32_t last_token,
                                       const std::vector<int32_t>& draft_tokens);

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_qkv_block(int32_t layer_idx, const tensor::Tensor& pos_tensor,
                           const tensor::Tensor& rmsnorm_output_2d,
                           const tensor::Tensor& query_2d) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<LLama2Layers> llama_layers_;
};
}  // namespace model

#endif