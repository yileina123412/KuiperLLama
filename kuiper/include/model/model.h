#ifndef KUIPER_INCLUDE_MODEL_MODEL_H_
#define KUIPER_INCLUDE_MODEL_MODEL_H_
#include <op/embedding.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"
// 核心作用：读取.bin文件，把文件映射到一块虚拟内存，并把指针交给下一个部门。是读取文件的作用。
/**
 * 作用
 * 加载模型文件，把.bin文件映射到内存
 * 管理模型配置 存储模型的各种参数：维度，层数等
 * 管理层和缓存 统一管理所有计算层和临时数据
 * 提供推理接口：封装推理的主要流程
 */

namespace model {
class Model {
 public:
  enum class QuantFormat : uint8_t {
    kNone = 0,
    kInt8Q8 = 1,
    kAwq4 = 2,  // 预留。姑且先留着吧
    kSQ4 = 3,   // SmoothQuant + RTN 4bit（export_smooth_rtn_pro_v3.py）
  };

  QuantFormat quant_format() const { return quant_format_; }
  // 构造函数  初始化模型的基本信息，设置模型类型，文件路径以及是否量化等参数
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model,
                 QuantFormat quant_format = QuantFormat::kInt8Q8);

  virtual base::Status init(base::DeviceType device_type) = 0;
  // predict: 完整的预测流程（包含后处理）
  virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               bool is_prompt, int& next) const = 0;
  // 纯前向计算
  virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;
  // prefill阶段和decode阶段
  virtual base::Status prefill(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;
  // 管理模型运行的各种缓存
  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;
  //  判断当前Token是否为句子结束标记
  virtual bool is_sentence_ending(int32_t token_idx) const;
  // 将Token ID（序列）转换回文本
  virtual std::string decode(int32_t token_idx) const;

  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  /////////////////////////////////////////////////////
  // 输入文本转换为Token ID序列
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  // kv 滑动窗口
  void set_kv_window_size(int32_t kv_window_size);
  int32_t kv_window_size() const;
  int64_t kv_total_tokens() const;  // 已写入的token总数
  void set_kv_total_tokens(int64_t n);
  void advance_kv_total_tokens(int32_t delta);
  void reset_kv_total_tokens();
  int32_t kv_valid_token_num() const;

  // prefix keep tokens: 前 P 个逻辑 token 固定保留，不被覆盖
  void set_kv_prefix_keep_tokens(int32_t n);
  int32_t kv_prefix_keep_tokens() const;

  // 投机解码
  // 一次性配置运行时 KV 参数（最小必要接口，供 demo 统一调用）
  void configure_kv_runtime(int32_t kv_window_size, int32_t kv_prefix_keep_tokens,
                            bool reset_kv_total_tokens = true);

  // ===== Speculative decode: KV checkpoint / rollback =====
  struct KVCheckpoint {
    int64_t kv_total_tokens = 0;
    tensor::Tensor key_cache;
    tensor::Tensor value_cache;
    bool valid = false;
  };

  KVCheckpoint kv_checkpoint() const;
  void kv_rollback(const KVCheckpoint& checkpoint);
  void kv_commit(const KVCheckpoint& checkpoint);

  // 轻量 checkpoint：只保存逻辑 token 计数（不拷贝整块KV）
  struct KVTokenCheckpoint {
    int64_t kv_total_tokens = 0;
  };
  KVTokenCheckpoint kv_token_checkpoint() const;
  void kv_token_rollback(const KVTokenCheckpoint& checkpoint);

  // 从KV Cache中切片获取特定层和位置的键值对
  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) const;

  // 从KV Cache中切片获取特定层、连续区间[start_pos, start_pos+token_num)的键值对（2D view）
  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache_block(int32_t layer_idx,
                                                                         int32_t start_pos,
                                                                         int32_t token_num) const;
  // Token序列转换为embedding向量
  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;
  // 将embedding输出填充为模型输入张量
  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

 protected:
  // 管理模型运行的各种缓存
  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);
  // 模型文件读取函数
  // 打开模型权重文件，读取模型配置信息，用mmap把整个文件映射到虚拟内存，设置起始指针
  virtual base::Status read_model_file();
  // 根据tokenizer类型创建相应的编码层
  virtual base::Status create_encode_layer();

  virtual base::Status gen_model_from_file();
  // 根据模型文件生成模型的详细参数  设置维度信息，派生参数，验证词表大小一致性
  virtual base::Status generate_model_infos(const ModelConfig& config) const;
  // 推理后处理（如softmax、采样等）
  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

  // kv滑动窗口
  int32_t logical_to_kv_slot(int64_t logical_pos) const;
  int32_t effective_kv_window_size() const;
  int32_t effective_kv_prefix_keep_tokens() const;

 private:
  virtual void init_mem() = 0;
  // 层的管理和创建
  virtual base::Status create_layers() = 0;

  virtual void create_param_layers() = 0;

  virtual void create_nonparam_layers() = 0;

  virtual void create_param_quant_layers() = 0;

 protected:
  // 模型配置
  // 量化分组大小 每个组内的权重个数
  int32_t group_size_ = 1;
  // 是否为量化模型
  bool is_quant_model_ = false;

  QuantFormat quant_format_ = QuantFormat::kNone;

  // 模型的配置信息  维度，层数，头数
  std::unique_ptr<TransformerConfig> config_;

  // 路径
  std::string token_path_;  // Token词表文件路径
  std::string model_path_;  // 模型权重文件路径
  // 核心组件
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;  // Token编解码层
  std::map<ModelBufferType, tensor::Tensor> buffers_;  // 缓存管理
  std::unique_ptr<sampler::Sampler> sampler_;          // 采样器
  std::shared_ptr<RawModelData> raw_model_data_;       // 原始模型数据
  // 设备信息
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;           // 设备类型
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;           // 模型类型
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;  // Tokenizer类型
  // kv滑动窗口
  int32_t kv_window_size_ = 0;   // 默认等于seq_len_
  int64_t kv_total_tokens_ = 0;  // 逻辑上已写入的token总数
  int32_t kv_prefix_keep_tokens_ = 0;
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_MODEL_H_
