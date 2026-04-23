#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model, QuantFormat quant_format)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {
  if (!is_quant_model_) {
    quant_format_ = QuantFormat::kNone;
  } else {
    // 兼容：如果调用方没传/误传 kNone，就仍然按原来的 int8 走
    quant_format_ = (quant_format == QuantFormat::kNone) ? QuantFormat::kInt8Q8 : quant_format;
  }
}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

void Model::set_kv_window_size(int32_t kv_window_size) {
  CHECK_GT(kv_window_size, 0);
  kv_window_size_ = kv_window_size;
}

int32_t Model::kv_window_size() const { return kv_window_size_; }

int64_t Model::kv_total_tokens() const { return kv_total_tokens_; }

void Model::set_kv_total_tokens(int64_t total_tokens) {
  CHECK_GE(total_tokens, 0);
  kv_total_tokens_ = total_tokens;
}

void Model::advance_kv_total_tokens(int32_t delta) {
  CHECK_GE(delta, 0);
  kv_total_tokens_ += delta;
}
void Model::reset_kv_total_tokens() { kv_total_tokens_ = 0; }
int32_t Model::kv_valid_token_num() const {
  const int32_t w = kv_window_size();
  if (w <= 0) return 0;
  return kv_total_tokens_ < w ? static_cast<int32_t>(kv_total_tokens_) : w;
}

// int32_t Model::logical_to_kv_slot(int64_t logical_pos) const {
//   CHECK_GE(logical_pos, 0);
//   const int32_t window = effective_kv_window_size();
//   return static_cast<int32_t>(logical_pos % window);
// }
// 重新计算位置
int32_t Model::logical_to_kv_slot(int64_t logical_pos) const {
  CHECK_GE(logical_pos, 0);
  const int32_t window = effective_kv_window_size();
  const int32_t prefix = effective_kv_prefix_keep_tokens();
  // 没有设置前缀
  if (prefix <= 0) {
    return static_cast<int32_t>(logical_pos % window);
  }
  if (logical_pos < prefix) {
    return static_cast<int32_t>(logical_pos);
  }
  const int32_t ring = window - prefix;
  CHECK_GT(ring, 0);
  const int64_t rel = logical_pos - prefix;
  return prefix + static_cast<int32_t>(rel % ring);
}

int32_t Model::effective_kv_window_size() const {
  CHECK(config_ != nullptr);
  CHECK_GT(config_->seq_len_, 0);

  if (kv_window_size_ <= 0) {
    return config_->seq_len_;
  }
  return kv_window_size_ < config_->seq_len_ ? kv_window_size_ : config_->seq_len_;
}

void Model::set_kv_prefix_keep_tokens(int32_t n) {
  CHECK_GE(n, 0);
  kv_prefix_keep_tokens_ = n;
}

int32_t Model::kv_prefix_keep_tokens() const { return kv_prefix_keep_tokens_; }

int32_t Model::effective_kv_prefix_keep_tokens() const {
  const int32_t window = effective_kv_window_size();
  if (window <= 1) return 0;
  int32_t p = kv_prefix_keep_tokens_;
  if (p < 0) p = 0;
  if (p >= window) p = window - 1;  // 至少给 tail 留 1 个槽位
  return p;
}

// 读取模型文件
base::Status Model::read_model_file() {
  using namespace base;
  // 检查模型路径是否存在
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  // 打开文件获取文件描述符，用于后续mmap操作
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }
  // 用FILE*打开，用于读取头部配置
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }
  // 读取模型基本配置
  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }
  // 如果是量化模型，还要读取量化参数
  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
  }

  // 生成模型详细参数，将读取的配置转换为内部使用的详细参数
  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }
  // 创建原始数据对象
  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }
  // 获取文件大小信息  mmap需要知道映射多少字节，用于后续的内存管理和边界检查
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  raw_model_data_->file_size = sb.st_size;
  // 核心：mmap映射
  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }
  // 设置权重起始位置
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
  }
  // 最终验证
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
  config_->immediate_dim_ = config.immediate_dim_;
#endif
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}
// 创建encode
base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}
// 从KV Cache中切片获取特定层和位置的键值对
/**
 * 对于每个位置 token_pos，都要写入一个新的 K,V
这意味着函数假设：每次调用都会生成新的 K,V
但问题是：Prefill 时都已经算过并存储了，Decode 时又要再算一遍！
 */

// 环形槽位
std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, config_->layer_num_);
  CHECK_GE(token_pos, 0);
  const int32_t slot = logical_to_kv_slot(static_cast<int64_t>(token_pos));
  // 本层的序号
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  // 当前位置
  int32_t cache_offset = layer_offset + slot * config_->kv_dim_;
  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache_block(int32_t layer_idx,
                                                                      int32_t start_pos,
                                                                      int32_t token_num) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, config_->layer_num_);
  CHECK_GE(start_pos, 0);
  CHECK_GT(token_num, 0);

  const int32_t window = effective_kv_window_size();
  const int32_t prefix = effective_kv_prefix_keep_tokens();

  CHECK_GT(window, 0);

  // 环形起点槽位
  const int32_t slot_start = logical_to_kv_slot(static_cast<int64_t>(start_pos));

  // 只返回“物理连续首段”
  int32_t first_chunk = token_num;
  if (prefix > 0 && start_pos < prefix) {
    // 逻辑还在前缀区：连续到 prefix 结束
    first_chunk = std::min(first_chunk, prefix - start_pos);
  } else {
    // 在尾部
    first_chunk = std::min(first_chunk, window - slot_start);
  }
  CHECK_GT(first_chunk, 0);

  const int64_t layer_stride = static_cast<int64_t>(config_->seq_len_) * config_->kv_dim_;
  const int64_t layer_offset = static_cast<int64_t>(layer_idx) * layer_stride;
  const int64_t cache_offset = layer_offset + static_cast<int64_t>(slot_start) * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, first_chunk, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, first_chunk, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
#if defined(QWEN3_SUPPORT)
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      config_->hidden_dim_ * sizeof(float), nullptr,
      input_embeddings.ptr<float>(index * config_->hidden_dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->hidden_dim_);

#else
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * config_->dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
#endif
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model