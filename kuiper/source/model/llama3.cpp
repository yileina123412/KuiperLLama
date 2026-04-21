#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/matmul_sq4.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "base/alloc.h"
#include "base/tick.h"
namespace model {
// 从 2D 张量中提取第 row 行，创建一个 1D 视图（View），避免数据复制。
static tensor::Tensor row_view_fp32(const tensor::Tensor& mat2d, int32_t row, int32_t dim,
                                    base::DeviceType device_type) {
  float* base_ptr = const_cast<float*>(mat2d.ptr<float>(static_cast<int64_t>(row) * dim));
  auto buf = std::make_shared<base::Buffer>(static_cast<size_t>(dim) * sizeof(float), nullptr,
                                            base_ptr, true);
  tensor::Tensor v(base::DataType::kDataTypeFp32, dim);
  CHECK(v.assign(buf));
  v.set_device_type(device_type);
  return v;
}
// 把里面所有层转到cuda
void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }
}

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model, QuantFormat quant_format)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model, quant_format) {}
// 初始化
base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  // mmap映射
  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  // 1.5g
  init_mem();
  // 旋转位置编码提前准备
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status LLama2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);  // ← 关键：无论 Prefill 还是 Decode 都要执行
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

base::Status LLama2Model::prefill(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    if (input.dims_size() == 1) {
      // 原逻辑：decode/prefill(单token) 不动
      attention_rms(layer_idx, input);
      attention_qkv(layer_idx, pos_tensor);
      attention_mha(layer_idx, pos_tensor);
      feed_forward(layer_idx, input);
    } else {
      // 新逻辑：prefill_block（2D）
      const int32_t token_num = input.get_dim(0);

      tensor::Tensor rms2d = get_buffer(ModelBufferType::kOutputRMSNorm2D);
      rms2d.reshape({token_num, config_->dim_});

      tensor::Tensor query2d = get_buffer(ModelBufferType::kQuery2D);
      query2d.reshape({token_num, config_->dim_});

      tensor::Tensor mha2d = get_buffer(ModelBufferType::kOutputMHA2D);
      mha2d.reshape({token_num, config_->dim_});

      tensor::Tensor attn2d = get_buffer(ModelBufferType::kAttnOutput2D);
      attn2d.reshape({token_num, config_->dim_});

      // 2D RMSNorm（RmsNormLayer 已支持 dims_size>1）
      auto rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
      CHECK_NE(rmsnorm_layer, nullptr);
      STATUS_CHECK(rmsnorm_layer->forward(input, rms2d));

      // 2D QKV + 一次写 KV cache
      attention_qkv_block(layer_idx, pos_tensor, rms2d, query2d);

      // 2D MHA：你之前的 MHA 会在 query.dims_size()==2 时走 prefill kernel
      auto mha_layer = llama_layers_->mha_layer_;
      CHECK_NE(mha_layer, nullptr);
      const int32_t start_pos = pos_tensor.index<int32_t>(0);
      std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(start_pos);
      std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);

      tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);  // 占位参数
      tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
      tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
      STATUS_CHECK(mha_layer->forward(query2d, score_storage, key_cache, val_cache, mha2d));

      // 2D WO（Matmul 的 check 你已经放宽到 2D 了）
      const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
      CHECK_NE(wo_layer, nullptr);
      STATUS_CHECK(wo_layer->forward(mha2d, attn2d));

      // ===== FFN (2D) =====
      // residual add: input += attn2d
      CHECK_NE(llama_layers_->add_layer_, nullptr);
      STATUS_CHECK(llama_layers_->add_layer_->forward(input, attn2d, input));

      // 2D ffn rmsnorm
      tensor::Tensor ffn_norm2d = get_buffer(ModelBufferType::kFFNRMSNorm2D);
      ffn_norm2d.reshape({token_num, config_->dim_});
      const auto& ffn_rms = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
      CHECK_NE(ffn_rms, nullptr);
      STATUS_CHECK(ffn_rms->forward(input, ffn_norm2d));

      // 2D W1/W3
      tensor::Tensor w1_out2d = get_buffer(ModelBufferType::kW1Output2D);
      tensor::Tensor w3_out2d = get_buffer(ModelBufferType::kW3Output2D);
      w1_out2d.reshape({token_num, config_->hidden_dim_});
      w3_out2d.reshape({token_num, config_->hidden_dim_});

      const auto& w1 = llama_layers_->w1_layers_.at(layer_idx);
      const auto& w3 = llama_layers_->w3_layers_.at(layer_idx);
      CHECK_NE(w1, nullptr);
      CHECK_NE(w3, nullptr);
      STATUS_CHECK(w1->forward(ffn_norm2d, w1_out2d));
      STATUS_CHECK(w3->forward(ffn_norm2d, w3_out2d));

      // SwiGLU (in-place): w1_out2d = silu(w1_out2d) * w3_out2d
      CHECK_NE(llama_layers_->swiglu_layer_, nullptr);
      STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_out2d, w3_out2d, w1_out2d));

      // 2D W2: hidden -> dim
      tensor::Tensor w2_out2d = get_buffer(ModelBufferType::kW2Output2D);
      w2_out2d.reshape({token_num, config_->dim_});
      const auto& w2 = llama_layers_->w2_layers_.at(layer_idx);
      CHECK_NE(w2, nullptr);
      STATUS_CHECK(w2->forward(w1_out2d, w2_out2d));

      // residual add: input += w2_out2d
      STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_out2d, input));
    }
  }
  if (input.dims_size() == 2) {
    const int32_t token_num = input.get_dim(0);
    tensor::Tensor last = row_view_fp32(input, token_num - 1, config_->dim_, device_type_);
    cls_logits(last);
  } else {
    cls_logits(input);
  }
  return base::error::Success();
}

void LLama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void LLama2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(llama_layers_ != nullptr);
  // 单独写一个分支
  if (quant_format() == QuantFormat::kSQ4) {
    auto cpu_device_type = base::DeviceType::kDeviceCPU;

    const uint8_t* begin = reinterpret_cast<const uint8_t*>(raw_model_data_->weight_data);
    const uint8_t* end =
        reinterpret_cast<const uint8_t*>(raw_model_data_->data) + raw_model_data_->file_size;

    model::QuantBlockCursor cursor(begin, end);

    const int32_t dim = config_->dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    // 读取一个SQ4量化块，并根据反量化策略返回对应的计算层
    // 读取一个SQ4量化块，始终按SQ4权重层构建（不做提前反量化）
    auto read_sq4_matmul = [&](const char* tag, int32_t layer_idx, int32_t rows,
                               int32_t cols) -> std::shared_ptr<op::Layer> {
      (void)tag;
      (void)layer_idx;

      model::QuantBlockPayload payload;
      STATUS_CHECK(cursor.ReadNextSQ4(rows, cols, group_size_, &payload, true));

      op::SQ4WeightView view;
      view.rows = payload.desc.rows;
      view.cols = payload.desc.cols;
      view.group_size = payload.desc.group_size;

      view.qweight_packed = payload.q;
      view.q_bytes = static_cast<size_t>(payload.desc.q_size);

      view.scales = payload.s;
      view.s_bytes = static_cast<size_t>(payload.desc.s_size);

      view.zeros_packed = payload.z;
      view.z_bytes = static_cast<size_t>(payload.desc.z_size);

      auto layer = std::make_shared<op::MatmulSQ4Layer>(device_type_, rows, cols);
      STATUS_CHECK(layer->set_sq4_weight(view, cpu_device_type, true));
      return layer;
    };

    // WQ
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->wq_layers_.push_back(read_sq4_matmul("wq", i, dim, dim));
    }
    // WK
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->wk_layers_.push_back(read_sq4_matmul("wk", i, config_->kv_dim_, dim));
    }
    // WV
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->wv_layers_.push_back(read_sq4_matmul("wv", i, config_->kv_dim_, dim));
    }
    // WO
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->wo_layers_.push_back(read_sq4_matmul("wo", i, dim, dim));
    }
    // W1
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->w1_layers_.push_back(read_sq4_matmul("w1", i, hidden_dim, dim));
    }
    // W2
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->w2_layers_.push_back(read_sq4_matmul("w2", i, dim, hidden_dim));
    }
    // W3
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      llama_layers_->w3_layers_.push_back(read_sq4_matmul("w3", i, hidden_dim, dim));
    }

    // lm_head / cls
    // llama_layers_->cls_layer_ = read_sq4_matmul(config_->vocab_size_, dim);
    if (config_->is_shared_weight_) {
      size_t skipped = 0;
      STATUS_CHECK(cursor.SkipNext(&skipped));  // 仍然消费掉 exporter 写入的 lm_head SQ4 block
      LOG(INFO) << "SQ4: skipped lm_head SQ4 block bytes=" << skipped
                << " (use fp32 tied embedding for cls)";
      // cls_layer_ 会在 embedding weight_ptr 就绪后再绑定
    } else {
      llama_layers_->cls_layer_ = read_sq4_matmul("lm_head", -1, config_->vocab_size_, dim);
    }

    // ---- 常数区（embedding + norms）对齐/边界/校验 ----
    // 接下来应当是 fp32 embedding + rmsnorm
    const size_t off = cursor.offset_bytes_from(begin);
    const size_t align = 16;  // 对应 exporter: _write_align_padding(f, 16)
    const size_t pad = (align - (off % align)) % align;

    const size_t emb_floats =
        static_cast<size_t>(std::abs(config_->vocab_size_)) * static_cast<size_t>(dim);
    const size_t norm_floats =
        static_cast<size_t>(2 * config_->layer_num_ + 1) * static_cast<size_t>(dim);
    const size_t const_bytes = (emb_floats + norm_floats) * sizeof(float);

    CHECK(cursor.remaining_bytes() >= pad + const_bytes)
        << "SQ4: not enough bytes for fp32 constants region";

    // 可选但推荐：验证 padding 全为 0（更容易定位导出/读取顺序错误）
    const uint8_t* p0 = cursor.ptr();
    for (size_t i = 0; i < pad; ++i) {
      CHECK_EQ(p0[i], 0) << "SQ4: non-zero padding byte, bin may be corrupted";
    }

    float* weight_ptr = reinterpret_cast<float*>(const_cast<uint8_t*>(cursor.ptr() + pad));
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight_ptr) % alignof(float), 0)
        << "SQ4: fp32 constants not aligned to float";

    // embedding
    llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
    llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim},
                                                weight_ptr, cpu_device_type);

    if (config_->is_shared_weight_) {
      auto cls = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
      cls->set_weight(0, {config_->vocab_size_, dim}, weight_ptr, cpu_device_type);
      llama_layers_->cls_layer_ = cls;
    }

    weight_ptr += static_cast<size_t>(config_->vocab_size_) * static_cast<size_t>(dim);

    // rmsnorm attention/ffn/final: 2*layer_num + 1
    for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
      auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim);
      rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
      llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
      weight_ptr += dim;
    }
    CHECK(weight_ptr <= reinterpret_cast<float*>(const_cast<uint8_t*>(end)))
        << "SQ4: constants region overflow";

    return;
  }

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  llama_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                              cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

void LLama2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

  const void* weight_embedding = raw_model_data_->weight(0);
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                              weight_embedding, cpu_device_type);

  // create all matmul layer
  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
  // create weight matrix for query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }

  // create weight matrix for key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  pos += config_->seq_len_ * config_->head_size_;

  llama_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(0), cpu_device_type);
  } else {
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(pos), cpu_device_type);
  }

  // create rmsnorm layer
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  // skip attention.wq attention.wk attention.wv attention.wo
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  // skip ffn.w1 ffn.w2 ffn.w3
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}
// 初始化内存空间（包括前向推理的输入输出缓冲区、kv cache 缓冲区，以及 prefill block 可能用到的 2D
// 临时缓冲区）
void LLama2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    llama_layers_->to_cuda(cuda_config_);
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache  申请内存
  // 0.043
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  // 0.043
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));

  // ===== Prefill block 2D temp buffers (capacity: [seq_len, dim]) =====
  tensor::Tensor rms_output_2d(base::DataType::kDataTypeFp32, config_->seq_len_, config_->dim_,
                               true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm2D, rms_output_2d));

  tensor::Tensor query_2d(base::DataType::kDataTypeFp32, config_->seq_len_, config_->dim_, true,
                          alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery2D, query_2d));

  // 复用内存（和你现在 1D 复用 kOutputMHA/kW2Output 的思路一致）
  CHECK(insert_buffer(ModelBufferType::kOutputMHA2D, rms_output_2d));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput2D, query_2d));

  // ===== Prefill block FFN 2D temp buffers =====
  tensor::Tensor ffn_norm_2d(base::DataType::kDataTypeFp32, config_->seq_len_, config_->dim_, true,
                             alloc);
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm2D, ffn_norm_2d));

  tensor::Tensor w1_output_2d(base::DataType::kDataTypeFp32, config_->seq_len_,
                              config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output_2d(base::DataType::kDataTypeFp32, config_->seq_len_,
                              config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output2D, w1_output_2d));
  CHECK(insert_buffer(ModelBufferType::kW3Output2D, w3_output_2d));

  // 复用 ffn_norm_2d 的内存给 W2 输出（同一块 data，不同 buffer key）
  CHECK(insert_buffer(ModelBufferType::kW2Output2D, ffn_norm_2d));
}

base::Status LLama2Model::create_layers() {
  using namespace base;
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
      llama_layers_->wk_layers_.size() != config_->layer_num_ ||
      llama_layers_->wv_layers_.size() != config_->layer_num_ ||
      llama_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
        !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
      llama_layers_->w2_layers_.size() != config_->layer_num_ ||
      llama_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!llama_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!llama_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!llama_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!llama_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}
// 嵌入层
op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !llama_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}
// 得到kqv向量
void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input  kvcache缓存
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
  // 关键：无论是 Prefill 还是 Decode，都要做这三个矩阵乘法

  // key
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  // value
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

void LLama2Model::attention_qkv_block(int32_t layer_idx, const tensor::Tensor& pos_tensor,
                                      const tensor::Tensor& rmsnorm_output_2d,
                                      const tensor::Tensor& query_2d) const {
  CHECK(llama_layers_ != nullptr);
  CHECK_EQ(rmsnorm_output_2d.dims_size(), 2);
  CHECK_EQ(query_2d.dims_size(), 2);

  const int32_t start_pos = pos_tensor.index<int32_t>(0);
  const int32_t token_num = rmsnorm_output_2d.get_dim(0);

  // 一次切出 KV cache 连续区间 view: [token_num, kv_dim]
  auto [key_block, val_block] = slice_kv_cache_block(layer_idx, start_pos, token_num);

  // wq
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr);
  STATUS_CHECK(query_layer->forward(rmsnorm_output_2d, query_2d));

  // wk/wv 直接写进 cache block（核心：一次写 KV cache）
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr);
  STATUS_CHECK(key_layer->forward(rmsnorm_output_2d, key_block));

  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr);
  STATUS_CHECK(value_layer->forward(rmsnorm_output_2d, val_block));

  // rope: 2D 语义使用 start_pos + t（你前面已实现）
  CHECK_NE(llama_layers_->rope_layer_, nullptr);
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query_2d, key_block, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

base::Status LLama2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);

  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = llama_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}
// 生成 Logits（对数概率）的函数，是 LLM 推理的最后一步，将隐藏状态转换为词表概率分布。
void LLama2Model::cls_logits(const tensor::Tensor& input) const {
  // rmsnorm
  CHECK(llama_layers_ != nullptr);
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));
  //
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(llama_layers_->cls_layer_, nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

}  // namespace model