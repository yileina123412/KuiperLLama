#include "op/mha.h"
#include <algorithm>
#include "kernels/cpu/mha_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                       int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                       int32_t head_num, int32_t head_size)
    : Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status MultiHeadAttention::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  const tensor::Tensor& mha_out = this->get_output(0);
  const tensor::Tensor& query_tensor = this->get_input(0);
  const tensor::Tensor& score_tensor = this->get_input(1);
  const tensor::Tensor& key_cache_tensor = this->get_input(2);
  const tensor::Tensor& value_cache_tensor = this->get_input(3);

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  int32_t runtime_window = kv_window_size_ > 0 ? kv_window_size_ : seq_len_;
  if (runtime_window > seq_len_) runtime_window = seq_len_;
  if (runtime_window <= 0) {
    runtime_window = seq_len_;
  }
  int32_t runtime_valid_len = kv_valid_len_;
  if (runtime_valid_len <= 0) {
    runtime_valid_len = pos_ + 1;
  }
  if (runtime_valid_len > runtime_window) {
    runtime_valid_len = runtime_window;
  }
  if (runtime_valid_len <= 0) {
    runtime_valid_len = 1;
  }

  // kernel::get_mha_kernel(device_type_)(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_,
  //                                      head_size_, mha_out, query_tensor, score_tensor,
  //                                      key_cache_tensor, value_cache_tensor, device_type_,
  //                                      cuda_config_ ? cuda_config_.get() : nullptr);

  int32_t runtime_prefix = kv_prefix_keep_tokens_;
  if (runtime_prefix < 0) runtime_prefix = 0;
  if (runtime_prefix >= runtime_window) runtime_prefix = runtime_window - 1;
  kernel::get_mha_kernel(device_type_)(
      pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_, head_size_, runtime_window,
      runtime_valid_len, runtime_prefix, mha_out, query_tensor, score_tensor, key_cache_tensor,
      value_cache_tensor, device_type_, cuda_config_ ? cuda_config_.get() : nullptr);
  return base::error::Success();
}

void MultiHeadAttention::set_pos(int32_t pos) { this->pos_ = pos; }

void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->layer_index_ = layer_idx; }

base::Status MultiHeadAttention::check() const {
  base::Status status;
  const int32_t input_tensor_num = 4;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    // mha score tensor
    status = check_tensor(get_input(i), device_type_, data_type_);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the matmul layer.";
      return status;
    }
  }
  return check_tensor(get_output(0), device_type_, data_type_);
}

void MultiHeadAttention::set_kv_window_size(int32_t kv_window_size) {
  this->kv_window_size_ = kv_window_size;
}

void MultiHeadAttention::set_kv_valid_len(int32_t kv_valid_len) {
  this->kv_valid_len_ = kv_valid_len;
}
void MultiHeadAttention::set_kv_prefix_keep_tokens(int32_t n) { this->kv_prefix_keep_tokens_ = n; }

}  // namespace op