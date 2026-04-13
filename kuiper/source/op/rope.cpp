#include "op/rope.h"
#include <cmath>
#include "kernels/cpu/rope_kernel.h"
#include "kernels/kernels_interface.h"
// 应用旋转位置编码
namespace op {
// 五个输入分别是：input_q,input_k,input_pos,sin_cache,cos_cache
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status RoPELayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input_q = this->get_input(0);
  tensor::Tensor input_k = this->get_input(1);
  tensor::Tensor input_pos = this->get_input(2);

  tensor::Tensor sin_cache = this->get_input(3);
  tensor::Tensor cos_cache = this->get_input(4);

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_rope_kernel(device_type_)(dim_, kv_dim_, head_size_, input_q, input_k, input_pos,
                                        sin_cache, cos_cache,
                                        cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

//  输入校验，确保张量维度、类型、设备类型合法
base::Status RoPELayer::check() const {
  // pos tensor: CPU int32[1]，2D 时它表示 start_pos
  auto status = check_tensor_with_dim(get_input(2), base::DeviceType::kDeviceCPU,
                                      base::DataType::kDataTypeInt32, 1);
  if (!status) return status;

  const auto q = get_input(0);
  const auto k = get_input(1);

  if (q.dims_size() == 1) {
    status = check_tensor_with_dim(q, device_type_, data_type_, dim_);
    if (!status) return status;
    status = check_tensor_with_dim(k, device_type_, data_type_, kv_dim_);
    if (!status) return status;
    return base::error::Success();
  }

  if (q.dims_size() == 2) {
    // q: [T, dim_]
    if (q.get_dim(1) != dim_) {
      return base::error::InvalidArgument("RoPE q last-dim mismatch");
    }
    // k: [T, kv_dim_]
    if (k.dims_size() != 2 || k.get_dim(0) != q.get_dim(0) || k.get_dim(1) != kv_dim_) {
      return base::error::InvalidArgument("RoPE k dim mismatch for batch");
    }
    status = check_tensor(q, device_type_, data_type_);
    if (!status) return status;
    status = check_tensor(k, device_type_, data_type_);
    if (!status) return status;
    return base::error::Success();
  }

  return base::error::InvalidArgument("RoPE only supports 1D/2D");
}

}  // namespace op