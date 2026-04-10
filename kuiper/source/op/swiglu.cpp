#include "op/swiglu.h"
#include "kernels/cpu/swiglu_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"
namespace op {
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
  reset_input_size(2);
  reset_output_size(1);
}

// base::Status SwiGLULayer::check() const {
//   base::Status status;
//   const int32_t input_tensor_num = 2;
//   for (int32_t i = 0; i < input_tensor_num; ++i) {
//     status = check_tensor_with_dim(get_input(0), device_type_, data_type_, hidden_dim_);
//     if (!status) {
//       LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the swiglu layer.";
//       return status;
//     }
//   }

//   status = check_tensor_with_dim(get_output(0), device_type_, data_type_, hidden_dim_);
//   if (!status) {
//     LOG(ERROR) << "The output tensor error in the swiglu layer.";
//     return status;
//   }
//   return base::error::Success();
// }

base::Status SwiGLULayer::check() const {
  const auto input1 = get_input(0);
  const auto input2 = get_input(1);
  const auto output = get_output(0);

  auto status = check_tensor(input1, device_type_, data_type_);
  if (!status) return status;
  status = check_tensor(input2, device_type_, data_type_);
  if (!status) return status;
  status = check_tensor(output, device_type_, data_type_);
  if (!status) return status;

  // 只支持 1D / 2D
  if (input1.dims_size() != 1 && input1.dims_size() != 2) {
    return base::error::InvalidArgument("SwiGLU only supports 1D/2D tensors");
  }

  // elementwise：要求 input1/input2/output shape 完全一致
  if (input2.dims_size() != input1.dims_size() || output.dims_size() != input1.dims_size()) {
    return base::error::InvalidArgument("SwiGLU dims_size mismatch");
  }
  if (input1.size() != input2.size() || input1.size() != output.size()) {
    return base::error::InvalidArgument("SwiGLU size mismatch");
  }
  if (input1.dims() != input2.dims() || input1.dims() != output.dims()) {
    return base::error::InvalidArgument("SwiGLU shape mismatch");
  }

  // hidden_dim_ 约束：最后一维必须是 hidden_dim_
  if (input1.dims_size() == 1) {
    if (input1.get_dim(0) != hidden_dim_) {
      return base::error::InvalidArgument("SwiGLU hidden_dim mismatch (1D)");
    }
  } else {  // 2D: [B, hidden_dim_]
    if (input1.get_dim(1) != hidden_dim_) {
      return base::error::InvalidArgument("SwiGLU hidden_dim mismatch (2D)");
    }
  }

  return base::error::Success();
}

base::Status SwiGLULayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_swiglu_kernel(device_type_)(input1, input2, output,
                                          cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

}  // namespace op
