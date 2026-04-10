#include "op/add.h"
#include "kernels/kernels_interface.h"
namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") {
  reset_input_size(2);
  reset_output_size(1);
}

// base::Status VecAddLayer::check() const {
//   tensor::Tensor input1 = this->get_input(0);
//   tensor::Tensor input2 = this->get_input(1);
//   int32_t size = input1.size();
//   base::Status status;
//   status = check_tensor_with_dim(input1, device_type_, data_type_, size);
//   if (!status) {
//     LOG(ERROR) << "The input tensor 1 error in the add layer.";
//     return status;
//   }

//   status = check_tensor_with_dim(input2, device_type_, data_type_, size);
//   if (!status) {
//     LOG(ERROR) << "The input tensor 2 error in the add layer.";
//     return status;
//   }

//   status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
//   if (!status) {
//     LOG(ERROR) << "The output tensor error in the add layer.";
//     return status;
//   }
//   return base::error::Success();
// }
base::Status VecAddLayer::check() const {
  const auto input1 = this->get_input(0);
  const auto input2 = this->get_input(1);
  const auto output = this->get_output(0);

  auto status = check_tensor(input1, device_type_, data_type_);
  if (!status) return status;
  status = check_tensor(input2, device_type_, data_type_);
  if (!status) return status;
  status = check_tensor(output, device_type_, data_type_);
  if (!status) return status;

  // 只支持 1D / 2D（decode 或 prefill block）
  if (input1.dims_size() != 1 && input1.dims_size() != 2) {
    return base::error::InvalidArgument("VecAdd only supports 1D/2D tensors");
  }

  // elementwise：要求 shape 完全一致（不做广播）
  if (input2.dims_size() != input1.dims_size() || output.dims_size() != input1.dims_size()) {
    return base::error::InvalidArgument("VecAdd dims_size mismatch");
  }
  if (input1.size() != input2.size() || input1.size() != output.size()) {
    return base::error::InvalidArgument("VecAdd size mismatch");
  }
  if (input1.dims() != input2.dims() || input1.dims() != output.dims()) {
    return base::error::InvalidArgument("VecAdd shape mismatch");
  }

  return base::error::Success();
}

base::Status VecAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_add_kernel(device_type_)(input1, input2, output,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

}  // namespace op