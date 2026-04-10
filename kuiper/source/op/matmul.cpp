#include "op/matmul.h"
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
      dim0_(dim0),
      dim1_(dim1),
      has_bias_(has_bias) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
  if (has_bias_) {
    bias_.resize(1);
  }
}

base::Status MatmulLayer::check() const {
  const auto input = get_input(0);
  const auto output = get_output(0);

  // input dims: [M] or [B, M]
  if (input.dims_size() == 1) {
    auto status = check_tensor_with_dim(input, device_type_, data_type_, dim1_);
    if (!status) return status;

    status = check_tensor_with_dim(output, device_type_, data_type_, dim0_);
    if (!status) return status;
  } else if (input.dims_size() == 2) {
    // 只要求最后一维是 dim1_
    if (input.get_dim(1) != dim1_) {
      return base::error::InvalidArgument("Matmul input dim mismatch at dim=1");
    }
    // output should be [B, dim0_]
    if (output.dims_size() != 2 || output.get_dim(0) != input.get_dim(0) ||
        output.get_dim(1) != dim0_) {
      return base::error::InvalidArgument("Matmul output dim mismatch for batch");
    }
    auto status = check_tensor(input, device_type_, data_type_);
    if (!status) return status;
    status = check_tensor(output, device_type_, data_type_);
    if (!status) return status;
  } else {
    return base::error::InvalidArgument("Matmul only supports 1D/2D input");
  }

  // weight check（保持你原逻辑）
  if (!is_quant_layer_) {
    auto status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
    if (!status) return status;
  } else {
    auto status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                        dim0_, dim1_);
    if (!status) return status;
    status =
        check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32, scales_.size());
    if (!status) return status;
  }

  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  if (is_quant_layer_) {
    kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                   group_size_, scales_,
                                                   cuda_config_ ? cuda_config_.get() : nullptr);
  } else {
    kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                            cuda_config_ ? cuda_config_.get() : nullptr);
  }

  if (has_bias_) {
    kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0),
                                         cuda_config_ ? cuda_config_->stream : nullptr);
  }

  return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                   base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  size_t size = dim * sizeof(float);
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    // LOG(INFO) << "bias:" << bias.index<float>(0);
    bias_.at(idx) = bias;
  } else {
    // is quant layer
    tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;

    const int32_t bias_size = static_cast<int32_t>(bias.size());
    CHECK(bias_size % group_size_ == 0);

    int32_t scale_nums = bias_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

void MatmulLayer::to_cuda() {
  LayerParam::to_cuda();
  if (has_bias_) {
    for (auto& bias : bias_) {
      bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

}  // namespace op