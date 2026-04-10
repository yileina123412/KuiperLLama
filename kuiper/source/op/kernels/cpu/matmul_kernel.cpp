#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"
namespace kernel {
// void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
//                        const tensor::Tensor& output, float scale,
//                        const CudaConfig* config) {
//   UNUSED(config);
//   CHECK(input.is_empty() == false);
//   CHECK(weight.is_empty() == false);
//   CHECK(output.is_empty() == false);
//   CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
//   CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
//   CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

//   const float* input_ptr = input.ptr<float>();
//   const float* weight_ptr = weight.ptr<float>();
//   const float* output_ptr = output.ptr<float>();

//   int32_t in_dim1 = 1;
//   int32_t in_dim0 = 1;
//   if (input.dims_size() == 2) {
//     in_dim0 = input.get_dim(0);
//     in_dim1 = input.get_dim(1);
//   } else if (input.dims_size() == 1) {
//     in_dim0 = input.get_dim(0);
//   } else {
//     LOG(FATAL) << "The input tensor has a wrong dim size.";
//   }

//   CHECK_EQ(weight.dims_size(), 2);
//   const int32_t wei_dim0 = weight.get_dim(0);
//   const int32_t wei_dim1 = weight.get_dim(1);
//   CHECK_EQ(in_dim0, wei_dim1);

//   CHECK_EQ(output.size(), wei_dim0 * in_dim1);
//   arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false, true);
//   arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
//   arma::fmat output_mat(const_cast<float*>(output_ptr), in_dim1, wei_dim0, false, true);
//   output_mat = ((input_mat * weight_mat)) * scale;
// }
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale, const CudaConfig* config) {
  UNUSED(config);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  CHECK_EQ(weight.dims_size(), 2);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);

  const float* x = input.ptr<float>();
  const float* w = weight.ptr<float>();
  float* y = const_cast<float*>(output.ptr<float>());

  if (input.dims_size() == 1) {
    CHECK_EQ(input.get_dim(0), M);
    CHECK_EQ(output.dims_size(), 1);
    CHECK_EQ(output.get_dim(0), K);
    for (int32_t k = 0; k < K; ++k) {
      float sum = 0.f;
      const float* wk = w + (size_t)k * M;
      for (int32_t i = 0; i < M; ++i) sum += x[i] * wk[i];
      y[k] = sum * scale;
    }
    return;
  }

  CHECK_EQ(input.dims_size(), 2);
  const int32_t B = input.get_dim(0);
  CHECK_EQ(input.get_dim(1), M);
  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(output.get_dim(0), B);
  CHECK_EQ(output.get_dim(1), K);

  for (int32_t b = 0; b < B; ++b) {
    const float* xb = x + (size_t)b * M;
    float* yb = y + (size_t)b * K;
    for (int32_t k = 0; k < K; ++k) {
      float sum = 0.f;
      const float* wk = w + (size_t)k * M;
      for (int32_t i = 0; i < M; ++i) sum += xb[i] * wk[i];
      yb[k] = sum * scale;
    }
  }
}
}  // namespace kernel