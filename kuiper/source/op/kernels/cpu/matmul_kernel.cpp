#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"
namespace kernel {

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