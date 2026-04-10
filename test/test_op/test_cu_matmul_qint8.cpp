#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"
#include "tensor/tensor.h"

static tensor::Tensor row_view_fp32(const tensor::Tensor& mat2d, int32_t row, int32_t cols) {
  CHECK_EQ(mat2d.dims_size(), 2);
  CHECK_EQ(mat2d.get_dim(1), cols);
  CHECK_GE(row, 0);
  CHECK_LT(row, mat2d.get_dim(0));
  CHECK(mat2d.data_type() == base::DataType::kDataTypeFp32);
  void* ptr = const_cast<float*>(mat2d.ptr<float>() + (size_t)row * cols);
  auto buffer = std::make_shared<base::Buffer>(sizeof(float) * (size_t)cols, nullptr, ptr, true);
  buffer->set_device_type(mat2d.device_type());

  tensor::Tensor view(base::DataType::kDataTypeFp32, cols);
  view.set_device_type(mat2d.device_type());
  CHECK(view.assign(buffer));
  return view;
}

TEST(test_matmul_cu, matmul_qint8_batch_matches_rowwise_1d) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int B = 7;   // batch / token_num
  const int M = 64;  // input dim (make it multiple of 4 for 1D quant kernel)
  const int K = 13;  // output dim
  const int group_size = 16;
  static_assert(M % 4 == 0, "M must be multiple of 4 for 1D quant kernel path");

  // input: [B, M] fp32 (CPU)
  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, B, M, true, alloc_cpu);
  for (int i = 0; i < B * M; ++i) {
    input_cpu.index<float>(i) = 0.01f * float(i + 1) + ((i % 7) - 3) * 0.1f;
  }

  // weight: [K, M] int8 (CPU)
  tensor::Tensor weight_cpu(base::DataType::kDataTypeInt8, K, M, true, alloc_cpu);
  for (int i = 0; i < K * M; ++i) {
    int v = (i * 13 + 7) % 127;  // 0..126
    v -= 63;                     // -63..63
    weight_cpu.index<int8_t>(i) = (int8_t)v;
  }

  // scales: [K*M/group_size] fp32 (CPU), must match group_idx = (k*M + i)/group_size
  const int scale_num = (K * M) / group_size;
  ASSERT_EQ(scale_num * group_size, K * M);

  tensor::Tensor scales_cpu(base::DataType::kDataTypeFp32, scale_num, true, alloc_cpu);
  for (int g = 0; g < scale_num; ++g) {
    scales_cpu.index<float>(g) = 0.001f * float(g + 1);  // deterministic, non-uniform
  }

  // move to CUDA
  tensor::Tensor input_cu = input_cpu.clone();
  tensor::Tensor weight_cu = weight_cpu.clone();
  tensor::Tensor scales_cu = scales_cpu.clone();
  input_cu.to_cuda(nullptr);
  weight_cu.to_cuda(nullptr);
  scales_cu.to_cuda(nullptr);

  // output from 2D batch kernel
  tensor::Tensor out_batch_cu(base::DataType::kDataTypeFp32, B, K, true, alloc_cu);

  // golden output: for each row, call 1D quant kernel, write into [B,K]
  tensor::Tensor out_golden_cu(base::DataType::kDataTypeFp32, B, K, true, alloc_cu);

  kernel::CudaConfig config;
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  config.stream = stream;

  // batch path (this is what prefill(2D) will use)
  kernel::get_matmul_kernel_quant8(base::DeviceType::kDeviceCUDA)(input_cu, weight_cu, out_batch_cu,
                                                                  group_size, scales_cu, &config);

  // golden: row-wise 1D
  for (int b = 0; b < B; ++b) {
    tensor::Tensor x_row = row_view_fp32(input_cu, b, M);
    tensor::Tensor y_row = row_view_fp32(out_golden_cu, b, K);
    kernel::get_matmul_kernel_quant8(base::DeviceType::kDeviceCUDA)(x_row, weight_cu, y_row,
                                                                    group_size, scales_cu, &config);
  }

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  out_batch_cu.to_cpu();
  out_golden_cu.to_cpu();

  for (int i = 0; i < out_batch_cu.size(); ++i) {
    ASSERT_NEAR(out_batch_cu.index<float>(i), out_golden_cu.index<float>(i), 1e-2f) << "i=" << i;
  }

  // cudaStreamDestroy(stream);
}