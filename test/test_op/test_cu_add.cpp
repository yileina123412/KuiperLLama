#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
TEST(test_add_cu, add1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }

  delete[] output;
}

TEST(test_add_cu, add1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }
  cudaStreamDestroy(stream);
  delete[] output;
}

TEST(test_add_cu, add_align1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 13;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f, 0.1f);
  }

  delete[] output;
}

TEST(test_add_cu, add_batch_cpu_vs_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int32_t B = 4;
  const int32_t H = 151;
  const int32_t size = B * H;

  tensor::Tensor a_cpu(base::DataType::kDataTypeFp32, B, H, true, alloc_cpu);
  tensor::Tensor b_cpu(base::DataType::kDataTypeFp32, B, H, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, B, H, true, alloc_cpu);

  for (int i = 0; i < size; ++i) {
    a_cpu.index<float>(i) = 0.01f * float(i + 1);
    b_cpu.index<float>(i) = -0.02f * float(i - 7);
  }

  tensor::Tensor a_cu = a_cpu.clone();
  tensor::Tensor b_cu = b_cpu.clone();
  a_cu.to_cuda(nullptr);
  b_cu.to_cuda(nullptr);

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, B, H, true, alloc_cu);

  // CPU ref
  kernel::get_add_kernel(base::DeviceType::kDeviceCPU)(a_cpu, b_cpu, out_cpu, nullptr);

  // CUDA
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(a_cu, b_cu, out_cu, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  out_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-6f) << "i=" << i;
  }
}