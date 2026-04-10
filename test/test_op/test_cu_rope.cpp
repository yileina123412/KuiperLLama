#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/cpu/rope_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
// TEST(test_rope_cu, rope_nostream) {
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
//   int32_t dim = 256;
//   int32_t head_size = 64;
//   int32_t kv_dim = 128;
//   int32_t pos = 3;
//   tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
//   input_pos.index<int32_t>(0) = pos;

//   std::random_device rd;
//   std::mt19937 mt(rd());
//   std::uniform_real_distribution<float> dist(0.f, 1.f);
//   tensor::Tensor input_q_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
//   tensor::Tensor input_k_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);

//   for (int i = 0; i < dim; ++i) {
//     input_q_cpu.index<float>(i) = dist(mt);
//     input_k_cpu.index<float>(i) = dist(mt);
//   }

//   tensor::Tensor input_q_gpu = input_q_cpu.clone();
//   tensor::Tensor input_k_gpu = input_k_cpu.clone();
//   input_q_gpu.to_cuda(nullptr);
//   input_k_gpu.to_cuda(nullptr);

//   kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(dim, kv_dim, head_size, input_q_cpu,
//                                                         input_k_cpu, input_pos, nullptr);

//   kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(dim, kv_dim, head_size, input_q_gpu,
//                                                          input_k_gpu, input_pos, nullptr);
//   cudaDeviceSynchronize();

//   input_q_gpu.to_cpu();
//   input_k_gpu.to_cpu();
//   for (int32_t i = 0; i < dim; ++i) {
//     ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f) << "ik: " << i;
//     ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f) << "iq: " << i;
//   }
// }

// TEST(test_rope_cu, rope_nostream2) {
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
//   int32_t dim = 512;
//   int32_t head_size = 128;
//   int32_t kv_dim = 32;
//   int32_t pos = 4;
//   tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
//   input_pos.index<int32_t>(0) = pos;

//   std::random_device rd;
//   std::mt19937 mt(rd());
//   std::uniform_real_distribution<float> dist(0.f, 1.f);
//   tensor::Tensor input_q_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
//   tensor::Tensor input_k_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);

//   for (int i = 0; i < dim; ++i) {
//     input_q_cpu.index<float>(i) = dist(mt);
//     input_k_cpu.index<float>(i) = dist(mt);
//   }

//   tensor::Tensor input_q_gpu = input_q_cpu.clone();
//   tensor::Tensor input_k_gpu = input_k_cpu.clone();
//   input_q_gpu.to_cuda(nullptr);
//   input_k_gpu.to_cuda(nullptr);

//   kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(dim, kv_dim, head_size, input_q_cpu,
//                                                         input_k_cpu, input_pos, nullptr);

//   kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(dim, kv_dim, head_size, input_q_gpu,
//                                                          input_k_gpu, input_pos, nullptr);
//   cudaDeviceSynchronize();

//   input_q_gpu.to_cpu();
//   input_k_gpu.to_cpu();
//   for (int32_t i = 0; i < dim; ++i) {
//     ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f) << "ik: " << i;
//     ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f) << "iq: " << i;
//   }
// }

// TEST(test_rope_cu, rope_stream1) {
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
//   int32_t dim = 512;
//   int32_t head_size = 128;
//   int32_t kv_dim = 32;
//   int32_t pos = 4;
//   tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
//   input_pos.index<int32_t>(0) = pos;

//   std::random_device rd;
//   std::mt19937 mt(rd());
//   std::uniform_real_distribution<float> dist(0.f, 1.f);
//   tensor::Tensor input_q_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
//   tensor::Tensor input_k_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);
//   for (int i = 0; i < dim; ++i) {
//     input_q_cpu.index<float>(i) = dist(mt);
//     input_k_cpu.index<float>(i) = dist(mt);
//   }

//   tensor::Tensor input_q_gpu = input_q_cpu.clone();
//   tensor::Tensor input_k_gpu = input_k_cpu.clone();
//   input_q_gpu.to_cuda(nullptr);
//   input_k_gpu.to_cuda(nullptr);

//   kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(dim, kv_dim, head_size, input_q_cpu,
//                                                         input_k_cpu, input_pos, nullptr);

//   kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(dim, kv_dim, head_size, input_q_gpu,
//                                                          input_k_gpu, input_pos, stream);
//   cudaDeviceSynchronize();

//   input_q_gpu.to_cpu();
//   input_k_gpu.to_cpu();
//   for (int32_t i = 0; i < dim; ++i) {
//     ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f) << "ik: " << i;
//     ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f) << "iq: " << i;
//   }
// }

TEST(test_rope_cu, rope_batch_cpu_vs_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int32_t T = 3;
  const int32_t dim = 16;
  const int32_t kv_dim = 8;
  const int32_t head_size = 8;

  const int32_t start_pos = 5;
  const int32_t max_seq_len = start_pos + T + 1;

  // pos: CPU int32[1], 2D 时它表示 start_pos
  tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input_pos.index<int32_t>(0) = start_pos;

  // sin/cos cache: 先在 CPU 上按同一套公式生成，再拷到 GPU，避免 CPU/GPU cache 生成差异干扰对比
  tensor::Tensor sin_cpu(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
  tensor::Tensor cos_cpu(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
  kernel::sin_cos_cache_calc_cpu(head_size, max_seq_len, sin_cpu.ptr<float>(),
                                 cos_cpu.ptr<float>());

  tensor::Tensor sin_cu = sin_cpu.clone();
  tensor::Tensor cos_cu = cos_cpu.clone();
  sin_cu.to_cuda(nullptr);
  cos_cu.to_cuda(nullptr);

  // q: [T, dim], k: [T, kv_dim]
  tensor::Tensor q_cpu(base::DataType::kDataTypeFp32, T, dim, true, alloc_cpu);
  tensor::Tensor k_cpu(base::DataType::kDataTypeFp32, T, kv_dim, true, alloc_cpu);

  for (int i = 0; i < (int)q_cpu.size(); ++i) q_cpu.index<float>(i) = 0.01f * float(i + 1);
  for (int i = 0; i < (int)k_cpu.size(); ++i) k_cpu.index<float>(i) = 0.02f * float(i - 3);

  tensor::Tensor q_cu = q_cpu.clone();
  tensor::Tensor k_cu = k_cpu.clone();
  q_cu.to_cuda(nullptr);
  k_cu.to_cuda(nullptr);

  // CPU reference (in-place)
  kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(dim, kv_dim, head_size, q_cpu, k_cpu,
                                                        input_pos, sin_cpu, cos_cpu, nullptr);

  // CUDA (in-place)
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(dim, kv_dim, head_size, q_cu, k_cu,
                                                         input_pos, sin_cu, cos_cu,
                                                         reinterpret_cast<void*>(stream));
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  q_cu.to_cpu();
  k_cu.to_cpu();

  for (int i = 0; i < (int)q_cpu.size(); ++i) {
    ASSERT_NEAR(q_cu.index<float>(i), q_cpu.index<float>(i), 1e-4f) << "q i=" << i;
  }
  for (int i = 0; i < (int)k_cpu.size(); ++i) {
    ASSERT_NEAR(k_cu.index<float>(i), k_cpu.index<float>(i), 1e-4f) << "k i=" << i;
  }
}