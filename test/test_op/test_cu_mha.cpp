#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

TEST(test_mha_cu, mha_prefill_batch_cpu_vs_cuda) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // small config (make head_size multiple of 4)
  const int32_t head_num = 4;
  const int32_t head_size = 8;
  const int32_t kv_mul = 2;                                // GQA: kv_heads = head_num / kv_mul = 2
  const int32_t kv_dim = (head_num / kv_mul) * head_size;  // 16
  const int32_t dim = head_num * head_size;                // 32

  const int32_t seq_len = 8;
  const int32_t layer_index = 0;

  const int32_t start_pos = 2;
  const int32_t T = 3;  // positions: 2,3,4

  // query: [T, dim]
  tensor::Tensor q_cpu(base::DataType::kDataTypeFp32, T, dim, true, alloc_cpu);
  for (int i = 0; i < (int)q_cpu.size(); ++i) {
    q_cpu.index<float>(i) = 0.01f * float(i - 7);
  }

  // key/value cache: [seq_len, kv_dim]
  tensor::Tensor kcache_cpu(base::DataType::kDataTypeFp32, seq_len, kv_dim, true, alloc_cpu);
  tensor::Tensor vcache_cpu(base::DataType::kDataTypeFp32, seq_len, kv_dim, true, alloc_cpu);
  for (int p = 0; p < seq_len; ++p) {
    for (int i = 0; i < kv_dim; ++i) {
      kcache_cpu.index<float>(p * kv_dim + i) = 0.001f * float(p * kv_dim + i + 1);
      vcache_cpu.index<float>(p * kv_dim + i) = 0.002f * float(p * kv_dim + i - 3);
    }
  }

  // score tensor: decode 用的草稿纸，这里 batch 分支不会用到，但保持接口一致
  tensor::Tensor score_cpu(base::DataType::kDataTypeFp32, head_num * seq_len, true, alloc_cpu);
  for (int i = 0; i < (int)score_cpu.size(); ++i) score_cpu.index<float>(i) = 0.f;

  // outputs: [T, dim]
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, T, dim, true, alloc_cpu);

  // CUDA tensors
  tensor::Tensor q_cu = q_cpu.clone();
  tensor::Tensor kcache_cu = kcache_cpu.clone();
  tensor::Tensor vcache_cu = vcache_cpu.clone();
  tensor::Tensor score_cu = score_cpu.clone();

  q_cu.to_cuda(nullptr);
  kcache_cu.to_cuda(nullptr);
  vcache_cu.to_cuda(nullptr);
  score_cu.to_cuda(nullptr);

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, T, dim, true, alloc_cu);

  // CPU reference
  kernel::get_mha_kernel(base::DeviceType::kDeviceCPU)(
      start_pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size, seq_len, start_pos + T,
      out_cpu, q_cpu, score_cpu, kcache_cpu, vcache_cpu, base::DeviceType::kDeviceCPU, nullptr);

  // CUDA
  kernel::CudaConfig config;
  cudaStreamCreate(&config.stream);

  kernel::get_mha_kernel(base::DeviceType::kDeviceCUDA)(
      start_pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size, seq_len, start_pos + T,
      out_cu, q_cu, score_cu, kcache_cu, vcache_cu, base::DeviceType::kDeviceCUDA, &config);

  cudaStreamSynchronize(config.stream);
  out_cu.to_cpu();

  for (int i = 0; i < (int)out_cpu.size(); ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-4f) << "i=" << i;
  }
}