#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
// 非量化
// THREAD_PER_BLOCK：每个block里的线程数  ROW_PER_BLOCK：每个block负责多少行输出
// 这个模板是共享内存和cub库的时候要用的，需要编译时已知而不是运行时已知
// M：输入长度  K：权重行数/输出长度
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  // 设置共享内存
  // 一个block里的共享数组
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;  // 线程块内的编号

  // 设置每个线程的负责的行数
  // 每个block负责计算的行数
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  // float4向量化
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;
// 遍历本block负责的行
#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    // 权重偏移
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);
// 每个线程处理若干个float4块
#pragma unroll
    // 遍历float4向量
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }
    // 规约前先等全部算完
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

__global__ void matmul_kernel_cu_fp32_batch(const float* input, const float* weight, float* output,
                                            int B, int M, int K) {
  // grid = (B, K)
  const int b = (int)blockIdx.x;
  const int k = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;

  extern __shared__ float ssum[];
  float sum = 0.f;

  const float* x = input + (size_t)b * M;
  const float* w = weight + (size_t)k * M;

  for (int i = tid; i < M; i += blockDim.x) sum += x[i] * w[i];
  ssum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) ssum[tid] += ssum[tid + offset];
    __syncthreads();
  }
  if (tid == 0) output[(size_t)b * K + k] = ssum[0];
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

__global__ void matmul_kernel_cu_qint8_batch(const float* input, const int8_t* weight,
                                             const float* scales, int group_size, float* output,
                                             int B, int M, int K) {
  // grid = (B, K)
  const int b = (int)blockIdx.x;
  const int k = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;

  extern __shared__ float ssum[];
  float sum = 0.f;

  const float* x = input + (size_t)b * M;
  const int8_t* w = weight + (size_t)k * M;

  for (int i = tid; i < M; i += blockDim.x) {
    const int weight_idx = k * M + i;
    const int group_idx = weight_idx / group_size;
    const float w_deq = scales[group_idx] * (float)w[i];
    sum += x[i] * w_deq;
  }

  ssum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) ssum[tid] += ssum[tid + offset];
    __syncthreads();
  }

  if (tid == 0) output[(size_t)b * K + k] = ssum[0];
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(!input.is_empty() && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(!weight.is_empty() && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(!output.is_empty());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t K = weight.get_dim(0);  // output dim
  const int32_t M = weight.get_dim(1);  // input dim

  // 1D: [M] -> [K]
  if (input.dims_size() == 1) {
    CHECK_EQ(M, input.get_dim(0));
    CHECK_EQ(output.dims_size(), 1);
    CHECK_EQ(output.get_dim(0), K);

    if (config && config->stream) {
      matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                                const_cast<float*>(output.ptr<float>()), M, K);
    }

    CHECK_EQ(scale, 1.f);  // 先这样，避免 silent wrong
    return;
  }

  // 2D batch: [B, M] -> [B, K]
  CHECK_EQ(input.dims_size(), 2);
  const int32_t B = input.get_dim(0);
  CHECK_EQ(M, input.get_dim(1));

  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(output.get_dim(0), B);
  CHECK_EQ(output.get_dim(1), K);

  const int threads = 256;
  dim3 grid(B, K);
  size_t shmem = threads * sizeof(float);

  if (config && config->stream) {
    matmul_kernel_cu_fp32_batch<<<grid, threads, shmem, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), B, M, K);
  } else {
    matmul_kernel_cu_fp32_batch<<<grid, threads, shmem>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), B, M, K);
  }

  CHECK_EQ(scale, 1.f);  // 先这样，避免 silent wrong
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  // 1D: [M] -> [K]（decode）
  if (input.dims_size() == 1) {
    int packet_size = 4;
    CHECK_EQ(M % packet_size, 0);
    CHECK_EQ(M, input.get_dim(0));

    CHECK_EQ(output.dims_size(), 1);
    CHECK_EQ(output.get_dim(0), K);

    if (config->stream) {
      matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
          input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
          const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                    scale.ptr<float>(), group_size,
                                                    const_cast<float*>(output.ptr<float>()), M, K);
    }
    return;
  }

  // 2D batch: [B, M] -> [B, K]（prefill）
  CHECK_EQ(input.dims_size(), 2);
  const int32_t B = input.get_dim(0);
  CHECK_EQ(M, input.get_dim(1));

  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(output.get_dim(0), B);
  CHECK_EQ(output.get_dim(1), K);

  const int threads = 256;
  dim3 grid(B, K);
  size_t shmem = threads * sizeof(float);

  if (config->stream) {
    matmul_kernel_cu_qint8_batch<<<grid, threads, shmem, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), B, M, K);
  } else {
    matmul_kernel_cu_qint8_batch<<<grid, threads, shmem>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), B, M, K);
  }
}

}  // namespace kernel