#include <base/cuda_config.h>
#include <base/tick.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
namespace kernel {
// 多头子注意力

// 满足cub模板参数，模板参数必须编译器就确定下来   以及保证host和device一致
constexpr static int thread_num = 256;

// x必须是可写的连续内存
// __restrict__ 是一个指针别名限定符，编译器可以假设
// 在该指针的有效作用域内，只有这个指针会访问所指向的内存 用作优化的
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  // 一个block计算处理一个向量x
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  // 寻找每个线程负责元素的最大值
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}
// 一个block计算一个头
__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t kv_window_size, int32_t kv_valid_len,
                                            int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }
  // 预加载query到共享内存
  // 动态声明共享内存的大小 同一个头内，所有时间不在算qk点积的时候都要用到当前的q
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  // 当前头指向的内存
  float* query_head = query + head * head_size;

  // 预加载query到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  const int window = max(1, min(kv_window_size, seq_len));
  const int valid_len = max(1, min(kv_valid_len, window));
  const int logical_start = pos - valid_len + 1;
  if (logical_start < 0) {
    return;
  }

  float* score_head = score_ptr + head * seq_len;
  // head当前的注意力头索引，kv_mul用于gqa，head_size表示一个自注意力头的维度
  // kv_dim = head_size * head_num，多头自注意力情况下的key,value 维度
  // kv_dim = head_size * head_num / kv_num，GQA情况下的key,value 维度
  int head_offset = (head / kv_mul) * head_size;
  // 计算自注意力分数
  // 一个线程计算一个的历史qv
  for (int t = threadIdx.x; t < valid_len; t += blockDim.x) {
    int logical_t = logical_start + t;
    int slot = logical_t % window;
    if (slot < 0) slot += window;
    float* key_head = key_cache + layer_offset + slot * kv_dim + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, valid_len);
  __syncthreads();

  float* output_head = output + head * head_size;
  // 使用自注意力分数对value矩阵加权
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int idx = 0; idx < valid_len; ++idx) {
      int logical_t = logical_start + idx;
      int slot = logical_t % window;
      if (slot < 0) slot += window;

      float* value_head = value_cache + layer_offset + slot * kv_dim + head_offset;
      value += score_head[idx] * value_head[i];
    }
    output_head[i] = value;
  }
}

__device__ __forceinline__ float dot_qk_fp32(const float* __restrict__ q,
                                             const float* __restrict__ k, int head_size) {
  float sum = 0.f;
  int i = 0;
  for (; i + 3 < head_size; i += 4) {
    float4 q4 = *reinterpret_cast<const float4*>(q + i);
    float4 k4 = *reinterpret_cast<const float4*>(k + i);
    sum += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
  }
  for (; i < head_size; ++i) sum += q[i] * k[i];
  return sum;
}

// grid = (head_num, T), block = 256 threads
// __global__ void multi_head_attention_prefill_kernel(int32_t start_pos, int32_t T, int32_t
// seq_len,
//                                                     const float* __restrict__ query_2d,  // [T,
//                                                     dim] float* __restrict__ output_2d,       //
//                                                     [T, dim] const float* __restrict__ key_cache,
//                                                     const float* __restrict__ value_cache,
//                                                     int32_t kv_dim, int32_t kv_mul,
//                                                     int32_t head_num, int32_t head_size,
//                                                     int32_t layer_offset) {
//   const int head = (int)blockIdx.x;
//   const int t = (int)blockIdx.y;
//   if (head >= head_num || t >= T) return;

//   const int32_t pos = start_pos + t;
//   if (pos >= seq_len) return;

//   extern __shared__ float smem[];
//   float* s_query = smem;             // [head_size]
//   float* s_w = s_query + head_size;  // [blockDim.x]
//   __shared__ float s_max;
//   __shared__ float s_sumexp;

//   const float scale = 1.f / sqrtf((float)head_size);
//   const int head_offset = (head / kv_mul) * head_size;

//   const int dim = head_num * head_size;
//   const float* q = query_2d + (size_t)t * dim + (size_t)head * head_size;
//   float* o = output_2d + (size_t)t * dim + (size_t)head * head_size;

//   // load q to shared
//   for (int i = threadIdx.x; i < head_size; i += blockDim.x) s_query[i] = q[i];
//   __syncthreads();

//   using BlockReduce = cub::BlockReduce<float, thread_num>;
//   __shared__ typename BlockReduce::TempStorage temp;

//   // pass1: max
//   float local_max = -FLT_MAX;
//   for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
//     const float* k = key_cache + layer_offset + (size_t)p * kv_dim + head_offset;
//     float s = dot_qk_fp32(s_query, k, head_size) * scale;
//     local_max = fmaxf(local_max, s);
//   }
//   float maxv = BlockReduce(temp).Reduce(local_max, cub::Max());
//   if (threadIdx.x == 0) s_max = maxv;
//   __syncthreads();
//   maxv = s_max;

//   // pass2: sumexp + accumulate output (unnormalized)
//   if (threadIdx.x == 0) s_sumexp = 0.f;
//   __syncthreads();

//   // init output accum
//   for (int i = threadIdx.x; i < head_size; i += blockDim.x) o[i] = 0.f;
//   __syncthreads();

//   for (int p0 = 0; p0 <= pos; p0 += blockDim.x) {
//     int p = p0 + threadIdx.x;
//     float w = 0.f;
//     if (p <= pos) {
//       const float* k = key_cache + layer_offset + (size_t)p * kv_dim + head_offset;
//       float s = dot_qk_fp32(s_query, k, head_size) * scale;
//       w = expf(s - maxv);
//     }
//     s_w[threadIdx.x] = w;
//     __syncthreads();

//     // tile sumexp
//     float tile_sum = BlockReduce(temp).Sum(w);
//     if (threadIdx.x == 0) s_sumexp += tile_sum;
//     __syncthreads();

//     // accumulate output for this tile
//     const int tile_len = min((int)blockDim.x, (int)(pos + 1 - p0));
//     for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
//       float acc = 0.f;
//       for (int j = 0; j < tile_len; ++j) {
//         const int pp = p0 + j;
//         const float ww = s_w[j];
//         const float* v = value_cache + layer_offset + (size_t)pp * kv_dim + head_offset;
//         acc += ww * v[i];
//       }
//       o[i] += acc;
//     }
//     __syncthreads();
//   }

//   float sumexp = s_sumexp;
//   for (int i = threadIdx.x; i < head_size; i += blockDim.x) o[i] /= sumexp;
// }
// grid = (head_num, T), block = 256 threads
__global__ void multi_head_attention_prefill_kernel(
    int32_t start_pos, int32_t T, int32_t seq_len, int32_t kv_window_size,
    const float* __restrict__ query_2d,  // [T, dim]
    float* __restrict__ output_2d,       // [T, dim]
    const float* __restrict__ key_cache, const float* __restrict__ value_cache, int32_t kv_dim,
    int32_t kv_mul, int32_t head_num, int32_t head_size, int32_t layer_offset) {
  const int head = (int)blockIdx.x;
  const int t = (int)blockIdx.y;
  if (head >= head_num || t >= T) return;

  const int32_t pos = start_pos + t;

  const int window = max(1, min(kv_window_size, seq_len));
  const int valid_len_t = max(1, min((int)pos + 1, window));
  const int logical_start = (int)pos - valid_len_t + 1;
  if (logical_start < 0) return;

  extern __shared__ float smem[];
  float* s_query = smem;             // [head_size]
  float* s_w = s_query + head_size;  // [blockDim.x]
  __shared__ float s_max;
  __shared__ float s_sumexp;

  const float scale = 1.f / sqrtf((float)head_size);
  const int head_offset = (head / kv_mul) * head_size;

  const int dim = head_num * head_size;
  const float* q = query_2d + (size_t)t * dim + (size_t)head * head_size;
  float* o = output_2d + (size_t)t * dim + (size_t)head * head_size;

  // load q to shared
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) s_query[i] = q[i];
  __syncthreads();

  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ typename BlockReduce::TempStorage temp;

  // pass1: max
  float local_max = -FLT_MAX;
  for (int idx = threadIdx.x; idx < valid_len_t; idx += blockDim.x) {
    int logical_p = logical_start + idx;
    int slot = logical_p % window;
    if (slot < 0) slot += window;

    const float* k = key_cache + layer_offset + (size_t)slot * kv_dim + head_offset;
    float s = dot_qk_fp32(s_query, k, head_size) * scale;
    local_max = fmaxf(local_max, s);
  }
  float maxv = BlockReduce(temp).Reduce(local_max, cub::Max());
  if (threadIdx.x == 0) s_max = maxv;
  __syncthreads();
  maxv = s_max;

  // pass2: sumexp + accumulate output (unnormalized)
  if (threadIdx.x == 0) s_sumexp = 0.f;
  __syncthreads();

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) o[i] = 0.f;
  __syncthreads();

  for (int idx0 = 0; idx0 < valid_len_t; idx0 += blockDim.x) {
    int idx = idx0 + threadIdx.x;
    float w = 0.f;

    if (idx < valid_len_t) {
      int logical_p = logical_start + idx;
      int slot = logical_p % window;
      if (slot < 0) slot += window;

      const float* k = key_cache + layer_offset + (size_t)slot * kv_dim + head_offset;
      float s = dot_qk_fp32(s_query, k, head_size) * scale;
      w = expf(s - maxv);
    }

    s_w[threadIdx.x] = w;
    __syncthreads();

    float tile_sum = BlockReduce(temp).Sum(w);
    if (threadIdx.x == 0) s_sumexp += tile_sum;
    __syncthreads();

    const int tile_len = min((int)blockDim.x, valid_len_t - idx0);
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
      float acc = 0.f;
      for (int j = 0; j < tile_len; ++j) {
        int logical_p = logical_start + (idx0 + j);
        int slot = logical_p % window;
        if (slot < 0) slot += window;

        const float ww = s_w[j];
        const float* v = value_cache + layer_offset + (size_t)slot * kv_dim + head_offset;
        acc += ww * v[i];
      }
      o[i] += acc;
    }
    __syncthreads();
  }

  float sumexp = s_sumexp;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) o[i] /= sumexp;
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, int32_t kv_window_size,
                   int32_t kv_valid_len, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
  cudaStream_t stream = config->stream;

  // ===== 1D decode：保持你原实现不变 =====
  if (query_tensor.dims_size() == 1) {
    float* query = const_cast<float*>(query_tensor.ptr<float>());
    float* score = const_cast<float*>(score_tensor.ptr<float>());
    float* output = const_cast<float*>(mha_out.ptr<float>());

    multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
        pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
        head_size, kv_window_size, kv_valid_len, layer_offset);
    return;
  }

  // ===== 2D prefill block =====
  CHECK_EQ(query_tensor.dims_size(), 2);
  CHECK_EQ(mha_out.dims_size(), 2);

  const int32_t start_pos = pos;
  const int32_t T = query_tensor.get_dim(0);
  const int32_t dim = query_tensor.get_dim(1);
  CHECK_EQ(dim, head_num * head_size);
  CHECK_EQ(mha_out.get_dim(0), T);
  CHECK_EQ(mha_out.get_dim(1), dim);

  const float* query2d = query_tensor.ptr<float>();
  float* out2d = const_cast<float*>(mha_out.ptr<float>());

  dim3 grid(head_num, T);
  size_t shmem = (size_t)(head_size + thread_num) * sizeof(float);

  // multi_head_attention_prefill_kernel<<<grid, thread_num, shmem, stream>>>(
  //     start_pos, T, seq_len, query2d, out2d, key_cache, value_cache, kv_dim, kv_mul, head_num,
  //     head_size, layer_offset);
  multi_head_attention_prefill_kernel<<<grid, thread_num, shmem, stream>>>(
      start_pos, T, seq_len, kv_window_size, query2d, out2d, key_cache, value_cache, kv_dim, kv_mul,
      head_num, head_size, layer_offset);
}
}  // namespace kernel