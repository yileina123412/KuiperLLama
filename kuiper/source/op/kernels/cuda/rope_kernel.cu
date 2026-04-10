#include "rope_kernel.cuh"
namespace kernel {

#if defined(LLAMA3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec =
        const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void rope_kernel_cu_fp32_batch(int start_pos, int T, int dim, int kv_dim, int head_size,
                                          float* q, float* k, const float* sin_cache,
                                          const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;  // pair index across all heads
  int t = (int)blockIdx.y;
  if (t >= T) return;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) return;

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  int pos = start_pos + t;

  // 保持和 LLAMA3_SUPPORT 的 1D rope_kernel_cu_fp32 完全一致的查表方式
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  float* q_t = q + (size_t)t * (size_t)dim;
  float q0 = q_t[v0_idx];
  float q1 = q_t[v1_idx];
  q_t[v0_idx] = fcr * q0 - fci * q1;
  q_t[v1_idx] = fcr * q1 + fci * q0;

  // 只对落在 kv_dim 覆盖范围内的 head 做 K 的 RoPE（兼容 GQA/MQA）
  if (i < kv_dim) {
    float* k_t = k + (size_t)t * (size_t)kv_dim;
    float k0 = k_t[v0_idx];
    float k1 = k_t[v1_idx];
    k_t[v0_idx] = fcr * k0 - fci * k1;
    k_t[v1_idx] = fcr * k1 + fci * k0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq =
        1.0f / pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec =
        const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq =
        1.0f / pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#else
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr =
      make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);

  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

__global__ void rope_kernel_cu_fp32_batch(int start_pos, int T, int dim, int kv_dim, int head_size,
                                          float* q, float* k, const float* sin_cache,
                                          const float* cos_cache) {
  int pair_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int t = (int)blockIdx.y;
  if (t >= T) return;

  int idx = pair_idx * 2;
  if (idx >= dim) return;

  int pos = start_pos + t;
  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);

  float* q_t = q + (size_t)t * dim;
  float* k_t = k + (size_t)t * kv_dim;

  rope_calc(fcr, fci, q_t, idx);
  if (idx >= kv_dim) return;
  rope_calc(fcr, fci, k_t, idx);
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#endif

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()));
  }
}

// void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor&
// input_q,
//                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
//                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
//                     void* stream) {
//   const int32_t pos = *input_pos.ptr<int32_t>(0);
//   int threads = 128;
//   int blocks = (dim + threads - 1) / threads;
//   if (stream) {
//     cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
//     rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
//         pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
//         sin_cache.ptr<float>(), cos_cache.ptr<float>());
//   } else {
//     rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
//                                              input_k.ptr<float>(), sin_cache.ptr<float>(),
//                                              cos_cache.ptr<float>());
//   }
// }

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                    void* stream) {
  const int32_t start_pos = *input_pos.ptr<int32_t>(0);
  const int threads = 128;

  if (input_q.dims_size() == 1) {
    const int32_t pos = start_pos;
    int blocks = (dim + threads - 1) / threads;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
          pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
          sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } else {
      rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                               input_k.ptr<float>(), sin_cache.ptr<float>(),
                                               cos_cache.ptr<float>());
    }
    return;
  }

  // 2D: q [T, dim], k [T, kv_dim]
  CHECK_EQ(input_q.dims_size(), 2);
  CHECK_EQ(input_k.dims_size(), 2);
  const int32_t T = input_q.get_dim(0);
  CHECK_EQ(input_q.get_dim(1), dim);
  CHECK_EQ(input_k.get_dim(0), T);
  CHECK_EQ(input_k.get_dim(1), kv_dim);

  const int32_t pairs = dim / 2;
  const int blocks = (pairs + threads - 1) / threads;
  dim3 grid(blocks, T);

  float* q = const_cast<float*>(input_q.ptr<float>());
  float* k = const_cast<float*>(input_k.ptr<float>());

  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32_batch<<<grid, threads, 0, stream_>>>(
        start_pos, T, dim, kv_dim, head_size, q, k, sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32_batch<<<grid, threads>>>(start_pos, T, dim, kv_dim, head_size, q, k,
                                                 sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}
}  // namespace kernel