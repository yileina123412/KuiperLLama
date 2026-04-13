#include "rope_kernel.h"
namespace kernel {
#if defined(LLAMA3_SUPPORT)
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t head_dim = i % head_size; head_dim < head_size / 2; head_dim++) {
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim * 2);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim * 2);

      int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; v++) {
        float* vec = const_cast<float*>(
            v == 0 ? input_q.ptr<float>()
                   : input_k.ptr<float>());  // the vector to rotate (query or key)
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size / 2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t head_dim = i % head_size; head_dim < head_size / 2; head_dim++) {
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim * 2);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim * 2);

      int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; v++) {
        float* vec = const_cast<float*>(
            v == 0 ? input_q.ptr<float>()
                   : input_k.ptr<float>());  // the vector to rotate (query or key)
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size / 2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}
#else
// 预计算和存储
// head_size: 每个头的维度大小，max_seq_len: 最大序列长度
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}
// RoPE核心计算逻辑
// 使用预计算的 sin/cos 缓存，对 Q 和 K 进行旋转变换
// dim: Q/K的维度大小，kv_dim: K的维度大小，head_size: 每个头的维度大小
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  // 获取起始位置 pos，单token时它表示当前token位置，2D时它表示start_pos
  const int32_t start_pos = *input_pos.ptr<int32_t>(0);
  // 2. 核心lambda：对【单个位置】的Q、K向量执行旋转
  auto rotate_one = [&](int32_t pos, float* q, float* k) {
    for (int32_t i = 0; i < dim; i += 2) {
      int32_t head_dim = i % head_size;
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim);
      // 旋转次数：i在K的维度内 → 旋转Q+K；否则只旋转Q
      int32_t rotn = i < kv_dim ? 2 : 1;
      for (int32_t v = 0; v < rotn; v++) {
        float* vec = (v == 0 ? q : k);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }
  };
  // 单批量推理
  if (input_q.dims_size() == 1) {
    rotate_one(start_pos, const_cast<float*>(input_q.ptr<float>()),
               const_cast<float*>(input_k.ptr<float>()));
    return;
  }

  CHECK_EQ(input_q.dims_size(), 2);
  CHECK_EQ(input_k.dims_size(), 2);
  const int32_t T = input_q.get_dim(0);
  CHECK_EQ(input_q.get_dim(1), dim);
  CHECK_EQ(input_k.get_dim(0), T);
  CHECK_EQ(input_k.get_dim(1), kv_dim);

  float* q0 = const_cast<float*>(input_q.ptr<float>());
  float* k0 = const_cast<float*>(input_k.ptr<float>());

  for (int32_t t = 0; t < T; ++t) {
    rotate_one(start_pos + t, q0 + (size_t)t * dim, k0 + (size_t)t * kv_dim);
  }
}
#endif
}  // namespace kernel