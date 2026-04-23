#include "../cpu/mha_kernel.h"
#include <cuda_runtime_api.h>
#include "../kernels_interface.h"
namespace kernel {
// 多头注意力的前向计算过程
// pos：当前解码到的token位置  seq_len：最大序列长度  head_size：每个head的维度
// query_tensor：当前位置的查询张量
// kv_mul：分组查询注意力的比例  等于 (Query头数) / (KV头数)。
// mha_out：存放融合了value的最终输出
// score_tensor:连续的草稿纸内存，咱村当前Quiery和历史key计算出的中间值
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, int32_t kv_window_size, int32_t kv_valid_len,
                int32_t kv_prefix_keep_tokens, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
  const int32_t layer_offset = layer_index * seq_len * kv_dim;
  const float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  const int32_t window = std::max(1, std::min(kv_window_size, seq_len));
  int32_t prefix = kv_prefix_keep_tokens;
  if (prefix < 0) prefix = 0;
  if (prefix >= window) prefix = window - 1;
  const int32_t tail_cap = window - prefix;

  auto logical_to_slot = [prefix, tail_cap](int32_t logical_pos) -> int32_t {
    if (logical_pos < prefix) return logical_pos;
    return prefix + ((logical_pos - prefix) % tail_cap);
  };

  // ===== prefill 2D =====
  if (query_tensor.dims_size() == 2) {
    const int32_t start_pos = pos;
    const int32_t T = query_tensor.get_dim(0);
    const int32_t dim = query_tensor.get_dim(1);
    CHECK_EQ(dim, head_num * head_size);

    CHECK_EQ(mha_out.dims_size(), 2);
    CHECK_EQ(mha_out.get_dim(0), T);
    CHECK_EQ(mha_out.get_dim(1), dim);

    const float* query = query_tensor.ptr<float>();
    const float* key_cache = key_cache_tensor.ptr<float>();
    const float* value_cache = value_cache_tensor.ptr<float>();
    float* out = const_cast<float*>(mha_out.ptr<float>());

    for (int32_t t = 0; t < T; ++t) {
      const int32_t cur_pos = start_pos + t;
      const int32_t seen = cur_pos + 1;

      const int32_t prefix_visible = std::min(prefix, seen);
      const int32_t tail_visible = std::min(std::max(seen - prefix, 0), tail_cap);
      const int32_t attn_len_t = std::max(1, prefix_visible + tail_visible);
      const int32_t tail_start = cur_pos - tail_visible + 1;

      for (int32_t h = 0; h < head_num; ++h) {
        const int32_t head_offset = (h / kv_mul) * head_size;
        const float* q = query + (size_t)t * dim + (size_t)h * head_size;
        float* o = out + (size_t)t * dim + (size_t)h * head_size;

        float max_score = -std::numeric_limits<float>::infinity();
        for (int32_t idx = 0; idx < attn_len_t; ++idx) {
          const int32_t logical_p =
              (idx < prefix_visible) ? idx : (tail_start + (idx - prefix_visible));
          const int32_t slot = logical_to_slot(logical_p);

          const float* k = key_cache + layer_offset + (size_t)slot * kv_dim + head_offset;
          float s = 0.f;
          for (int32_t i = 0; i < head_size; ++i) s += q[i] * k[i];
          s *= scale;
          if (s > max_score) max_score = s;
        }

        float sum_exp = 0.f;
        for (int32_t i = 0; i < head_size; ++i) o[i] = 0.f;

        for (int32_t idx = 0; idx < attn_len_t; ++idx) {
          const int32_t logical_p =
              (idx < prefix_visible) ? idx : (tail_start + (idx - prefix_visible));
          const int32_t slot = logical_to_slot(logical_p);

          const float* k = key_cache + layer_offset + (size_t)slot * kv_dim + head_offset;
          float s = 0.f;
          for (int32_t i = 0; i < head_size; ++i) s += q[i] * k[i];
          s *= scale;

          const float w = std::exp(s - max_score);
          sum_exp += w;

          const float* v = value_cache + layer_offset + (size_t)slot * kv_dim + head_offset;
          for (int32_t i = 0; i < head_size; ++i) o[i] += w * v[i];
        }

        const float inv = 1.f / sum_exp;
        for (int32_t i = 0; i < head_size; ++i) o[i] *= inv;
      }
    }
    return;
  }

  // ===== decode 1D =====
  const int32_t seen = pos + 1;

  // decode 用 runtime 传入的 kv_valid_len（再夹紧到 seen）
  const int32_t valid_req = std::max(1, std::min(kv_valid_len, window));
  const int32_t valid = std::min(valid_req, seen);

  const int32_t prefix_visible = std::min(prefix, valid);
  const int32_t tail_visible = std::max(0, valid - prefix_visible);
  const int32_t attn_len = std::max(1, prefix_visible + tail_visible);
  const int32_t tail_start = pos - tail_visible + 1;

  std::shared_ptr<base::DeviceAllocator> allocator;
  if (device_type == base::DeviceType::kDeviceCPU) {
    allocator = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    allocator = base::CUDADeviceAllocatorFactory::get_instance();
  }

  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                             query_head_addr);
    query_mat.set_device_type(device_type);

    for (int32_t idx = 0; idx < attn_len; ++idx) {
      const int32_t logical_t =
          (idx < prefix_visible) ? idx : (tail_start + (idx - prefix_visible));
      const int32_t slot = logical_to_slot(logical_t);

      const int32_t cache_offset = slot * kv_dim + (h / kv_mul) * head_size;
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + cache_offset;

      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             const_cast<float*>(key_head_addr));
      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr,
                               score_head_addr + idx);
      key_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
    }

    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, attn_len, false, nullptr,
                                     score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);

    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                           config ? config->stream : nullptr, false);

    for (int32_t i = 0; i < head_size; ++i) {
      float acc = 0.f;
      for (int32_t idx = 0; idx < attn_len; ++idx) {
        const int32_t logical_t =
            (idx < prefix_visible) ? idx : (tail_start + (idx - prefix_visible));
        const int32_t slot = logical_to_slot(logical_t);

        const int32_t cache_offset = slot * kv_dim + (h / kv_mul) * head_size;
        const float* value_head_addr =
            value_cache_tensor.ptr<float>() + layer_offset + cache_offset;
        acc += score_head_addr[idx] * value_head_addr[i];
      }
      output_head_ptr[i] = acc;
    }
  }
}
}  // namespace kernel