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
                const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                CudaConfig* config) {
  // ===== 新增：batch prefill (query: [T, dim]) =====
  if (query_tensor.dims_size() == 2) {
    const int32_t start_pos = pos;
    const int32_t T = query_tensor.get_dim(0);
    const int32_t dim = query_tensor.get_dim(1);
    CHECK_EQ(dim, head_num * head_size);

    CHECK_EQ(mha_out.dims_size(), 2);
    CHECK_EQ(mha_out.get_dim(0), T);
    CHECK_EQ(mha_out.get_dim(1), dim);

    // cache layout: [layer, seq, kv_dim] flattened
    const int32_t layer_offset = layer_index * seq_len * kv_dim;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_size));

    const float* query = query_tensor.ptr<float>();
    const float* key_cache = key_cache_tensor.ptr<float>();
    const float* value_cache = value_cache_tensor.ptr<float>();
    float* out = const_cast<float*>(mha_out.ptr<float>());

    const int32_t window = std::max(1, std::min(kv_window_size, seq_len));

    for (int32_t t = 0; t < T; ++t) {
      const int32_t cur_pos = start_pos + t;

      const int32_t valid_len_t = std::max(1, std::min(cur_pos + 1, window));
      const int32_t logical_start = cur_pos - valid_len_t + 1;

      for (int32_t h = 0; h < head_num; ++h) {
        const int32_t head_offset = (h / kv_mul) * head_size;
        const float* q = query + (size_t)t * dim + (size_t)h * head_size;
        float* o = out + (size_t)t * dim + (size_t)h * head_size;

        // pass1: max score
        float max_score = -std::numeric_limits<float>::infinity();
        for (int32_t idx = 0; idx < valid_len_t; ++idx) {
          const int32_t logical_p = logical_start + idx;
          int32_t slot = logical_p % window;
          if (slot < 0) slot += window;
          const float* k = key_cache + layer_offset + (size_t)slot * kv_dim + head_offset;
          float s = 0.f;
          for (int32_t i = 0; i < head_size; ++i) s += q[i] * k[i];
          s *= scale;
          if (s > max_score) max_score = s;
        }

        // pass2: sumexp + weighted value
        float sum_exp = 0.f;
        for (int32_t i = 0; i < head_size; ++i) o[i] = 0.f;

        for (int32_t idx = 0; idx < valid_len_t; ++idx) {
          const int32_t logical_p = logical_start + idx;
          int32_t slot = logical_p % window;
          if (slot < 0) slot += window;
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
  // decode
  // 定位本层cache，本层的kv缓存         缩放系数  缩放点积注意力的那个长度
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  const int32_t window = std::max(1, std::min(kv_window_size, seq_len));
  const int32_t valid_len = std::max(1, std::min(kv_valid_len, window));
  const int32_t logical_start = pos - valid_len + 1;
  CHECK_GE(logical_start, 0);

  auto logical_to_slot = [window](int32_t logical_pos) -> int32_t {
    int32_t s = logical_pos % window;
    if (s < 0) s += window;
    return s;
  };
  // 选择内存分配器
  std::shared_ptr<base::DeviceAllocator> allocator;
  if (device_type == base::DeviceType::kDeviceCPU) {
    allocator = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    allocator = base::CUDADeviceAllocatorFactory::get_instance();
  }
  // 逐个head计算attention
  for (int32_t h = 0; h < head_num; ++h) {
    // 每个头独立计算自己的q和v的匹配度

    // 获取当前头对应的那部分score数组和query数组的指针
    // 多头跟别计算各自跟历史记录的点积
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    // 按照头切割query
    float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);
    // 零拷贝视图 将query_head_addr包装到tensor里面 为了和后面接口一致
    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                             query_head_addr);
    query_mat.set_device_type(device_type);

    // 计算q和历史的key的点积
    for (int32_t t = 0; t < valid_len; t++) {
      const int32_t logical_t = logical_start + t;
      const int32_t slot = logical_to_slot(logical_t);

      // 每个头只和每个头的对应的算
      int32_t cache_offset = slot * kv_dim + (h / kv_mul) * head_size;
      // key的头的指针
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + cache_offset;

      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             const_cast<float*>(key_head_addr));

      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr,
                               score_head_addr + t);
      key_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
    }

    // 计算注意力分数  对socre做softmax转换成概率  存放softmax的结果
    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, valid_len, false, nullptr,
                                     score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);
    // 存放结果
    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                           config ? config->stream : nullptr, false);

    for (int32_t i = 0; i < head_size; i++) {
      float acc = 0.f;
      for (int32_t idx = 0; idx < valid_len; ++idx) {
        const int32_t logical_t = logical_start + idx;
        const int32_t slot = logical_to_slot(logical_t);

        int32_t cache_offset = slot * kv_dim + (h / kv_mul) * head_size;

        const float* value_head_addr =
            value_cache_tensor.ptr<float>() + layer_offset + cache_offset;
        acc += score_head_addr[idx] * value_head_addr[i];
      }
      output_head_ptr[i] = acc;
    }
    // tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
    //                              output_head_ptr);
    // output_tensor.set_device_type(device_type);
    // // 准备value缓存数据，以和前面计算的权重加权
    // int32_t cache_offset = (h / kv_mul) * head_size;
    // float* value_head_addr =
    //     const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
    // tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
    //                             value_head_addr);
    // get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos,
    //                                   head_size, kv_dim, config ? config->stream : nullptr);
  }
}
}  // namespace kernel