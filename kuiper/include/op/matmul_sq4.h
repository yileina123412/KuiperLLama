#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "op/layer.h"

namespace op {

// 这是“从 mmap 里读出来的 SQ4 权重视图”，不拷贝数据。
// 约定：rows=out_features(dim0), cols=in_features(dim1)
struct SQ4WeightView {
  int32_t rows = 0;
  int32_t cols = 0;
  int32_t group_size = 128;

  const void* qweight_packed = nullptr;  // byte array, len=q_bytes
  size_t q_bytes = 0;

  const void* scales = nullptr;  // float array bytes, len=s_bytes
  size_t s_bytes = 0;

  const void* zeros_packed = nullptr;  // byte array, len=z_bytes
  size_t z_bytes = 0;
};

class MatmulSQ4Layer : public Layer {
 public:
  explicit MatmulSQ4Layer(base::DeviceType device_type, int32_t dim0, int32_t dim1);

  base::Status check() const override;
  base::Status forward() override;
  void to_cuda() override;

  // 绑定权重（zero-copy 指向 mmap 区域）
  base::Status set_sq4_weight(const SQ4WeightView& view,
                              base::DeviceType weight_device_type = base::DeviceType::kDeviceCPU,
                              bool strict_size_check = true);

  int32_t dim0() const { return dim0_; }
  int32_t dim1() const { return dim1_; }
  int32_t group_size() const { return group_size_; }

  const tensor::Tensor& qweight_packed() const { return qweight_packed_; }
  const tensor::Tensor& scales() const { return scales_; }
  const tensor::Tensor& zeros_packed() const { return zeros_packed_; }

 private:
  static size_t expected_q_bytes(int32_t rows, int32_t cols);
  static size_t expected_group_count(int32_t rows, int32_t cols, int32_t group_size);
  static size_t expected_s_bytes(int32_t rows, int32_t cols, int32_t group_size);
  static size_t expected_z_bytes(int32_t rows, int32_t cols, int32_t group_size);

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
  int32_t group_size_ = 128;

  // 存储方式：全部是 1D tensor（bytes / floats），维度信息由 dim0_/dim1_/group_size_ 提供
  tensor::Tensor qweight_packed_;  // int8, [q_bytes]
  tensor::Tensor scales_;          // fp32, [group_count]
  tensor::Tensor zeros_packed_;    // int8, [z_bytes]
};

}  // namespace op