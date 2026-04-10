#ifndef KUIPER_INCLUDE_OP_ROPE_H_
#define KUIPER_INCLUDE_OP_ROPE_H_
#include "layer.h"
namespace op {
class RoPELayer : public Layer {
 public:
  explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;        // query的完整维度   隐藏层维度
  int32_t kv_dim_ = 0;     // Key/value的维度
  int32_t head_size_ = 0;  // 每个注意力头的维度
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ROPE_H_
