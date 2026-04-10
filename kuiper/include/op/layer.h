#ifndef KUIPER_INCLUDE_OP_LAYER_H_
#define KUIPER_INCLUDE_OP_LAYER_H_
#include <base/cuda_config.h>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

// 底层类，当llama3把某一层的内存指针给它的时候，把int8张量包装成fp32张量
// 这里layer没有还原，而是后面部分通过指针偏移声明fp32的张量
// 在推理的时候，gpu计算的时候才在寄存器还原扔掉。

/**
 * 神经网络层的基础架构，为整个llm推理框架提供统一的层接口和权重管理功能。
 * 所有层的基类模板
 */
namespace op {
// 纯虚基类 记录基本的元数据
class Layer;
enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};
// 最抽象的层接口  统一定义接口
// 通用函数都是纯虚函数，一些返回通用变量的在这里实现，以及设置权重的设置一个默认实现
class BaseLayer {
 public:
  explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name = "");

  base::DataType data_type() const;

  LayerType layer_type() const;
  // 初始化
  virtual base::Status init() = 0;

  virtual base::Status forward() = 0;
  // 前向计算
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;
  // 输入/输出管理  存取第 idx 个输张量
  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t input_size() const = 0;

  virtual size_t output_size() const = 0;

  virtual base::Status check() const = 0;

  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;
  // 提供默认实现
  virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

  virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                  const void* weight_ptr,
                                  base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

  const std::string& get_layer_name() const;

  void set_layer_name(const std::string& layer_name);

  base::DeviceType device_type() const;

  void set_device_type(base::DeviceType device_type);

 protected:
  std::string layer_name_;                                           // 层名
  LayerType layer_type_ = LayerType::kLayerUnknown;                  //层类型
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;      //数据类型
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;  // 设备类型
};

class Layer : public BaseLayer {
 public:
  explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

  base::Status init() override;

  base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                            base::DataType data_type) const;
  // 张量校验，可变参数校验张量的设备，数据类型和维度
  base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                     base::DataType data_type, ...) const;

  base::Status check() const override;

  base::Status forward() override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1) override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  size_t input_size() const override;

  size_t output_size() const override;
  // 预分配输入槽位数量
  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

  virtual void to_cuda();
  // 设置cuda 身体热爱名，device id等
  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

  std::shared_ptr<kernel::CudaConfig> cuda_config() const;

 protected:
  std::vector<tensor::Tensor> inputs_;               // 输入张量
  std::vector<tensor::Tensor> outputs_;              // 输出张量
  std::shared_ptr<kernel::CudaConfig> cuda_config_;  // cuda stream
};
// 带权重的层
class LayerParam : public Layer {
 public:
  explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                      bool is_quant_layer = false, std::string layer_name = "");

  size_t weight_size() const;

  void reset_weight_size(size_t size);

  tensor::Tensor& get_weight(int32_t idx);

  const tensor::Tensor& get_weight(int32_t idx) const;

  void to_cuda() override;
  // 设置权重方式
  base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;
  // 权重设置(指针切割方式)  核心函数  从大块连续内存中切割出权重张量(零拷贝)
  /**
   * idx:第几个权重，索引
   * dims：权重维度，形状
   * weight_ptr:指向权重数据的指针
   * device_type:权重的设备类型
   */
  base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                          base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

  void set_scales(const tensor::Tensor& scales);

  void set_group_size(int32_t group_size);

  int32_t get_scale_num() const;

 protected:
  int32_t group_size_ = 0;               // 量化分组大小  一组内的数量
  bool is_quant_layer_ = false;          // 是否为量化层
  tensor::Tensor scales_;                // 量化scale参数
  std::vector<tensor::Tensor> weights_;  // 权重列表
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_LAYER_H_
