#ifndef RAW_MODEL_DATA_H
#define RAW_MODEL_DATA_H
#include <cstddef>
#include <cstdint>
/**
 * 原始模型数据的封装和管理  统一管理从模型文件映射的原始数据
 */
namespace model {
// 统一管理mmap的映射的资源
struct RawModelData {
  ~RawModelData();
  int32_t fd = -1;              // 文件描述符
  size_t file_size = 0;         // 文件大小
  void* data = nullptr;         // mmap映射的起始地址
  void* weight_data = nullptr;  // 权重指针
  // 进行数据偏移，根据数据类型决定偏移多少  获取第几个数据
  virtual const void* weight(size_t offset) const = 0;
};
// 多态处理不同数据类型
// float数据类型
struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;
};

}  // namespace model
#endif  // RAW_MODEL_DATA_H
