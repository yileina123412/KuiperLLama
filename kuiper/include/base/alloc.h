#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"
namespace base {
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};
// 作用：基类，内存管理的通用方法：包括allocate free memcpy
class DeviceAllocator {
 public:
  // explicit 禁止隐式类型转换 防止传参错误导致莫名构造临时对象
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}
  virtual ~DeviceAllocator() = default;
  // 返回设备的类型
  virtual DeviceType device_type() const { return device_type_; }
  // 纯虚函数 基类不提供实现
  virtual void release(void* ptr) const = 0;

  virtual void* allocate(size_t byte_size) const = 0;
  // 由于是转移变量，不需要进行分设备，本身就是往不同设备转的
  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                      bool need_sync = false) const;

  // 将指定内存区域的所有字节设置为0  将内存内容清零 跨平台，gpu和cpu都能用
  // 分配内存后清零，确保计算正确 void* stream：cuda流，用于异步操作
  //  bool need_sync： 是否需要同步等待完成  操作提交后，CPU 必须等 GPU 完成清零，才能继续后续操作
  virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};
// 直接继承父类
class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();
  // override：表示这个成员函数是重写（覆盖）基类中的虚函数。
  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};
// 纯数据结构，存储内存地址，大小，这块内存是否被占用  没有释放开辟内存的权限 是个记账工具
// 内存块的元数据，记录每块显存的状态
struct CudaMemoryBuffer {
  void* data;        // 指针
  size_t byte_size;  // 内存大小
  bool busy;         // 是否正在使用

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};
// 显存池，负责cuda的部分内存申请和回收
class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

 private:
  // 这里使用map可以根据内存块的大小进行排序，快速找到大于等于所需尺寸的最小内存块
  // 这里的id是gpu的id,可以允许多个gpu,但我这里的设备是一个，因此没咋用到

  // 跟踪每个设备商空闲内存的总大小，之记录小内存的，作用是处理小内存池的清理
  mutable std::map<int, size_t> no_busy_cnt_;
  // 大内存缓冲区 >=1MB
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  // 小内存缓冲区 <1MB
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};
// 工厂模式 通过get_instance统一管理对象的生成
// 原因：只提供一个allocator分配器，防止混乱和浪费？
class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};
// 单例工厂  直接调用get_instance就可以申请CUDADeviceAllocator 而且只能有一个CUDADeviceAllocator

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};
}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_