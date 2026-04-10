#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
// cuda的信息 封装cuda相关信息
// cuda stream用于异步执行gpu任务，允许多个操作并发执行
// 这个结构体可以自动销毁stream
struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      // 销毁cuda stream流
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
