#ifndef MATMUL_SQ4_KERNEL_CU_CUH
#define MATMUL_SQ4_KERNEL_CU_CUH

#include "../kernels_interface.h"
#include "tensor/tensor.h"

namespace kernel {

void matmul_kernel_cu_sq4(const tensor::Tensor& input, const tensor::Tensor& qweight_packed,
                          const tensor::Tensor& scales, const tensor::Tensor& zeros_packed,
                          const tensor::Tensor& output, int32_t rows, int32_t cols,
                          int32_t group_size, const CudaConfig* config);

}  // namespace kernel

#endif  // MATMUL_SQ4_KERNEL_CU_CUH