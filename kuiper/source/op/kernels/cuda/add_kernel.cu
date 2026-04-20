#include "add_kernel.cuh"

namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) return;
  out[tid] = in1[tid] + in2[tid];
}

// output/input1: [rows, cols], input2: [cols]
__global__ void add_bias_rowwise_cu_fp32(int32_t rows, int32_t cols, const float* in,
                                         const float* bias, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t total = rows * cols;
  if (tid >= total) return;
  int32_t c = tid % cols;
  out[tid] = in[tid] + bias[c];
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  const int32_t thread_num = 512;
  cudaStream_t stream_ = stream ? static_cast<CUstream_st*>(stream) : nullptr;

  // Case A: elementwise same-shape add
  if (input1.size() == input2.size() && input1.size() == output.size()) {
    int32_t size = static_cast<int32_t>(input1.size());
    int32_t block_num = (size + thread_num - 1) / thread_num;
    if (stream_) {
      add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
          size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    } else {
      add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                    const_cast<float*>(output.ptr<float>()));
    }
    return;
  }

  // Case B: row-wise broadcast: [B, C] + [C] -> [B, C]
  if (input1.dims_size() == 2 && output.dims_size() == 2 && input2.dims_size() == 1) {
    const int32_t rows = input1.get_dim(0);
    const int32_t cols = input1.get_dim(1);

    CHECK_EQ(output.get_dim(0), rows);
    CHECK_EQ(output.get_dim(1), cols);
    CHECK_EQ(input2.get_dim(0), cols);

    const int32_t total = rows * cols;
    const int32_t block_num = (total + thread_num - 1) / thread_num;
    if (stream_) {
      add_bias_rowwise_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
          rows, cols, input1.ptr<float>(), input2.ptr<float>(),
          const_cast<float*>(output.ptr<float>()));
    } else {
      add_bias_rowwise_cu_fp32<<<block_num, thread_num>>>(rows, cols, input1.ptr<float>(),
                                                          input2.ptr<float>(),
                                                          const_cast<float*>(output.ptr<float>()));
    }
    return;
  }

  LOG(FATAL) << "add_kernel_cu shape mismatch: input1.size=" << input1.size()
             << ", input2.size=" << input2.size() << ", output.size=" << output.size()
             << ", input1.dims_size=" << input1.dims_size()
             << ", input2.dims_size=" << input2.dims_size()
             << ", output.dims_size=" << output.dims_size();
}
}  // namespace kernel