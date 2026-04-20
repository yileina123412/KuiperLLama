#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include "../kernels_interface.h"
#include "tensor/tensor.h"
// // 从一个byte里面拆出两个4bit
// __device__ __forceinline__ uint8_t load_nibble(const uint8_t* p, size_t idx) {
//   const uint8_t byte = p[idx >> 1];
//   return (idx & 1) ? (byte >> 4) : (byte & 0x0F);
// }

// __device__ __forceinline__ float dequant_sq4(const uint8_t* q, const float* s, const uint8_t* z,
//                                              int k, int i, int M, int group_size) {
//   const size_t w_idx = (size_t)k * (size_t)M + (size_t)i;
//   const uint8_t q4 = load_nibble(q, w_idx);

//   const int groups_per_row = M / group_size;
//   const size_t g_idx = (size_t)k * (size_t)groups_per_row + (size_t)(i / group_size);

//   const uint8_t z4 = load_nibble(z, g_idx);
//   const float scale = s[g_idx];

//   return (float)((int)q4 - (int)z4) * scale;
// }

// // 1D: [M] -> [K]
// __global__ void matmul_sq4_vec_kernel(const float* x, const uint8_t* q, const float* s,
//                                       const uint8_t* z, float* y, int M, int K, int group_size) {
//   const int k = (int)blockIdx.x;
//   const int tid = (int)threadIdx.x;

//   extern __shared__ float ssum[];
//   float sum = 0.f;

//   for (int i = tid; i < M; i += blockDim.x) {
//     const float w = dequant_sq4(q, s, z, k, i, M, group_size);
//     sum += x[i] * w;
//   }

//   ssum[tid] = sum;
//   __syncthreads();

//   for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
//     if (tid < offset) ssum[tid] += ssum[tid + offset];
//     __syncthreads();
//   }

//   if (tid == 0) y[k] = ssum[0];
// }

// // 2D: [B, M] -> [B, K]
// __global__ void matmul_sq4_batch_kernel(const float* x, const uint8_t* q, const float* s,
//                                         const uint8_t* z, float* y, int B, int M, int K,
//                                         int group_size) {
//   const int b = (int)blockIdx.x;
//   const int k = (int)blockIdx.y;
//   const int tid = (int)threadIdx.x;

//   extern __shared__ float ssum[];
//   float sum = 0.f;

//   const float* xb = x + (size_t)b * M;

//   for (int i = tid; i < M; i += blockDim.x) {
//     const float w = dequant_sq4(q, s, z, k, i, M, group_size);
//     sum += xb[i] * w;
//   }

//   ssum[tid] = sum;
//   __syncthreads();

//   for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
//     if (tid < offset) ssum[tid] += ssum[tid + offset];
//     __syncthreads();
//   }

//   if (tid == 0) y[(size_t)b * K + k] = ssum[0];
// }

// void matmul_kernel_cu_sq4(const tensor::Tensor& input, const tensor::Tensor& qweight_packed,
//                           const tensor::Tensor& scales, const tensor::Tensor& zeros_packed,
//                           const tensor::Tensor& output, int32_t rows, int32_t cols,
//                           int32_t group_size, const kernel::CudaConfig* config) {
//   //   CHECK(config != nullptr);

//   //   CHECK(!input.is_empty());
//   //   CHECK(!qweight_packed.is_empty());
//   //   CHECK(!scales.is_empty());
//   //   CHECK(!zeros_packed.is_empty());
//   //   CHECK(!output.is_empty());

//   //   CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
//   //   CHECK(qweight_packed.device_type() == base::DeviceType::kDeviceCUDA);
//   //   CHECK(scales.device_type() == base::DeviceType::kDeviceCUDA);
//   //   CHECK(zeros_packed.device_type() == base::DeviceType::kDeviceCUDA);
//   //   CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

//   //   CHECK(input.data_type() == base::DataType::kDataTypeFp32);
//   //   CHECK(output.data_type() == base::DataType::kDataTypeFp32);
//   //   CHECK(qweight_packed.data_type() == base::DataType::kDataTypeInt8);
//   //   CHECK(zeros_packed.data_type() == base::DataType::kDataTypeInt8);
//   //   CHECK(scales.data_type() == base::DataType::kDataTypeFp32);

//   //   CHECK(group_size > 0);
//   //   CHECK(cols % group_size == 0);

//   const int32_t K = rows;
//   const int32_t M = cols;

//   const int threads = 256;
//   const size_t shmem = threads * sizeof(float);

//   const uint8_t* q = reinterpret_cast<const uint8_t*>(qweight_packed.ptr<int8_t>());
//   const float* s = scales.ptr<float>();
//   const uint8_t* z = reinterpret_cast<const uint8_t*>(zeros_packed.ptr<int8_t>());

//   if (input.dims_size() == 1) {
//     // CHECK_EQ(input.get_dim(0), M);
//     // CHECK_EQ(output.dims_size(), 1);
//     // CHECK_EQ(output.get_dim(0), K);

//     if (config->stream) {
//       matmul_sq4_vec_kernel<<<K, threads, shmem, config->stream>>>(
//           input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), M, K,
//           group_size);
//     } else {
//       matmul_sq4_vec_kernel<<<K, threads, shmem>>>(
//           input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), M, K,
//           group_size);
//     }
//     return;
//   }

//   //   CHECK_EQ(input.dims_size(), 2);
//   const int32_t B = input.get_dim(0);
//   //   CHECK_EQ(input.get_dim(1), M);

//   //   CHECK_EQ(output.dims_size(), 2);
//   //   CHECK_EQ(output.get_dim(0), B);
//   //   CHECK_EQ(output.get_dim(1), K);

//   dim3 grid(B, K);
//   if (config->stream) {
//     matmul_sq4_batch_kernel<<<grid, threads, shmem, config->stream>>>(
//         input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), B, M, K,
//         group_size);
//   } else {
//     matmul_sq4_batch_kernel<<<grid, threads, shmem>>>(
//         input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), B, M, K,
//         group_size);
//   }
// }