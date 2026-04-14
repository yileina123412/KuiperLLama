#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include "matmul_sq4_kernel.cuh"

namespace kernel {

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
//                           int32_t group_size, const CudaConfig* config) {
//   CHECK(config != nullptr);

//   CHECK(!input.is_empty());
//   CHECK(!qweight_packed.is_empty());
//   CHECK(!scales.is_empty());
//   CHECK(!zeros_packed.is_empty());
//   CHECK(!output.is_empty());

//   CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
//   CHECK(qweight_packed.device_type() == base::DeviceType::kDeviceCUDA);
//   CHECK(scales.device_type() == base::DeviceType::kDeviceCUDA);
//   CHECK(zeros_packed.device_type() == base::DeviceType::kDeviceCUDA);
//   CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

//   CHECK(input.data_type() == base::DataType::kDataTypeFp32);
//   CHECK(output.data_type() == base::DataType::kDataTypeFp32);
//   CHECK(qweight_packed.data_type() == base::DataType::kDataTypeInt8);
//   CHECK(zeros_packed.data_type() == base::DataType::kDataTypeInt8);
//   CHECK(scales.data_type() == base::DataType::kDataTypeFp32);

//   CHECK(group_size > 0);
//   CHECK(cols % group_size == 0);

//   const int32_t K = rows;
//   const int32_t M = cols;

//   const int threads = 256;
//   const size_t shmem = threads * sizeof(float);

//   const uint8_t* q = reinterpret_cast<const uint8_t*>(qweight_packed.ptr<int8_t>());
//   const float* s = scales.ptr<float>();
//   const uint8_t* z = reinterpret_cast<const uint8_t*>(zeros_packed.ptr<int8_t>());

//   if (input.dims_size() == 1) {
//     CHECK_EQ(input.get_dim(0), M);
//     CHECK_EQ(output.dims_size(), 1);
//     CHECK_EQ(output.get_dim(0), K);

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

//   CHECK_EQ(input.dims_size(), 2);
//   const int32_t B = input.get_dim(0);
//   CHECK_EQ(input.get_dim(1), M);

//   CHECK_EQ(output.dims_size(), 2);
//   CHECK_EQ(output.get_dim(0), B);
//   CHECK_EQ(output.get_dim(1), K);

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

__device__ __forceinline__ uint8_t nibble_lo(uint8_t byte) { return byte & 0x0F; }
__device__ __forceinline__ uint8_t nibble_hi(uint8_t byte) { return byte >> 4; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFF, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_256(float v) {
  // assumes blockDim.x == 256
  __shared__ float warp_sums[8];  // 256 / 32 = 8 warps
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();

  // Only lanes [0..7] are valid to load warp_sums[]
  float w = 0.0f;
  if (warp == 0 && lane < 8) {
    w = warp_sums[lane];
  }
  if (warp == 0) {
    w = warp_reduce_sum(w);
  }
  return w;  // correct sum in thread0
}

template <int GS, int SHIFT>
__global__ void matmul_sq4_vec_kernel_opt(const float* __restrict__ x,
                                          const uint8_t* __restrict__ q,
                                          const float* __restrict__ s,
                                          const uint8_t* __restrict__ z, float* __restrict__ y,
                                          int M, int K) {
  const int k = (int)blockIdx.x;
  float sum = 0.f;

  const int groups_per_row = M >> SHIFT;
  const int group_base = k * groups_per_row;

  // process 2 weights per iteration, i is even -> one q byte holds q(i), q(i+1)
  for (int i = (int)threadIdx.x * 2; i < M; i += (int)blockDim.x * 2) {
    const int w0 = k * M + i;
    const int q_byte_idx = w0 >> 1;
    const uint8_t qb = q[q_byte_idx];

    const float x0 = x[i];
    const uint8_t q0 = nibble_lo(qb);

    const int g0 = i >> SHIFT;
    const int gi0 = group_base + g0;
    const uint8_t zb0 = z[gi0 >> 1];
    const uint8_t z0 = (gi0 & 1) ? nibble_hi(zb0) : nibble_lo(zb0);
    const float s0 = s[gi0];
    sum += x0 * (float(int(q0) - int(z0)) * s0);

    const int i1 = i + 1;
    if (i1 < M) {
      const float x1 = x[i1];
      const uint8_t q1 = nibble_hi(qb);

      const int g1 = i1 >> SHIFT;
      const int gi1 = group_base + g1;
      uint8_t z1;
      float s1;
      if (gi1 == gi0) {
        z1 = z0;
        s1 = s0;
      } else {
        const uint8_t zb1 = z[gi1 >> 1];
        z1 = (gi1 & 1) ? nibble_hi(zb1) : nibble_lo(zb1);
        s1 = s[gi1];
      }
      sum += x1 * (float(int(q1) - int(z1)) * s1);
    }
  }

  const float total = block_reduce_sum_256(sum);
  if (threadIdx.x == 0) y[k] = total;
}

template <int GS, int SHIFT>
__global__ void matmul_sq4_batch_kernel_opt(const float* __restrict__ x,
                                            const uint8_t* __restrict__ q,
                                            const float* __restrict__ s,
                                            const uint8_t* __restrict__ z, float* __restrict__ y,
                                            int B, int M, int K) {
  const int b = (int)blockIdx.x;
  const int k = (int)blockIdx.y;
  float sum = 0.f;

  const float* xb = x + (size_t)b * (size_t)M;

  const int groups_per_row = M >> SHIFT;
  const int group_base = k * groups_per_row;

  for (int i = (int)threadIdx.x * 2; i < M; i += (int)blockDim.x * 2) {
    const int w0 = k * M + i;
    const int q_byte_idx = w0 >> 1;
    const uint8_t qb = q[q_byte_idx];

    const float x0 = xb[i];
    const uint8_t q0 = nibble_lo(qb);

    const int g0 = i >> SHIFT;
    const int gi0 = group_base + g0;
    const uint8_t zb0 = z[gi0 >> 1];
    const uint8_t z0 = (gi0 & 1) ? nibble_hi(zb0) : nibble_lo(zb0);
    const float s0 = s[gi0];
    sum += x0 * (float(int(q0) - int(z0)) * s0);

    const int i1 = i + 1;
    if (i1 < M) {
      const float x1 = xb[i1];
      const uint8_t q1 = nibble_hi(qb);

      const int g1 = i1 >> SHIFT;
      const int gi1 = group_base + g1;
      uint8_t z1;
      float s1;
      if (gi1 == gi0) {
        z1 = z0;
        s1 = s0;
      } else {
        const uint8_t zb1 = z[gi1 >> 1];
        z1 = (gi1 & 1) ? nibble_hi(zb1) : nibble_lo(zb1);
        s1 = s[gi1];
      }
      sum += x1 * (float(int(q1) - int(z1)) * s1);
    }
  }

  const float total = block_reduce_sum_256(sum);
  if (threadIdx.x == 0) y[(size_t)b * (size_t)K + (size_t)k] = total;
}

void matmul_kernel_cu_sq4(const tensor::Tensor& input, const tensor::Tensor& qweight_packed,
                          const tensor::Tensor& scales, const tensor::Tensor& zeros_packed,
                          const tensor::Tensor& output, int32_t rows, int32_t cols,
                          int32_t group_size, const CudaConfig* config) {
  CHECK(config != nullptr);

  CHECK(!input.is_empty());
  CHECK(!qweight_packed.is_empty());
  CHECK(!scales.is_empty());
  CHECK(!zeros_packed.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(qweight_packed.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(scales.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(zeros_packed.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(input.data_type() == base::DataType::kDataTypeFp32);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);
  CHECK(qweight_packed.data_type() == base::DataType::kDataTypeInt8);
  CHECK(zeros_packed.data_type() == base::DataType::kDataTypeInt8);
  CHECK(scales.data_type() == base::DataType::kDataTypeFp32);

  CHECK(group_size == 32 || group_size == 64 || group_size == 128)
      << "SQ4 kernel only supports group_size in {32,64,128}";
  CHECK(cols % group_size == 0);

  // NOTE: In MatmulLayer, weight dims are [rows, cols] where:
  //   output dim = rows (K)
  //   input  dim = cols (M)
  const int32_t K = rows;
  const int32_t M = cols;

  // Kernel assumes blockDim.x == 256 (see block_reduce_sum_256).
  const int threads = 256;
  const size_t shmem = 0;

  const uint8_t* q = reinterpret_cast<const uint8_t*>(qweight_packed.ptr<int8_t>());
  const float* s = scales.ptr<float>();
  const uint8_t* z = reinterpret_cast<const uint8_t*>(zeros_packed.ptr<int8_t>());

  // small helpers to reduce duplicated launch boilerplate
  auto launch_vec = [&](auto kernel_fn) {
    if (config->stream) {
      kernel_fn<<<K, threads, shmem, config->stream>>>(
          input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      kernel_fn<<<K, threads, shmem>>>(input.ptr<float>(), q, s, z,
                                       const_cast<float*>(output.ptr<float>()), M, K);
    }
  };

  auto launch_batch = [&](auto kernel_fn, int32_t B) {
    dim3 grid(B, K);
    if (config->stream) {
      kernel_fn<<<grid, threads, shmem, config->stream>>>(
          input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), B, M, K);
    } else {
      kernel_fn<<<grid, threads, shmem>>>(input.ptr<float>(), q, s, z,
                                          const_cast<float*>(output.ptr<float>()), B, M, K);
    }
  };

  if (input.dims_size() == 1) {
    CHECK_EQ(input.get_dim(0), M);
    CHECK_EQ(output.dims_size(), 1);
    CHECK_EQ(output.get_dim(0), K);

    switch (group_size) {
      case 32:
        launch_vec(matmul_sq4_vec_kernel_opt<32, 5>);
        break;
      case 64:
        launch_vec(matmul_sq4_vec_kernel_opt<64, 6>);
        break;
      case 128:
        launch_vec(matmul_sq4_vec_kernel_opt<128, 7>);
        break;
      default:
        LOG(FATAL) << "SQ4: unsupported group_size=" << group_size;
    }
    return;
  }

  CHECK_EQ(input.dims_size(), 2);
  const int32_t B = input.get_dim(0);
  CHECK_EQ(input.get_dim(1), M);

  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(output.get_dim(0), B);
  CHECK_EQ(output.get_dim(1), K);

  switch (group_size) {
    case 32:
      launch_batch(matmul_sq4_batch_kernel_opt<32, 5>, B);
      break;
    case 64:
      launch_batch(matmul_sq4_batch_kernel_opt<64, 6>, B);
      break;
    case 128:
      launch_batch(matmul_sq4_batch_kernel_opt<128, 7>, B);
      break;
    default:
      LOG(FATAL) << "SQ4: unsupported group_size=" << group_size;
  }
}

}  // namespace kernel