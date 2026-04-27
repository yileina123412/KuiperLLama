#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include "matmul_sq4_kernel.cuh"

namespace kernel {

#define TILE_SIZE 16

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

__global__ void matmul_sq4_batch_kernel_tiled(const float* __restrict__ x,
                                              const uint8_t* __restrict__ q,
                                              const float* __restrict__ s,
                                              const uint8_t* __restrict__ z, float* __restrict__ y,
                                              int B, int M, int K, int group_size) {
  // 16x16 的线程块
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 映射到全局输出矩阵的行 (Batch) 和 列 (K)
  int row_b = blockIdx.y * TILE_SIZE + ty;
  int col_k = blockIdx.x * TILE_SIZE + tx;

  // 申请极速的共享内存 (用来存 X 和 解压后的 W)
  __shared__ float sX[TILE_SIZE][TILE_SIZE];
  __shared__ float sW[TILE_SIZE][TILE_SIZE];  // 注意：这里存的是解压后的 FP32!

  float sum = 0.0f;

  // 沿着 M 维度分块推进
  for (int m = 0; m < M; m += TILE_SIZE) {
    // ==========================================
    // 1. 读入 Input (X) 并存入共享内存
    // ==========================================
    if (row_b < B && m + tx < M) {
      sX[ty][tx] = x[row_b * M + (m + tx)];
    } else {
      sX[ty][tx] = 0.0f;
    }

    // ==========================================
    // 2. 读入 4-bit 权重，瞬间解压，并存入共享内存
    // ==========================================
    if (col_k < K && m + ty < M) {
      // 计算全局的 1D 索引
      int w_idx = col_k * M + (m + ty);

      // --- 解压 4-bit 权重 (q) ---
      int q_byte_idx = w_idx >> 1;  // 除以 2
      uint8_t qb = q[q_byte_idx];
      // 偶数索引取低 4 位，奇数索引取高 4 位
      uint8_t q_val = ((m + ty) & 1) ? (qb >> 4) : (qb & 0x0F);

      // --- 解压 Zero Point (z) 和 Scale (s) ---
      int g_idx = w_idx / group_size;
      float scale = s[g_idx];

      uint8_t zb = z[g_idx >> 1];
      uint8_t z_val = (g_idx & 1) ? (zb >> 4) : (zb & 0x0F);

      // --- 极其关键：在此处完成反量化！---
      sW[ty][tx] = (float((int)q_val - (int)z_val)) * scale;
    } else {
      sW[ty][tx] = 0.0f;
    }

    // 等待整个 Block 把这 16x16 的数据全部加载且解压完毕
    __syncthreads();

// ==========================================
// 3. 在极速共享内存中进行矩阵乘法 (此时全是 FP32 纯粹的计算)
// ==========================================
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += sX[ty][i] * sW[i][tx];
    }

    // 计算完这块，同步一下，准备覆盖加载下一块
    __syncthreads();
  }

  // 将最终结果写回全局显存
  if (row_b < B && col_k < K) {
    y[row_b * K + col_k] = sum;
  }
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
  dim3 threads_tiled(16, 16);
  dim3 grid_tiled((K + 15) / 16, (B + 15) / 16);

  // const size_t shmem = 0;

  // 2. 直接调用新写的 Tiled 算子，彻底抛弃 switch 模板！
  if (config->stream) {
    matmul_sq4_batch_kernel_tiled<<<grid_tiled, threads_tiled, shmem, config->stream>>>(
        input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), B, M, K, group_size);
  } else {
    matmul_sq4_batch_kernel_tiled<<<grid_tiled, threads_tiled, shmem>>>(
        input.ptr<float>(), q, s, z, const_cast<float*>(output.ptr<float>()), B, M, K, group_size);
  }

  // switch (group_size) {
  //   case 32:
  //     launch_batch(matmul_sq4_batch_kernel_opt<32, 5>, B);
  //     break;
  //   case 64:
  //     launch_batch(matmul_sq4_batch_kernel_opt<64, 6>, B);
  //     break;
  //   case 128:
  //     launch_batch(matmul_sq4_batch_kernel_opt<128, 7>, B);
  //     break;
  //   default:
  //     LOG(FATAL) << "SQ4: unsupported group_size=" << group_size;
  // }
}

}  // namespace kernel