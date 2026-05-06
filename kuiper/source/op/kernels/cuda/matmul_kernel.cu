#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {

// 引入 const_cast 解决 const 指针转换报错问题
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(const_cast<float*>(&(pointer)))[0])
#define FETCH_CHAR4(pointer) (reinterpret_cast<const char4*>(const_cast<int8_t*>(&(pointer)))[0])
#define TILE_SIZE 16
// 非量化
// THREAD_PER_BLOCK：每个block里的线程数  ROW_PER_BLOCK：每个block负责多少行输出
// 这个模板是共享内存和cub库的时候要用的，需要编译时已知而不是运行时已知
// M：输入长度  K：权重行数/输出长度
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  // 设置共享内存
  // 一个block里的共享数组
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;  // 线程块内的编号

  // 设置每个线程的负责的行数
  // 每个block负责计算的行数
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  // float4向量化
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;
// 遍历本block负责的行
#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    // 权重偏移
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);
// 每个线程处理若干个float4块
#pragma unroll
    // 遍历float4向量
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }
    // 规约前先等全部算完
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

__global__ void matmul_kernel_cu_fp32_batch(const float* input, const float* weight, float* output,
                                            int B, int M, int K) {
  // block 维度: (TILE_SIZE, TILE_SIZE) -> (16, 16)
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 当前线程负责计算 output[row, col]
  int row = blockIdx.y * TILE_SIZE + ty;  // 映射到 B
  int col = blockIdx.x * TILE_SIZE + tx;  // 映射到 K

  // 申请共享内存，用于缓存一小块 Input 和 Weight
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sW[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  // 沿着 M 维度，以 TILE_SIZE 为步长分块步进
  for (int m = 0; m < M; m += TILE_SIZE) {
    // 1. 协作将 Input 数据读入共享内存
    if (row < B && m + tx < M) {
      sA[ty][tx] = input[row * M + (m + tx)];
    } else {
      sA[ty][tx] = 0.0f;
    }

    // 2. 协作将 Weight 数据读入共享内存
    // 注意：用户的 weight 是按 [K, M] 扁平化存储的
    // 所以 col 是行索引，m + ty 是列索引
    if (col < K && m + ty < M) {
      sW[ty][tx] = weight[col * M + (m + ty)];
    } else {
      sW[ty][tx] = 0.0f;
    }

    // 等待一个 Block 内的所有线程读完这一小块数据
    __syncthreads();

// 3. 在极速的共享内存中完成 16x16 的局部矩阵乘法
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += sA[ty][i] * sW[i][tx];
    }

    // 等待计算完成，然后再进入下一次 m 循环覆盖共享内存
    __syncthreads();
  }

  // 4. 将最终结果写回全局显存
  if (row < B && col < K) {
    output[row * K + col] = sum;
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

// ==========================================
// [极其重大的优化] 2D (Prefill) Tiled QInt8 算子 (包含提前反量化)
// ==========================================
__global__ void matmul_kernel_cu_qint8_batch(const float* input, const int8_t* weight,
                                             const float* scales, int group_size, float* output,
                                             int B, int M, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_SIZE + ty;  // 映射到 B
  int col = blockIdx.x * TILE_SIZE + tx;  // 映射到 K

  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  // 共享内存里存的是解压后的 float 数据！
  __shared__ float sW[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  for (int m = 0; m < M; m += TILE_SIZE) {
    // 1. 读 Input
    if (row < B && m + tx < M) {
      sA[ty][tx] = input[row * M + (m + tx)];
    } else {
      sA[ty][tx] = 0.0f;
    }

    // 2. 读 Weight，并同时完成【反量化】操作！
    if (col < K && m + ty < M) {
      const int weight_idx = col * M + (m + ty);
      const int group_idx = weight_idx / group_size;
      // 将 int8 转 float 并乘上 scale，存入极速的共享内存
      sW[ty][tx] = static_cast<float>(weight[weight_idx]) * scales[group_idx];
    } else {
      sW[ty][tx] = 0.0f;
    }

    __syncthreads();

// 3. 此时的计算逻辑与 FP32 完全一致，极其纯粹且高效
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += sA[ty][i] * sW[i][tx];
    }

    __syncthreads();
  }

  if (row < B && col < K) {
    output[row * K + col] = sum;
  }
}

// ==========================================
// [FP32 极致优化版] 借鉴 mysgemm_v7 架构
// BM=128, BN=128, BK=8, TM=8, TN=8
// ==========================================
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8,
          const int TN = 8>
__global__ void __launch_bounds__(256)
    matmul_kernel_cu_fp32_batch_opt(const float* input, const float* weight, float* output, int B,
                                    int M, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;

  // 线程映射：每个 block 256 线程，切分为 16x16 的网格
  int tx = (tid % (BN / TN)) * TN;
  int ty = (tid / (BN / TN)) * TM;

  // 双缓冲共享内存，大小 [2][BK][BM/BN]
  // 为了避免 Bank Conflict，我们让线程连续写内层维度
  __shared__ float As[2][BK][BM];
  __shared__ float Bs[2][BK][BN];

  // 全局显存搬运坐标 (每个线程搬运 4 个元素)
  int load_a_row = tid % BM;
  int load_a_col = (tid / BM) * 4;
  int load_b_row = tid % BN;
  int load_b_col = (tid / BN) * 4;

  int global_a_row = by * BM + load_a_row;
  int global_b_row = bx * BN + load_b_row;

  // 寄存器分块 (类似 accum[TM][TN], a_frag, b_frag)
  float accum[TM][TN] = {0.f};
  float a_frag[2][TM];
  float b_frag[2][TN];
  float4 ldg_a_reg;
  float4 ldg_b_reg;

  // ----------------------------------------
  // Prologue: 预取第一个块 (Block 0)
  // ----------------------------------------
  if (global_a_row < B && load_a_col < M) {
    ldg_a_reg = FETCH_FLOAT4(input[global_a_row * M + load_a_col]);
  } else {
    ldg_a_reg = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  As[0][load_a_col][load_a_row] = ldg_a_reg.x;
  As[0][load_a_col + 1][load_a_row] = ldg_a_reg.y;
  As[0][load_a_col + 2][load_a_row] = ldg_a_reg.z;
  As[0][load_a_col + 3][load_a_row] = ldg_a_reg.w;

  if (global_b_row < K && load_b_col < M) {
    // 由于 Weight 是 [K, M]，所以 load_b_col 沿着 M 维前进，可以完美 float4 读取！
    ldg_b_reg = FETCH_FLOAT4(weight[global_b_row * M + load_b_col]);
  } else {
    ldg_b_reg = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  Bs[0][load_b_col][load_b_row] = ldg_b_reg.x;
  Bs[0][load_b_col + 1][load_b_row] = ldg_b_reg.y;
  Bs[0][load_b_col + 2][load_b_row] = ldg_b_reg.z;
  Bs[0][load_b_col + 3][load_b_row] = ldg_b_reg.w;

  __syncthreads();

// 预取第一个计算碎片到寄存器
#pragma unroll
  for (int m = 0; m < TM; m += 4) {
    FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][0][ty + m]);
  }
#pragma unroll
  for (int n = 0; n < TN; n += 4) {
    FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][0][tx + n]);
  }

  int write_idx = 1;
  int load_idx;

  // ----------------------------------------
  // Main Loop: 软件流水线
  // ----------------------------------------
  for (int k_idx = 0; k_idx < M; k_idx += BK) {
    int k_next = k_idx + BK;

    // 1. 全局显存异步预取下一个大块 (隐藏访存延迟)
    if (k_next < M) {
      if (global_a_row < B && k_next + load_a_col < M) {
        ldg_a_reg = FETCH_FLOAT4(input[global_a_row * M + k_next + load_a_col]);
      } else {
        ldg_a_reg = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      if (global_b_row < K && k_next + load_b_col < M) {
        ldg_b_reg = FETCH_FLOAT4(weight[global_b_row * M + k_next + load_b_col]);
      } else {
        ldg_b_reg = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }

    load_idx = write_idx ^ 1;

// 2. 当前块的计算 (前 BK-1 步)
#pragma unroll
    for (int bk = 0; bk < BK - 1; bk++) {
// 预取下一步的寄存器碎片
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(As[load_idx][bk + 1][ty + m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(Bs[load_idx][bk + 1][tx + n]);
      }
// 乘加计算
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
        }
      }
    }

    // 3. 将预取好的下一个大块写入共享内存 (Double Buffering)
    if (k_next < M) {
      As[write_idx][load_a_col][load_a_row] = ldg_a_reg.x;
      As[write_idx][load_a_col + 1][load_a_row] = ldg_a_reg.y;
      As[write_idx][load_a_col + 2][load_a_row] = ldg_a_reg.z;
      As[write_idx][load_a_col + 3][load_a_row] = ldg_a_reg.w;

      Bs[write_idx][load_b_col][load_b_row] = ldg_b_reg.x;
      Bs[write_idx][load_b_col + 1][load_b_row] = ldg_b_reg.y;
      Bs[write_idx][load_b_col + 2][load_b_row] = ldg_b_reg.z;
      Bs[write_idx][load_b_col + 3][load_b_row] = ldg_b_reg.w;

      __syncthreads();

// 为下一次循环的 bk=0 预取寄存器
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[write_idx][0][ty + m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[write_idx][0][tx + n]);
      }
      write_idx ^= 1;
    }

// 4. 计算当前块的最后一步 (第 BK-1 步)
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
      for (int n = 0; n < TN; n++) {
        accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
      }
    }
  }

// ----------------------------------------
// Epilogue: 结果写回全局显存
// ----------------------------------------
#pragma unroll
  for (int m = 0; m < TM; m++) {
    int global_y = by * BM + ty + m;
    if (global_y < B) {
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        int global_x = bx * BN + tx + n;
        if (global_x < K) {
          float4 ctmp;
          ctmp.x = accum[m][n];
          ctmp.y = accum[m][n + 1];
          ctmp.z = accum[m][n + 2];
          ctmp.w = accum[m][n + 3];
          FETCH_FLOAT4(output[global_y * K + global_x]) = ctmp;
        }
      }
    }
  }
}

// ==========================================
// [QInt8 极致优化版] 融合反量化流水线
// ==========================================
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8,
          const int TN = 8>
__global__ void __launch_bounds__(256)
    matmul_kernel_cu_qint8_batch_opt(const float* input, const int8_t* weight, const float* scales,
                                     int group_size, float* output, int B, int M, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;

  int tx = (tid % (BN / TN)) * TN;
  int ty = (tid / (BN / TN)) * TM;

  __shared__ float As[2][BK][BM];
  __shared__ float Bs[2][BK][BN];  // 共享内存中存放的依然是反量化后的 FP32 数据！

  int load_a_row = tid % BM;
  int load_a_col = (tid / BM) * 4;
  int load_b_row = tid % BN;
  int load_b_col = (tid / BN) * 4;

  int global_a_row = by * BM + load_a_row;
  int global_b_row = bx * BN + load_b_row;

  float accum[TM][TN] = {0.f};
  float a_frag[2][TM];
  float b_frag[2][TN];
  float4 ldg_a_reg;
  float4 ldg_b_reg;

  // Prologue (Block 0)
  if (global_a_row < B && load_a_col < M) {
    ldg_a_reg = FETCH_FLOAT4(input[global_a_row * M + load_a_col]);
  } else {
    ldg_a_reg = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  As[0][load_a_col][load_a_row] = ldg_a_reg.x;
  As[0][load_a_col + 1][load_a_row] = ldg_a_reg.y;
  As[0][load_a_col + 2][load_a_row] = ldg_a_reg.z;
  As[0][load_a_col + 3][load_a_row] = ldg_a_reg.w;

  if (global_b_row < K && load_b_col < M) {
    int weight_idx = global_b_row * M + load_b_col;
    int group_idx = weight_idx / group_size;
    float scale = scales[group_idx];

    // 绝妙的优化：一次读取 4 个 int8_t (char4)，在加载到寄存器时完成反量化
    char4 tmp_c = FETCH_CHAR4(weight[weight_idx]);
    ldg_b_reg.x = static_cast<float>(tmp_c.x) * scale;
    ldg_b_reg.y = static_cast<float>(tmp_c.y) * scale;
    ldg_b_reg.z = static_cast<float>(tmp_c.z) * scale;
    ldg_b_reg.w = static_cast<float>(tmp_c.w) * scale;
  } else {
    ldg_b_reg = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  Bs[0][load_b_col][load_b_row] = ldg_b_reg.x;
  Bs[0][load_b_col + 1][load_b_row] = ldg_b_reg.y;
  Bs[0][load_b_col + 2][load_b_row] = ldg_b_reg.z;
  Bs[0][load_b_col + 3][load_b_row] = ldg_b_reg.w;

  __syncthreads();

#pragma unroll
  for (int m = 0; m < TM; m += 4) {
    FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][0][ty + m]);
  }
#pragma unroll
  for (int n = 0; n < TN; n += 4) {
    FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][0][tx + n]);
  }

  int write_idx = 1;
  int load_idx;

  // Main Loop
  for (int k_idx = 0; k_idx < M; k_idx += BK) {
    int k_next = k_idx + BK;

    if (k_next < M) {
      if (global_a_row < B && k_next + load_a_col < M) {
        ldg_a_reg = FETCH_FLOAT4(input[global_a_row * M + k_next + load_a_col]);
      } else {
        ldg_a_reg = make_float4(0.f, 0.f, 0.f, 0.f);
      }

      if (global_b_row < K && k_next + load_b_col < M) {
        int weight_idx = global_b_row * M + k_next + load_b_col;
        int group_idx = weight_idx / group_size;
        float scale = scales[group_idx];

        char4 tmp_c = FETCH_CHAR4(weight[weight_idx]);
        ldg_b_reg.x = static_cast<float>(tmp_c.x) * scale;
        ldg_b_reg.y = static_cast<float>(tmp_c.y) * scale;
        ldg_b_reg.z = static_cast<float>(tmp_c.z) * scale;
        ldg_b_reg.w = static_cast<float>(tmp_c.w) * scale;
      } else {
        ldg_b_reg = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }

    load_idx = write_idx ^ 1;

#pragma unroll
    for (int bk = 0; bk < BK - 1; bk++) {
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(As[load_idx][bk + 1][ty + m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(Bs[load_idx][bk + 1][tx + n]);
      }

#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
        }
      }
    }

    if (k_next < M) {
      As[write_idx][load_a_col][load_a_row] = ldg_a_reg.x;
      As[write_idx][load_a_col + 1][load_a_row] = ldg_a_reg.y;
      As[write_idx][load_a_col + 2][load_a_row] = ldg_a_reg.z;
      As[write_idx][load_a_col + 3][load_a_row] = ldg_a_reg.w;

      Bs[write_idx][load_b_col][load_b_row] = ldg_b_reg.x;
      Bs[write_idx][load_b_col + 1][load_b_row] = ldg_b_reg.y;
      Bs[write_idx][load_b_col + 2][load_b_row] = ldg_b_reg.z;
      Bs[write_idx][load_b_col + 3][load_b_row] = ldg_b_reg.w;

      __syncthreads();

#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[write_idx][0][ty + m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[write_idx][0][tx + n]);
      }
      write_idx ^= 1;
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
      for (int n = 0; n < TN; n++) {
        accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
      }
    }
  }

#pragma unroll
  for (int m = 0; m < TM; m++) {
    int global_y = by * BM + ty + m;
    if (global_y < B) {
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        int global_x = bx * BN + tx + n;
        if (global_x < K) {
          float4 ctmp;
          ctmp.x = accum[m][n];
          ctmp.y = accum[m][n + 1];
          ctmp.z = accum[m][n + 2];
          ctmp.w = accum[m][n + 3];
          FETCH_FLOAT4(output[global_y * K + global_x]) = ctmp;
        }
      }
    }
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(!input.is_empty() && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(!weight.is_empty() && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(!output.is_empty());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t K = weight.get_dim(0);  // output dim
  const int32_t M = weight.get_dim(1);  // input dim

  // 1D: [M] -> [K]
  if (input.dims_size() == 1) {
    CHECK_EQ(M, input.get_dim(0));
    CHECK_EQ(output.dims_size(), 1);
    CHECK_EQ(output.get_dim(0), K);

    if (config && config->stream) {
      matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                                const_cast<float*>(output.ptr<float>()), M, K);
    }

    CHECK_EQ(scale, 1.f);  // 先这样，避免 silent wrong
    return;
  }

  // 2D batch: [B, M] -> [B, K]
  CHECK_EQ(input.dims_size(), 2);
  const int32_t B = input.get_dim(0);
  CHECK_EQ(M, input.get_dim(1));

  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(output.get_dim(0), B);
  CHECK_EQ(output.get_dim(1), K);

  // const int threads = 256;
  // dim3 grid(B, K);
  // size_t shmem = threads * sizeof(float);

  // // 对应 BM=128, BN=128 架构
  // dim3 threads(256);
  // dim3 grid((K + 128 - 1) / 128, (B + 128 - 1) / 128);

  // if (config && config->stream) {
  //   matmul_kernel_cu_fp32_batch_opt<32, 128, 8, 4, 8><<<grid, threads, 0, config->stream>>>(
  //       input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), B, M,
  //       K);
  // } else {
  //   matmul_kernel_cu_fp32_batch_opt<32, 128, 8, 4, 8><<<grid, threads>>>(
  //       input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), B, M,
  //       K);
  // }

  dim3 threads(TILE_SIZE, TILE_SIZE);  // 16x16 = 256
  // Grid 向上取整，确保覆盖全部数据
  dim3 grid((K + TILE_SIZE - 1) / TILE_SIZE, (B + TILE_SIZE - 1) / TILE_SIZE);

  if (config && config->stream) {
    matmul_kernel_cu_fp32_batch<<<grid, threads, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), B, M, K);
  } else {
    matmul_kernel_cu_fp32_batch<<<grid, threads>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), B, M, K);
  }

  CHECK_EQ(scale, 1.f);  // 先这样，避免 silent wrong
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  // 1D: [M] -> [K]（decode）
  if (input.dims_size() == 1) {
    int packet_size = 4;
    CHECK_EQ(M % packet_size, 0);
    CHECK_EQ(M, input.get_dim(0));

    CHECK_EQ(output.dims_size(), 1);
    CHECK_EQ(output.get_dim(0), K);

    if (config->stream) {
      matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
          input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
          const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                    scale.ptr<float>(), group_size,
                                                    const_cast<float*>(output.ptr<float>()), M, K);
    }
    return;
  }

  // 2D batch: [B, M] -> [B, K]（prefill）
  CHECK_EQ(input.dims_size(), 2);
  const int32_t B = input.get_dim(0);
  CHECK_EQ(M, input.get_dim(1));

  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(output.get_dim(0), B);
  CHECK_EQ(output.get_dim(1), K);

  // dim3 threads(256);
  // dim3 grid((K + 128 - 1) / 128, (B + 128 - 1) / 128);
  // if (config->stream) {
  //   matmul_kernel_cu_qint8_batch_opt<32, 128, 8, 4, 8><<<grid, threads, 0, config->stream>>>(
  //       input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
  //       const_cast<float*>(output.ptr<float>()), B, M, K);
  // } else {
  //   matmul_kernel_cu_qint8_batch_opt<32, 128, 8, 4, 8>
  //       <<<grid, threads>>>(input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(),
  //                           group_size, const_cast<float*>(output.ptr<float>()), B, M, K);
  // }

  dim3 threads(TILE_SIZE, TILE_SIZE);  // 16x16
  dim3 grid((K + TILE_SIZE - 1) / TILE_SIZE, (B + TILE_SIZE - 1) / TILE_SIZE);

  if (config->stream) {
    matmul_kernel_cu_qint8_batch<<<grid, threads, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), B, M, K);
  } else {
    matmul_kernel_cu_qint8_batch<<<grid, threads>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), B, M, K);
  }
}

}  // namespace kernel