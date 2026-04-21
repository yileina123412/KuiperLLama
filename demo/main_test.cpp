// #include <cublas_v2.h>
// #include <cuda_runtime_api.h>
// #include <glog/logging.h>
// #include <gtest/gtest.h>
// #include <algorithm>
// #include <cmath>
// #include <cstdint>
// #include <vector>
// #include "../source/op/kernels/cpu/matmul_kernel.h"
// #include "../source/op/kernels/kernels_interface.h"
// #include "/home/furina/code_learnning/cpp/cuda/KuiperLLama/test/utils.cuh"
// #include "base/buffer.h"
// #include "tensor/tensor.h"
// using namespace kernel;

// static inline int clamp_int(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

// static void build_q8_and_sq4_from_fp32(const tensor::Tensor& w_fp32, int K, int M, int
// group_size,
//                                        tensor::Tensor& w_q8, tensor::Tensor& s_q8,
//                                        tensor::Tensor& qweight_sq4_packed, tensor::Tensor& s_sq4,
//                                        tensor::Tensor& zeros_sq4_packed) {
//   CHECK_EQ(w_fp32.dims_size(), 2);
//   CHECK_EQ(w_fp32.get_dim(0), K);
//   CHECK_EQ(w_fp32.get_dim(1), M);
//   CHECK_EQ((K * M) % group_size, 0);

//   const int N = K * M;
//   const int group_cnt = N / group_size;

//   // 临时保存每个4bit量化值（0~15）
//   std::vector<uint8_t> q4((size_t)N, 0);

//   for (int g = 0; g < group_cnt; ++g) {
//     const int begin = g * group_size;
//     const int end = begin + group_size;

//     float max_abs = 0.f;
//     for (int i = begin; i < end; ++i) {
//       max_abs = std::max(max_abs, std::fabs(w_fp32.index<float>(i)));
//     }

//     const float scale_q8 = (max_abs < 1e-12f) ? 1.0f : (max_abs / 127.0f);
//     const float scale_sq4 = (max_abs < 1e-12f) ? 1.0f : (max_abs / 7.0f);  // 对应零点8

//     s_q8.index<float>(g) = scale_q8;
//     s_sq4.index<float>(g) = scale_sq4;

//     for (int i = begin; i < end; ++i) {
//       const float v = w_fp32.index<float>(i);

//       // qint8: deq = scale * q
//       int q8 = (int)std::round(v / scale_q8);
//       q8 = clamp_int(q8, -127, 127);
//       w_q8.index<int8_t>(i) = (int8_t)q8;

//       // sq4: deq = (q - z) * scale, 这里固定 z=8
//       int q4v = (int)std::round(v / scale_sq4) + 8;
//       q4v = clamp_int(q4v, 0, 15);
//       q4[i] = (uint8_t)q4v;
//     }
//   }

//   // pack q4: 两个4bit塞进1个byte，低位在前
//   const int q_bytes = (N + 1) / 2;
//   for (int bi = 0; bi < q_bytes; ++bi) {
//     const int i0 = bi * 2;
//     const int i1 = i0 + 1;
//     const uint8_t lo = q4[i0] & 0x0F;
//     const uint8_t hi = (i1 < N) ? (q4[i1] & 0x0F) : 0;
//     const uint8_t packed = (uint8_t)(lo | (hi << 4));
//     qweight_sq4_packed.index<int8_t>(bi) = (int8_t)packed;
//   }

//   // pack zeros: 每组一个4bit zero-point，这里全是8 => 0x88
//   const int z_bytes = (group_cnt + 1) / 2;
//   for (int bi = 0; bi < z_bytes; ++bi) {
//     uint8_t lo = 8;
//     uint8_t hi = 8;
//     const int g1 = bi * 2 + 1;
//     if (g1 >= group_cnt) hi = 0;  // 奇数尾巴
//     zeros_sq4_packed.index<int8_t>(bi) = (int8_t)(lo | (hi << 4));
//   }
// }

// int main() {
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

//   // 你可按显存调大：M/K越大越适合Ncu观察
//   const int B = 16;            // batch
//   const int M = 1024;          // input dim
//   const int K = 1024;          // output dim
//   const int group_size = 128;  // SQ4支持: 32/64/128

//   CHECK_EQ((K * M) % group_size, 0);

//   // input fp32 [B, M]
//   tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, B, M, true, alloc_cpu);
//   for (int i = 0; i < B * M; ++i) {
//     input_cpu.index<float>(i) = 0.01f * float((i % 97) - 48);
//   }

//   // 原始权重 fp32 [K, M]（同一组矩阵源）
//   tensor::Tensor w_fp32_cpu(base::DataType::kDataTypeFp32, K, M, true, alloc_cpu);
//   for (int i = 0; i < K * M; ++i) {
//     w_fp32_cpu.index<float>(i) = 0.02f * float((i * 17 + 13) % 101 - 50);
//   }

//   const int group_cnt = (K * M) / group_size;
//   const int q_bytes = (K * M + 1) / 2;
//   const int z_bytes = (group_cnt + 1) / 2;

//   // qint8 需要
//   tensor::Tensor w_q8_cpu(base::DataType::kDataTypeInt8, K, M, true, alloc_cpu);
//   tensor::Tensor s_q8_cpu(base::DataType::kDataTypeFp32, group_cnt, true, alloc_cpu);

//   // sq4 需要
//   tensor::Tensor qweight_sq4_cpu(base::DataType::kDataTypeInt8, q_bytes, true, alloc_cpu);
//   tensor::Tensor s_sq4_cpu(base::DataType::kDataTypeFp32, group_cnt, true, alloc_cpu);
//   tensor::Tensor zeros_sq4_cpu(base::DataType::kDataTypeInt8, z_bytes, true, alloc_cpu);

//   build_q8_and_sq4_from_fp32(w_fp32_cpu, K, M, group_size, w_q8_cpu, s_q8_cpu, qweight_sq4_cpu,
//                              s_sq4_cpu, zeros_sq4_cpu);

//   // to cuda
//   tensor::Tensor input_cu = input_cpu.clone();
//   input_cu.to_cuda(nullptr);
//   tensor::Tensor w_q8_cu = w_q8_cpu.clone();
//   w_q8_cu.to_cuda(nullptr);
//   tensor::Tensor s_q8_cu = s_q8_cpu.clone();
//   s_q8_cu.to_cuda(nullptr);

//   tensor::Tensor qweight_sq4_cu = qweight_sq4_cpu.clone();
//   qweight_sq4_cu.to_cuda(nullptr);
//   tensor::Tensor s_sq4_cu = s_sq4_cpu.clone();
//   s_sq4_cu.to_cuda(nullptr);
//   tensor::Tensor zeros_sq4_cu = zeros_sq4_cpu.clone();
//   zeros_sq4_cu.to_cuda(nullptr);

//   tensor::Tensor out_q8_cu(base::DataType::kDataTypeFp32, B, K, true, alloc_cu);
//   tensor::Tensor out_sq4_cu(base::DataType::kDataTypeFp32, B, K, true, alloc_cu);

//   kernel::CudaConfig cfg;
//   cudaStream_t stream;
//   ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
//   cfg.stream = stream;

//   // warmup
//   for (int i = 0; i < 10; ++i) {
//     kernel::get_matmul_kernel_quant8(base::DeviceType::kDeviceCUDA)(input_cu, w_q8_cu, out_q8_cu,
//                                                                     group_size, s_q8_cu, &cfg);

//     kernel::get_matmul_kernel_sq4(base::DeviceType::kDeviceCUDA)(
//         input_cu, qweight_sq4_cu, s_sq4_cu, zeros_sq4_cu, out_sq4_cu, K, M, group_size, &cfg);
//   }
//   ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

//   // timing
//   const int iters = 50;
//   cudaEvent_t st, ed;
//   ASSERT_EQ(cudaEventCreate(&st), cudaSuccess);
//   ASSERT_EQ(cudaEventCreate(&ed), cudaSuccess);

//   ASSERT_EQ(cudaEventRecord(st, stream), cudaSuccess);
//   for (int i = 0; i < iters; ++i) {
//     kernel::get_matmul_kernel_quant8(base::DeviceType::kDeviceCUDA)(input_cu, w_q8_cu, out_q8_cu,
//                                                                     group_size, s_q8_cu, &cfg);
//   }
//   ASSERT_EQ(cudaEventRecord(ed, stream), cudaSuccess);
//   ASSERT_EQ(cudaEventSynchronize(ed), cudaSuccess);
//   float q8_ms = 0.f;
//   ASSERT_EQ(cudaEventElapsedTime(&q8_ms, st, ed), cudaSuccess);

//   ASSERT_EQ(cudaEventRecord(st, stream), cudaSuccess);
//   for (int i = 0; i < iters; ++i) {
//     kernel::get_matmul_kernel_sq4(base::DeviceType::kDeviceCUDA)(
//         input_cu, qweight_sq4_cu, s_sq4_cu, zeros_sq4_cu, out_sq4_cu, K, M, group_size, &cfg);
//   }
//   ASSERT_EQ(cudaEventRecord(ed, stream), cudaSuccess);
//   ASSERT_EQ(cudaEventSynchronize(ed), cudaSuccess);
//   float sq4_ms = 0.f;
//   ASSERT_EQ(cudaEventElapsedTime(&sq4_ms, st, ed), cudaSuccess);

//   // 防止被当成“无副作用”路径，顺便看数值是否正常
//   out_q8_cu.to_cpu();
//   out_sq4_cu.to_cpu();
//   double sum_q8 = 0.0, sum_sq4 = 0.0;
//   for (int i = 0; i < out_q8_cu.size(); ++i) sum_q8 += out_q8_cu.index<float>(i);
//   for (int i = 0; i < out_sq4_cu.size(); ++i) sum_sq4 += out_sq4_cu.index<float>(i);

//   LOG(INFO) << "[NCU_BENCH] q8 avg ms = " << (q8_ms / iters)
//             << ", sq4 avg ms = " << (sq4_ms / iters)
//             << ", speedup(q8/sq4) = " << ((sq4_ms > 1e-6f) ? (q8_ms / sq4_ms) : 0.0f);
//   LOG(INFO) << "[NCU_BENCH] checksum q8=" << sum_q8 << ", sq4=" << sum_sq4;

//   ASSERT_TRUE(std::isfinite(sum_q8));
//   ASSERT_TRUE(std::isfinite(sum_sq4));

//   cudaEventDestroy(st);
//   cudaEventDestroy(ed);
//   // cudaStreamDestroy(stream);
//   return 0;
// }