#include "op/matmul_sq4.h"
#include <glog/logging.h>
#include <cmath>
#include <cstdlib>
#include "kernels/kernels_interface.h"
namespace op {

MatmulSQ4Layer::MatmulSQ4Layer(base::DeviceType device_type, int32_t dim0, int32_t dim1)
    : Layer(device_type, LayerType::kLayerMatmul, "MatmulSQ4"), dim0_(dim0), dim1_(dim1) {
  reset_input_size(1);
  reset_output_size(1);
}

size_t MatmulSQ4Layer::expected_q_bytes(int32_t rows, int32_t cols) {
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  return (n + 1) / 2;  // 2 weights per byte
}

size_t MatmulSQ4Layer::expected_group_count(int32_t rows, int32_t cols, int32_t group_size) {
  // 对齐 exporter：tensor.view(rows, -1, group_size) => 要求 cols % group_size == 0
  return static_cast<size_t>(rows) * (static_cast<size_t>(cols) / static_cast<size_t>(group_size));
}

size_t MatmulSQ4Layer::expected_s_bytes(int32_t rows, int32_t cols, int32_t group_size) {
  return expected_group_count(rows, cols, group_size) * sizeof(float);
}

size_t MatmulSQ4Layer::expected_z_bytes(int32_t rows, int32_t cols, int32_t group_size) {
  const size_t g = expected_group_count(rows, cols, group_size);
  return (g + 1) / 2;  // 2 zeros per byte
}

base::Status MatmulSQ4Layer::set_sq4_weight(const SQ4WeightView& view,
                                            base::DeviceType weight_device_type,
                                            bool strict_size_check) {
  if (view.rows != dim0_ || view.cols != dim1_) {
    return base::error::InvalidArgument("MatmulSQ4: view dims mismatch");
  }
  if (view.group_size <= 0) {
    return base::error::InvalidArgument("MatmulSQ4: group_size <= 0");
  }
  if (view.cols % view.group_size != 0) {
    return base::error::ModelParseError("MatmulSQ4: exporter requires cols % group_size == 0");
  }
  if (!view.qweight_packed || !view.scales || !view.zeros_packed) {
    return base::error::InvalidArgument("MatmulSQ4: null weight pointer");
  }
  if ((view.s_bytes % sizeof(float)) != 0) {
    return base::error::ModelParseError("MatmulSQ4: s_bytes not multiple of 4");
  }

  group_size_ = view.group_size;

  if (strict_size_check) {
    const size_t eq = expected_q_bytes(view.rows, view.cols);
    const size_t es = expected_s_bytes(view.rows, view.cols, view.group_size);
    const size_t ez = expected_z_bytes(view.rows, view.cols, view.group_size);
    if (view.q_bytes != eq) return base::error::ModelParseError("MatmulSQ4: q_bytes mismatch");
    if (view.s_bytes != es) return base::error::ModelParseError("MatmulSQ4: s_bytes mismatch");
    if (view.z_bytes != ez) return base::error::ModelParseError("MatmulSQ4: z_bytes mismatch");
  }

  // 用 Tensor(ptr=...) 包装外部 mmap 指针（zero-copy）
  qweight_packed_ =
      tensor::Tensor(base::DataType::kDataTypeInt8, static_cast<int32_t>(view.q_bytes), false,
                     nullptr, const_cast<void*>(view.qweight_packed));
  qweight_packed_.set_device_type(weight_device_type);

  const int32_t group_count = static_cast<int32_t>(view.s_bytes / sizeof(float));
  scales_ = tensor::Tensor(base::DataType::kDataTypeFp32, group_count, false, nullptr,
                           const_cast<void*>(view.scales));
  scales_.set_device_type(weight_device_type);

  zeros_packed_ = tensor::Tensor(base::DataType::kDataTypeInt8, static_cast<int32_t>(view.z_bytes),
                                 false, nullptr, const_cast<void*>(view.zeros_packed));
  zeros_packed_.set_device_type(weight_device_type);

  return base::error::Success();
}

base::Status MatmulSQ4Layer::check() const {
  const auto input = get_input(0);
  const auto output = get_output(0);

  // input dims: [M] or [B, M]
  if (input.dims_size() == 1) {
    auto st = check_tensor_with_dim(input, device_type_, data_type_, dim1_);
    if (!st) return st;
    st = check_tensor_with_dim(output, device_type_, data_type_, dim0_);
    if (!st) return st;
  } else if (input.dims_size() == 2) {
    if (input.get_dim(1) != dim1_) {
      return base::error::InvalidArgument("MatmulSQ4 input dim mismatch at dim=1");
    }
    if (output.dims_size() != 2 || output.get_dim(0) != input.get_dim(0) ||
        output.get_dim(1) != dim0_) {
      return base::error::InvalidArgument("MatmulSQ4 output dim mismatch for batch");
    }
    auto st = check_tensor(input, device_type_, data_type_);
    if (!st) return st;
    st = check_tensor(output, device_type_, data_type_);
    if (!st) return st;
  } else {
    return base::error::InvalidArgument("MatmulSQ4 only supports 1D/2D input");
  }

  if (qweight_packed_.is_empty() || scales_.is_empty() || zeros_packed_.is_empty()) {
    return base::error::InvalidArgument("MatmulSQ4 weight not set");
  }
  if (qweight_packed_.device_type() != device_type_ || scales_.device_type() != device_type_ ||
      zeros_packed_.device_type() != device_type_) {
    return base::error::InvalidArgument("MatmulSQ4 weight device_type mismatch");
  }
  if (qweight_packed_.data_type() != base::DataType::kDataTypeInt8 ||
      zeros_packed_.data_type() != base::DataType::kDataTypeInt8 ||
      scales_.data_type() != base::DataType::kDataTypeFp32) {
    return base::error::InvalidArgument("MatmulSQ4 weight dtype mismatch");
  }
  if (dim1_ % group_size_ != 0) {
    return base::error::ModelParseError("MatmulSQ4: dim1 % group_size != 0");
  }

  // 尺寸 sanity check（避免 loader 顺序读错）
  const size_t eq = expected_q_bytes(dim0_, dim1_);
  const size_t es = expected_s_bytes(dim0_, dim1_, group_size_);
  const size_t ez = expected_z_bytes(dim0_, dim1_, group_size_);
  if (qweight_packed_.size() != eq) return base::error::ModelParseError("MatmulSQ4: q size bad");
  if (scales_.byte_size() != es) return base::error::ModelParseError("MatmulSQ4: scales size bad");
  if (zeros_packed_.size() != ez) return base::error::ModelParseError("MatmulSQ4: zeros size bad");

  return base::error::Success();
}

base::Status MatmulSQ4Layer::forward() {
  auto st = check();
  if (!st) return st;

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  kernel::get_matmul_kernel_sq4(device_type_)(get_input(0), qweight_packed_, scales_, zeros_packed_,
                                              get_output(0), dim0_, dim1_, group_size_,
                                              cuda_config_ ? cuda_config_.get() : nullptr);

  //   // ---- DEBUG: one-shot compare CUDA output vs CPU reference ----
  //   static bool debug_done = false;
  //   const char* dbg = std::getenv("KUIPER_DEBUG_SQ4");
  //   if (!debug_done && dbg && dbg[0] != '0') {
  //     debug_done = true;

  //     // 1) 拷贝 input/output/weights 到 CPU
  //     tensor::Tensor x_cpu = get_input(0).clone();
  //     x_cpu.to_cpu();

  //     tensor::Tensor y_cpu = get_output(0).clone();
  //     y_cpu.to_cpu();

  //     tensor::Tensor q_cpu = qweight_packed_.clone();
  //     q_cpu.to_cpu();
  //     tensor::Tensor s_cpu = scales_.clone();
  //     s_cpu.to_cpu();
  //     tensor::Tensor z_cpu = zeros_packed_.clone();
  //     z_cpu.to_cpu();

  //     auto load_nibble = [](const uint8_t* p, size_t idx) -> uint8_t {
  //       const uint8_t byte = p[idx >> 1];
  //       return (idx & 1) ? (byte >> 4) : (byte & 0x0F);
  //     };

  //     const int32_t K = dim0_;
  //     const int32_t M = dim1_;
  //     const int32_t gsz = group_size_;
  //     CHECK_EQ(M % gsz, 0);
  //     const int32_t groups_per_row = M / gsz;

  //     const float* xptr = x_cpu.ptr<float>();
  //     const float* yptr = y_cpu.ptr<float>();

  //     const uint8_t* qptr = reinterpret_cast<const uint8_t*>(q_cpu.ptr<int8_t>());
  //     const float* sptr = s_cpu.ptr<float>();
  //     const uint8_t* zptr = reinterpret_cast<const uint8_t*>(z_cpu.ptr<int8_t>());

  //     const int k_check = std::min<int32_t>(K, 8);
  //     float max_abs_diff = 0.f;

  //     if (x_cpu.dims_size() == 1) {
  //       // [M] -> [K]
  //       for (int k = 0; k < k_check; ++k) {
  //         float ref = 0.f;
  //         for (int i = 0; i < M; ++i) {
  //           const size_t w_idx = (size_t)k * (size_t)M + (size_t)i;
  //           const uint8_t q4 = load_nibble(qptr, w_idx);

  //           const size_t g_idx = (size_t)k * (size_t)groups_per_row + (size_t)(i / gsz);
  //           const uint8_t z4 = load_nibble(zptr, g_idx);
  //           const float sc = sptr[g_idx];

  //           const float w = float(int(q4) - int(z4)) * sc;
  //           ref += xptr[i] * w;
  //         }
  //         const float got = yptr[k];
  //         max_abs_diff = std::max(max_abs_diff, std::fabs(ref - got));
  //       }
  //     } else if (x_cpu.dims_size() == 2) {
  //       // [B, M] -> [B, K]，只检查 b=0
  //       const int32_t B = x_cpu.get_dim(0);
  //       CHECK_GT(B, 0);
  //       const float* xb = xptr + (size_t)0 * M;
  //       const float* yb = yptr + (size_t)0 * K;

  //       for (int k = 0; k < k_check; ++k) {
  //         float ref = 0.f;
  //         for (int i = 0; i < M; ++i) {
  //           const size_t w_idx = (size_t)k * (size_t)M + (size_t)i;
  //           const uint8_t q4 = load_nibble(qptr, w_idx);

  //           const size_t g_idx = (size_t)k * (size_t)groups_per_row + (size_t)(i / gsz);
  //           const uint8_t z4 = load_nibble(zptr, g_idx);
  //           const float sc = sptr[g_idx];

  //           const float w = float(int(q4) - int(z4)) * sc;
  //           ref += xb[i] * w;
  //         }
  //         const float got = yb[k];
  //         max_abs_diff = std::max(max_abs_diff, std::fabs(ref - got));
  //       }
  //     } else {
  //       LOG(ERROR) << "KUIPER_DEBUG_SQ4: unsupported input dims_size=" << x_cpu.dims_size();
  //     }

  //     LOG(INFO) << "KUIPER_DEBUG_SQ4: max_abs_diff(first " << k_check << " rows) = " <<
  //     max_abs_diff;
  //   }

  return base::error::Success();
}

void MatmulSQ4Layer::to_cuda() {
  Layer::to_cuda();
  if (!qweight_packed_.is_empty())
    qweight_packed_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  if (!scales_.is_empty()) scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  if (!zeros_packed_.is_empty())
    zeros_packed_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
}

}  // namespace op