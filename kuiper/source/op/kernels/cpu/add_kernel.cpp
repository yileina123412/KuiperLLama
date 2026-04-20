#include "add_kernel.h"
#include <armadillo>
#include "base/base.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  // Case A: elementwise same-shape add
  if (input1.size() == input2.size() && input1.size() == output.size()) {
    arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
    arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
    output_vec = input_vec1 + input_vec2;
    return;
  }

  // Case B: row-wise broadcast: [B, C] + [C] -> [B, C]
  if (input1.dims_size() == 2 && output.dims_size() == 2 && input2.dims_size() == 1) {
    const int32_t rows = input1.get_dim(0);
    const int32_t cols = input1.get_dim(1);

    CHECK_EQ(output.get_dim(0), rows);
    CHECK_EQ(output.get_dim(1), cols);
    CHECK_EQ(input2.get_dim(0), cols);

    const float* in = input1.ptr<float>();
    const float* b = input2.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    for (int32_t r = 0; r < rows; ++r) {
      const int32_t base = r * cols;
      for (int32_t c = 0; c < cols; ++c) {
        out[base + c] = in[base + c] + b[c];
      }
    }
    return;
  }

  LOG(FATAL) << "add_kernel_cpu shape mismatch: input1.size=" << input1.size()
             << ", input2.size=" << input2.size() << ", output.size=" << output.size();
}
}  // namespace kernel