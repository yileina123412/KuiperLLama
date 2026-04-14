#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "base/base.h"
#include "model/quant_block.h"

namespace model {

// 对齐 export_smooth_rtn_pro_v3.py：BLOCK_TYPE_SQ4 = 1
static constexpr int32_t kBlockTypeSQ4 = 1;

struct QuantBlockPayload {
  QuantBlockDesc desc{};
  const uint8_t* q = nullptr;  // packed nibble weights
  const uint8_t* s = nullptr;  // fp32 scales bytes
  const uint8_t* z = nullptr;  // packed nibble zeros
  size_t total_bytes = 0;      // desc(32) + q + s + z
};

// SQ4 假设：按“每行沿 cols 方向分组”
// group_count = rows * ceil(cols / group_size)
// scales: float32 per group
// zeros: 4bit per group (packed two per byte)
// q: 4bit per weight (packed two per byte)
inline size_t SQ4GroupsPerRow(int32_t cols, int32_t group_size) {
  return (static_cast<size_t>(cols) + static_cast<size_t>(group_size) - 1) /
         static_cast<size_t>(group_size);
}

inline size_t SQ4GroupCount(int32_t rows, int32_t cols, int32_t group_size) {
  return static_cast<size_t>(rows) * SQ4GroupsPerRow(cols, group_size);
}

inline size_t SQ4ExpectedQBytes(int32_t rows, int32_t cols) {
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  return (n + 1) / 2;  // 2 weights per byte
}

inline size_t SQ4ExpectedSBytes(int32_t rows, int32_t cols, int32_t group_size) {
  return SQ4GroupCount(rows, cols, group_size) * sizeof(float);
}

inline size_t SQ4ExpectedZBytes(int32_t rows, int32_t cols, int32_t group_size) {
  const size_t g = SQ4GroupCount(rows, cols, group_size);
  return (g + 1) / 2;  // 2 zeros per byte
}

inline base::Status ValidateBasicDesc(const QuantBlockDesc& d) {
  if (d.rows <= 0 || d.cols <= 0) {
    return base::error::ModelParseError("QuantBlock: rows/cols <= 0");
  }
  if (d.q_size < 0 || d.s_size < 0 || d.z_size < 0) {
    return base::error::ModelParseError("QuantBlock: q/s/z size < 0");
  }
  if (d.group_size <= 0) {
    return base::error::ModelParseError("QuantBlock: group_size <= 0");
  }
  if (d.reserved != 0) {
    return base::error::ModelParseError("QuantBlock: reserved != 0");
  }
  return base::error::Success();
}

inline base::Status ValidateSQ4Desc(const QuantBlockDesc& d, int32_t expected_rows,
                                    int32_t expected_cols, int32_t expected_group_size,
                                    bool strict_sizes) {
  if (d.block_type != kBlockTypeSQ4) {
    return base::error::ModelParseError("QuantBlock: not SQ4 block_type");
  }
  if (!IsValidSQ4Desc(d)) {
    return base::error::ModelParseError("QuantBlock: IsValidSQ4Desc failed");
  }

  if (expected_rows > 0 && d.rows != expected_rows) {
    return base::error::ModelParseError("QuantBlock: rows mismatch");
  }
  if (expected_cols > 0 && d.cols != expected_cols) {
    return base::error::ModelParseError("QuantBlock: cols mismatch");
  }
  if (expected_group_size > 0 && d.group_size != expected_group_size) {
    return base::error::ModelParseError("QuantBlock: group_size mismatch");
  }

  // scales 是 fp32 bytes，至少保证 4 字节对齐长度（不要求地址对齐）
  if ((d.s_size % static_cast<int32_t>(sizeof(float))) != 0) {
    return base::error::ModelParseError("QuantBlock: s_size not multiple of 4");
  }

  if (strict_sizes) {
    const size_t expect_q = SQ4ExpectedQBytes(d.rows, d.cols);
    const size_t expect_s = SQ4ExpectedSBytes(d.rows, d.cols, d.group_size);
    const size_t expect_z = SQ4ExpectedZBytes(d.rows, d.cols, d.group_size);

    if (static_cast<size_t>(d.q_size) != expect_q) {
      return base::error::ModelParseError("QuantBlock: q_size mismatch (formula)");
    }
    if (static_cast<size_t>(d.s_size) != expect_s) {
      return base::error::ModelParseError("QuantBlock: s_size mismatch (formula)");
    }
    if (static_cast<size_t>(d.z_size) != expect_z) {
      return base::error::ModelParseError("QuantBlock: z_size mismatch (formula)");
    }
  }

  return base::error::Success();
}

// 一个“读指针游标”：每次读一个 block，然后自动推进 cur_。
class QuantBlockCursor {
 public:
  QuantBlockCursor(const void* begin, const void* end)
      : cur_(reinterpret_cast<const uint8_t*>(begin)),
        end_(reinterpret_cast<const uint8_t*>(end)) {}

  size_t remaining_bytes() const { return (end_ >= cur_) ? static_cast<size_t>(end_ - cur_) : 0; }

  size_t offset_bytes_from(const void* begin) const {
    const auto* b = reinterpret_cast<const uint8_t*>(begin);
    return (cur_ >= b) ? static_cast<size_t>(cur_ - b) : 0;
  }

  const uint8_t* ptr() const { return cur_; }

  base::Status PeekDesc(QuantBlockDesc* out_desc) const {
    if (!out_desc) return base::error::InvalidArgument("PeekDesc: out_desc is null");
    if (remaining_bytes() < sizeof(QuantBlockDesc)) {
      return base::error::ModelParseError("PeekDesc: truncated desc");
    }
    *out_desc = ReadQuantBlockDesc(cur_);
    return base::error::Success();
  }

  base::Status ReadNext(QuantBlockPayload* out, bool basic_validate = true) {
    if (!out) return base::error::InvalidArgument("ReadNext: out is null");
    if (remaining_bytes() < sizeof(QuantBlockDesc)) {
      return base::error::ModelParseError("ReadNext: truncated desc");
    }

    const QuantBlockDesc d = ReadQuantBlockDesc(cur_);
    if (basic_validate) {
      base::Status st = ValidateBasicDesc(d);
      if (!st) return st;
    }

    // 防止负数转 size_t 溢出：basic_validate 已经检查 <0
    const size_t total = QuantBlockTotalBytes(d);
    if (total > remaining_bytes()) {
      return base::error::ModelParseError("ReadNext: truncated payload");
    }

    const uint8_t* q = cur_ + sizeof(QuantBlockDesc);
    const uint8_t* s = q + static_cast<size_t>(d.q_size);
    const uint8_t* z = s + static_cast<size_t>(d.s_size);

    out->desc = d;
    out->q = q;
    out->s = s;
    out->z = z;
    out->total_bytes = total;

    cur_ += total;
    return base::error::Success();
  }

  base::Status ReadNextSQ4(int32_t expected_rows, int32_t expected_cols,
                           int32_t expected_group_size, QuantBlockPayload* out,
                           bool strict_sizes = true) {
    base::Status st = ReadNext(out, true);
    if (!st) return st;
    return ValidateSQ4Desc(out->desc, expected_rows, expected_cols, expected_group_size,
                           strict_sizes);
  }

  base::Status SkipNext(size_t* skipped_bytes = nullptr) {
    QuantBlockPayload tmp;
    base::Status st = ReadNext(&tmp, true);
    if (!st) return st;
    if (skipped_bytes) *skipped_bytes = tmp.total_bytes;
    return base::error::Success();
  }

 private:
  const uint8_t* cur_ = nullptr;
  const uint8_t* end_ = nullptr;
};

}  // namespace model