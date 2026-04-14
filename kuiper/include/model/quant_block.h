#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace model {

// 对齐导出脚本 export_smooth_rtn_pro_v3.py：struct.pack("<8i", ...)
// 一共 8 个 int32 = 32 字节
struct QuantBlockDesc {
  int32_t block_type = 0;  // 1 = SQ4（你脚本里的 BLOCK_TYPE_SQ4）
  int32_t rows = 0;
  int32_t cols = 0;
  int32_t q_size = 0;      // q_bytes 字节数（packed nibble）
  int32_t s_size = 0;      // scales 字节数（fp32）
  int32_t z_size = 0;      // zeros 字节数（packed nibble）
  int32_t group_size = 0;  // 128
  int32_t reserved = 0;    // 0
};

static_assert(sizeof(QuantBlockDesc) == 32, "QuantBlockDesc must be exactly 32 bytes");

// 为了避免未对齐访问（mmap 指针不一定 4 字节对齐），用 memcpy 解析
inline QuantBlockDesc ReadQuantBlockDesc(const void* p) {
  QuantBlockDesc d;
  std::memcpy(&d, p, sizeof(QuantBlockDesc));
  return d;
}

// 用于推进指针：descriptor(32B) + q + s + z
inline size_t QuantBlockTotalBytes(const QuantBlockDesc& d) {
  return sizeof(QuantBlockDesc) + (size_t)d.q_size + (size_t)d.s_size + (size_t)d.z_size;
}

// （可选）你也可以在 loader 里做更严格检查
inline bool IsValidSQ4Desc(const QuantBlockDesc& d) {
  if (d.rows <= 0 || d.cols <= 0) return false;
  if (d.q_size <= 0 || d.s_size <= 0 || d.z_size <= 0) return false;
  if (d.group_size <= 0) return false;
  return true;
}

}  // namespace model