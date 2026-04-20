#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace model {

// 对齐导出脚本 export_smooth_rtn_pro_v3.py
// 一共 8 个 int32  32 字节
struct QuantBlockDesc {
  int32_t block_type = 0;  // 1 = SQ4
  int32_t rows = 0;
  int32_t cols = 0;
  int32_t q_size = 0;      // q_bytes 字节数
  int32_t s_size = 0;      // scales 字节数
  int32_t z_size = 0;      // zeros 字节数
  int32_t group_size = 0;  // 128
  int32_t reserved = 0;    // 0
};

// static_assert(sizeof(QuantBlockDesc) == 32, "QuantBlockDesc must be exactly 32 bytes");

// 为了避免未对齐访问（mmap 指针不一定 4 字节对齐），用 memcpy 解析
// 对mmap返回的指针做类型转换会有风险
inline QuantBlockDesc ReadQuantBlockDesc(const void* p) {
  QuantBlockDesc d;
  // d = *(QuantBlockDesc*)p;
  std::memcpy(&d, p, sizeof(QuantBlockDesc));
  return d;
}

// 一个块内总共的字节数
inline size_t QuantBlockTotalBytes(const QuantBlockDesc& d) {
  return sizeof(QuantBlockDesc) + (size_t)d.q_size + (size_t)d.s_size + (size_t)d.z_size;
}

inline bool IsValidSQ4Desc(const QuantBlockDesc& d) {
  if (d.rows <= 0 || d.cols <= 0) return false;
  if (d.q_size <= 0 || d.s_size <= 0 || d.z_size <= 0) return false;
  if (d.group_size <= 0) return false;
  return true;
}

}  // namespace model