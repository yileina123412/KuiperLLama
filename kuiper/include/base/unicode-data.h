#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
// Unicode字符处理数据库模块 为文本预处理提供字符查询和转换功能
// LLaMA等大语言模型中，文本需要先转换为token序列，提供了文本规范化所需的Unicode字符数据
struct range_nfd {
  uint32_t first;
  uint32_t last;
  uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;  // 字符属性
extern const std::unordered_set<uint32_t> unicode_set_whitespace;              // 空白集合
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase;     // 大写->小写
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase;     // 小写->大写
extern const std::vector<range_nfd> unicode_ranges_nfd;                        // NFD分解
