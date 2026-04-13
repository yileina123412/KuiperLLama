#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
// Unicode字符处理数据库模块 为文本预处理提供字符查询和转换功能
// LLaMA等大语言模型中，文本需要先转换为token序列，提供了文本规范化所需的Unicode字符数据

// unicode-data 模块存储了 Unicode 标准数据库 (UCD) 的离线副本
struct range_nfd {
  uint32_t first;
  uint32_t last;
  uint32_t nfd;
};
// Unicode 码点空间的最大大小
static const uint32_t MAX_CODEPOINTS = 0x110000;
// 记录了每个码点属于哪一类。比如它标注了哪些范围是“字母”，哪些是“数字”，哪些是“标点”。
extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;  // 字符属性

// 不仅包含我们常用的空格和换行，还包含全世界各种语言里定义的几十种“看不见”的空白字符。
extern const std::unordered_set<uint32_t> unicode_set_whitespace;           // 空白集合
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase;  // 大写->小写
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase;  // 小写->大写

// 记录了合成字符如何拆分。例如将 é 拆分为 e + ´ 的数据支持。
extern const std::vector<range_nfd> unicode_ranges_nfd;  // NFD分解
