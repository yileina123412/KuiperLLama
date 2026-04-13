#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Unicode 和 UTF-8 的转换是一个纯粹的数学算法，它不需要查表
// 将磁盘上的 UTF-8 字节流转成程序好处理的 Unicode 码点序列。

// 这是一个 Unicode 文本预处理引擎,专门为 Tokenizer (分词器)
// 提供底层的字符编码转换和正则表达式分割功能。
// TODO: prefix all symbols with "llama_"

// 文本预处理和分词的基础组件

// 把原始字符串修改成分词器可以用的格式
// 字符分类标志常量
struct codepoint_flags {
  // 支持一个字符拥有多个属性
  enum {
    UNDEFINED = 0x0001,        // 未定义的字符
    NUMBER = 0x0002,           // regex: \p{N} 数字字符
    LETTER = 0x0004,           // regex: \p{L}  字母字符（a-z, A-Z, 中文等）
    SEPARATOR = 0x0008,        // regex: \p{Z}  分隔符（空格、换行等）
    ACCENT_MARK = 0x0010,      // regex: \p{M}
    PUNCTUATION = 0x0020,      // regex: \p{P}
    SYMBOL = 0x0040,           // regex: \p{S}  数学/货币符号（$, +, = 等）
    CONTROL = 0x0080,          // regex: \p{C}
    MASK_CATEGORIES = 0x00FF,  // 掩码：提取前8位（所有分类标记）
  };

  // codepoint type
  uint16_t is_undefined : 1;
  uint16_t is_number : 1;       // regex: \p{N}
  uint16_t is_letter : 1;       // regex: \p{L}
  uint16_t is_separator : 1;    // regex: \p{Z}
  uint16_t is_accent_mark : 1;  // regex: \p{M}
  uint16_t is_punctuation : 1;  // regex: \p{P}
  uint16_t is_symbol : 1;       // regex: \p{S}
  uint16_t is_control : 1;      // regex: \p{C}
  // helper flags
  uint16_t is_whitespace : 1;  // regex: \s
  uint16_t is_lowercase : 1;
  uint16_t is_uppercase : 1;
  uint16_t is_nfd : 1;

  // decode from uint16
  inline codepoint_flags(const uint16_t flags = 0) { *reinterpret_cast<uint16_t*>(this) = flags; }

  inline uint16_t as_uint() const { return *reinterpret_cast<const uint16_t*>(this); }

  inline uint16_t category_flag() const { return this->as_uint() & MASK_CATEGORIES; }
};

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8(uint32_t cp);
uint32_t unicode_cpt_from_utf8(const std::string& utf8, size_t& offset);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& utf8);

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t>& cpts);

codepoint_flags unicode_cpt_flags(const uint32_t cp);
codepoint_flags unicode_cpt_flags(const std::string& utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string& utf8);

uint32_t unicode_tolower(uint32_t cp);

std::vector<std::string> unicode_regex_split(const std::string& text,
                                             const std::vector<std::string>& regex_exprs);
