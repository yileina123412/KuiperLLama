#include "base/base.h"
#include <string>
namespace base {
Status::Status(int code, std::string err_message) : code_(code), message_(std::move(err_message)) {}
// 赋值操作符
Status& Status::operator=(int code) {
  code_ = code;
  return *this;
};
// 比较操作符
bool Status::operator==(int code) const {
  if (code_ == code) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(int code) const {
  if (code_ != code) {
    return true;
  } else {
    return false;
  }
};
// 类型转换操作符
Status::operator int() const { return code_; }

Status::operator bool() const { return code_ == kSuccess; }  // 只有成功时返回true

int32_t Status::get_err_code() const { return code_; }

const std::string& Status::get_err_msg() const { return message_; }

void Status::set_err_msg(const std::string& err_msg) { message_ = err_msg; }

namespace error {
// 工厂函数，快速创建不同状态

// 创建成功状态
Status Success(const std::string& err_msg) { return Status{kSuccess, err_msg}; }
// 工厂函数，创建和返回一个错误状态对象
// 返回一个status对象
Status FunctionNotImplement(const std::string& err_msg) {
  return Status{kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) { return Status{kPathNotValid, err_msg}; }

Status ModelParseError(const std::string& err_msg) { return Status{kModelParseError, err_msg}; }

Status InternalError(const std::string& err_msg) { return Status{kInternalError, err_msg}; }

Status InvalidArgument(const std::string& err_msg) { return Status{kInvalidArgument, err_msg}; }

Status KeyHasExits(const std::string& err_msg) { return Status{kKeyValueHasExist, err_msg}; }
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.get_err_msg();
  return os;
}

}  // namespace base