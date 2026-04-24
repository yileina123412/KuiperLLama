#include <base/base.h>
#include <base/tick.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>
#include "model/llama3.h"

// ===== Greedy 采样（确定性）=====
static int32_t greedy_sample(const tensor::Tensor& logits_any_device) {
  tensor::Tensor logits = logits_any_device;
  if (logits.device_type() == base::DeviceType::kDeviceCUDA) {
    logits.to_cpu();
  }
  const float* p = logits.ptr<float>();
  const int32_t vocab = static_cast<int32_t>(logits.size());

  float max_val = -1e30f;
  int32_t max_idx = 0;
  for (int32_t i = 0; i < vocab; ++i) {
    if (p[i] > max_val) {
      max_val = p[i];
      max_idx = i;
    }
  }
  return max_idx;
}

// ===== 配置结构体 =====
struct SpeculativeConfig {
  // 路径配置
  std::string draft_checkpoint = "/home/furina/models/stories110M.bin";
  std::string large_checkpoint = "/home/furina/models/tinyllama.bin";
  std::string tokenizer_path = "/home/furina/models/tokenizer.model";

  // 运行时配置
  base::DeviceType device = base::DeviceType::kDeviceCUDA;
  int32_t kv_window = 2048;
  int32_t kv_prefix = 0;

  // 生成配置
  std::string prompt = "做一下自我介绍";
  int32_t draft_len = 4;
  int32_t max_new_tokens = 100;
  bool strict_mode = true;

  // Step4: 先固定确定性（greedy-only）
  bool deterministic = true;

  std::string output_csv = "speculative_result.csv";
};

// ===== 验收结果结构体 =====
struct VerificationResult {
  std::vector<int32_t> accepted_tokens;  // 接受的 token 序列
  std::vector<int32_t> rejected_tokens;  // 被拒绝的 token 序列
  int32_t accept_count = 0;
  int32_t reject_count = 0;
  int32_t first_mismatch_pos = -1;  // 第一个不匹配的位置（-1 表示全部接受）
};

struct BaselineResult {
  std::vector<int32_t> tokens;
  double duration_s = 0.0;
};

static void build_context_state_for_model(const model::LLama2Model& model_ref,
                                          const std::vector<int32_t>& context_tokens, int32_t& pos,
                                          int32_t& last_token) {
  CHECK(!context_tokens.empty());

  if (context_tokens.size() >= 2) {
    std::vector<int32_t> prefill_tokens(context_tokens.begin(), context_tokens.end() - 1);
    const auto& prefill_emb = model_ref.embedding(prefill_tokens);
    auto [prefill_input_tokens, prefill_input_embeddings, prefill_token_num] = prefill_emb;
    (void)prefill_input_tokens;
    (void)prefill_token_num;

    tensor::Tensor pos_tensor = model_ref.get_buffer(model::ModelBufferType::kInputPos);
    pos_tensor.index<int32_t>(0) = 0;
    int dummy_next = -1;
    model_ref.prefill(prefill_input_embeddings, pos_tensor, dummy_next);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      LOG(FATAL) << "baseline prefill sync failed: " << cudaGetErrorString(err);
    }

    pos = static_cast<int32_t>(context_tokens.size()) - 1;
    last_token = context_tokens.back();
  } else {
    pos = 0;
    last_token = context_tokens[0];
  }
}

static int32_t forward_one_step_for_model(const model::LLama2Model& model_ref, int32_t pos,
                                          int32_t input_token) {
  tensor::Tensor pos_tensor = model_ref.get_buffer(model::ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;

  std::vector<int32_t> cur_tokens = {input_token};
  const auto& emb = model_ref.embedding(cur_tokens);
  tensor::Tensor input = model_ref.fill_input(pos_tensor, emb, false);

  int dummy_next = -1;
  model_ref.forward(input, pos_tensor, dummy_next);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    LOG(FATAL) << "baseline forward sync failed at pos=" << pos
               << ", err=" << cudaGetErrorString(err);
  }

  const tensor::Tensor& logits = model_ref.get_buffer(model::ModelBufferType::kForwardOutput);
  return greedy_sample(logits);
}

static BaselineResult run_large_baseline_greedy(const model::LLama2Model& large_model,
                                                const std::string& prompt, int32_t max_new_tokens) {
  BaselineResult out;
  std::vector<int32_t> prompt_tokens = large_model.encode(prompt);
  if (prompt_tokens.empty()) {
    LOG(ERROR) << "baseline prompt_tokens empty";
    return out;
  }

  int32_t pos = 0;
  int32_t last_token = -1;
  build_context_state_for_model(large_model, prompt_tokens, pos, last_token);

  auto t0 = std::chrono::steady_clock::now();
  out.tokens.reserve(max_new_tokens);
  for (int32_t i = 0; i < max_new_tokens; ++i) {
    int32_t tok = forward_one_step_for_model(large_model, pos, last_token);
    out.tokens.push_back(tok);
    last_token = tok;
    pos += 1;
  }
  auto t1 = std::chrono::steady_clock::now();

  out.duration_s = std::chrono::duration<double>(t1 - t0).count();
  return out;
}

static bool compare_token_sequences(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                                    int32_t& mismatch_idx) {
  mismatch_idx = -1;
  const int32_t n = static_cast<int32_t>(std::min(a.size(), b.size()));
  for (int32_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      mismatch_idx = i;
      return false;
    }
  }
  if (a.size() != b.size()) {
    mismatch_idx = n;
    return false;
  }
  return true;
}

// ===== 主 Speculative Decode 逻辑 =====
// ===== 主 Speculative Decode 逻辑 =====
class SpeculativeDecoder {
 public:
  SpeculativeDecoder(const model::LLama2Model& draft_model, const model::LLama2Model& large_model,
                     const SpeculativeConfig& config)
      : draft_model_(draft_model), large_model_(large_model), config_(config) {}

  // 执行 speculative decode（阶段1：严格逐 token 验收版本）
  VerificationResult decode(const std::string& prompt, int32_t num_gen_tokens) {
    VerificationResult result;
    if (!config_.strict_mode) {
      LOG(WARNING) << "Current implementation is strict_mode only.";
    }
    CHECK(config_.deterministic) << "Stage1 requires deterministic=true (greedy-only).";

    // 1) prompt tokenize
    std::vector<int32_t> prompt_tokens = large_model_.encode(prompt);
    if (prompt_tokens.empty()) {
      LOG(ERROR) << "Empty prompt tokens!";
      return result;
    }

    // 2) 大模型先走 prompt prefill，拿到 decode 起点状态
    int32_t large_pos = 0;
    int32_t large_last_token = -1;
    build_context_state(large_model_, prompt_tokens, large_pos, large_last_token);

    // committed = 最终已确认上下文（prompt + 已生成）
    std::vector<int32_t> committed = prompt_tokens;

    while (static_cast<int32_t>(result.accepted_tokens.size()) < num_gen_tokens) {
      const int32_t remain = num_gen_tokens - static_cast<int32_t>(result.accepted_tokens.size());
      const int32_t k = std::min(config_.draft_len, remain);

      // 3) 小模型：基于 committed 上下文，草拟 k 个 token
      int32_t draft_pos = 0;
      int32_t draft_last_token = -1;
      build_context_state(draft_model_, committed, draft_pos, draft_last_token);

      std::vector<int32_t> draft_tokens;
      draft_tokens.reserve(k);
      for (int32_t i = 0; i < k; ++i) {
        int32_t tok = forward_one_step(draft_model_, draft_pos, draft_last_token);
        draft_tokens.push_back(tok);
        draft_last_token = tok;
        draft_pos += 1;
      }

      // 4) 大模型逐 token 严格验收
      bool mismatch = false;
      for (int32_t i = 0; i < static_cast<int32_t>(draft_tokens.size()); ++i) {
        int32_t large_tok = forward_one_step(large_model_, large_pos, large_last_token);

        if (large_tok == draft_tokens[i]) {
          // 接受
          result.accepted_tokens.push_back(large_tok);
          result.accept_count += 1;
          committed.push_back(large_tok);

          large_last_token = large_tok;
          large_pos += 1;
        } else {
          // mismatch：从当前位开始，草稿全部丢弃，输出大模型 token
          mismatch = true;
          if (result.first_mismatch_pos < 0) {
            result.first_mismatch_pos = static_cast<int32_t>(result.accepted_tokens.size());
          }

          // 当前位输出大模型 token（保证零精度损失）
          result.accepted_tokens.push_back(large_tok);
          result.accept_count += 1;
          committed.push_back(large_tok);

          // 统计被拒 token（当前位及之后的草稿）
          for (int32_t j = i; j < static_cast<int32_t>(draft_tokens.size()); ++j) {
            result.rejected_tokens.push_back(draft_tokens[j]);
          }
          result.reject_count += (static_cast<int32_t>(draft_tokens.size()) - i);

          large_last_token = large_tok;
          large_pos += 1;
          break;
        }

        if (static_cast<int32_t>(result.accepted_tokens.size()) >= num_gen_tokens) {
          break;
        }
      }

      (void)mismatch;
    }

    return result;
  }

 private:
  const model::LLama2Model& draft_model_;
  const model::LLama2Model& large_model_;
  const SpeculativeConfig& config_;

  // 统一：给定完整上下文 token，构建模型 decode 起始状态
  // 输出：pos=最后一个上下文 token 的位置，last_token=最后一个上下文 token
  void build_context_state(const model::LLama2Model& model_ref,
                           const std::vector<int32_t>& context_tokens, int32_t& pos,
                           int32_t& last_token) const {
    CHECK(!context_tokens.empty());

    if (context_tokens.size() >= 2) {
      std::vector<int32_t> prefill_tokens(context_tokens.begin(), context_tokens.end() - 1);
      const auto& prefill_emb = model_ref.embedding(prefill_tokens);
      auto [prefill_input_tokens, prefill_input_embeddings, prefill_token_num] = prefill_emb;
      (void)prefill_input_tokens;
      (void)prefill_token_num;

      tensor::Tensor pos_tensor = model_ref.get_buffer(model::ModelBufferType::kInputPos);
      pos_tensor.index<int32_t>(0) = 0;
      int dummy_next = -1;
      model_ref.prefill(prefill_input_embeddings, pos_tensor, dummy_next);

      cudaError_t err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        LOG(FATAL) << "prefill sync failed: " << cudaGetErrorString(err);
      }

      pos = static_cast<int32_t>(context_tokens.size()) - 1;
      last_token = context_tokens.back();
    } else {
      pos = 0;
      last_token = context_tokens[0];
    }
  }

  // 单步 forward + greedy
  int32_t forward_one_step(const model::LLama2Model& model_ref, int32_t pos,
                           int32_t input_token) const {
    tensor::Tensor pos_tensor = model_ref.get_buffer(model::ModelBufferType::kInputPos);
    pos_tensor.index<int32_t>(0) = pos;

    std::vector<int32_t> cur_tokens = {input_token};
    const auto& emb = model_ref.embedding(cur_tokens);
    tensor::Tensor input = model_ref.fill_input(pos_tensor, emb, false);

    int dummy_next = -1;
    model_ref.forward(input, pos_tensor, dummy_next);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      LOG(FATAL) << "forward sync failed at pos=" << pos << ", err=" << cudaGetErrorString(err);
    }

    const tensor::Tensor& logits = model_ref.get_buffer(model::ModelBufferType::kForwardOutput);
    return greedy_sample(logits);
  }
};
// ===== Main =====
int main(int argc, char* argv[]) {
  // ===== 初始化日志 =====
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  // ===== 统一运行时配置（Step5）=====
  SpeculativeConfig config;

  LOG(INFO) << "Loading draft model: " << config.draft_checkpoint;
  model::LLama2Model draft_model(base::TokenizerType::kEncodeSpe, config.tokenizer_path,
                                 config.draft_checkpoint, false);

  auto draft_init = draft_model.init(base::DeviceType::kDeviceCUDA);
  if (!draft_init) {
    LOG(FATAL) << "Draft model init failed";
  }

  LOG(INFO) << "Loading large model: " << config.large_checkpoint;
  model::LLama2Model large_model(base::TokenizerType::kEncodeSpe, config.tokenizer_path,
                                 config.large_checkpoint, false);
  auto large_init = large_model.init(config.device);
  if (!large_init) {
    LOG(FATAL) << "Large model init failed";
  }

  // ===== 统一运行时窗口配置（Step2 + Step6）=====
  draft_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);
  large_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);

  //   // baseline 专用大模型（和 speculative 使用的 large_model 分开，避免状态污染）
  //   model::LLama2Model baseline_large_model(base::TokenizerType::kEncodeSpe,
  //   config.tokenizer_path,
  //                                           config.large_checkpoint, false);
  //   auto baseline_init = baseline_large_model.init(config.device);
  //   if (!baseline_init) {
  //     LOG(FATAL) << "Baseline large model init failed";
  //   }
  //   baseline_large_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);

  LOG(INFO) << "Models loaded successfully";

  LOG(INFO) << "Unified KV config: window=" << config.kv_window << ", prefix=" << config.kv_prefix;
  // ===== 可选覆盖默认配置（按需改）=====
  config.draft_len = 4;
  config.max_new_tokens = 100;
  config.strict_mode = true;
  config.deterministic = true;
  config.kv_prefix = 0;  // 改成 >0 即启用前缀保护

  // ===== Speculative decode =====
  // ===== Step7: 一致性基线（先跑 baseline，避免第二个大模型占显存）=====
  large_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);
  BaselineResult baseline =
      run_large_baseline_greedy(large_model, config.prompt, config.max_new_tokens);

  // ===== 再跑 speculative 前，重置两边 KV 运行态 =====
  draft_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);
  large_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);

  // ===== Speculative decode =====
  SpeculativeDecoder decoder(draft_model, large_model, config);
  auto start = std::chrono::steady_clock::now();
  VerificationResult result = decoder.decode(config.prompt, config.max_new_tokens);
  auto end = std::chrono::steady_clock::now();

  double spec_duration = std::chrono::duration<double>(end - start).count();

  int32_t mismatch_idx = -1;
  bool exact_match = compare_token_sequences(result.accepted_tokens, baseline.tokens, mismatch_idx);

  // ===== 统计 =====
  const double spec_tps =
      result.accepted_tokens.empty()
          ? 0.0
          : static_cast<double>(result.accepted_tokens.size()) / std::max(spec_duration, 1e-9);
  const double base_tps = baseline.tokens.empty() ? 0.0
                                                  : static_cast<double>(baseline.tokens.size()) /
                                                        std::max(baseline.duration_s, 1e-9);
  const double speedup = (spec_duration > 0.0) ? (baseline.duration_s / spec_duration) : 0.0;

  // ===== 输出结果 =====
  LOG(INFO) << "========== Speculative Decode Result ==========";
  LOG(INFO) << "Generated tokens: " << result.accepted_tokens.size();
  LOG(INFO) << "Accept count: " << result.accept_count;
  LOG(INFO) << "Reject count: " << result.reject_count;
  LOG(INFO) << "First mismatch at (internal draft verify): " << result.first_mismatch_pos;
  LOG(INFO) << "Speculative duration: " << spec_duration << " s";
  LOG(INFO) << "Baseline duration: " << baseline.duration_s << " s";
  LOG(INFO) << "Speculative tokens/s: " << spec_tps;
  LOG(INFO) << "Baseline tokens/s: " << base_tps;
  LOG(INFO) << "Speedup (baseline/spec): " << speedup;

  if (!exact_match) {
    LOG(ERROR) << "[STEP7 FAIL] token sequence mismatch, first idx = " << mismatch_idx
               << ", spec_token="
               << (mismatch_idx >= 0 &&
                           mismatch_idx < static_cast<int32_t>(result.accepted_tokens.size())
                       ? result.accepted_tokens[mismatch_idx]
                       : -1)
               << ", base_token="
               << (mismatch_idx >= 0 && mismatch_idx < static_cast<int32_t>(baseline.tokens.size())
                       ? baseline.tokens[mismatch_idx]
                       : -1);
  } else {
    LOG(INFO) << "[STEP7 PASS] token sequence EXACT MATCH with baseline.";
  }

  // ===== Step8: P0 DoD =====
  const bool dod_exact_match = exact_match;
  const bool dod_len_ok =
      (static_cast<int32_t>(result.accepted_tokens.size()) == config.max_new_tokens);
  const bool dod_deterministic = config.deterministic;

  const bool p0_pass = dod_exact_match && dod_len_ok && dod_deterministic;

  LOG(INFO) << "========== P0 DoD ==========";
  LOG(INFO) << "DoD-1 exact token match: " << (dod_exact_match ? "PASS" : "FAIL");
  LOG(INFO) << "DoD-2 generated length == max_new_tokens: " << (dod_len_ok ? "PASS" : "FAIL");
  LOG(INFO) << "DoD-3 deterministic mode enabled: " << (dod_deterministic ? "PASS" : "FAIL");
  LOG(INFO) << "P0 STATUS: " << (p0_pass ? "PASS" : "FAIL");

  std::string decoded_text = large_model.decode(result.accepted_tokens);
  LOG(INFO) << "Decoded text: " << decoded_text;

  return p0_pass ? 0 : 2;
}