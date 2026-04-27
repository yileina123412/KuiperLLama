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
  std::string prompt = "Once upon a time, there was a little girl named Lily";
  int32_t draft_len = 4;
  int32_t max_new_tokens = 100;
  bool strict_mode = true;

  // Step4: 先固定确定性（greedy-only）
  bool deterministic = true;

  // 模式开关
  bool scan_mode = true;          // true: 扫描评测
  bool single_demo_mode = false;  // true: 单条demo输出文本

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

static void build_context_state_for_model(model::LLama2Model& model_ref,
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
    model_ref.set_kv_total_tokens(pos + 1);
  } else {
    pos = 0;
    last_token = context_tokens[0];
    model_ref.set_kv_total_tokens(1);
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

static BaselineResult run_large_baseline_greedy(model::LLama2Model& large_model,
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
    int32_t tok = large_model.decode_one_greedy(pos, last_token);
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
  SpeculativeDecoder(model::LLama2Model& draft_model, model::LLama2Model& large_model,
                     const SpeculativeConfig& config)
      : draft_model_(draft_model), large_model_(large_model), config_(config) {}

  VerificationResult decode(const std::string& prompt, int32_t num_gen_tokens) {
    VerificationResult result;
    CHECK(config_.deterministic) << "Stage2 currently supports deterministic=true only.";

    std::vector<int32_t> prompt_tokens = large_model_.encode(prompt);
    if (prompt_tokens.empty()) {
      LOG(ERROR) << "Empty prompt tokens!";
      return result;
    }

    std::vector<int32_t> committed_all = prompt_tokens;

    DecodeState draft_state;
    DecodeState large_state;
    build_state_from_context(draft_model_, committed_all, draft_state);
    build_state_from_context(large_model_, committed_all, large_state);

    while (static_cast<int32_t>(result.accepted_tokens.size()) < num_gen_tokens) {
      const int32_t remain = num_gen_tokens - static_cast<int32_t>(result.accepted_tokens.size());
      const int32_t k = std::min(config_.draft_len, remain);

      // 保存 draft 暂存态（S2-3）
      const auto draft_ckpt = draft_model_.kv_token_checkpoint();
      const DecodeState draft_state_before = draft_state;

      // 小模型草拟
      std::vector<int32_t> draft_tokens = draft_k_tokens(draft_state, k);
      if (draft_tokens.empty()) {
        break;
      }

      // 大模型批量验证（S2-2）
      auto verify = large_model_.verify_draft_batch_block(large_state.pos, large_state.last_token,
                                                          draft_tokens);

      const int32_t old_size = static_cast<int32_t>(result.accepted_tokens.size());
      std::vector<int32_t> round_committed;

      // 接受前缀
      for (int32_t i = 0; i < verify.accepted_prefix_len; ++i) {
        const int32_t t = draft_tokens[i];
        result.accepted_tokens.push_back(t);
        committed_all.push_back(t);
        round_committed.push_back(t);
        result.accept_count += 1;
      }

      if (verify.all_accepted) {
        if (!draft_tokens.empty()) {
          large_state.pos += static_cast<int32_t>(draft_tokens.size());
          large_state.last_token = draft_tokens.back();
        }
      } else {
        // mismatch 位置：使用大模型 token
        result.accepted_tokens.push_back(verify.mismatch_large_token);
        committed_all.push_back(verify.mismatch_large_token);
        round_committed.push_back(verify.mismatch_large_token);
        result.accept_count += 1;

        // 拒绝剩余草稿
        for (int32_t i = verify.accepted_prefix_len; i < static_cast<int32_t>(draft_tokens.size());
             ++i) {
          result.rejected_tokens.push_back(draft_tokens[i]);
        }
        result.reject_count +=
            (static_cast<int32_t>(draft_tokens.size()) - verify.accepted_prefix_len);

        if (result.first_mismatch_pos < 0) {
          result.first_mismatch_pos = old_size + verify.accepted_prefix_len;
        }

        // 大模型状态前进：accepted_prefix + mismatch_token
        large_state.pos += verify.accepted_prefix_len + 1;
        large_state.last_token = verify.mismatch_large_token;

        // draft 回退（S2-3）
        draft_model_.kv_token_rollback(draft_ckpt);
        draft_state = draft_state_before;

        // 回放“本轮已提交token”，避免全量重建上下文
        for (int32_t t : round_committed) {
          (void)draft_model_.decode_one_greedy(draft_state.pos, draft_state.last_token);
          draft_state.last_token = t;
          draft_state.pos += 1;
        }
      }
    }

    if (static_cast<int32_t>(result.accepted_tokens.size()) > num_gen_tokens) {
      result.accepted_tokens.resize(num_gen_tokens);
    }
    return result;
  }

 private:
  struct DecodeState {
    int32_t pos = 0;
    int32_t last_token = -1;
    bool ready = false;
  };

  model::LLama2Model& draft_model_;
  model::LLama2Model& large_model_;
  const SpeculativeConfig& config_;

  void build_state_from_context(model::LLama2Model& model_ref,
                                const std::vector<int32_t>& context_tokens, DecodeState& st) const {
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

      st.pos = static_cast<int32_t>(context_tokens.size()) - 1;
      st.last_token = context_tokens.back();
      st.ready = true;

      model_ref.set_kv_total_tokens(st.pos + 1);
    } else {
      st.pos = 0;
      st.last_token = context_tokens[0];
      st.ready = true;
      model_ref.set_kv_total_tokens(1);
    }
  }

  std::vector<int32_t> draft_k_tokens(DecodeState& draft_state, int32_t k) {
    std::vector<int32_t> out =
        draft_model_.draft_block_greedy(draft_state.pos, draft_state.last_token, k);
    if (!out.empty()) {
      draft_state.pos += static_cast<int32_t>(out.size());
      draft_state.last_token = out.back();
    }
    return out;
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

  // ===== 配置覆盖（按需）=====
  config.strict_mode = true;
  config.deterministic = true;
  config.kv_prefix = 0;
  config.max_new_tokens = 128;

  if (config.single_demo_mode) {
    // ===== 单条 demo 模式（有输出文本）=====
    config.draft_len = 2;

    large_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);
    BaselineResult baseline =
        run_large_baseline_greedy(large_model, config.prompt, config.max_new_tokens);

    draft_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);
    large_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);

    SpeculativeDecoder decoder(draft_model, large_model, config);
    auto t0 = std::chrono::steady_clock::now();
    VerificationResult result = decoder.decode(config.prompt, config.max_new_tokens);
    auto t1 = std::chrono::steady_clock::now();

    const double spec_s = std::chrono::duration<double>(t1 - t0).count();
    const double speedup = (spec_s > 0.0) ? (baseline.duration_s / spec_s) : 0.0;

    int32_t mismatch_idx = -1;
    bool exact_match =
        compare_token_sequences(result.accepted_tokens, baseline.tokens, mismatch_idx);

    LOG(INFO) << "========== SINGLE DEMO ==========";
    LOG(INFO) << "exact_match=" << (exact_match ? "Y" : "N") << ", speedup=" << speedup
              << ", accept_count=" << result.accept_count
              << ", reject_count=" << result.reject_count;

    std::string spec_text = large_model.decode(result.accepted_tokens);
    std::string base_text = large_model.decode(baseline.tokens);
    LOG(INFO) << "[SPEC TEXT] " << spec_text;
    LOG(INFO) << "[BASE TEXT] " << base_text;

    return exact_match ? 0 : 2;
  }

  // ===== 扫描评测模式 =====
  config.scan_mode = true;
  const std::vector<int32_t> draft_len_grid = {2, 4};
  const std::vector<std::string> prompts = {
      "Once upon a time, there was a little girl named Lily",
      "Explain the concept of overfitting in machine learning in simple terms.",
      "写一段关于春天的短文，100字左右。",
      "Please provide three practical tips to improve coding productivity."};

  std::ofstream csv(config.output_csv);
  csv << "prompt_id,draft_len,baseline_s,spec_s,baseline_tps,spec_tps,speedup,accept_count,reject_"
         "count,accept_rate,exact_match\n";

  int total_runs = 0;
  int exact_failures = 0;
  std::map<int32_t, double> speedup_sum;
  std::map<int32_t, int> speedup_cnt;

  for (int pidx = 0; pidx < static_cast<int>(prompts.size()); ++pidx) {
    const std::string& prompt = prompts[pidx];

    large_model.configure_kv_runtime(config.kv_window, config.kv_prefix, true);
    BaselineResult baseline = run_large_baseline_greedy(large_model, prompt, config.max_new_tokens);

    for (int32_t dlen : draft_len_grid) {
      SpeculativeConfig run_cfg = config;
      run_cfg.prompt = prompt;
      run_cfg.draft_len = dlen;

      draft_model.configure_kv_runtime(run_cfg.kv_window, run_cfg.kv_prefix, true);
      large_model.configure_kv_runtime(run_cfg.kv_window, run_cfg.kv_prefix, true);

      SpeculativeDecoder decoder(draft_model, large_model, run_cfg);

      auto t0 = std::chrono::steady_clock::now();
      VerificationResult result = decoder.decode(run_cfg.prompt, run_cfg.max_new_tokens);
      auto t1 = std::chrono::steady_clock::now();
      const double spec_s = std::chrono::duration<double>(t1 - t0).count();

      int32_t mismatch_idx = -1;
      bool exact_match =
          compare_token_sequences(result.accepted_tokens, baseline.tokens, mismatch_idx);

      const double base_tps =
          baseline.tokens.empty()
              ? 0.0
              : static_cast<double>(baseline.tokens.size()) / std::max(baseline.duration_s, 1e-9);
      const double spec_tps =
          result.accepted_tokens.empty()
              ? 0.0
              : static_cast<double>(result.accepted_tokens.size()) / std::max(spec_s, 1e-9);
      const double speedup = (spec_s > 0.0) ? (baseline.duration_s / spec_s) : 0.0;

      const int denom = std::max(1, result.accept_count + result.reject_count);
      const double accept_rate =
          static_cast<double>(result.accept_count) / static_cast<double>(denom);

      csv << pidx << "," << dlen << "," << baseline.duration_s << "," << spec_s << "," << base_tps
          << "," << spec_tps << "," << speedup << "," << result.accept_count << ","
          << result.reject_count << "," << accept_rate << "," << (exact_match ? 1 : 0) << "\n";

      total_runs += 1;
      if (!exact_match) {
        exact_failures += 1;
      }

      speedup_sum[dlen] += speedup;
      speedup_cnt[dlen] += 1;

      LOG(INFO) << "[SCAN] prompt_id=" << pidx << ", draft_len=" << dlen << ", speedup=" << speedup
                << ", exact_match=" << (exact_match ? "Y" : "N") << ", accept_rate=" << accept_rate;
    }
  }

  csv.close();
  LOG(INFO) << "Scan CSV written to: " << config.output_csv;

  int32_t best_dlen = -1;
  double best_avg_speedup = -1.0;
  for (auto& kv : speedup_sum) {
    const int32_t dlen = kv.first;
    const double avg = kv.second / std::max(1, speedup_cnt[dlen]);
    LOG(INFO) << "[AVG] draft_len=" << dlen << ", avg_speedup=" << avg;
    if (avg > best_avg_speedup) {
      best_avg_speedup = avg;
      best_dlen = dlen;
    }
  }

  const bool dod_exact = (exact_failures == 0);
  const bool dod_perf = (best_avg_speedup >= 1.10);
  const bool dod_stability =
      (total_runs == static_cast<int>(prompts.size()) * static_cast<int>(draft_len_grid.size()));
  const bool p1_pass = dod_exact && dod_perf && dod_stability;

  LOG(INFO) << "========== P1 DoD ==========";
  LOG(INFO) << "DoD-1 exact token match all runs: " << (dod_exact ? "PASS" : "FAIL");
  LOG(INFO) << "DoD-2 best avg speedup >= 1.10: " << (dod_perf ? "PASS" : "FAIL");
  LOG(INFO) << "DoD-3 scan coverage complete: " << (dod_stability ? "PASS" : "FAIL");
  LOG(INFO) << "Best draft_len: " << best_dlen;
  LOG(INFO) << "Best avg speedup: " << best_avg_speedup;
  LOG(INFO) << "P1 STATUS: " << (p1_pass ? "PASS" : "FAIL");

  return p1_pass ? 0 : 3;
}