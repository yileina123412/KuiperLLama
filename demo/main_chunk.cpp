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
#include "model/llama3.h"
/*
`sample_from_logits(...)` 的输入就是**模型输出的
logits**：未归一化、每个词的打分（越大越可能）。函数做的就是**从这些 logits 里按采样策略选出下一个
token**。流程如下：

1) **把 logits 拷到 CPU**
如果 logits 在 GPU，就先 `to_cpu()`，便于做采样。

2) **屏蔽特殊 token**
把 `<unk>` 和 `<bos>` 的分数设为极小值，避免被选中。

3) **重复惩罚（repetition penalty）**
在最近的历史窗口里出现过的 token 会被削弱概率，减少重复输出。

4) **温度（temperature）**
把所有分数除以 temperature：
- 温度低 → 更保守
- 温度高 → 更随机
- 温度接近 0 → 退化成贪心（直接选最大值）。

5) **Top‑k 筛选**
只保留分数最高的 k 个 token，缩小候选集合。

6) **Softmax 归一化**
对保留的候选做 softmax，得到概率分布。

7) **Top‑p（核采样）**
按概率从大到小累加，直到累计达到 p，只保留这一部分候选，再重新归一化。

8) **按概率采样**
用离散分布从候选里随机采样一个 token，作为输出。

---

**一句话概括**：
输入是 logits（未归一化打分），输出是“按温度 + top‑k + top‑p + 重复惩罚”采样出的下一个 token。*/
static int32_t sample_from_logits(const tensor::Tensor& logits_any_device,
                                  const std::vector<int32_t>& history, float temperature = 0.85f,
                                  int top_k = 40, float top_p = 0.92f,
                                  float repetition_penalty = 1.10f) {
  tensor::Tensor logits = logits_any_device;  // 拷贝一个 view，避免改模型内部 buffer
  if (logits.device_type() == base::DeviceType::kDeviceCUDA) {
    logits.to_cpu();
  }

  const int32_t vocab = static_cast<int32_t>(logits.size());
  const float* p = logits.ptr<float>();
  std::vector<float> scores(p, p + vocab);
  // --- Debug: check logits range / NaN / Inf ---
  {
    float max_logit = -1e30f;
    float min_logit = 1e30f;
    int nan_cnt = 0, inf_cnt = 0;
    for (int i = 0; i < vocab; ++i) {
      float v = scores[i];
      if (std::isnan(v)) nan_cnt++;
      if (std::isinf(v)) inf_cnt++;
      max_logit = std::max(max_logit, v);
      min_logit = std::min(min_logit, v);
    }
    if (nan_cnt > 0 || inf_cnt > 0) {
      printf("[DEBUG] logits NaN=%d Inf=%d min=%f max=%f\n", nan_cnt, inf_cnt, min_logit,
             max_logit);
    }
  }

  if (vocab > 0) scores[0] = -1e30f;  // <unk>
  if (vocab > 1) scores[1] = -1e30f;  // <bos>

  // repetition penalty
  const int32_t repeat_window = 96;
  int32_t start = static_cast<int32_t>(history.size()) - repeat_window;
  if (start < 0) start = 0;
  for (int32_t i = start; i < static_cast<int32_t>(history.size()); ++i) {
    int32_t t = history[i];
    if (t >= 0 && t < vocab) {
      if (scores[t] > 0.f)
        scores[t] /= repetition_penalty;
      else
        scores[t] *= repetition_penalty;
    }
  }

  // temperature = 0 等价 greedy
  if (temperature <= 1e-6f) {
    return static_cast<int32_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());
  }

  for (float& x : scores) x /= temperature;

  // top-k
  std::vector<int32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0);
  if (top_k > 0 && top_k < vocab) {
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    idx.resize(top_k);
  } else {
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  }

  // softmax（数值稳定）
  float m = -1e30f;
  for (int i : idx) m = std::max(m, scores[i]);

  std::vector<float> probs;
  probs.reserve(idx.size());
  float s = 0.f;
  for (int i : idx) {
    float e = std::exp(scores[i] - m);
    probs.push_back(e);
    s += e;
  }
  for (float& x : probs) x /= (s + 1e-12f);
  // --- Debug: check softmax sum and invalid probs ---
  {
    float sum_p = 0.f;
    int nan_cnt = 0, inf_cnt = 0;
    for (float v : probs) {
      if (std::isnan(v)) nan_cnt++;
      if (std::isinf(v)) inf_cnt++;
      sum_p += v;
    }
    if (nan_cnt > 0 || inf_cnt > 0 || sum_p < 0.98f || sum_p > 1.02f) {
      printf("[DEBUG] softmax sum=%f NaN=%d Inf=%d\n", sum_p, nan_cnt, inf_cnt);
    }
  }

  // top-p
  if (top_p < 1.f) {
    std::vector<int> order(probs.size());
    std::iota(order.begin(), order.end(), 0);
    // probs 已按 idx 对应的logit降序，理论上可不排序；这里保持稳妥
    std::sort(order.begin(), order.end(), [&](int a, int b) { return probs[a] > probs[b]; });

    float cum = 0.f;
    int keep = 0;
    for (int j : order) {
      cum += probs[j];
      keep++;
      if (cum >= top_p) break;
    }
    keep = std::max(1, keep);

    std::vector<int32_t> idx2;
    std::vector<float> probs2;
    idx2.reserve(keep);
    probs2.reserve(keep);
    for (int n = 0; n < keep; ++n) {
      int j = order[n];
      idx2.push_back(idx[j]);
      probs2.push_back(probs[j]);
    }

    float z = 0.f;
    for (float x : probs2) z += x;
    for (float& x : probs2) x /= (z + 1e-12f);

    static thread_local std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int> dist(probs2.begin(), probs2.end());
    return idx2[dist(gen)];
  }

  static thread_local std::mt19937 gen(std::random_device{}());
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return idx[dist(gen)];
}

int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 const std::string& csv_path, bool enable_chunk_prefill, int prefill_chunk_size,
                 const std::string& run_tag, bool need_output = false) {
  auto start_time = std::chrono::steady_clock::now();
  // 进行文本编码，得到 token 序列
  auto tokens = model.encode(sentence);
  const bool use_chunked_prefill = enable_chunk_prefill && (prefill_chunk_size > 0);

  int32_t prompt_len = tokens.size();
  printf("\nprompt_len: %d \n", prompt_len);
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = false;  // 改成：prompt 一次性 prefill，while 里只有 decode

  std::vector<int32_t> words;

  // ===== 1) Prompt: prefill（支持 chunked） =====
  if (prompt_len >= 2) {
    std::vector<int32_t> prefill_tokens(tokens.begin(), tokens.end() - 1);  // [0..L-2]

    // ---- prefill 内存峰值统计（仅估算）----
    size_t mem_free = 0, mem_total = 0;
    size_t min_free = std::numeric_limits<size_t>::max();

    if (!use_chunked_prefill) {
      // ---- 原始：一次性 prefill ----
      const auto& prefill_emb = model.embedding(prefill_tokens);
      auto [prefill_input_tokens, prefill_input_embeddings, prefill_token_num] = prefill_emb;
      (void)prefill_input_tokens;
      (void)prefill_token_num;

      pos_tensor.index<int32_t>(0) = 0;  // start_pos = 0
      int dummy_next = -1;
      cudaDeviceSynchronize();
      cudaMemGetInfo(&mem_free, &mem_total);
      min_free = std::min(min_free, mem_free);
      model.prefill(prefill_input_embeddings, pos_tensor, dummy_next);
      cudaDeviceSynchronize();
      cudaMemGetInfo(&mem_free, &mem_total);
      min_free = std::min(min_free, mem_free);
    } else {
      // ---- 新增：chunked prefill ----
      int32_t start_pos = 0;
      const int32_t total = static_cast<int32_t>(prefill_tokens.size());

      for (int32_t i = 0; i < total; i += prefill_chunk_size) {
        const int32_t len = std::min(prefill_chunk_size, total - i);
        std::vector<int32_t> chunk(prefill_tokens.begin() + i, prefill_tokens.begin() + i + len);

        const auto& chunk_emb = model.embedding(chunk);
        auto [chunk_input_tokens, chunk_input_embeddings, chunk_token_num] = chunk_emb;
        (void)chunk_input_tokens;
        (void)chunk_token_num;

        pos_tensor.index<int32_t>(0) = start_pos;  // 每个 chunk 的起始位置
        int dummy_next = -1;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&mem_free, &mem_total);
        min_free = std::min(min_free, mem_free);
        model.prefill(chunk_input_embeddings, pos_tensor, dummy_next);
        cudaDeviceSynchronize();
        cudaMemGetInfo(&mem_free, &mem_total);
        min_free = std::min(min_free, mem_free);

        start_pos += len;  // 位置递增
      }
    }

    cudaError_t prefill_sync_err = cudaDeviceSynchronize();
    if (prefill_sync_err != cudaSuccess) {
      LOG(FATAL) << "prefill cudaDeviceSynchronize failed: "
                 << cudaGetErrorString(prefill_sync_err);
    }

    auto end_ttft = std::chrono::steady_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(end_ttft - start_time).count();
    int chunk_count = 1;
    if (use_chunked_prefill && prompt_len >= 2) {
      const int32_t total = static_cast<int32_t>(prefill_tokens.size());
      chunk_count = (total + prefill_chunk_size - 1) / prefill_chunk_size;
    }

    printf("\nTTFT[%s]: %lf ms (chunks=%d)\n", run_tag.c_str(), ttft_ms, chunk_count);
    printf("\nTTFT[%s]: %lf ms/token\n", run_tag.c_str(), ttft_ms / prompt_len);

    if (mem_total > 0 && min_free != std::numeric_limits<size_t>::max()) {
      double peak_used_gb = double(mem_total - min_free) / (1024.0 * 1024.0 * 1024.0);
      printf("PREFILL_PEAK[%s]: %.3f GB\n", run_tag.c_str(), peak_used_gb);
    }

    // ===== 2) 从最后一个 prompt token 开始 decode =====
    pos = prompt_len - 1;
    next = tokens.back();  // 作为 decode 的输入 token（上下文已在 KV cache 里）
    is_prompt = false;
  } else {
    // prompt 只有 1 个 token：没有 prefill 的意义，直接从它开始 decode
    pos = 0;
    next = tokens[0];
    is_prompt = false;
  }
  // // 临时decode only
  // if (prompt_len >= 2) {
  //   for (int32_t i = 0; i < prompt_len - 1; i++) {
  //     pos_tensor.index<int32_t>(0) = i;
  //     std::vector<int32_t> warm_tokens = {tokens[i]};
  //     const auto& warm_emb = model.embedding(warm_tokens);
  //     tensor::Tensor warm_input = model.fill_input(pos_tensor, warm_emb, false);
  //     int dummy_next = -1;
  //     model.forward(warm_input, pos_tensor, dummy_next);
  //   }
  //   cudaError_t warm_sync_err = cudaDeviceSynchronize();
  //   if (warm_sync_err != cudaSuccess) {
  //     LOG(FATAL) << "warmup cudaDeviceSynchronize failed: " << cudaGetErrorString(warm_sync_err);
  //   }

  //   auto end_ttft = std::chrono::steady_clock::now();
  //   double ttft_ms = std::chrono::duration<double, std::milli>(end_ttft - start_time).count();
  //   printf("\nTTFT(warmup-decode): %lf ms\n", ttft_ms);

  //   pos = prompt_len - 1;
  //   next = tokens.back();
  //   is_prompt = false;
  // } else {
  //   pos = 0;
  //   next = tokens[0];
  //   is_prompt = false;
  // }

  std::vector<int32_t> history = tokens;  // 用于 repetition penalty

  const int32_t max_new_tokens = total_steps;  // 希望新生成多少个
  const int32_t min_new_tokens = 256;          // 可按需改，比如 64/128/256
  int32_t generated = 0;

  std::ofstream csv_file(csv_path);
  csv_file << "step,token_id,time_ms\n";  // CSV 表头

  auto prev_time = std::chrono::steady_clock::now();

  while (generated < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;

    std::vector<int32_t> cur_tokens = {next};
    const auto& token_embedding = model.embedding(cur_tokens);
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);

    int dummy_next = -1;
    model.forward(input, pos_tensor, dummy_next);  // 只算 logits，不用内置 argmax

    // 关键：确保 CUDA stream 上的 logits 已经写完
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      LOG(FATAL) << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err);
    }

    tensor::Tensor logits = model.get_buffer(model::ModelBufferType::kForwardOutput);
    if (logits.device_type() == base::DeviceType::kDeviceCUDA) {
      logits.to_cpu();
    }
    // ---- DEBUG: logits stats (min/max/NaN/Inf) ----
    // {
    //   const float* lp = logits.ptr<float>();
    //   int32_t n = static_cast<int32_t>(logits.size());
    //   float minv = 1e30f;
    //   float maxv = -1e30f;
    //   int nan_cnt = 0;
    //   int inf_cnt = 0;
    //   for (int i = 0; i < n; ++i) {
    //     float v = lp[i];
    //     if (std::isnan(v)) nan_cnt++;
    //     if (std::isinf(v)) inf_cnt++;
    //     if (v < minv) minv = v;
    //     if (v > maxv) maxv = v;
    //   }
    //   if (nan_cnt > 0 || inf_cnt > 0 || maxv > 100.f || minv < -100.f) {
    //     printf("[LOGITS] step=%d min=%f max=%f NaN=%d Inf=%d\n", generated, minv, maxv, nan_cnt,
    //            inf_cnt);
    //   }
    //   printf("[LOGITS] step=%d min=%f max=%f NaN=%d Inf=%d\n", generated, minv, maxv, nan_cnt,
    //          inf_cnt);
    // }
    // next = sample_from_logits(logits, history, 0.85f, 40, 0.92f, 1.10f);
    next = sample_from_logits(logits, history, 0.70f, 20, 0.90f, 1.05f);
    // next = sample_from_logits(logits, history, 0.0f, 1, 1.0f, 1.0f);

    // if (model.is_sentence_ending(next)) {
    //   break;
    // }

    // history.push_back(next);
    // words.push_back(next);
    // printf("Step %d: Token ID = %d\n", pos, next);
    // pos += 1;
    // 到达最小长度之前，忽略 EOS，避免太早停
    // if (model.is_sentence_ending(next) && generated >= min_new_tokens) {
    //   break;
    // }

    history.push_back(next);
    words.push_back(next);
    // printf("Step %d: Token ID = %d\n", generated, next);
    auto cur_time = std::chrono::steady_clock::now();
    double step_time_ms = std::chrono::duration<double, std::milli>(cur_time - prev_time).count();
    csv_file << generated << "," << next << "," << step_time_ms << "\n";
    prev_time = cur_time;

    pos += 1;        // 绝对位置，驱动 KV cache
    generated += 1;  // 新生成 token 计数
  }

  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  csv_file.close();
  LOG(INFO) << "Generation steps saved to generation_steps.csv";
  return generated;
}

struct ExpCase {
  int window;
  int prefix;
  int steps;
  std::string tag;
};

int main(int argc, char* argv[]) {
  // if (argc != 3) {
  //   LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
  //   return -1;
  // }
  // const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  // const char* tokenizer_path = argv[2];
  // const char* checkpoint_path = "/home/furina/models/stories110M.bin";  // e.g.
  // const char* checkpoint_path = "/home/furina/models/stories260K.bin";  // e.g.
  // const char* checkpoint_path = "/home/furina/models/llama-160m.bin";  // e.g.
  // const char* checkpoint_path = "/home/furina/models/tinyllama_int8.bin";  // e.g.
  //   const char* checkpoint_path = "/home/furina/models/tinyllama.bin";  // e.g.
  const char* checkpoint_path = "/home/furina/models/tinyllama_longer.bin";  // e.g.
  // const char* checkpoint_path = "/home/furina/models/llama2_7b_smooth_pro_v3.bin";
  // const char* checkpoint_path = "/home/furina/models/tinyllama_4bit.bin";
  // const char* checkpoint_path = "/home/furina/models/llama2_7b_smooth_pro_v3_32.bin";

  // const char* tokenizer_path = "/home/furina/models/tok512.model";  // stories260K
  const char* tokenizer_path = "/home/furina/models/tokenizer.model";
  // const char* checkpoint_path =
  //     "/home/furina/models/llama32_1bnq.bin";  // e.g.
  //                                                                                  //
  //                                                                                  out/model.bin
  // const char* tokenizer_path =
  //     "/home/furina/models/tokenizer.json";
  // model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path,
  // true,
  //                          model::Model::QuantFormat::kSQ4);
  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  model.set_kv_window_size(2048);  // 先用 256 做实验
  model.reset_kv_total_tokens();   // 可选，当前阶段主要用 pos 计算 valid_len
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }

  const std::string& sentence =
      R"(You are a meticulous technical assistant. Read the following continuous report and output a concise summary plus a checklist of actions. Keep your output readable and structured.

REPORT START
Section 1: Background
In early May, the team reviewed the GPU inference pipeline with a focus on the prefill stage. The system uses a transformer with sliding‑window attention and optional prefix‑keep. The primary objective is to minimize time‑to‑first‑token (TTFT) while keeping memory pressure stable. The team observed that prefill dominates latency for long prompts, so chunked prefill was proposed to split prompts into blocks and reduce transient pressure.

Section 2: Correctness
Chunked prefill must be equivalent to baseline prefill in terms of logits. The report notes that sampling randomness can hide discrepancies, so the recommended verification is to compare logits or run greedy decoding to ensure deterministic parity.

Section 3: Metrics
We track TTFT, steps per second, and per‑step latency into CSV. Two additional metrics are proposed: prefill phase duration and peak memory usage. Memory is estimated by sampling cudaMemGetInfo before and after each prefill chunk, tracking the minimum free memory during prefill.

Section 4: Expected Behavior
Chunked prefill shows benefits when the prompt is long enough and the attention window is large enough to avoid truncation. If the window is too small, prompt tokens are clipped, and chunked prefill offers minimal advantage. Therefore, evaluation should use prompts exceeding 3000 tokens with window size at least 4096.

Section 5: Trade‑offs
Small chunks reduce peak pressure but increase kernel launch overhead. Large chunks reduce overhead but may diminish memory benefits. The suggested starting point is chunk size 256, followed by 512 for comparison.

The remainder of the report repeats the same technical content to extend the prompt length for stress testing while maintaining readability.

REPEATED SECTION A
In early May, the team reviewed the GPU inference pipeline with a focus on the prefill stage. The system uses a transformer with sliding‑window attention and optional prefix‑keep. The primary objective is to minimize time‑to‑first‑token (TTFT) while keeping memory pressure stable. The team observed that prefill dominates latency for long prompts, so chunked prefill was proposed to split prompts into blocks and reduce transient pressure.
Chunked prefill must be equivalent to baseline prefill in terms of logits. Sampling randomness can hide discrepancies, so logits or greedy decoding should be compared for deterministic parity.
We track TTFT, steps per second, and per‑step latency into CSV. Additional metrics include prefill duration and peak memory usage via cudaMemGetInfo sampling.
Chunked prefill shows benefits when prompts are long and the attention window is large enough to avoid truncation.

REPEATED SECTION B
In early May, the team reviewed the GPU inference pipeline with a focus on the prefill stage. The system uses a transformer with sliding‑window attention and optional prefix‑keep. The primary objective is to minimize time‑to‑first‑token (TTFT) while keeping memory pressure stable. The team observed that prefill dominates latency for long prompts, so chunked prefill was proposed to split prompts into blocks and reduce transient pressure.
Chunked prefill must be equivalent to baseline prefill in terms of logits. Sampling randomness can hide discrepancies, so logits or greedy decoding should be compared for deterministic parity.
We track TTFT, steps per second, and per‑step latency into CSV. Additional metrics include prefill duration and peak memory usage via cudaMemGetInfo sampling.
Chunked prefill shows benefits when prompts are long and the attention window is large enough to avoid truncation.

REPEATED SECTION C
In early May, the team reviewed the GPU inference pipeline with a focus on the prefill stage. The system uses a transformer with sliding‑window attention and optional prefix‑keep. The primary objective is to minimize time‑to‑first‑token (TTFT) while keeping memory pressure stable. The team observed that prefill dominates latency for long prompts, so chunked prefill was proposed to split prompts into blocks and reduce transient pressure.
Chunked prefill must be equivalent to baseline prefill in terms of logits. Sampling randomness can hide discrepancies, so logits or greedy decoding should be compared for deterministic parity.
We track TTFT, steps per second, and per‑step latency into CSV. Additional metrics include prefill duration and peak memory usage via cudaMemGetInfo sampling.
Chunked prefill shows benefits when prompts are long and the attention window is large enough to avoid truncation.

REPEATED SECTION D
In early May, the team reviewed the GPU inference pipeline with a focus on the prefill stage. The system uses a transformer with sliding‑window attention and optional prefix‑keep. The primary objective is to minimize time‑to‑first‑token (TTFT) while keeping memory pressure stable. The team observed that prefill dominates latency for long prompts, so chunked prefill was proposed to split prompts into blocks and reduce transient pressure.
Chunked prefill must be equivalent to baseline prefill in terms of logits. Sampling randomness can hide discrepancies, so logits or greedy decoding should be compared for deterministic parity.
We track TTFT, steps per second, and per‑step latency into CSV. Additional metrics include prefill duration and peak memory usage via cudaMemGetInfo sampling.
Chunked prefill shows benefits when prompts are long and the attention window is large enough to avoid truncation.

REPEATED SECTION E
In early May, the team reviewed the GPU inference pipeline with a focus on the prefill stage. The system uses a transformer with sliding‑window attention and optional prefix‑keep. The primary objective is to minimize time‑to‑first‑token (TTFT) while keeping memory pressure stable. The team observed that prefill dominates latency for long prompts, so chunked prefill was proposed to split prompts into blocks and reduce transient pressure.
Chunked prefill must be equivalent to baseline prefill in terms of logits. Sampling randomness can hide discrepancies, so logits or greedy decoding should be compared for deterministic parity.
We track TTFT, steps per second, and per‑step latency into CSV. Additional metrics include prefill duration and peak memory usage via cudaMemGetInfo sampling.
Chunked prefill shows benefits when prompts are long and the attention window is large enough to avoid truncation.

REPEATED SECTION F
In early May, the team reviewed the GPU inference pipeline with a focus on the prefill stage. The system uses a transformer with sliding‑window attention and optional prefix‑keep. The primary objective is to minimize time‑to‑first‑token (TTFT) while keeping memory pressure stable. The team observed that prefill dominates latency for long prompts, so chunked prefill was proposed to split prompts into blocks and reduce transient pressure.
Chunked prefill must be equivalent to baseline prefill in terms of logits. Sampling randomness can hide discrepancies, so logits or greedy decoding should be compared for deterministic parity.
We track TTFT, steps per second, and per‑step latency into CSV. Additional metrics include prefill duration and peak memory usage via cudaMemGetInfo sampling.
Chunked prefill shows benefits when prompts are long and the attention window is large enough to avoid truncation.

REPORT END
Please output a concise summary and a checklist of next steps.)";

  const std::vector<ExpCase> cases = {
      {4096, 64, 300, "prefix_w4096_p64"},
  };
  for (const auto& c : cases) {
    model.set_kv_window_size(c.window);
    model.set_kv_prefix_keep_tokens(c.prefix);
    model.reset_kv_total_tokens();

    const bool enable_chunk_prefill = false;  // true 开启 chunked prefill
    int prefill_chunk_size = 256;             // chunk 大小（token）

    // === 1) baseline：不开启 chunked prefill ===
    {
      const bool enable_chunk_prefill = false;
      std::string csv_name = "generation_steps_" + c.tag + "_baseline.csv";

      auto start = std::chrono::steady_clock::now();
      printf("\n=== CASE: %s (window=%d, prefix=%d) [baseline] ===\n", c.tag.c_str(), c.window,
             c.prefix);
      fflush(stdout);

      int gen_steps = generate(model, sentence, c.steps, csv_name, enable_chunk_prefill,
                               prefill_chunk_size, "baseline", true);

      auto end = std::chrono::steady_clock::now();
      double duration = std::chrono::duration<double>(end - start).count();
      double sps = gen_steps / std::max(duration, 1e-9);

      printf("case=%s, generated=%d, time=%.3f s, steps/s=%.3f, csv=%s\n", c.tag.c_str(), gen_steps,
             duration, sps, csv_name.c_str());
      fflush(stdout);
    }

    // === 2) chunked：开启 chunked prefill ===
    {
      const bool enable_chunk_prefill = true;
      std::string csv_name = "generation_steps_" + c.tag + "_chunked.csv";

      auto start = std::chrono::steady_clock::now();
      printf("\n=== CASE: %s (window=%d, prefix=%d) [chunked] ===\n", c.tag.c_str(), c.window,
             c.prefix);
      fflush(stdout);

      int gen_steps = generate(model, sentence, c.steps, csv_name, enable_chunk_prefill,
                               prefill_chunk_size, "chunked", true);

      auto end = std::chrono::steady_clock::now();
      double duration = std::chrono::duration<double>(end - start).count();
      double sps = gen_steps / std::max(duration, 1e-9);

      printf("case=%s, generated=%d, time=%.3f s, steps/s=%.3f, csv=%s\n", c.tag.c_str(), gen_steps,
             duration, sps, csv_name.c_str());
      fflush(stdout);
    }
    // === 3) chunked：开启 chunked prefill 512 ===
    {
      prefill_chunk_size = 512;  // 可以改成 512 看看更大 chunk 的效果
      const bool enable_chunk_prefill = true;
      std::string csv_name = "generation_steps_" + c.tag + "_chunked.csv";

      auto start = std::chrono::steady_clock::now();
      printf("\n=== CASE: %s (window=%d, prefix=%d) [chunked] ===\n", c.tag.c_str(), c.window,
             c.prefix);
      fflush(stdout);

      int gen_steps = generate(model, sentence, c.steps, csv_name, enable_chunk_prefill,
                               prefill_chunk_size, "chunked", true);

      auto end = std::chrono::steady_clock::now();
      double duration = std::chrono::duration<double>(end - start).count();
      double sps = gen_steps / std::max(duration, 1e-9);

      printf("case=%s, generated=%d, time=%.3f s, steps/s=%.3f, csv=%s\n", c.tag.c_str(), gen_steps,
             duration, sps, csv_name.c_str());
      fflush(stdout);
    }
    // === 4) chunked：开启 chunked prefill 128 ===
    {
      prefill_chunk_size = 128;  // 可以改成 128 看看更小 chunk 的效果
      const bool enable_chunk_prefill = true;
      std::string csv_name = "generation_steps_" + c.tag + "_chunked.csv";

      auto start = std::chrono::steady_clock::now();
      printf("\n=== CASE: %s (window=%d, prefix=%d) [chunked] ===\n", c.tag.c_str(), c.window,
             c.prefix);
      fflush(stdout);

      int gen_steps = generate(model, sentence, c.steps, csv_name, enable_chunk_prefill,
                               prefill_chunk_size, "chunked", true);

      auto end = std::chrono::steady_clock::now();
      double duration = std::chrono::duration<double>(end - start).count();
      double sps = gen_steps / std::max(duration, 1e-9);

      printf("case=%s, generated=%d, time=%.3f s, steps/s=%.3f, csv=%s\n", c.tag.c_str(), gen_steps,
             duration, sps, csv_name.c_str());
      fflush(stdout);
    }
  }

  return 0;
}
