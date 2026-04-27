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
                 const std::string& csv_path, bool need_output = false) {
  auto start_time = std::chrono::steady_clock::now();
  // 进行文本编码，得到 token 序列
  auto tokens = model.encode(sentence);

  int32_t prompt_len = tokens.size();
  printf("\nprompt_len: %d \n", prompt_len);
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = false;  // 改成：prompt 一次性 prefill，while 里只有 decode

  std::vector<int32_t> words;

  // ===== 1) Prompt: 一次性 prefill (2D)，写 KV cache =====
  if (prompt_len >= 2) {
    std::vector<int32_t> prefill_tokens(tokens.begin(), tokens.end() - 1);  // [0..L-2]
    const auto& prefill_emb = model.embedding(prefill_tokens);
    auto [prefill_input_tokens, prefill_input_embeddings, prefill_token_num] = prefill_emb;
    (void)prefill_input_tokens;
    (void)prefill_token_num;

    pos_tensor.index<int32_t>(0) = 0;  // start_pos = 0
    int dummy_next = -1;
    model.prefill(prefill_input_embeddings, pos_tensor, dummy_next);

    cudaError_t prefill_sync_err = cudaDeviceSynchronize();
    if (prefill_sync_err != cudaSuccess) {
      LOG(FATAL) << "prefill cudaDeviceSynchronize failed: "
                 << cudaGetErrorString(prefill_sync_err);
    }

    auto end_ttft = std::chrono::steady_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(end_ttft - start_time).count();
    printf("\nTTFT(prefill): %lf ms\n", ttft_ms);
    printf("\nTTFT(prefill): %lf ms\n", ttft_ms / prompt_len);

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

    const tensor::Tensor& logits = model.get_buffer(model::ModelBufferType::kForwardOutput);
    // next = sample_from_logits(logits, history, 0.85f, 40, 0.92f, 1.10f);
    next = sample_from_logits(logits, history, 0.70f, 20, 0.90f, 1.05f);

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
  // const char* checkpoint_path =
  //     "/home/furina/models/stories110M.bin";  // e.g.
  // const char* checkpoint_path = "/home/furina/models/tinyllama_int8.bin";  // e.g.
  const char* checkpoint_path = "/home/furina/models/tinyllama.bin";  // e.g.
  // const char* checkpoint_path = "/home/furina/models/llama2_7b_smooth_pro_v3.bin";
  // const char* checkpoint_path = "/home/furina/models/tinyllama_4bit.bin";
  // const char* checkpoint_path = "/home/furina/models/llama2_7b_smooth_pro_v3_32.bin";

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
  const std::string& sentence = "hello the world,my name is bob.please tell me something.";

  const std::vector<ExpCase> cases = {
      {2048, 0, 1000, "baseline_w2048_p0"},
      {256, 0, 1000, "window_w256_p0"},
      {256, 64, 1000, "prefix_w256_p64"},
      {128, 32, 1000, "prefix_w128_p32"},
  };
  for (const auto& c : cases) {
    model.set_kv_window_size(c.window);
    model.set_kv_prefix_keep_tokens(c.prefix);
    model.reset_kv_total_tokens();

    std::string csv_name = "generation_steps_" + c.tag + ".csv";

    auto start = std::chrono::steady_clock::now();
    printf("\n=== CASE: %s (window=%d, prefix=%d) ===\n", c.tag.c_str(), c.window, c.prefix);
    fflush(stdout);

    int gen_steps = generate(model, sentence, c.steps, csv_name, true);

    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    double sps = gen_steps / std::max(duration, 1e-9);

    printf("case=%s, generated=%d, time=%.3f s, steps/s=%.3f, csv=%s\n", c.tag.c_str(), gen_steps,
           duration, sps, csv_name.c_str());
    fflush(stdout);
  }
  // const std::string& sentence = "this is a test,please answer me in English.";

  // auto start = std::chrono::steady_clock::now();
  // printf("Generating...\n");
  // fflush(stdout);
  // int steps = generate(model, sentence, 1000, true);
  // auto end = std::chrono::steady_clock::now();
  // auto duration = std::chrono::duration<double>(end - start).count();
  // printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  // fflush(stdout);
  return 0;
}
