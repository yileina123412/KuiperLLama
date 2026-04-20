#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"
int32_t generate1(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                  bool need_output = false) {
  auto start_time = std::chrono::steady_clock::now();  // 记录开始时间
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;

    if (prompt_len >= 2) {
      std::vector<int32_t> prefill_tokens(tokens.begin(), tokens.end() - 1);
      const auto& prefill_emb = model.embedding(prefill_tokens);
      auto [prefill_input_tokens, prefill_input_embeddings, prefill_token_num] = prefill_emb;

      pos_tensor.index<int32_t>(0) = 0;  // start_pos
      int dummy_next = -1;
      model.prefill(prefill_input_embeddings, pos_tensor, dummy_next);

      // 2) 让下一步 decode 从最后一个 prompt token 开始
      pos = prompt_len - 1;
      is_prompt = false;
      next = tokens.back();  // teacher-forcing 最后一个 prompt token
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};  // 极其关键：把刚才预测出来的 next，变成新的输入！
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (model.is_sentence_ending(next)) {
      break;  // 停止标志
    }

    if (is_prompt) {
      auto end_ttft = std::chrono::steady_clock::now();
      double ttft_ms = std::chrono::duration<double, std::milli>(end_ttft - start_time).count();
      printf("\nTTFT: %lf ms\n", ttft_ms);
      next = tokens.at(pos + 1);  // 强制把用户的下一个词作为答案
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto start_time = std::chrono::steady_clock::now();
  // 进行文本编码，得到 token 序列
  auto tokens = model.encode(sentence);

  int32_t prompt_len = tokens.size();
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

    auto end_ttft = std::chrono::steady_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(end_ttft - start_time).count();
    printf("\nTTFT(prefill): %lf ms\n", ttft_ms);

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

  // ===== 3) Decode loop (1D) =====
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;

    std::vector<int32_t> cur_tokens = {next};
    const auto& token_embedding = model.embedding(cur_tokens);
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);

    model.predict(input, pos_tensor, is_prompt, next);  // next 被更新为“预测出来的 token”
    if (model.is_sentence_ending(next)) {
      break;
    }

    words.push_back(next);
    pos += 1;
  }

  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
  // if (argc != 3) {
  //   LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
  //   return -1;
  // }
  // const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  // const char* tokenizer_path = argv[2];
  // const char* checkpoint_path =
  //     "/home/furina/models/stories110M.bin";  // e.g.
  const char* checkpoint_path = "/home/furina/models/tinyllama_int8.bin";  // e.g.
  // const char* checkpoint_path = "/home/furina/models/llama2_7b_smooth_pro_v3.bin";
  // const char* checkpoint_path = "/home/furina/models/tinyllama_4bit.bin";

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
  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path, true);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  const std::string& sentence = "hello";
  // const std::string& sentence = "this is a test,please answer me in English.";

  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}
