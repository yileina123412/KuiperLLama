// #include <base/base.h>
// #include <base/tick.h>
// #include <cuda_runtime_api.h>
// #include <glog/logging.h>
// #include <algorithm>
// #include <chrono>
// #include <cmath>
// #include <fstream>
// #include <numeric>
// #include <random>
// #include <vector>
// #include "model/llama3.h"

// // ===== Greedy 采样（确定性）=====
// static int32_t greedy_sample(const tensor::Tensor& logits_any_device) {
//   tensor::Tensor logits = logits_any_device;
//   if (logits.device_type() == base::DeviceType::kDeviceCUDA) {
//     logits.to_cpu();
//   }
//   const float* p = logits.ptr<float>();
//   const int32_t vocab = static_cast<int32_t>(logits.size());

//   float max_val = -1e30f;
//   int32_t max_idx = 0;
//   for (int32_t i = 0; i < vocab; ++i) {
//     if (p[i] > max_val) {
//       max_val = p[i];
//       max_idx = i;
//     }
//   }
//   return max_idx;
// }

// // ===== 配置结构体 =====
// struct SpeculativeConfig {
//   int32_t draft_len = 4;         // 小模型一次草拟多少个 token
//   int32_t max_new_tokens = 100;  // 最大新增 token 数
//   bool strict_mode = true;       // 严格验收模式：第一个 mismatch 就丢弃所有草稿
//   std::string output_csv = "speculative_result.csv";
// };

// // ===== 验收结果结构体 =====
// struct VerificationResult {
//   std::vector<int32_t> accepted_tokens;  // 接受的 token 序列
//   std::vector<int32_t> rejected_tokens;  // 被拒绝的 token 序列
//   int32_t accept_count = 0;
//   int32_t reject_count = 0;
//   int32_t first_mismatch_pos = -1;  // 第一个不匹配的位置（-1 表示全部接受）
// };

// // ===== 主 Speculative Decode 逻辑 =====
// class SpeculativeDecoder {
//  public:
//   SpeculativeDecoder(const model::LLama2Model& draft_model, const model::LLama2Model&
//   large_model,
//                      const SpeculativeConfig& config)
//       : draft_model_(draft_model), large_model_(large_model), config_(config) {}

//   // 执行 speculative decode
//   VerificationResult decode(const std::string& prompt, int32_t num_gen_tokens) {
//     VerificationResult result;

//     // 1. Tokenize prompt（使用大模型的 tokenizer，两个模型应该一样）
//     std::vector<int32_t> prompt_tokens = large_model_.encode(prompt);
//     int32_t prompt_len = prompt_tokens.size();

//     if (prompt_tokens.empty()) {
//       LOG(ERROR) << "Empty prompt tokens!";
//       return result;
//     }

//     LOG(INFO) << "Prompt length: " << prompt_len;

//     // 2. Prefill：用大模型处理 prompt（累积 KV cache）
//     this->prefill_large_model(prompt_tokens);

//     // 3. Decode 循环
//     std::vector<int32_t> generated_tokens;
//     int32_t current_pos = prompt_len;

//     while (static_cast<int32_t>(generated_tokens.size()) < num_gen_tokens) {
//       // ===== 第一步：小模型草拟 K 个 token =====
//       std::vector<int32_t> draft_tokens = this->draft_k_tokens(current_pos, config_.draft_len);

//       if (draft_tokens.empty()) {
//         LOG(INFO) << "Draft returned empty, stopping.";
//         break;
//       }

//       // ===== 第二步：大模型逐 token 验收 =====
//       VerificationResult verify_result = this->verify_draft_tokens(current_pos, draft_tokens);

//       // ===== 第三步：处理验收结果 =====
//       if (config_.strict_mode) {
//         // 严格模式：第一个 mismatch 后全部丢弃
//         if (verify_result.first_mismatch_pos >= 0) {
//           // 接受 mismatch 前的 token
//           int32_t accept_up_to = verify_result.first_mismatch_pos;
//           for (int32_t i = 0; i < accept_up_to; ++i) {
//             generated_tokens.push_back(draft_tokens[i]);
//           }
//           result.accept_count += accept_up_to;

//           // 在 mismatch 位置用大模型的 token
//           int32_t large_token = verify_result.accepted_tokens[accept_up_to];
//           generated_tokens.push_back(large_token);
//           result.accept_count += 1;
//           result.reject_count += (draft_tokens.size() - accept_up_to);

//           current_pos += (accept_up_to + 1);

//           // 记录第一次 mismatch
//           if (result.first_mismatch_pos < 0) {
//             result.first_mismatch_pos = accept_up_to;
//           }
//         } else {
//           // 全部接受
//           for (int32_t token : draft_tokens) {
//             generated_tokens.push_back(token);
//             result.accept_count += 1;
//           }
//           current_pos += draft_tokens.size();
//         }
//       }
//     }

//     result.accepted_tokens = generated_tokens;
//     return result;
//   }

//  private:
//   const model::LLama2Model& draft_model_;
//   const model::LLama2Model& large_model_;
//   const SpeculativeConfig& config_;

//   // ===== Prefill：用大模型处理完整 prompt =====
//   void prefill_large_model(const std::vector<int32_t>& prompt_tokens) {
//     if (prompt_tokens.size() < 2) {
//       return;  // 太短，不 prefill
//     }

//     std::vector<int32_t> prefill_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
//     const auto& prefill_emb = large_model_.embedding(prefill_tokens);
//     auto [prefill_input_tokens, prefill_input_embeddings, prefill_token_num] = prefill_emb;
//     (void)prefill_input_tokens;
//     (void)prefill_token_num;

//     tensor::Tensor pos_tensor = large_model_.get_buffer(model::ModelBufferType::kInputPos);
//     pos_tensor.index<int32_t>(0) = 0;

//     int dummy_next = -1;
//     large_model_.prefill(prefill_input_embeddings, pos_tensor, dummy_next);

//     cudaError_t err = cudaDeviceSynchronize();
//     if (err != cudaSuccess) {
//       LOG(FATAL) << "Large model prefill sync failed: " << cudaGetErrorString(err);
//     }

//     LOG(INFO) << "Prefill done, prompt_len=" << prompt_tokens.size();
//   }

//   // ===== 小模型草拟 K 个 token =====
//   std::vector<int32_t> draft_k_tokens(int32_t start_pos, int32_t draft_len) {
//     std::vector<int32_t> draft_tokens;
//     int32_t cur_pos = start_pos;
//     int32_t next_token = -1;

//     // 需要准备小模型的初始状态
//     // 这里简化：直接用小模型的 forward 逐 token
//     for (int32_t i = 0; i < draft_len; ++i) {
//       if (i == 0) {
//         // 从大模型的最后一个 token 开始
//         if (draft_tokens.empty() && start_pos > 0) {
//           // 这里需要你的大模型已经有该位置的上下文
//           next_token = 0;  // 临时占位
//         }
//       }

//       // TODO: 这里需要实现小模型独立的 forward
//       // 暂时用大模型替代（后续会改为真正的小模型）
//       tensor::Tensor pos_tensor = large_model_.get_buffer(model::ModelBufferType::kInputPos);
//       pos_tensor.index<int32_t>(0) = cur_pos;

//       std::vector<int32_t> cur_tokens = {next_token};
//       const auto& token_embedding = large_model_.embedding(cur_tokens);
//       tensor::Tensor input = large_model_.fill_input(pos_tensor, token_embedding, false);

//       int dummy_next = -1;
//       large_model_.forward(input, pos_tensor, dummy_next);

//       cudaError_t err = cudaDeviceSynchronize();
//       if (err != cudaSuccess) {
//         LOG(ERROR) << "Draft forward sync failed";
//         break;
//       }

//       const tensor::Tensor& logits =
//           large_model_.get_buffer(model::ModelBufferType::kForwardOutput);
//       int32_t sampled = greedy_sample(logits);
//       draft_tokens.push_back(sampled);
//       next_token = sampled;
//       cur_pos += 1;
//     }

//     return draft_tokens;
//   }

//   // ===== 大模型逐 token 验收 =====
//   VerificationResult verify_draft_tokens(int32_t start_pos,
//                                          const std::vector<int32_t>& draft_tokens) {
//     VerificationResult result;

//     tensor::Tensor pos_tensor = large_model_.get_buffer(model::ModelBufferType::kInputPos);
//     int32_t cur_pos = start_pos;
//     int32_t next_token = -1;

//     for (size_t i = 0; i < draft_tokens.size(); ++i) {
//       pos_tensor.index<int32_t>(0) = cur_pos;

//       std::vector<int32_t> cur_tokens = {next_token};
//       const auto& token_embedding = large_model_.embedding(cur_tokens);
//       tensor::Tensor input = large_model_.fill_input(pos_tensor, token_embedding, false);

//       int dummy_next = -1;
//       large_model_.forward(input, pos_tensor, dummy_next);

//       cudaError_t err = cudaDeviceSynchronize();
//       if (err != cudaSuccess) {
//         LOG(ERROR) << "Verify forward sync failed at pos " << cur_pos;
//         break;
//       }

//       const tensor::Tensor& logits =
//           large_model_.get_buffer(model::ModelBufferType::kForwardOutput);
//       int32_t large_token = greedy_sample(logits);

//       result.accepted_tokens.push_back(large_token);

//       // 与草稿对比
//       if (large_token != draft_tokens[i]) {
//         result.first_mismatch_pos = static_cast<int32_t>(i);
//         LOG(INFO) << "First mismatch at draft pos " << i << ": draft=" << draft_tokens[i]
//                   << ", large=" << large_token;
//         break;
//       }

//       next_token = large_token;
//       cur_pos += 1;
//     }

//     return result;
//   }
// };

// // ===== Main =====
// int main(int argc, char* argv[]) {
//   // ===== 初始化日志 =====
//   google::InitGoogleLogging(argv[0]);
//   FLAGS_logtostderr = 1;

//   // ===== 配置路径 =====
//   const char* draft_checkpoint = "/home/furina/models/stories110M.bin";
//   const char* large_checkpoint = "/home/furina/models/tinyllama.bin";
//   const char* tokenizer_path = "/home/furina/models/tokenizer.model";

//   LOG(INFO) << "Loading draft model: " << draft_checkpoint;
//   model::LLama2Model draft_model(base::TokenizerType::kEncodeSpe, tokenizer_path,
//   draft_checkpoint,
//                                  false);
//   auto draft_init = draft_model.init(base::DeviceType::kDeviceCUDA);
//   if (!draft_init) {
//     LOG(FATAL) << "Draft model init failed";
//   }

//   LOG(INFO) << "Loading large model: " << large_checkpoint;
//   model::LLama2Model large_model(base::TokenizerType::kEncodeSpe, tokenizer_path,
//   large_checkpoint,
//                                  false);
//   auto large_init = large_model.init(base::DeviceType::kDeviceCUDA);
//   if (!large_init) {
//     LOG(FATAL) << "Large model init failed";
//   }

//   LOG(INFO) << "Both models loaded successfully";

//   // ===== 配置 =====
//   SpeculativeConfig config;
//   config.draft_len = 4;
//   config.max_new_tokens = 100;
//   config.strict_mode = true;

//   // ===== Speculative decode =====
//   SpeculativeDecoder decoder(draft_model, large_model, config);
//   std::string prompt = "做一下自我介绍";

//   auto start = std::chrono::steady_clock::now();
//   VerificationResult result = decoder.decode(prompt, config.max_new_tokens);
//   auto end = std::chrono::steady_clock::now();

//   double duration = std::chrono::duration<double>(end - start).count();

//   // ===== 输出结果 =====
//   LOG(INFO) << "========== Speculative Decode Result ==========";
//   LOG(INFO) << "Generated tokens: " << result.accepted_tokens.size();
//   LOG(INFO) << "Accept count: " << result.accept_count;
//   LOG(INFO) << "Reject count: " << result.reject_count;
//   LOG(INFO) << "First mismatch at: " << result.first_mismatch_pos;
//   LOG(INFO) << "Duration: " << duration << " s";

//   std::string decoded_text = large_model.decode(result.accepted_tokens);
//   LOG(INFO) << "Decoded text: " << decoded_text;

//   return 0;
// }