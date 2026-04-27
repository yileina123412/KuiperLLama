# 框架概述
计并构建了一个高效、灵活且可扩展的深度学习模型推理框架，旨在实现提供快速、准确的模型推理服务。通过优化模型加载和推理流程，大幅提升了处理速度和资源利用效率，能有效支撑多种业务需求。
## 流程图

<img src="imgs/流程图.png"  />




# 改进
进行的优化：
- PD分离
- 4bit量化
- 滑动窗口

## PD分离
解耦 Prefill与 Decode阶段。将 Prefill 重构为 Batch SGEMM 压榨 SM 吞吐。
以llama1.1b模型为例，在非量化情况下，prefill阶段token数量为17：

| llama1.1b | prefill | Decode Speed  |
| --------- | ------- | ------------- |
| PD分离      | 47.5ms  | 62.1 Tokens/s |
| 不分离       | 93ms    | 62.3 Tokens/s |

进行PD分离后，prefill阶段的速度提升，从访存瓶颈转化为计算瓶颈


## 4bit量化
使用SmoothQuant 对模型进行4bit量化。并手写了反量化算子，成功加载了Llama2-7B模型。
首字延迟控制在331.2ms 内，生成阶段达到 16.26 Tokens/s 。

### 各模型性能对比
prefill阶段token数量为17：

|                | prefill | Decode Speed   | 模型大小  | 峰值显存占用 |
| -------------- | ------- | -------------- | ----- | ------ |
| llama1.1b-8bit | 48.3ms  | 71.5 Tokens/s  | 1.4GB | 2.180g |
| llama1.1b      | 47.5ms  | 62.1 Tokens/s  | 4.4GB | 4.924  |
| llama7b-4bit   | 331.2ms   | 15.90 Tokens/s | 4.8GB | 7.438  |

分析：
- 由于decode阶段是访存瓶颈，8bit量化后权重数据减少，因此8int量化的decode速度快
- 由于prefill阶段是计算瓶颈，主要的时间都花在了矩阵乘法本身上，而int8中由于多了反量化操作，导致结果差不多

### 显存占用对比
| 显存          | llama 1.1b 8bit | llama 1.1b | llama7b-4bit |
| ----------- | --------------- | ---------- | ------------ |
| 权重          | 1.356g          | 4.101      | 4.57 GB      |
| k cache     | 0.043           | 0.043      | 1 GB         |
| v cache     | 0.043           | 0.043      | 1 GB         |
| prefill中间计算 | 0.133g          | 0.133      | 0.26 GB      |

### 不同量化算子性能对比

| **NCU 指标 (Metric)**  | **FP32 (基线)** | **INT8** | **4-bit** |
| -------------------- | ------------- | -------- | --------- |
| **SM % (算力利用率)**     | 80.50         | 69.38    | 70.27     |
| **Memory % (带宽利用率)** | 80.50         | 69.38     | 70.27     |
| **寄存器数/线程**          | 36            | 36       | 38        |
| 时间                   | 1.48ms         | 1.77ms    | 1.80ms     |

### Nsight Compute (NCU) 性能剖析
为了验证 2D Tiling 与 Shared Memory 复用的物理收益，本引擎使用 Nsight Compute 进行了严格的 Profiling。总体数据显示，优化后的 Prefill SGEMM 算子组（FP32, INT8, W4A16）耗时均实现了较大下降。
<img src="imgs/ncu三个算子.png"  />
通过消除全局显存的冗余访问，算子的物理瓶颈已成功从 Memory Bound（显存带宽受限）推演至 Compute Bound（CUDA Cores 算力受限）。

#### Roofline
fp32矩阵乘法算子
<img src="imgs/roofline.png"  />

通过 NCU 的 Roofline 模型分析，我的算子已经完全越过了 Ridge Point（脊点），处于斜线右侧。这证明我手写的 Shared Memory 2D Tiling 完美消除了全局访存瓶颈（Memory Bound），算子已进入 Compute Bound（算力受限） 状态。

至于未能触碰顶端横线，是因为当前 C++ 内核使用的是标准 CUDA Cores 进行标量 FP32 乘加运算，而 NCU 的横线代表的是 Tensor Cores 的理论峰值。如果要跨越这段垂直距离，必须引入 mma.sync 汇编指令并重构数据对齐格式。基于工程 ROI 考量，这一步我选择交由 vLLM 或 AWQ 官方底层库来实现。

## 滑动窗口
- **痛点：** 传统的标准 KV Cache 随着生成长度的增加，显存占用呈线性爆炸。在受限的 8G 显卡上，长文本生成最终必然指向 OOM（Out of Memory）崩溃。
- **成果：** 通过纯 C++ 手写 **环形缓冲区（Ring Buffer）**，利用指针取模逻辑（`pos % window_size`），将 KV Cache 的物理内存占用强行“锁死”。
- **价值：** 彻底打破了硬件显存对上下文长度的物理限制，赋予了模型在端侧 7×24 小时不宕机、无限轮次对话的工业级鲁棒性。

### 前缀保护

- **痛点：** 简单粗暴的环形覆盖会拔掉注意力机制的“定海神针”，导致模型在游标转满一圈后瞬间智商清零，输出无意义乱码（就像你日志里出现的 `zzgnzezo...`）。
- **成果：** 敏锐洞察到 Attention 机制的数值坍塌风险，在底层精准引入了 **Attention Sink（静态前缀保护）** 机制。通过保留最初的 64 个 Token 不被覆盖，稳住了 Softmax 的概率分布。
- **价值：** 在完全不修改 RoPE（旋转位置编码）逻辑、不牺牲生成连贯性的前提下，在极小显存里实现了高逻辑性的长文本输出。

### 效果
突破了上下文的限制。并且改善了在 Decode 阶段（典型的访存瓶颈 Memory Bound），随着生成文本变长，GPU 搬运 KV 数据的耗时呈线性恶化，导致模型“越跑越慢”。

截断了 Attention 算子的矩阵乘法边界。在实测中，对比 2048 的全量窗口（61.89 steps/s），128 的极限小窗口将 Decode 吞吐率暴力拉升了 **18%**（达到 72.89 steps/s）。


# 基本使用

## 第三方依赖
> 借助企业级开发库，更快地搭建出大模型推理框架
1. google glog https://github.com/google/glog
2. google gtest https://github.com/google/googletest
3. sentencepiece https://github.com/google/sentencepiece
4. armadillo + openblas https://arma.sourceforge.net/download.html
5. Cuda Toolkit


## 模型下载地址
1. LLama2 https://pan.baidu.com/s/1PF5KqvIvNFR8yDIY1HmTYA?pwd=ma8r 或 https://huggingface.co/fushenshen/lession_model/tree/main

2. Tiny LLama 
- TinyLLama模型 https://huggingface.co/karpathy/tinyllamas/tree/main
- TinyLLama分词器 https://huggingface.co/yahma/llama-7b-hf/blob/main/tokenizer.model

3. Qwen2.5/LLama
   
   请参考本项目配套课程，课程参加方式请看本文开头。


## 模型导出
```shell
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
# 使用--hf标签从hugging face中加载模型， 指定--version3可以导出量化模型
# 其他使用方法请看export.py中的命令行参数实例
```


## 编译方法
```shell
  mkdir build 
  cd build
  # 需要安装上述的第三方依赖
  cmake ..
  # 或者开启 USE_CPM 选项，自动下载第三方依赖
  cmake -DUSE_CPM=ON ..
  make -j16
```

## 生成文本的方法
```shell
./llama_infer llama2_7b.bin tokenizer.model

```

# LLama3.2 推理

- 以 meta-llama/Llama-3.2-1B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir meta-llama/Llama-3.2-1B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export.py Llama-3.2-1B.bin --hf=meta-llama/Llama-3.2-1B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/llama_infer Llama-3.2-1B.bin meta-llama/Llama-3.2-1B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/llama3_infer.py
```

# Qwen2.5 推理

- 以 Qwen2.5-0.5B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B --local-dir Qwen/Qwen2.5-0.5B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export_qwen2.py Qwen2.5-0.5B.bin --hf=Qwen/Qwen2.5-0.5B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DQWEN2_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/qwen2_infer.py
```

## Qwen3推理
和上面同理，我们先从huggingface仓库中将模型下载到本地。
1. tools/export_qwen3/load.py中导出为pth，模型的输入`model_name`和输出地址`output_file`依次需要填写；
2. 导出pth格式的模型后，再用同文件夹下的write_bin.py导出qwen.bin；
3. 用CMake选项`QWEN3_SUPPORT`重新编译项目，其他步骤就都是一样的了。