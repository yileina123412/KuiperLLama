import pandas as pd
import matplotlib.pyplot as plt


def plot_csv_data(file_path):
    try:
        # 1. 读取 CSV 文件
        # Pandas 会自动将第一行识别为表头 (step, token_id, times_ms)
        df = pd.read_csv(file_path)

        # 2. 提取数据
        # 第一列为 x (step)，第三列为 y (times_ms)
        x = df["step"]
        y = df["time_ms"]

        # 3. 创建图表
        plt.figure(figsize=(10, 6))  # 设置画布大小

        # 绘制折线图，可以添加标记点 (marker)
        plt.plot(x, y, marker="o", linestyle="-", color="b", label="Latency per Step")

        # 4. 添加修饰
        plt.title("Step vs Time (ms)", fontsize=14)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)  # 添加网格线
        plt.legend()

        # 5. 显示图表
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请检查路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")


def compare_two_csv(file1, file2, label1="File A", label2="File B"):
    try:
        # 1. 读取两个 CSV 文件
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # 2. 提取数据
        # 假设第一列列名是 'step'，第三列是 'times_ms'
        x = df1["step"]
        y1 = df1["time_ms"]
        y2 = df2["time_ms"]

        # 3. 创建画布
        plt.figure(figsize=(12, 6))

        # 4. 绘制两条曲线
        plt.plot(
            x, y1, label=label1, color="blue", linewidth=1.5, marker=".", alpha=0.8
        )
        plt.plot(x, y2, label=label2, color="red", linewidth=1.5, marker=".", alpha=0.8)

        # 5. 图表修饰
        plt.title("Performance Comparison: Times (ms) per Step", fontsize=14)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Times (ms)", fontsize=12)

        # 添加网格，方便对比数值差
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        # 显示图例，区分哪条线是哪个文件
        plt.legend()

        # 6. 显示结果
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"处理文件时出错: {e}")


def compare_multiple_csv(file_list, labels=None):
    # 如果没提供标签，就用文件名代替
    if labels is None:
        labels = file_list

    # 设置一组对比明显的颜色
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 蓝、橙、绿、红

    plt.figure(figsize=(12, 7))

    for i, file in enumerate(file_list):
        try:
            # 读取数据
            df = pd.read_csv(file)

            # 提取数据：第一列做X，第三列做Y
            # .iloc[:, 0] 表示取第一列，.iloc[:, 2] 表示取第三列
            x = df.iloc[:, 0]
            y = df.iloc[:, 2]

            # 绘图
            plt.plot(
                x,
                y,
                label=labels[i],
                color=colors[i % len(colors)],
                linewidth=1.5,
                alpha=0.8,
            )

        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    # 图表修饰
    plt.title("Comparison of 4 Files: Step vs Times (ms)", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Times (ms)", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")  # 显示图例

    plt.tight_layout()
    plt.show()


# 调用函数
if __name__ == "__main__":
    my_files = [
        "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_baseline_w2048_p0.csv",
        "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_window_w256_p0.csv",
        "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_prefix_w128_p32.csv",
        "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_prefix_w256_p64.csv",
    ]

    # 2. 为每个文件设置一个易读的标签
    my_labels = [
        "Config A (w2048_p0)",
        "Config B (w256_p0)",
        "Config C (w128_p32)",
        "Config D (w256_p64)",
    ]

    compare_multiple_csv(my_files, my_labels)
    # compare_two_csv(
    #     "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_window_w256_p0.csv",
    #     "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_baseline_w2048_p0.csv",
    #     label1="w256",
    #     label2="w2048",
    # )
    # plot_csv_data(
    #     "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_window_w256_p0.csv"
    # )
    # plot_csv_data(
    #     "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_baseline_w2048_p0.csv"
    # )
    # plot_csv_data(
    #     "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_prefix_w128_p32.csv"
    # )
    # plot_csv_data(
    #     "/home/furina/code_learnning/cpp/cuda/KuiperLLama/build/generation_steps_prefix_w256_p64.csv"
    # )
