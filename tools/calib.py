# 终端里可能需要先运行: pip install datasets
from datasets import load_dataset
import random

def generate_calib_data(output_file="calib_1k.txt", num_samples=1000):
    print("正在从 HuggingFace 下载 WikiText-2 数据集...")
    # 下载维基百科的原始文本集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    print("正在清洗和过滤数据...")
    valid_texts = []
    for item in dataset:
        text = item["text"].strip()
        # 过滤掉太短的废话标题和空行，保证校准数据有足够的长度和复杂度
        if len(text) > 80: 
            valid_texts.append(text)
            
    # 随机打乱一下，保证数据的多样性
    random.shuffle(valid_texts)
    
    # 截取前 1000 条
    final_texts = valid_texts[:num_samples]
    
    print(f"正在将 {len(final_texts)} 条校准数据写入 {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in final_texts:
            # 替换掉换行符，保证一行是一条样本
            f.write(line.replace('\n', ' ') + "\n")
            
    print("✅ 校准数据集生成完毕！")

if __name__ == "__main__":
    generate_calib_data()