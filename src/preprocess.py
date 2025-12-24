# src/preprocess.py
import os
import json
import glob
import nltk
import re
import argparse
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer

# 确保下载了 punkt 分词器
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 全局 ROUGE 打分器 (ROUGE-1, 2, L)
_ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# =========================================
# 核心处理逻辑 (保持完全不变)
# =========================================

# --------------------------
# 文本清洗
# --------------------------
def clean_text(text):
    patterns = [
        r"\(CNN\)", r"\[CNN\]", r"cnn", r"CNN",
        r"\(Reuters\)", r"\[Reuters\]",
        r"—", r"--", r"---"
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------
# 缩写保护：避免误切分
# --------------------------
def protect_abbreviations(text):
    abbr = {
        r"U\.S\.A\.": "USA_PROTECTED",
        r"U\.S\.": "US_PROTECTED",
        r"Mr\.": "Mr_PROTECTED",
        r"Mrs\.": "Mrs_PROTECTED",
        r"Dr\.": "Dr_PROTECTED",
        r"Ms\.": "Ms_PROTECTED",
    }
    for a, token in abbr.items():
        text = re.sub(a, token, text)
    return text


def restore_abbreviations(sentences):
    restore = {
        "USA_PROTECTED": "U.S.A.",
        "US_PROTECTED": "U.S.",
        "Mr_PROTECTED": "Mr.",
        "Mrs_PROTECTED": "Mrs.",
        "Dr_PROTECTED": "Dr.",
        "Ms_PROTECTED": "Ms.",
    }
    restored = []
    for s in sentences:
        for token, val in restore.items():
            s = s.replace(token, val)
        restored.append(s)
    return restored


# --------------------------
# 句子切分
# --------------------------
def split_into_sentences(text):
    text = protect_abbreviations(text)
    raw = nltk.sent_tokenize(text)
    raw = restore_abbreviations(raw)
    return [s.strip() for s in raw if len(s.strip()) > 1]


# --------------------------
# 解析 .story 文件
# --------------------------
def parse_story(file_path):
    with open(file_path, "r", encoding="utf8", errors='ignore') as f:
        lines = f.readlines()

    article_lines = []
    highlights = []
    is_highlight = False

    for line in lines:
        line = line.strip()
        if line == "@highlight":
            is_highlight = True
            continue
        if is_highlight:
            if len(line) > 0: highlights.append(line)
        else:
            if len(line) > 0: article_lines.append(line)

    article = clean_text(" ".join(article_lines))
    highlights = [clean_text(h) for h in highlights]
    return article, highlights


# --------------------------
# [核心修改] 贪婪 ROUGE 标签对齐
# --------------------------
def align_labels(sentences, highlights, max_oracle_sents=3):
    """
    使用 Greedy ROUGE 策略生成 Oracle Labels。
    原理：每次选择能让当前 ROUGE 分数提升最大的一句话，直到达到数量限制或无法提升。
    这是抽取式摘要的"金标准"做法。
    """
    if not sentences or not highlights:
        return [0] * len(sentences)

    # 将摘要列表拼接成参考文本
    abstract_text = " ".join(highlights)

    selected_indices = []
    current_summary_list = []
    best_score = 0.0

    # 循环选择句子
    while len(selected_indices) < max_oracle_sents:
        best_gain = 0.0
        best_idx = -1

        for i, sent in enumerate(sentences):
            if i in selected_indices:
                continue

            # 尝试加入这句话
            trial_summary = " ".join(current_summary_list + [sent])

            # 计算 ROUGE 分数
            scores = _ROUGE_SCORER.score(abstract_text, trial_summary)
            # 综合指标：通常使用 ROUGE-1 + ROUGE-2 的平均或总和
            current_score = scores['rouge1'].fmeasure + scores['rouge2'].fmeasure

            # 看看是否有提升
            if current_score > best_score:
                gain = current_score - best_score
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

        # 如果找到了能提升分数的句子
        if best_idx != -1:
            best_score += best_gain
            selected_indices.append(best_idx)
            current_summary_list.append(sentences[best_idx])
        else:
            # 如果遍历一圈都无法提升分数，提前结束
            break

    # 生成 0/1 标签
    labels = [0] * len(sentences)
    for idx in selected_indices:
        labels[idx] = 1

    return labels


# =========================================
# 流程控制 (修改为分批处理)
# =========================================

def save_shard(data, output_dir, prefix, shard_idx):
    """保存单个分片文件"""
    filename = f"{prefix}.{shard_idx}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # print(f"Saved shard: {path} ({len(data)} samples)") # 可选：减少打印刷屏


def process_file_list(file_paths, output_dir, split_name, shard_size):
    """
    处理给定的文件路径列表，并分批保存。
    """
    dataset_buffer = []
    shard_idx = 0
    total_processed = 0

    desc = f"Processing {split_name}"
    for fp in tqdm(file_paths, desc=desc):
        try:
            article, highlights = parse_story(fp)

            if len(article.strip()) == 0 or len(highlights) == 0:
                continue

            sentences = split_into_sentences(article)
            if len(sentences) == 0:
                continue

            # 生成高质量标签
            labels = align_labels(sentences, highlights)

            # 如果全是0，或者全是1，通常是异常数据，可以考虑过滤（这里保留以防万一）
            if sum(labels) == 0:
                # 极其罕见的情况，可以做一个保底：选前三句
                labels[:min(3, len(labels))] = [1] * min(3, len(labels))

            # 添加到缓存
            dataset_buffer.append({
                "id": os.path.basename(fp),
                "sentences": sentences,
                "labels": labels,
                "highlights": highlights
            })
            total_processed += 1

            # 检查缓存是否达到分片大小
            if len(dataset_buffer) >= shard_size:
                save_shard(dataset_buffer, output_dir, split_name, shard_idx)
                dataset_buffer = []  # 清空缓存
                shard_idx += 1

        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue

    # 循环结束后，保存剩余的数据
    if len(dataset_buffer) > 0:
        save_shard(dataset_buffer, output_dir, split_name, shard_idx)
    
    print(f"Finished {split_name}: {total_processed} usable samples saved.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CNN/DailyMail dataset with sharding.")
    parser.add_argument("--raw_path", type=str, 
                        default="/root/codes/NLP/extractive_summarization_bilstm_attention/data/raw_stories",
                        help="Path to the directory containing .story files")
    parser.add_argument("--output_dir", type=str, 
                        default="/root/codes/NLP/extractive_summarization_bilstm_attention_origin/data/labeled_stories",
                        help="Directory to save processed JSON shards")
    parser.add_argument("--shard_size", type=int, default=5000, 
                        help="Number of samples per JSON file")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for splitting")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 扩展用户路径
    raw_path_expanded = os.path.expanduser(args.raw_path)
    print(f"Scanning for .story files in: {raw_path_expanded}")

    # 1. 获取所有文件路径 (只扫描，不读取内容)
    # 支持 raw_path 下直接是 .story，也支持 raw_path 下有 cnn/stories 和 dailymail/stories 子目录的情况
    # 这里的逻辑稍微增强一点，直接递归搜索或者搜索特定模式
    all_files = []
    # 尝试直接搜索
    all_files.extend(glob.glob(os.path.join(raw_path_expanded, "*.story")))
    # 尝试搜索子目录 (适配 CNN/DailyMail 常见结构)
    all_files.extend(glob.glob(os.path.join(raw_path_expanded, "*", "stories", "*.story")))
    
    # 去重
    all_files = sorted(list(set(all_files)))
    
    print(f"Total files found: {len(all_files)}")

    if len(all_files) == 0:
        print("No .story files found! Please check raw_path.")
        return

    # 2. 划分数据集 (在文件路径层面划分，而不是数据层面)
    # 这里的逻辑改为先 split path，再 process
    print("Splitting file list into Train/Val/Test...")
    train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=args.seed)
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=args.seed) # 0.9 * 0.1 = 0.09

    print(f"Split Plan -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # 3. 分别处理并分片保存
    # 处理 Train
    process_file_list(train_files, args.output_dir, "train", args.shard_size)
    
    # 处理 Val
    process_file_list(val_files, args.output_dir, "val", args.shard_size)
    
    # 处理 Test
    process_file_list(test_files, args.output_dir, "test", args.shard_size)

    print(f"\nPreprocessing completed! Data saved to {args.output_dir}")
    print(f"Shard size: {args.shard_size}")


if __name__ == "__main__":
    main()