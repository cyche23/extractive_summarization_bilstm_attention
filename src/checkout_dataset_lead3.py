import json
from rouge_score import rouge_scorer
import os
import glob
from tqdm import tqdm


def compute_rouge_for_sample(sample, rouge_types=("rouge1", "rougeL")):
    """
    计算一条样本中：
    label=1 的句子 vs highlights 的 ROUGE 分数
    """

    sentences = sample["sentences"]
    labels = sample["labels"]
    highlights = sample["highlights"]

    # 1. 取出 label=1 的句子
    selected_sentences = [
        sent for sent, label in zip(sentences, labels) if label == 1
    ]

    if len(selected_sentences) == 0:
        return None

    # 2. 拼接 candidate 和 reference
    candidate = " ".join(selected_sentences)
    reference = " ".join(highlights)

    # 3. 初始化 ROUGE 计算器
    scorer = rouge_scorer.RougeScorer(
        rouge_types=rouge_types,
        use_stemmer=True
    )

    # scores = scorer.score(candidate, reference)
    scores = scorer.score(reference, candidate) # 顺序：target, prediction


    # 4. 只返回 F1（论文中最常用）
    rouge_result = {
        rouge_type: {
            "precision": scores[rouge_type].precision,
            "recall": scores[rouge_type].recall,
            "f1": scores[rouge_type].fmeasure,
        }
        for rouge_type in rouge_types
    }

    return rouge_result


def evaluate_dataset(samples):
    total = {}
    count = 0

    for sample in samples:
        scores = compute_rouge_for_sample(sample)
        if scores is None:
            continue

        for k, v in scores.items():
            total.setdefault(k, {"p": 0, "r": 0, "f": 0})
            total[k]["p"] += v["precision"]
            total[k]["r"] += v["recall"]
            total[k]["f"] += v["f1"]

        count += 1

    avg = {
        k: {
            "precision": v["p"] / count,
            "recall": v["r"] / count,
            "f1": v["f"] / count
        }
        for k, v in total.items()
    }

    return avg


def evaluate_lead3(samples): 
    """
    计算一条样本中：
    label=1 的句子 vs highlights 的 ROUGE 分数
    """
    total = {}
    count = 0
    for sample in samples:
        sentences = sample["sentences"]
        labels = sample["labels"]
        highlights = sample["highlights"]

        # 1. 取出 前三句
        selected_sentences = [
            sent for sent in sentences[:3]
        ]

        if len(selected_sentences) == 0:
            return None

        # 2. 拼接 candidate 和 reference
        candidate = " ".join(selected_sentences)
        reference = " ".join(highlights)

        # 3. 初始化 ROUGE 计算器
        rouge_types=("rouge1", "rougeL")
        scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types,
            use_stemmer=True
        )

        scores = scorer.score(candidate, reference)
        # scores = scorer.score(reference, candidate) # 顺序：target, prediction

        # 4. 只返回 F1（论文中最常用）
        scores = {
            rouge_type: {
                "precision": scores[rouge_type].precision,
                "recall": scores[rouge_type].recall,
                "f1": scores[rouge_type].fmeasure,
            }
            for rouge_type in rouge_types
        }

        if scores is None:
            continue

        for k, v in scores.items():
            total.setdefault(k, {"p": 0, "r": 0, "f": 0})
            total[k]["p"] += v["precision"]
            total[k]["r"] += v["recall"]
            total[k]["f"] += v["f1"]

        count += 1

    avg = {
        k: {
            "precision": v["p"] / count,
            "recall": v["r"] / count,
            "f1": v["f"] / count
        }
        for k, v in total.items()
    }

    return avg


if __name__ == "__main__":
    # 示例数据（可替换为从文件中读取）
    # samples = []
    # with open("/root/codes/NLP/extractive_summarization_bilstm_attention_origin/data/test_data/train.0.json", 'r', encoding='utf8') as f:
    #         samples = json.load(f)
    path = "/root/codes/NLP/extractive_summarization_bilstm_attention_origin/data/labeled_stories_mini/val"

    target_files = []

    # 1. 显式通配符匹配 (如 "data/train/*.json")
    if '*' in path or '?' in path or '[' in path:
        target_files = sorted(glob.glob(path))
    
    # 2. 具体文件匹配
    elif os.path.isfile(path):
        target_files = [path]
        
    # 3. 前缀匹配
    else:
        # 匹配 preprocess.py 生成的 shard 格式: prefix.*.json
        prefix_pattern = path + ".*.json"
        target_files = sorted(glob.glob(prefix_pattern))
        
        # 备用匹配: prefix*.json
        extra_files = sorted(glob.glob(path + "*.json"))
        
        # 合并并去重
        target_files = sorted(list(set(target_files + extra_files)))

    if not target_files:
        raise FileNotFoundError(
            f"No matching JSON files found for path: '{path}'\n"
            f"Fix: Do not pass a raw directory path. "
            f"Pass a file prefix (e.g. 'data/train') or a glob pattern (e.g. 'data/train*.json')."
        )

    print(f"Loading data from {len(target_files)} file(s) matching '{path}'...")
    
    samples = []
    for file_p in tqdm(target_files, desc="Loading JSON shards"):
        try:
            with open(file_p, 'r', encoding='utf8') as f:
                shard_data = json.load(f)
                samples.extend(shard_data)
        except Exception as e:
            print(f"[Warning] Failed to load {file_p}: {e}")

    print(f"Total samples loaded: {len(samples)}")
    rouge_scores = evaluate_dataset(samples)

    print("ROUGE scores:")
    for k, v in rouge_scores.items():
        print(f"{k}: P={v['precision']:.4f}, R={v['recall']:.4f}, F1={v['f1']:.4f}")

    rouge_scores = evaluate_lead3(samples)

    print("Lead3 Baseline ROUGE scores:")
    for k, v in rouge_scores.items():
        print(f"{k}: P={v['precision']:.4f}, R={v['recall']:.4f}, F1={v['f1']:.4f}")
