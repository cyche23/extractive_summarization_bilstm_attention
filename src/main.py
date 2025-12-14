import json
from rouge_score import rouge_scorer


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



if __name__ == "__main__":
    # 示例数据（可替换为从文件中读取）
    samples = []
    with open("/root/codes/NLP/extractive_summarization_bilstm_attention/data/labeled_stories_mini/test.json", 'r', encoding='utf8') as f:
            samples = json.load(f)

    rouge_scores = evaluate_dataset(samples)

    print("ROUGE scores:")
    for k, v in rouge_scores.items():
        print(f"{k}: P={v['precision']:.4f}, R={v['recall']:.4f}, F1={v['f1']:.4f}")
