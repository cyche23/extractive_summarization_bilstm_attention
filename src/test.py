# src/test.py
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import pickle
import sys
from rouge_score import rouge_scorer 

# 引入项目模块
from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from inference import predict_summary
from utils import load_model
from train import get_cache_path

# ======================
# 固定随机种子
# ======================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ======================
# 辅助函数: 从训练缓存加载词表
# ======================
def load_vocab_from_cache(cache_path):
    """
    从训练生成的 pickle 缓存中提取 vocab 对象。
    这是保证训练和测试 token ID 一致性的唯一正确方法。
    """
    print(f"[*] Loading vocab from training cache: {cache_path}")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Training cache not found at {cache_path}. Please run train.py first to generate it.")
    
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
        
    if 'vocab' not in cache_data:
        raise ValueError("The provided cache file does not contain a 'vocab' key.")
        
    vocab = cache_data['vocab']
    print(f"[*] Successfully loaded vocab with size: {len(vocab)}")
    return vocab

# ======================
# 测试 / 推理逻辑
# ======================
@torch.no_grad()
def test_epoch(model, dataloader, device, strategy="topk"):
    model.eval()

    # 存储所有样本的分数 (F1, Precision, Recall)
    results = {
        "r1_f": [], "r1_p": [], "r1_r": [],
        "rl_f": [], "rl_p": [], "rl_r": []
    }

    # ✅ [关键修改] 使用 rougeLsum 且开启 Stemmer (学术界标准)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeLsum'], use_stemmer=True)

    print(f"[*] Starting inference using strategy: {strategy}")

    for batch in tqdm(dataloader, desc="Testing"):
        # 1. 解包数据
        word_ids_list = batch["word_ids"]
        lengths_list = batch["lengths"]
        raw_sents_list = batch["raw_sents"]   # 原始句子文本
        refs_list = batch["highlights"]       # 参考摘要文本
        
        # 2. 逐样本处理 (Batch Inference)
        for i in range(len(word_ids_list)):
            word_ids = word_ids_list[i].to(device)
            lengths = lengths_list[i].to(device)

            if word_ids.numel() == 0:
                continue

            # Forward
            output = model(word_ids, lengths)
            if len(output) == 3:
                logits, _, vectors = output
            else:
                logits, _ = output
                vectors = None

            sent_scores = logits

            # 3. 生成摘要 (Inference)
            pred_sents, selected_indices = predict_summary(
                article_sents=raw_sents_list[i],
                sent_scores=sent_scores,
                sent_vectors=vectors,
                strategy=strategy
            )

            # 4. 计算 ROUGE
            # 将句子列表拼接成文本块 (换行符分隔)
            pred_text = "\n".join(pred_sents)
            ref_text = "\n".join(refs_list[i])

            scores = scorer.score(pred_text, ref_text)
            
            # ✅ [关键修改] 记录 Precision, Recall 和 F1
            results["r1_f"].append(scores['rouge1'].fmeasure)
            results["r1_p"].append(scores['rouge1'].precision)
            results["r1_r"].append(scores['rouge1'].recall)
            
            results["rl_f"].append(scores['rougeLsum'].fmeasure)
            results["rl_p"].append(scores['rougeLsum'].precision)
            results["rl_r"].append(scores['rougeLsum'].recall)

    # 计算平均分
    final_scores = {}
    for k, v in results.items():
        final_scores[k] = np.mean(v) if v else 0.0

    return final_scores

# ======================
# 主函数
# ======================
def main():
    parser = argparse.ArgumentParser()
    
    # 路径参数
    parser.add_argument("--test_json", required=True, help="Path or prefix for test data (e.g., 'data/test')")
    parser.add_argument("--vocab_source", required=True, help="Path to the TRAIN cache file containing vocab (e.g., 'data/train.pkl')")
    parser.add_argument("--model_path", required=True, help="Path to the trained .pt model file")
    parser.add_argument("--glove_path", default=None, help="Optional: Path to GloVe vectors (only needed if lazy loading, usually not needed for inference)")

    # 配置参数
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing (default 1 is safest for eval)")
    parser.add_argument("--strategy", default="topk", choices=["topk", "dynamic"], help="Summary selection strategy")
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    setup_seed(42)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ================= 1. 加载词表 =================
    vocab = load_vocab_from_cache(args.vocab_source)

    # ================= 2. 加载测试集 =================
    test_dataset = SummDataset(
        args.test_json,
        vocab=vocab,
        build_vocab=False,
        cache_path=get_cache_path(args.test_json)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"[*] Test Dataset Loaded. Size: {len(test_dataset)}")

    # ================= 3. 构建模型结构 =================
    model = ExtractiveSummarizer(
        vocab=vocab,
        embed_dim=300,
        hidden_size=256,
        glove_path=args.glove_path 
    ).to(device)

    # ================= 4. 加载模型权重 =================
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    load_model(model, args.model_path)
    print(f"[*] Model weights loaded from {args.model_path}")

    # ================= 5. 开始测试 =================
    print("\n" + "="*30)
    print("Running Evaluation on Test Set...")
    print("="*30)
    
    scores = test_epoch(model, test_loader, device, strategy=args.strategy)

    print("\n" + "="*30)
    print("FINAL TEST RESULTS (Precision / Recall / F1)")
    print("="*30)
    print(f"ROUGE-1    : P={scores['r1_p']:.4f} | R={scores['r1_r']:.4f} | F1={scores['r1_f']:.4f}")
    print(f"ROUGE-Lsum : P={scores['rl_p']:.4f} | R={scores['rl_r']:.4f} | F1={scores['rl_f']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()