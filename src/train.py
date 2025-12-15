# src/train.py
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import random
import sys

# 引入 dataset 和 model
from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from utils import save_model, print_monitor_info
from inference import predict_summary
from rouge_score import rouge_scorer
import numpy as np

# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for batch in tqdm(dataloader, desc="Train"):
        ids = batch["ids"]
        word_ids_list = batch["word_ids"]
        lengths_list = batch["lengths"]
        labels_list = batch["labels"]

        optimizer.zero_grad()
        batch_loss = 0.0
        count = 0
        for word_ids, lengths, labels in zip(word_ids_list, lengths_list, labels_list):
            word_ids = word_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            if word_ids.numel() == 0:
                continue

            logits, attn = model(word_ids, lengths)
            loss = criterion(logits, labels)
            batch_loss += loss
            count += 1

        if count == 0:
            continue
        batch_loss = batch_loss / count
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
    return total_loss

@torch.no_grad()
def eval_epoch(model, dataloader, device, strategy="topk", debug=True):
    model.eval()
    r1_f, rl_f = [], []
    
    # 监测数据容器
    record_label = []
    record_idx = []
    logits_list = []
    count = 0
    monitor_limit = 10 

    for batch in tqdm(dataloader, desc="Val"):
        ids_list = batch["word_ids"]
        lens_list = batch["lengths"]
        raw_sents_batch = batch["raw_sents"]
        refs_batch = batch["highlights"]
        labels_list = batch["labels"]

        for i in range(len(ids_list)):
            word_ids = ids_list[i].to(device)
            lengths = lens_list[i].to(device)

            if word_ids.numel() == 0:
                continue

            output = model(word_ids, lengths)
            if len(output) == 3:
                logits, _, vectors = output
            else:
                logits, _ = output
                vectors = None

            sent_scores = logits

            pred_sents, selected_indices = predict_summary(
                article_sents=raw_sents_batch[i],
                sent_scores=sent_scores,
                sent_vectors=vectors,
                strategy=strategy
            )

            ref_sents = refs_batch[i]
            pred_text = "\n".join(pred_sents)
            ref_text = "\n".join(ref_sents)

            scores = scorer.score(pred_text, ref_text)
            r1_f.append(scores['rouge1'].fmeasure)
            rl_f.append(scores['rougeL'].fmeasure)

            if debug and count < monitor_limit:
                record_idx.append(selected_indices)
                record_label.append([k for k, val in enumerate(labels_list[i]) if val == 1])
                logits_list.append(sent_scores)
                count += 1
        
    if debug:
        print_monitor_info(record_idx, record_label, logits_list, count)

    return {
        "r1_f": float(np.mean(r1_f)) if r1_f else 0.0,
        "rl_f": float(np.mean(rl_f)) if rl_f else 0.0
    }

def get_cache_path(data_prefix):
    """
    根据数据前缀生成缓存路径。
    例如: prefix="data/train" -> cache="data/train.pkl"
    """
    # 简单的直接拼接，这对于 prefix 形式是安全的
    # 它会在 data 目录下生成 train.pkl
    return data_prefix + ".pkl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True, help="Data prefix (e.g., 'data/train')")
    parser.add_argument("--val_json", required=True, help="Data prefix (e.g., 'data/val')")
    parser.add_argument("--glove_path", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val_strategy", default="topk") 
    
    # 为了兼容旧脚本保留参数，但不使用
    parser.add_argument("--vocab_path", type=str, default=None)

    args = parser.parse_args()

    setup_seed(1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ================= 1. 路径处理与自洽性检查 =================
    
    # 自动生成缓存路径
    train_cache = get_cache_path(args.train_json)
    val_cache = get_cache_path(args.val_json)

    print(f"[*] Dataset Config:")
    print(f"    Train Prefix: {args.train_json} -> Cache: {train_cache}")
    print(f"    Val   Prefix: {args.val_json}   -> Cache: {val_cache}")

    # ================= 2. 加载数据集 =================

    # 加载训练集
    # 如果 train.pkl 存在，直接加载；否则读取 train.*.json 并构建词表
    train_dataset = SummDataset(
        args.train_json,
        build_vocab=True,
        cache_path=train_cache
    )

    # 加载验证集
    # 关键逻辑：这里传入 train_dataset.vocab
    # dataset.py 内部逻辑：
    #   - 如果 val_cache 存在：直接加载 val_cache (忽略传入的 vocab) -> 潜在风险点
    #   - 如果 val_cache 不存在：使用传入的 vocab 处理数据 -> 安全
    val_dataset = SummDataset(
        args.val_json,
        vocab=train_dataset.vocab,
        build_vocab=False,
        cache_path=val_cache
    )

    # ================= 3. 关键 BUG 防御：词表一致性检查 =================
    # 如果 Train 重新生成了(新词表)，但 Val 读取了旧 Cache(旧词表)，这里会检测出来
    
    vocab_size_train = len(train_dataset.vocab)
    vocab_size_val = len(val_dataset.vocab)
    
    print(f"[*] Vocab Check: Train={vocab_size_train}, Val={vocab_size_val}")
    
    # 检查1: 词表大小必须一致
    if vocab_size_train != vocab_size_val:
        print("\n" + "!"*50)
        print("[CRITICAL ERROR] Vocab size mismatch detected!")
        print(f"Train vocab: {vocab_size_train}, Val vocab (from cache): {vocab_size_val}")
        print("Reason: You likely rebuilt the Train Cache but kept an old Val Cache.")
        print(f"Fix: Please delete '{val_cache}' and restart.")
        print("!"*50 + "\n")
        sys.exit(1) # 强制退出，防止训练出无意义的模型

    # 检查2: 随机抽查 Token ID 一致性 (更深层的检查)
    # 检查 <unk> 和 <pad> 是否一致，或者随机词
    test_token = "<unk>"
    if train_dataset.vocab.get(test_token) != val_dataset.vocab.get(test_token):
        print(f"[Error] Token '{test_token}' ID mismatch between train and val datasets.")
        sys.exit(1)

    # ================= 4. 构建 Loader 与模型 =================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,
        collate_fn=collate_fn
    )

    model = ExtractiveSummarizer(
        train_dataset.vocab, 
        embed_dim=300, 
        hidden_size=256, 
        glove_path=args.glove_path
    ).to(device)

    # 优化器配置
    optimizer = torch.optim.Adam([
        {"params": model.embedding.parameters(), "lr": 2e-4},
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.decoder.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    # ================= 5. 训练循环 =================
    
    # 初始评估
    print("\nInitial Evaluation:")
    val_scores = eval_epoch(model, val_loader, device, strategy=args.val_strategy, debug=True)
    print(f"Init ROUGE-1: {val_scores['r1_f']:.4f} | ROUGE-L: {val_scores['rl_f']:.4f}")

    best_rl = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch} loss: {train_loss:.4f}")

        # 验证时开启 debug=True 以便观察 monitor info
        val_scores = eval_epoch(model, val_loader, device, strategy=args.val_strategy, debug=True)
        print(f"ROUGE-1: {val_scores['r1_f']:.4f} | ROUGE-L: {val_scores['rl_f']:.4f}")

        if val_scores["rl_f"] > best_rl:
            best_rl = val_scores["rl_f"]
            save_model(model, args.save_path)
            print(f"Model saved (ROUGE-L={best_rl:.4f})")

if __name__ == "__main__":
    main()
