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
import numpy as np

# 引入 dataset 和 model
from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from utils import save_model, print_monitor_info
from inference import predict_summary
from rouge_score import rouge_scorer

# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

# ========================================================
# [新增模块] 模型权重与梯度健康监测
# ========================================================
def monitor_model_weights(model, epoch):
    """
    监测模型各层的权重分布、梯度情况和异常值。
    重点关注: Decoder (分类头) 是否坍塌，LSTM 梯度是否消失。
    """
    print(f"\n{'='*20} Model Health Monitor (Epoch {epoch}) {'='*20}")
    print(f"{'Layer Name':<40} | {'Mean':<8} | {'Std':<8} | {'Grad Norm':<10} | {'Status'} | {'Slice (First 5)'}")
    print("-" * 110)

    has_nan = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            grad = param.grad
            
            # 1. 基础统计
            mean_val = data.mean().item()
            std_val = data.std().item()
            min_val = data.min().item()
            max_val = data.max().item()
            
            # 2. 梯度统计
            grad_norm = 0.0
            if grad is not None:
                grad_norm = grad.norm().item()
            
            # 3. 状态检查
            status = "OK"
            if torch.isnan(data).any() or torch.isinf(data).any():
                status = "NaN/Inf DETECTED!"
                has_nan = True
            elif std_val < 1e-6:
                status = "COLLAPSED (Zero Var)"
            elif grad is not None and grad_norm < 1e-9:
                status = "Vanishing Grad"
            
            # 4. 切片采样 (取前5个数值)
            # view(-1) 把 tensor 展平，避免维度不同导致的打印问题
            slice_vals = data.view(-1)[:5].cpu().numpy()
            slice_str = str(np.round(slice_vals, 4))

            # 仅打印关键层（为了避免刷屏，过滤掉太细碎的bias，但保留decoder的所有参数）
            # 或者你可以选择打印所有层
            is_important = "decoder" in name or "sent_lstm" in name or "encoder" in name
            
            if is_important:
                print(f"{name:<40} | {mean_val:8.4f} | {std_val:8.4f} | {grad_norm:10.4f} | {status:<15} | {slice_str}")

    print("-" * 110)
    if has_nan:
        print("[CRITICAL WARNING] Model parameters contain NaN or Inf! Training is likely broken.")
        # 可以选择在这里 sys.exit(1)
    print("\n")


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    # 注意：如果使用了 pos_weight，请确保在这里正确传递，或者在外部定义 criterion
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

            # Forward
            logits, attn = model(word_ids, lengths)
            
            # Loss Calculation
            loss = criterion(logits, labels)
            batch_loss += loss
            count += 1

        if count == 0:
            continue
            
        # Average loss over the batch (document batch)
        batch_loss = batch_loss / count
        
        # Backward
        batch_loss.backward()
        
        # [可选] 梯度裁剪，防止梯度爆炸 (Exploding Gradients)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
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
    monitor_limit = 5 # 稍微减少一点打印量

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
    parser.add_argument("--vocab_path", type=str, default=None)

    args = parser.parse_args()

    setup_seed(1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ================= 1. 路径处理与自洽性检查 =================
    train_cache = get_cache_path(args.train_json)
    val_cache = get_cache_path(args.val_json)

    print(f"[*] Dataset Config:")
    print(f"    Train Prefix: {args.train_json} -> Cache: {train_cache}")
    print(f"    Val   Prefix: {args.val_json}   -> Cache: {val_cache}")

    # ================= 2. 加载数据集 =================
    train_dataset = SummDataset(
        args.train_json,
        build_vocab=True,
        cache_path=train_cache
    )

    val_dataset = SummDataset(
        args.val_json,
        vocab=train_dataset.vocab,
        build_vocab=False,
        cache_path=val_cache
    )

    # ================= 3. 词表检查 =================
    vocab_size_train = len(train_dataset.vocab)
    vocab_size_val = len(val_dataset.vocab)
    print(f"[*] Vocab Check: Train={vocab_size_train}, Val={vocab_size_val}")
    
    if vocab_size_train != vocab_size_val:
        print("[CRITICAL ERROR] Vocab size mismatch detected! Please delete val cache.")
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
    # 注意：这里我们使用 model.decoder.parameters() 而不是 model.attention
    optimizer = torch.optim.Adam([
        {"params": model.embedding.parameters(), "lr": 2e-4},
        {"params": model.encoder.parameters(), "lr": 1e-4}, # Encoder 学习率
        {"params": model.decoder.parameters(), "lr": 1e-4}, # Decoder 学习率 (如果发现不收敛，可以尝试改回 1e-3)
    ], weight_decay=1e-4)

    # ================= 5. 训练循环 =================
    print("\nInitial Evaluation:")
    val_scores = eval_epoch(model, val_loader, device, strategy=args.val_strategy, debug=True)
    print(f"Init ROUGE-1: {val_scores['r1_f']:.4f} | ROUGE-L: {val_scores['rl_f']:.4f}")

    best_rl = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 1. 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch} loss: {train_loss:.4f}")

        # 2. [新增] 监测模型健康状态 (在验证之前检查)
        monitor_model_weights(model, epoch)

        # 3. 验证
        val_scores = eval_epoch(model, val_loader, device, strategy=args.val_strategy, debug=True)
        print(f"ROUGE-1: {val_scores['r1_f']:.4f} | ROUGE-L: {val_scores['rl_f']:.4f}")

        if val_scores["rl_f"] > best_rl:
            best_rl = val_scores["rl_f"]
            save_model(model, args.save_path)
            print(f"Model saved (ROUGE-L={best_rl:.4f})")

if __name__ == "__main__":
    main()