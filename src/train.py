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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 引入 dataset 和 model
from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from utils import save_model, print_monitor_info, monitor_model_weights, debug_lead3_data
from inference import predict_summary
from rouge_score import rouge_scorer

# 设置绘图后端，防止在无显示器的服务器上报错
plt.switch_backend('agg')

# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

scorer = rouge_scorer.RougeScorer(['rouge1','rougeLsum'], use_stemmer=True)

# ========================================================
# [新增模块] 早停机制 (Early Stopping)
# ========================================================
class EarlyStopping:
    """
    当验证集指标在连续 patience 个 epoch 内没有提升时，强制停止训练。
    """
    def __init__(self, patience=5, delta=0.0005):
        """
        Args:
            patience (int): 容忍多少个 epoch 指标不提升
            delta (float): 只有提升超过 delta 才算提升
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"[EarlyStopping] Score didn't improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ========================================================
# [新增模块] 绘图功能 (Plotting)
# ========================================================
def plot_training_curves(history, lead3_score, output_dir):
    """
    绘制 Loss 和 ROUGE 曲线，包含 Lead-3 基准线。
    """
    epochs = range(1, len(history['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- 子图 1: Training Loss ---
    ax1.plot(epochs, history['loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    # 强制 x 轴显示整数刻度
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend()

    # --- 子图 2: Validation ROUGE ---
    ax2.plot(epochs, history['r1'], 'g-s', label='Val ROUGE-1', linewidth=2)
    ax2.plot(epochs, history['rlsum'], 'm-^', label='Val ROUGE-Lsum', linewidth=2)
    
    # 绘制 Lead-3 基准线
    if lead3_score:
        ax2.axhline(y=lead3_score['r1'], color='g', linestyle='--', alpha=0.5, label=f"Lead-3 R1 ({lead3_score['r1']:.3f})")
        ax2.axhline(y=lead3_score['rlsum'], color='m', linestyle='--', alpha=0.5, label=f"Lead-3 RLsum ({lead3_score['rlsum']:.3f})")

    ax2.set_title('Validation ROUGE Scores', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()

    # 防止布局重叠
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[Plot] Training curves saved to: {save_path}")
    plt.close()

# ========================================================
# [新增模块] Lead-3 基准计算
# ========================================================
def calc_lead3_baseline(dataloader, device):
    """
    计算验证集的 Lead-3 (前三句作为摘要) 的基准分数。
    这为评估模型是否真的学到了东西提供重要参考。
    """
    print("Calculating Lead-3 Baseline on Validation Set...")
    r1_scores = []
    rl_scores = []
    rlsum_scores = []
    
    for batch in tqdm(dataloader, desc="Lead-3 Baseline"):
        raw_sents_batch = batch["raw_sents"]
        refs_batch = batch["highlights"]
        
        for i in range(len(raw_sents_batch)):
            # 简单粗暴：直接选前3句
            pred_sents = raw_sents_batch[i][:3]
            pred_text = "\n".join(pred_sents)
            ref_text = "\n".join(refs_batch[i])
            
            scores = scorer.score(pred_text, ref_text)
            r1_scores.append(scores['rouge1'].fmeasure)
            rlsum_scores.append(scores['rougeLsum'].fmeasure)
            
    lead3_r1 = float(np.mean(r1_scores))
    lead3_rl = float(np.mean(rl_scores))
    lead3_rlsum = float(np.mean(rlsum_scores))
    print(f"[*] Lead-3 Baseline -> ROUGE-1: {lead3_r1:.4f} | ROUGE-L: {lead3_rl:.4f} | ROUGE-Lsum: {lead3_rlsum:.4f}")
    return {'r1': lead3_r1, 'rlsum': lead3_rlsum}

def label_smoothing_loss(logits, targets, smoothing=0.1):
    """
    手动实现二分类的 Label Smoothing
    targets: [0, 1, 1, 0] -> [0.05, 0.95, 0.95, 0.05] (假设 smoothing=0.1)
    """
    # 将 0/1 标签软化
    # new_target = target * (1 - epsilon) + 0.5 * epsilon
    with torch.no_grad():
        soft_targets = targets * (1.0 - smoothing) + 0.5 * smoothing
    
    # 使用 BCEWithLogitsLoss 计算平滑后的 Loss
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(logits, soft_targets)

# ========================================================
# 原有训练逻辑 (保持不变)
# ========================================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss() 
    
    for batch in tqdm(dataloader, desc="Train"):
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
            logits, _ = model(word_ids, lengths)
            
            # Loss Calculation
            # loss = criterion(logits, labels)
            loss = label_smoothing_loss(logits, labels, smoothing=0.1)
            batch_loss += loss
            count += 1

        if count == 0:
            continue
            
        # Average loss over the batch (document batch)
        batch_loss = batch_loss / count
        
        # Backward
        batch_loss.backward()
        
        # 梯度裁剪 (保持原有逻辑，此处注释掉若之前没加)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()

        total_loss += batch_loss.item()
    return total_loss

@torch.no_grad()
def eval_epoch(model, dataloader, device, strategy="topk", debug=True):
    model.eval()
    r1_f, rlsum_f = [], []
    
    count = 0
    monitor_limit = 3

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
            rlsum_f.append(scores['rougeLsum'].fmeasure)

            count += 1
        
    if debug:
        # monitor_model_weights(model) # 按需开启，避免刷屏
        pass

    return {
        "r1_f": float(np.mean(r1_f)) if r1_f else 0.0,
        "rlsum_f": float(np.mean(rlsum_f)) if rlsum_f else 0.0
    }

def get_cache_path(data_prefix):
    return data_prefix + ".pkl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True, help="Data prefix (e.g., 'data/train')")
    parser.add_argument("--val_json", required=True, help="Data prefix (e.g., 'data/val')")
    parser.add_argument("--glove_path", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=15) # 默认加到15，配合早停
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="checkpoints/tmp/model.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val_strategy", default="topk") 
    parser.add_argument("--vocab_path", type=str, default=None)
    # 早停参数
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()

    setup_seed(1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 路径检查
    train_cache = get_cache_path(args.train_json)
    val_cache = get_cache_path(args.val_json)
    
    # 确保保存模型的目录存在，用于存放 plots
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # ================= 2. 加载数据集 =================
    train_dataset = SummDataset(args.train_json, build_vocab=True, cache_path=train_cache)
    val_dataset = SummDataset(args.val_json, vocab=train_dataset.vocab, build_vocab=False, cache_path=val_cache)

    vocab_size_train = len(train_dataset.vocab)
    vocab_size_val = len(val_dataset.vocab)
    print(f"[*] Vocab Check: Train={vocab_size_train}, Val={vocab_size_val}")
    
    if vocab_size_train != vocab_size_val:
        print("[CRITICAL ERROR] Vocab size mismatch detected! Please delete val cache.")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ================= [新增] 计算 Lead-3 基准 =================
    lead3_scores = calc_lead3_baseline(val_loader, device)
    # debug_lead3_data(val_loader)

    # ================= 4. 模型与优化器 =================
    model = ExtractiveSummarizer(
        train_dataset.vocab, 
        embed_dim=300, 
        hidden_size=256, 
        glove_path=args.glove_path
    ).to(device)

    # 学习率配置 (保持你最后一次成功的配置)
    optimizer = torch.optim.Adam([
        # {"params": model.embedding.parameters(), "lr": 1e-4},
        {"params": model.encoder.parameters(), "lr": 8e-4},
        {"params": model.decoder.parameters(), "lr": 5e-5},
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-7
    )
    
    # [新增] 初始化早停对象
    early_stopping = EarlyStopping(patience=args.patience, delta=0)

    # ================= 5. 训练循环 =================
    print("\nInitial Evaluation:")
    val_scores = eval_epoch(model, val_loader, device, strategy=args.val_strategy, debug=True)
    print(f"Init ROUGE-1: {val_scores['r1_f']:.4f} | ROUGE-Lsum: {val_scores['rlsum_f']:.4f}")

    best_rlsum = 0.0
    
    # [新增] 用于记录绘图数据
    history = {'loss': [], 'r1': [], 'rlsum': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 1. 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        # 归一化 Loss 便于观察 (可选，取决于 dataloader 长度)
        avg_train_loss = train_loss / len(train_loader) 
        print(f"Epoch {epoch} Avg Loss: {avg_train_loss:.4f} (Total: {train_loss:.2f})")

        # 2. 验证
        val_scores = eval_epoch(model, val_loader, device, strategy=args.val_strategy, debug=True)
        current_rouge_1 = val_scores['r1_f']
        current_rouge_lsum = val_scores['rlsum_f']
        print(f"ROUGE-1: {current_rouge_1:.4f} | ROUGE-Lsum: {current_rouge_lsum:.4f}")

        # 3. 记录历史
        history['loss'].append(avg_train_loss)
        history['r1'].append(current_rouge_1)
        history['rlsum'].append(current_rouge_lsum)

        # 4. 调度器更新
        scheduler.step(current_rouge_lsum)

        # 5. 保存最佳模型
        if current_rouge_lsum > best_rlsum:
            best_rlsum = current_rouge_lsum
            save_model(model, args.save_path)
            print(f"Model saved (ROUGE-Lsum={best_rlsum:.4f})")
        
        # 6. [新增] 实时绘图 (每个 epoch 都刷新图片，方便实时监控)
        plot_training_curves(history, lead3_scores, os.path.dirname(args.save_path))

        # 7. [新增] 早停检查
        # early_stopping(current_rouge_l)
        early_stopping(current_rouge_lsum)
        if early_stopping.early_stop:
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}! Best ROUGE-L was {early_stopping.best_score:.4f}")
            break

    print(f"\nTraining Finished. Best ROUGE-L: {best_rlsum:.4f}")
    # 最后再画一次确保完整
    plot_training_curves(history, lead3_scores, os.path.dirname(args.save_path))

if __name__ == "__main__":
    main()