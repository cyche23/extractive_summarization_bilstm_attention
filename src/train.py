# src/train.py
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import random

from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from utils import save_model

import numpy as np
from inference import predict_summary
# from evaluate import rouge_1, rouge_l
from rouge_score import rouge_scorer

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
            # move to device
            word_ids = word_ids.to(device)          # [num_sent, max_len]
            lengths = lengths.to(device)
            labels = labels.to(device)

            if word_ids.numel() == 0:
                continue

            logits, attn = model(word_ids, lengths)  # [num_sent], [num_sent]
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
def eval_epoch(model, dataloader, device, strategy="topk"):
    model.eval()

    r1_f, rl_f = [], []

    record_label = []
    record_idx = []
    count=0

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

            # forward
            output = model(word_ids, lengths)
            if len(output) == 3:
                logits, attn, vectors = output
            else:
                logits, attn = output
                vectors = None

            # sent_scores = torch.sigmoid(logits)
            sent_scores = logits

            # generate summary
            pred_sents, selected_indices = predict_summary(
                article_sents=raw_sents_batch[i],
                sent_scores=sent_scores,
                sent_vectors=vectors,
                strategy=strategy
            )

            ref_sents = refs_batch[i]

            # r1 = rouge_1(pred_sents, ref_sents)
            # rl = rouge_l(pred_sents, ref_sents)
            # r1_f.append(r1["f"])
            # rl_f.append(rl["f"])
        
            pred_text = "\n".join(pred_sents)
            ref_text = "\n".join(ref_sents)

            # 计算 ROUGE
            # scores = scorer.score(ref_text, pred_text)  # 注意：rouge_score 要求 (target, prediction)
            # 但注意：有些习惯是 scorer.score(prediction, target)，请核对！
            # 实际上：rouge_scorer 的 scorer.score(target, prediction) 是错误的！
            # 正确顺序是：scorer.score(prediction, target)
            # 参见官方文档：https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py#L85
            scores = scorer.score(pred_text, ref_text)

            r1_f.append(scores['rouge1'].fmeasure)
            rl_f.append(scores['rougeL'].fmeasure)

            if count < 10:
                record_idx.append(selected_indices)
                record_label.append([k for k, val in enumerate(labels_list[i]) if val == 1])
                count += 1
        
    print("Record Index and Score:")
    for i in range(count):
        print(record_idx[i], record_label[i])

    return {
        "r1_f": float(np.mean(r1_f)) if r1_f else 0.0,
        "rl_f": float(np.mean(rl_f)) if rl_f else 0.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--glove_path", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)  # number of articles per batch
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--val_strategy", default="topk") # strategy: 'topk', 'dynamic', 'mmr', 'dynamic_mmr'
    parser.add_argument("--vocab_path", type=str, default=None)

    args = parser.parse_args()

    setup_seed(1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build dataset and vocab
    # dataset = SummDataset(args.train_json, build_vocab=True, save_vocab_path="./vocab/vocab.pkl")
    # vocab = dataset.vocab
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # train dataset (build vocab)
    if args.vocab_path:
        train_dataset = SummDataset(
            args.train_json,
            build_vocab=False,
            load_vocab_path=args.vocab_path
        )
    else:
        train_dataset = SummDataset(
            args.train_json,
            build_vocab=True,
            save_vocab_path="./vocab/vocab.pkl"
        )

    # val dataset (load vocab!)
    val_dataset = SummDataset(
        args.val_json,
        vocab=train_dataset.vocab,
        build_vocab=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,          # 验证建议 batch=1，逻辑最清晰
        shuffle=False,
        collate_fn=collate_fn
    )


    # model
    model = ExtractiveSummarizer(train_dataset.vocab, embed_dim=300, hidden_size=256, glove_path=args.glove_path).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    optimizer = torch.optim.Adam([
        {
            "params": model.embedding.parameters(),
            "lr": 2e-4
        },
        {
            "params": model.encoder.parameters(),
            "lr": 1e-3
        },
        {
            "params": model.attention.parameters(),
            "lr": 1e-3
        },
    ], weight_decay=1e-5)

    val_scores = eval_epoch(
        model, val_loader, device, strategy=args.val_strategy
    )
    print(
        f"ROUGE-1 F1: {val_scores['r1_f']:.4f} | "
        f"ROUGE-L F1: {val_scores['rl_f']:.4f}"
    )

    best_rl = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        print(f"Epoch {epoch} loss: {train_loss:.4f}")

        val_scores = eval_epoch(
            model, val_loader, device, strategy=args.val_strategy
        )
        print(
            f"ROUGE-1 F1: {val_scores['r1_f']:.4f} | "
            f"ROUGE-L F1: {val_scores['rl_f']:.4f}"
        )

        # save best model
        if val_scores["rl_f"] > best_rl:
            best_rl = val_scores["rl_f"]
            save_model(model, args.save_path)
            print(f"Model saved (ROUGE-L={best_rl:.4f})")
    
    # save_model(model, args.save_path)

if __name__ == "__main__":
    main()
