# src/test.py
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from inference import predict_summary
from evaluate import rouge_1, rouge_l
from utils import load_model


# ======================
# 固定随机种子（可复现）
# ======================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ======================
# 测试 / 推理函数
# ======================
@torch.no_grad()
def test_epoch(model, dataloader, device, strategy="topk"):
    model.eval()

    r1_f, rl_f = [], []

    for batch in tqdm(dataloader, desc="Test"):
        ids_list = batch["word_ids"]
        lens_list = batch["lengths"]
        raw_sents_batch = batch["raw_sents"]
        refs_batch = batch["highlights"]

        for i in range(len(ids_list)):
            word_ids = ids_list[i].to(device)
            lengths = lens_list[i].to(device)

            if word_ids.numel() == 0:
                continue

            # forward
            output = model(word_ids, lengths)
            if isinstance(output, tuple):
                logits, vectors = output
            else:
                logits = output
                vectors = None

            sent_scores = torch.sigmoid(logits)

            # 生成摘要
            pred_sents = predict_summary(
                article_sents=raw_sents_batch[i],
                sent_scores=sent_scores,
                sent_vectors=vectors,
                strategy=strategy
            )

            ref_sents = refs_batch[i]

            # ROUGE
            print("Predicted Scores:")
            for sent in sent_scores:
                print(f"  {sent}")
            r1 = rouge_1(pred_sents, ref_sents)
            rl = rouge_l(pred_sents, ref_sents)

            r1_f.append(r1["f"])
            rl_f.append(rl["f"])

    return {
        "r1_f": float(np.mean(r1_f)) if r1_f else 0.0,
        "rl_f": float(np.mean(rl_f)) if rl_f else 0.0
    }


# ======================
# main
# ======================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_json", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--glove_path", default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--strategy", default="topk")
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    setup_seed(32)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ======================
    # Dataset
    # ======================
    test_dataset = SummDataset(
        args.test_json,
        build_vocab=False,
        load_vocab_path=args.vocab_path
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ======================
    # Model
    # ======================
    model = ExtractiveSummarizer(
        test_dataset.vocab,
        embed_dim=300,
        hidden_size=256,
        glove_path=args.glove_path
    ).to(device)

    load_model(model, args.model_path)
    print(f"Loaded model from {args.model_path}")

    # ======================
    # Test
    # ======================
    scores = test_epoch(
        model,
        test_loader,
        device,
        strategy=args.strategy
    )

    print("\n====== Test Results ======")
    print(f"ROUGE-1 F1: {scores['r1_f']:.4f}")
    print(f"ROUGE-L F1: {scores['rl_f']:.4f}")


if __name__ == "__main__":
    main()
