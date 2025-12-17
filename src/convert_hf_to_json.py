import json
import os
import argparse
import random
from datasets import load_dataset
from tqdm import tqdm


def normalize_ratios(train_r, val_r, test_r):
    """
    将 train / val / test 比例归一化，避免用户输入比例和 != 1 的情况
    例如：0.8 / 0.1 / 0.1 或 8 / 1 / 1 都是合法的
    """
    total = train_r + val_r + test_r
    return train_r / total, val_r / total, test_r / total


def convert_and_split(
    dataset_name,
    hf_splits,
    out_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    max_samples=None,
    drop_empty_labels=True,
    shuffle=True,
    seed=42,
    samples_per_file=10000  # <--- [新增参数] 每个文件的样本数量
):
    """
    核心函数：从 HuggingFace 加载 extractive 数据集，
    并在本地进行灵活划分，最终导出为 SummDataset 可直接使用的 JSON。
    """

    print("Loading HuggingFace dataset...")
    all_samples = []

    # -------------------------------------------------
    # Step 1: 读取 HuggingFace 数据（逻辑不变）
    # -------------------------------------------------
    for split in hf_splits:
        ds = load_dataset(dataset_name, split=split)

        for i, sample in enumerate(tqdm(ds, desc=f"Reading HF split={split}")):
            src = sample["src"]       # 文章句子列表
            labels = sample["labels"] # extractive 标签
            tgt = sample["tgt"]       # gold summary（句级）

            # 基本一致性检查
            if len(src) != len(labels):
                continue

            # 丢弃没有任何正标签的样本（避免模型学成全 0）
            if drop_empty_labels and sum(labels) == 0:
                continue

            # 转为 SummDataset 期望的格式
            all_samples.append({
                "id": f"{split}_{i}",
                "sentences": src,
                "labels": labels,
                "highlights": tgt
            })

            # debug：限制最大样本数
            if max_samples and len(all_samples) >= max_samples:
                break

    print(f"Total usable samples after filtering: {len(all_samples)}")

    # -------------------------------------------------
    # Step 2: 打乱 + 按比例重新划分（逻辑不变）
    # -------------------------------------------------
    if shuffle:
        random.seed(seed)
        random.shuffle(all_samples)

    train_ratio, val_ratio, test_ratio = normalize_ratios(
        train_ratio, val_ratio, test_ratio
    )

    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = all_samples[:n_train]
    val_data = all_samples[n_train:n_train + n_val]
    test_data = all_samples[n_train + n_val:]

    # -------------------------------------------------
    # Step 3: 写出 JSON 文件（[修改] 支持分批写入）
    # -------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    def dump(data, name):
        """
        将数据切分为多个批次并保存。
        例如：train.0.json, train.1.json ...
        """
        total_len = len(data)
        if total_len == 0:
            print(f"[{name}] No data to save.")
            return

        # 计算需要切分多少个文件
        # 如果 samples_per_file <= 0，则不切分，保存为一个大文件
        chunk_size = samples_per_file if samples_per_file > 0 else total_len
        
        # 向上取整计算 chunk 数量
        num_chunks = (total_len + chunk_size - 1) // chunk_size

        print(f"[{name}] Saving {total_len} samples into {num_chunks} files (batch size: {chunk_size})...")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_len)
            
            chunk_data = data[start_idx:end_idx]
            
            # 生成文件名：例如 train.0.json, val.0.json
            filename = f"{name}.{i}.json"
            path = os.path.join(out_dir, filename)
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
            # 仅在只有1个文件时不打印过于详细的日志，文件多时避免刷屏，可选
            # print(f"  -> Saved {path} ({len(chunk_data)} samples)")

    dump(train_data, "train")
    dump(val_data, "val")
    dump(test_data, "test")


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(
        description="Convert HF extractive dataset to SummDataset JSON with flexible split"
    )

    parser.add_argument(
        "--dataset",
        default="ereverter/cnn_dailymail_extractive",
        help="HuggingFace dataset name"
    )

    parser.add_argument(
        "--hf_splits",
        nargs="+",
        default=["train"],
        help="Which HF splits to use."
    )

    parser.add_argument(
        "--out_dir",
        default="data/cnn_dm_extractive",
        help="Output directory for train/val/test JSON files"
    )

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (debug only)"
    )

    # -------------------------------------------------
    # [新增] 命令行参数：控制每个 JSON 文件的样本数
    # -------------------------------------------------
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=10000,
        help="Number of samples per JSON file (e.g., train.0.json). Set to 0 for single file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and split"
    )

    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable shuffling before split"
    )

    args = parser.parse_args()

    convert_and_split(
        dataset_name=args.dataset,
        hf_splits=args.hf_splits,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        samples_per_file=args.samples_per_file  # 传入新参数
    )


if __name__ == "__main__":
    main()