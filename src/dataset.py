# src/dataset.py
import json
import os
import glob
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from collections import Counter
import pickle
from tqdm import tqdm

try:
    import spacy
    # 加载英文模型，禁用非必要组件
    spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
    # 创建批量处理函数，极大提升分词速度
    spacy_tokenizer = spacy_nlp.tokenizer
except ImportError:
    print("[Error] 请安装 spacy: pip install spacy")
    exit()
except OSError:
    print("[Error] 请下载 spacy 模型: python -m spacy download en_core_web_sm")
    exit()

def batch_word_tokenize(texts: List[str]) -> List[List[str]]:
    """
    批量分词优化版 (比单句分词快5-10倍)
    """
    # 基础清洗: 去除多余空白
    cleaned_texts = [" ".join(text.split()) if isinstance(text, str) else "" for text in texts]
    
    # 批量处理
    docs = spacy_nlp.pipe(cleaned_texts, batch_size=1000)
    
    # 提取token文本
    return [[token.text for token in doc] for doc in docs]

PAD = "<pad>"
UNK = "<unk>"

class SummDataset(Dataset):
    """
    支持分片加载的 Dataset:
    - 自动识别并聚合分片 JSON 文件
    - 初始化时预处理所有句子
    - 统一使用 cache_path 管理数据和词表的持久化，移除了独立的 vocab_path
    """
    def __init__(self, json_path, 
             max_sent_len=100, 
             min_freq=1,
             vocab=None, 
             build_vocab=True, 
             cache_path=None):
        
        # 移除了 save_vocab_path 和 load_vocab_path 参数
        self.max_sent_len = max_sent_len
        
        # === 核心修改 1: 智能加载数据 ===
        # 不再假设 json_path 只是一个单独的文件
        self.data = self._load_sharded_data(json_path)
        
        # 尝试加载预处理缓存 (包含 data 和 vocab)
        if cache_path and os.path.exists(cache_path):
            print(f"Loading preprocessed data (and vocab) from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.vocab = cache['vocab']
                self.preprocessed_data = cache['data']
            print(f"Loaded cache with {len(self.preprocessed_data)} items and vocab size {len(self.vocab)}")
            return
        
        # === 核心优化: 只分词一次 ===
        print("Collecting and tokenizing all sentences...")
        
        # 1. 收集所有需要处理的句子
        all_sentences_text = []
        sentence_map = []  # 记录每个句子在原始数据中的位置 (item_idx, sent_idx)
        
        for item_idx, item in enumerate(tqdm(self.data, desc="Collecting sentences")):
            sents = item.get("sentences", [])
            for sent_idx, s in enumerate(sents):
                if isinstance(s, str):
                    all_sentences_text.append(s.lower())
                    sentence_map.append((item_idx, sent_idx))
        
        # 2. 一次性批量分词
        all_tokenized = []
        batch_size = 2000
        for i in tqdm(range(0, len(all_sentences_text), batch_size), desc="Batch tokenization"):
            batch = all_sentences_text[i:i+batch_size]
            tokenized_batch = batch_word_tokenize(batch)
            all_tokenized.extend(tokenized_batch)
        
        # 3. 构建或加载词汇表 (逻辑已简化)
        if vocab is not None:
            # 优先使用传入的 vocab 对象 (通常用于验证集/测试集复用训练集的 vocab)
            self.vocab = vocab
            print(f"Used provided vocab with size: {len(self.vocab)}")
        elif build_vocab:
            # 仅在允许构建时重新统计
            print("Building vocabulary from pre-tokenized data...")
            counter = Counter()
            for tokens in tqdm(all_tokenized, desc="Counting tokens"):
                counter.update(tokens)
            
            # 构建词汇表
            self.vocab = {PAD: 0, UNK: 1}
            idx = 2
            for tok, freq in counter.most_common():
                if freq < min_freq:
                    break
                if tok not in self.vocab:
                    self.vocab[tok] = idx
                    idx += 1
            print(f"Built vocab size: {len(self.vocab)}")
            # 注意：此处移除了 save_vocab_path 的保存逻辑，统一留到最后保存到 cache
        else:
            # 如果不构建且没有传入 vocab，也没有命中 cache，则无法继续
            raise ValueError(
                "No vocabulary available. Since `build_vocab=False` and no cache was found, "
                "you must provide a `vocab` dictionary argument explicitly."
            )
    
        # 4. 转换为ID
        print("Converting tokens to IDs...")
        all_ids = []
        all_lengths = []
        for tokens in tqdm(all_tokenized, desc="ID conversion"):
            tokens = tokens[:self.max_sent_len]
            ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
            all_ids.append(ids)
            all_lengths.append(len(ids))
        
        # 5. 重组为原始数据结构
        item_sents = [[] for _ in range(len(self.data))]
        item_lens = [[] for _ in range(len(self.data))]
        
        for (item_idx, sent_idx), ids, length in zip(sentence_map, all_ids, all_lengths):
            item_sents[item_idx].append(ids)
            item_lens[item_idx].append(length)
        
        # 6. 构建最终预处理数据
        self.preprocessed_data = []
        for item_idx, item in enumerate(tqdm(self.data, desc="Finalizing data structure")):
            self.preprocessed_data.append({
                "id": item.get("id"),
                "sentences": item.get("sentences", []),
                "sent_ids": item_sents[item_idx],
                "sent_lens": item_lens[item_idx],
                "labels": item.get("labels", []),
                "highlights": item.get("highlights", [])
            })
        
        # 7. 保存缓存 (包含 vocab 和 data)
        if cache_path:
            print(f"Saving preprocessed data and vocab to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'data': self.preprocessed_data
                }, f)

    def _load_sharded_data(self, path):
        """
        加载逻辑保持不变
        """
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
        
        combined_data = []
        for file_p in tqdm(target_files, desc="Loading JSON shards"):
            try:
                with open(file_p, 'r', encoding='utf8') as f:
                    shard_data = json.load(f)
                    combined_data.extend(shard_data)
            except Exception as e:
                print(f"[Warning] Failed to load {file_p}: {e}")
        
        print(f"Total samples loaded: {len(combined_data)}")
        return combined_data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

def collate_fn(batch, pad_idx=0):
    """
    collate_fn 保持不变
    """
    ids = []
    word_id_tensors = []
    length_tensors = []
    label_tensors = []
    highlights = []
    raw_sents_list = []

    for item in batch:
        ids.append(item["id"])
        sent_ids = item["sent_ids"]
        sent_lens = item["sent_lens"]
        
        raw_sents_list.append(item.get("sentences", []))

        if len(sent_ids) == 0:
            word_id_tensors.append(torch.zeros((0, 0), dtype=torch.long))
            length_tensors.append(torch.zeros((0,), dtype=torch.long))
            label_tensors.append(torch.zeros((0,), dtype=torch.float))
            highlights.append(item.get("highlights", []))
            continue

        max_len = max(len(s) for s in sent_ids)
        padded = []
        for s in sent_ids:
            padded.append(s + [pad_idx] * (max_len - len(s)))

        word_id_tensors.append(torch.tensor(padded, dtype=torch.long))
        length_tensors.append(torch.tensor(sent_lens, dtype=torch.long))
        label_tensors.append(torch.tensor(item.get("labels", [0] * len(sent_ids)), dtype=torch.float))
        highlights.append(item.get("highlights", []))

    return {
        "ids": ids,
        "word_ids": word_id_tensors,
        "lengths": length_tensors,
        "labels": label_tensors,
        "highlights": highlights,
        "raw_sents": raw_sents_list
    }