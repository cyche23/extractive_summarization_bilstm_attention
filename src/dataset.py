# # src/dataset.py
# import json
# import os
# from typing import List, Dict, Any
# import torch
# from torch.utils.data import Dataset
# from collections import Counter
# import pickle
# from tqdm import tqdm  # 添加进度条显示

# try:
#     import spacy
#     # 加载英文模型，禁用非必要组件
#     spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
#     # 创建批量处理函数，极大提升分词速度
#     spacy_tokenizer = spacy_nlp.tokenizer
# except ImportError:
#     print("[Error] 请安装 spacy: pip install spacy")
#     exit()
# except OSError:
#     print("[Error] 请下载 spacy 模型: python -m spacy download en_core_web_sm")
#     exit()

# def batch_word_tokenize(texts: List[str]) -> List[List[str]]:
#     """
#     批量分词优化版 (比单句分词快5-10倍)
#     1. 预处理所有文本
#     2. 使用spacy的pipe批量处理
#     3. 保持原始顺序
#     """
#     # 基础清洗: 去除多余空白
#     cleaned_texts = [" ".join(text.split()) if isinstance(text, str) else "" for text in texts]
    
#     # 批量处理
#     docs = spacy_nlp.pipe(cleaned_texts, batch_size=1000)
    
#     # 提取token文本
#     return [[token.text for token in doc] for doc in docs]

# PAD = "<pad>"
# UNK = "<unk>"

# class SummDataset(Dataset):
#     """
#     优化后的Dataset:
#     - 初始化时预处理所有句子，避免重复tokenization
#     - 使用批量分词加速初始化
#     - 缓存ID转换结果
#     """
#     def __init__(self, json_path, 
#              max_sent_len=100, 
#              min_freq=1,
#              vocab=None, 
#              build_vocab=True, 
#              save_vocab_path=None,
#              load_vocab_path=None,
#              cache_path=None):
    
#         self.max_sent_len = max_sent_len
#         self.data = self._load(json_path)
        
#         # 尝试加载预处理缓存
#         if cache_path and os.path.exists(cache_path):
#             print(f"Loading preprocessed data from {cache_path}")
#             with open(cache_path, 'rb') as f:
#                 cache = pickle.load(f)
#                 self.vocab = cache['vocab']
#                 self.preprocessed_data = cache['data']
#             print(f"Loaded cache with {len(self.preprocessed_data)} items")
#             return
        
#         # === 核心优化: 只分词一次 ===
#         print("Collecting and tokenizing all sentences...")
        
#         # 1. 收集所有需要处理的句子
#         all_sentences_text = []
#         sentence_map = []  # 记录每个句子在原始数据中的位置 (item_idx, sent_idx)
        
#         for item_idx, item in enumerate(tqdm(self.data, desc="Collecting sentences")):
#             sents = item.get("sentences", [])
#             for sent_idx, s in enumerate(sents):
#                 if isinstance(s, str):
#                     all_sentences_text.append(s.lower())
#                     sentence_map.append((item_idx, sent_idx))
        
#         # 2. 一次性批量分词 (关键优化)
#         all_tokenized = []
#         batch_size = 2000
#         for i in tqdm(range(0, len(all_sentences_text), batch_size), desc="Batch tokenization"):
#             batch = all_sentences_text[i:i+batch_size]
#             tokenized_batch = batch_word_tokenize(batch)
#             all_tokenized.extend(tokenized_batch)
        
#         # 3. 构建词汇表 (使用已分词结果)
#         if vocab is not None:
#             self.vocab = vocab
#         elif build_vocab:
#             print("Building vocabulary from pre-tokenized data...")
#             counter = Counter()
#             for tokens in tqdm(all_tokenized, desc="Counting tokens"):
#                 counter.update(tokens)
            
#             # 构建词汇表
#             self.vocab = {PAD: 0, UNK: 1}
#             idx = 2
#             for tok, freq in counter.most_common():
#                 if freq < min_freq:
#                     break
#                 if tok not in self.vocab:
#                     self.vocab[tok] = idx
#                     idx += 1
#             print(f"Built vocab size: {len(self.vocab)}")
            
#             if save_vocab_path:
#                 with open(save_vocab_path, 'wb') as f:
#                     pickle.dump(self.vocab, f)
#         else:
#             if not load_vocab_path:
#                 raise ValueError("load_vocab_path required if build_vocab=False")
#             with open(load_vocab_path, 'rb') as f:
#                 self.vocab = pickle.load(f)
#             print(f"Loaded vocab size: {len(self.vocab)}")
    
#         # 4. 转换为ID (复用已分词结果)
#         print("Converting tokens to IDs...")
#         all_ids = []
#         all_lengths = []
#         for tokens in tqdm(all_tokenized, desc="ID conversion"):
#             tokens = tokens[:self.max_sent_len]
#             ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
#             all_ids.append(ids)
#             all_lengths.append(len(ids))
        
#         # 5. 重组为原始数据结构
#         item_sents = [[] for _ in range(len(self.data))]
#         item_lens = [[] for _ in range(len(self.data))]
        
#         for (item_idx, sent_idx), ids, length in zip(sentence_map, all_ids, all_lengths):
#             item_sents[item_idx].append(ids)
#             item_lens[item_idx].append(length)
        
#         # 6. 构建最终预处理数据
#         self.preprocessed_data = []
#         for item_idx, item in enumerate(tqdm(self.data, desc="Finalizing data structure")):
#             self.preprocessed_data.append({
#                 "id": item.get("id"),
#                 "sentences": item.get("sentences", []),
#                 "sent_ids": item_sents[item_idx],
#                 "sent_lens": item_lens[item_idx],
#                 "labels": item.get("labels", []),
#                 "highlights": item.get("highlights", [])
#             })
        
#         # 7. 保存缓存
#         if cache_path:
#             print(f"Saving preprocessed data to {cache_path}")
#             with open(cache_path, 'wb') as f:
#                 pickle.dump({
#                     'vocab': self.vocab,
#                     'data': self.preprocessed_data
#                 }, f)

#     def _load(self, path):
#         with open(path, 'r', encoding='utf8') as f:
#             data = json.load(f)
#         return data

#     def __len__(self):
#         return len(self.preprocessed_data)

#     def __getitem__(self, idx):
#         """直接返回预处理结果，无实时计算"""
#         return self.preprocessed_data[idx]

# # 保持collate_fn不变，因为输入格式未改变
# def collate_fn(batch, pad_idx=0):
#     """
#     修改点：增加了 raw_sents 的传递，供推理模块使用
#     """
#     ids = []
#     word_id_tensors = []
#     length_tensors = []
#     label_tensors = []
#     highlights = []
#     # 新增列表用于存储原始文本
#     raw_sents_list = []

#     for item in batch:
#         ids.append(item["id"])
#         sent_ids = item["sent_ids"]
#         sent_lens = item["sent_lens"]

#         #  获取 Dataset 中的原始句子列表
#         raw_sents_list.append(item.get("sentences", []))

#         if len(sent_ids) == 0:
#             word_id_tensors.append(torch.zeros((0, 0), dtype=torch.long))
#             length_tensors.append(torch.zeros((0,), dtype=torch.long))
#             label_tensors.append(torch.zeros((0,), dtype=torch.float))
#             highlights.append(item.get("highlights", []))
#             continue

#         max_len = max(len(s) for s in sent_ids)
#         padded = []
#         for s in sent_ids:
#             padded.append(s + [pad_idx] * (max_len - len(s)))

#         word_id_tensors.append(torch.tensor(padded, dtype=torch.long))
#         length_tensors.append(torch.tensor(sent_lens, dtype=torch.long))
#         label_tensors.append(torch.tensor(item.get("labels", [0] * len(sent_ids)), dtype=torch.float))
#         highlights.append(item.get("highlights", []))

#     return {
#         "ids": ids,
#         "word_ids": word_id_tensors,
#         "lengths": length_tensors,
#         "labels": label_tensors,
#         "highlights": highlights,
#         "raw_sents": raw_sents_list  # 返回这个关键数据！
#     }


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
    - 缓存机制保持不变
    """
    def __init__(self, json_path, 
             max_sent_len=100, 
             min_freq=1,
             vocab=None, 
             build_vocab=True, 
             save_vocab_path=None,
             load_vocab_path=None,
             cache_path=None):
    
        self.max_sent_len = max_sent_len
        
        # === 核心修改 1: 智能加载数据 ===
        # 不再假设 json_path 只是一个单独的文件
        self.data = self._load_sharded_data(json_path)
        
        # 尝试加载预处理缓存
        if cache_path and os.path.exists(cache_path):
            print(f"Loading preprocessed data from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.vocab = cache['vocab']
                self.preprocessed_data = cache['data']
            print(f"Loaded cache with {len(self.preprocessed_data)} items")
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
        
        # 3. 构建词汇表
        if vocab is not None:
            self.vocab = vocab
        elif build_vocab:
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
            
            if save_vocab_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_vocab_path), exist_ok=True)
                with open(save_vocab_path, 'wb') as f:
                    pickle.dump(self.vocab, f)
        else:
            if not load_vocab_path:
                raise ValueError("load_vocab_path required if build_vocab=False")
            with open(load_vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print(f"Loaded vocab size: {len(self.vocab)}")
    
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
        
        # 7. 保存缓存
        if cache_path:
            print(f"Saving preprocessed data to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'data': self.preprocessed_data
                }, f)

    def _load_sharded_data(self, path):
        """
        修改后的加载逻辑：
        1. 移除了 `os.path.isdir` 的自动加载，防止混淆同目录下的 train/val/test。
        2. 强制要求 path 必须匹配特定的文件模式（前缀或通配符）。
        """
        target_files = []

        # 1. 显式通配符匹配 (如 "data/train/*.json")
        if '*' in path or '?' in path or '[' in path:
            target_files = sorted(glob.glob(path))
        
        # 2. 具体文件匹配
        elif os.path.isfile(path):
            target_files = [path]
            
        # 3. 前缀匹配 (最常用的分片加载方式)
        # 例如 path="data/train"，将匹配 "data/train.0.json", "data/train.1.json"
        # 此时不会匹配到 "data/val.0.json"
        else:
            # 匹配 preprocess.py 生成的 shard 格式: prefix.*.json
            prefix_pattern = path + ".*.json"
            target_files = sorted(glob.glob(prefix_pattern))
            
            # 备用匹配: prefix*.json (兼容其他命名习惯，但需谨慎)
            # 注意：这行也可能导致问题，但比起加载整个目录安全得多
            # 如果 path 没带斜杠，比如 "train"，匹配 "train*.json" 是合理的
            extra_files = sorted(glob.glob(path + "*.json"))
            
            # 合并并去重
            target_files = sorted(list(set(target_files + extra_files)))

        if not target_files:
            # 提供明确的错误提示，告知用户不能只传目录路径
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
    collate_fn: 将一个 batch 的数据整理成 Tensor
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
        
        # 传递原始文本供 Evaluation 使用
        raw_sents_list.append(item.get("sentences", []))

        if len(sent_ids) == 0:
            # 处理空数据异常
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