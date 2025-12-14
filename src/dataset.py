# # src/dataset.py
# import json
# import os
# from typing import List, Dict
# import torch
# from torch.utils.data import Dataset
# from collections import Counter
# import random
# import pickle

# # try:
# #     # prefer nltk tokenizer for english
# #     import nltk
# #     nltk.data.find('tokenizers/punkt')
# #     from nltk.tokenize import word_tokenize
# # except Exception:
# #     # fallback to simple split
# #     def word_tokenize(x):
# #         return x.split()

# try:
#     import spacy

#     # 加载英文模型，禁用 parser, ner 等组件以极大提升速度
#     # 如果报错，请在终端运行: python -m spacy download en_core_web_sm
#     spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
# except ImportError:
#     print("[Error] 请安装 spacy: pip install spacy")
#     exit()
# except OSError:
#     print("[Error] 请下载 spacy 模型: python -m spacy download en_core_web_sm")
#     exit()

# def word_tokenize(text):
#     """
#     使用 spaCy 分词 (针对 GloVe 840B 优化)
#     自动处理: 标点粘连 (apple.), 缩写 (don't), 奇怪符号
#     """
#     if not isinstance(text, str):
#         return []

#     # 1. 基础清洗: 去除多余的 Tab、换行、连续空格
#     text = " ".join(text.split())

#     # 2. spaCy 智能分词
#     doc = spacy_nlp(text)

#     # 3. 返回 Token 文本列表
#     return [token.text for token in doc]


# PAD = "<pad>"
# UNK = "<unk>"

# class SummDataset(Dataset):
#     """
#     Dataset expects JSON list where each element is dict:
#       { "id": "...", "sentences": [...], "labels": [...], "highlights": [...] }
#     Tokenization is performed on the fly during __getitem__ or in collate.
#     """
#     def __init__(self, json_path, 
#                  max_sent_len=100, 
#                  min_freq=1,
#                  vocab=None, 
#                  build_vocab=True, 
#                  save_vocab_path=None,
#                  load_vocab_path=None):
#         self.data = self._load(json_path)
#         self.max_sent_len = max_sent_len
#         if vocab:
#             self.vocab = vocab
#         elif build_vocab:
#             self.vocab = self.build_vocab(self.data, min_freq=min_freq)
#             if save_vocab_path:
#                 with open(save_vocab_path, 'wb') as f:
#                     pickle.dump(self.vocab, f)
#         else:
#             if load_vocab_path:
#                 with open(load_vocab_path, 'rb') as f:
#                     self.vocab = pickle.load(f)
#                 print(f"Load vocab size: {len(self.vocab)}")
#             else:
#                 raise ValueError("load_vocab_path required if build_vocab=False")

#     def _load(self, path):
#         with open(path, 'r', encoding='utf8') as f:
#             data = json.load(f)
#         return data

#     def build_vocab(self, data, min_freq=1):
#         counter = Counter()
#         for item in data:
#             for s in item.get("sentences", []):
#                 toks = word_tokenize(s.lower())
#                 counter.update(toks)
#         # special tokens
#         vocab = {PAD:0, UNK:1}
#         idx = 2
#         for tok, freq in counter.most_common():
#             if freq < min_freq:
#                 break
#             if tok in vocab:
#                 continue
#             vocab[tok] = idx
#             idx += 1
#         print(f"Built vocab size: {len(vocab)}")
#         return vocab

#     def __len__(self):
#         return len(self.data)

#     def sentence_to_ids(self, sentence):
#         toks = word_tokenize(sentence.lower())
#         toks = toks[:self.max_sent_len]
#         ids = [self.vocab.get(t, self.vocab.get(UNK)) for t in toks]
#         return ids, len(ids)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         sents = item.get("sentences", [])
#         labels = item.get("labels", [])
#         sent_ids = []
#         sent_lens = []
#         for s in sents:
#             ids, l = self.sentence_to_ids(s)
#             sent_ids.append(ids)
#             sent_lens.append(l)
#         return {
#             "id": item.get("id"),
#             "sentences": sents,
#             "sent_ids": sent_ids,
#             "sent_lens": sent_lens,
#             "labels": labels,
#             "highlights": item.get("highlights", [])
#         }

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
#                  max_sent_len=100, 
#                  min_freq=1,
#                  vocab=None, 
#                  build_vocab=True, 
#                  save_vocab_path=None,
#                  load_vocab_path=None,
#                  cache_path=None):
#         """
#         新增cache_path参数: 可保存/加载预处理结果，避免重复初始化
#         """
#         self.max_sent_len = max_sent_len
#         self.data = self._load(json_path)
        
#         # 尝试加载预处理缓存
#         if cache_path and os.path.exists(cache_path):
#             print(f"Loading preprocessed data from {cache_path}")
#             with open(cache_path, 'rb') as f:
#                 cache = pickle.load(f)
#                 # self.vocab = cache['vocab']
#                 self.preprocessed_data = cache['data']
#             print(f"Loaded cache with {len(self.preprocessed_data)} items")
#         else:
#             # 构建或加载词汇表
#             if vocab:
#                 self.vocab = vocab
#             elif build_vocab:
#                 self.vocab = self.build_vocab(self.data, min_freq=min_freq)
#                 if save_vocab_path:
#                     with open(save_vocab_path, 'wb') as f:
#                         pickle.dump(self.vocab, f)
#             else:
#                 if not load_vocab_path:
#                     raise ValueError("load_vocab_path required if build_vocab=False")
#                 with open(load_vocab_path, 'rb') as f:
#                     self.vocab = pickle.load(f)
#                 print(f"Loaded vocab size: {len(self.vocab)}")
            
#             # 预处理所有数据 (核心优化)
#             self.preprocessed_data = self._preprocess_all()
            
#             # 保存预处理结果
#             if cache_path:
#                 print(f"Saving preprocessed data to {cache_path}")
#                 with open(cache_path, 'wb') as f:
#                     pickle.dump({
#                         'vocab': self.vocab,
#                         'data': self.preprocessed_data
#                     }, f)

#     def _load(self, path):
#         with open(path, 'r', encoding='utf8') as f:
#             data = json.load(f)
#         return data

#     def build_vocab(self, data, min_freq=1):
#         """优化: 批量分词加速词汇表构建"""
#         print("Building vocabulary (batch tokenization)...")
#         all_sentences = []
#         for item in tqdm(data, desc="Collecting sentences"):
#             all_sentences.extend([s.lower() for s in item.get("sentences", []) if isinstance(s, str)])
        
#         # 批量分词
#         all_tokens = []
#         batch_size = 1000
#         for i in tqdm(range(0, len(all_sentences), batch_size), desc="Tokenizing"):
#             batch = all_sentences[i:i+batch_size]
#             tokens_batch = batch_word_tokenize(batch)
#             for tokens in tokens_batch:
#                 all_tokens.extend(tokens)
        
#         # 统计词频
#         counter = Counter(all_tokens)
        
#         # 构建词汇表
#         vocab = {PAD: 0, UNK: 1}
#         idx = 2
#         for tok, freq in counter.most_common():
#             if freq < min_freq:
#                 break
#             if tok not in vocab:
#                 vocab[tok] = idx
#                 idx += 1
#         print(f"Built vocab size: {len(vocab)}")
#         return vocab

#     def _preprocess_all(self):
#         """核心优化: 一次性预处理所有数据"""
#         print("Preprocessing all data (batch tokenization and ID conversion)...")
#         preprocessed = []
        
#         # 收集所有需要处理的句子
#         all_sentences = []
#         indices = []  # 记录每个句子在原始数据中的位置 (item_idx, sent_idx)
        
#         for item_idx, item in enumerate(tqdm(self.data, desc="Collecting sentences for preprocessing")):
#             sents = item.get("sentences", [])
#             for sent_idx, s in enumerate(sents):
#                 if isinstance(s, str):
#                     all_sentences.append(s.lower())
#                     indices.append((item_idx, sent_idx))
        
#         # 批量分词
#         print(f"Tokenizing {len(all_sentences)} sentences in batches...")
#         all_tokenized = []
#         batch_size = 2000
#         for i in tqdm(range(0, len(all_sentences), batch_size), desc="Batch tokenization"):
#             batch = all_sentences[i:i+batch_size]
#             tokenized_batch = batch_word_tokenize(batch)
#             all_tokenized.extend(tokenized_batch)
        
#         # 转换为ID
#         print("Converting tokens to IDs...")
#         all_ids = []
#         all_lengths = []
#         for tokens in tqdm(all_tokenized, desc="ID conversion"):
#             # 截断
#             tokens = tokens[:self.max_sent_len]
#             # 转换为ID
#             ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
#             all_ids.append(ids)
#             all_lengths.append(len(ids))
        
#         # 重组为原始数据结构
#         print("Reconstructing dataset structure...")
#         # 为每个item创建存储空间
#         item_sents = [[] for _ in range(len(self.data))]
#         item_lens = [[] for _ in range(len(self.data))]
        
#         for (item_idx, sent_idx), ids, length in zip(indices, all_ids, all_lengths):
#             item_sents[item_idx].append(ids)
#             item_lens[item_idx].append(length)
        
#         # 构建最终预处理数据
#         for item_idx, item in enumerate(tqdm(self.data, desc="Finalizing preprocessing")):
#             preprocessed.append({
#                 "id": item.get("id"),
#                 "sentences": item.get("sentences", []),  # 保留原始句子
#                 "sent_ids": item_sents[item_idx],        # 预处理的ID
#                 "sent_lens": item_lens[item_idx],        # 预处理的长度
#                 "labels": item.get("labels", []),
#                 "highlights": item.get("highlights", [])
#             })
        
#         print(f"Preprocessing completed. Total items: {len(preprocessed)}")
#         return preprocessed

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
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from collections import Counter
import pickle
from tqdm import tqdm  # 添加进度条显示

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
    1. 预处理所有文本
    2. 使用spacy的pipe批量处理
    3. 保持原始顺序
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
    优化后的Dataset:
    - 初始化时预处理所有句子，避免重复tokenization
    - 使用批量分词加速初始化
    - 缓存ID转换结果
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
        self.data = self._load(json_path)
        
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
        
        # 2. 一次性批量分词 (关键优化)
        all_tokenized = []
        batch_size = 2000
        for i in tqdm(range(0, len(all_sentences_text), batch_size), desc="Batch tokenization"):
            batch = all_sentences_text[i:i+batch_size]
            tokenized_batch = batch_word_tokenize(batch)
            all_tokenized.extend(tokenized_batch)
        
        # 3. 构建词汇表 (使用已分词结果)
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
                with open(save_vocab_path, 'wb') as f:
                    pickle.dump(self.vocab, f)
        else:
            if not load_vocab_path:
                raise ValueError("load_vocab_path required if build_vocab=False")
            with open(load_vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print(f"Loaded vocab size: {len(self.vocab)}")
    
        # 4. 转换为ID (复用已分词结果)
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
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'data': self.preprocessed_data
                }, f)

    def _load(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        """直接返回预处理结果，无实时计算"""
        return self.preprocessed_data[idx]

# 保持collate_fn不变，因为输入格式未改变
def collate_fn(batch, pad_idx=0):
    """
    修改点：增加了 raw_sents 的传递，供推理模块使用
    """
    ids = []
    word_id_tensors = []
    length_tensors = []
    label_tensors = []
    highlights = []
    # 新增列表用于存储原始文本
    raw_sents_list = []

    for item in batch:
        ids.append(item["id"])
        sent_ids = item["sent_ids"]
        sent_lens = item["sent_lens"]

        #  获取 Dataset 中的原始句子列表
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
        "raw_sents": raw_sents_list  # 返回这个关键数据！
    }