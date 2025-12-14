# src/dataset.py
import json
import os
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from collections import Counter
import random
import pickle

# try:
#     # prefer nltk tokenizer for english
#     import nltk
#     nltk.data.find('tokenizers/punkt')
#     from nltk.tokenize import word_tokenize
# except Exception:
#     # fallback to simple split
#     def word_tokenize(x):
#         return x.split()

# ==========================================
# [升级] 使用 spaCy 强力分词器
# ==========================================
try:
    import spacy

    # 加载英文模型，禁用 parser, ner 等组件以极大提升速度
    # 如果报错，请在终端运行: python -m spacy download en_core_web_sm
    spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
except ImportError:
    print("[Error] 请安装 spacy: pip install spacy")
    exit()
except OSError:
    print("[Error] 请下载 spacy 模型: python -m spacy download en_core_web_sm")
    exit()

def word_tokenize(text):
    """
    使用 spaCy 分词 (针对 GloVe 840B 优化)
    自动处理: 标点粘连 (apple.), 缩写 (don't), 奇怪符号
    """
    if not isinstance(text, str):
        return []

    # 1. 基础清洗: 去除多余的 Tab、换行、连续空格
    text = " ".join(text.split())

    # 2. spaCy 智能分词
    doc = spacy_nlp(text)

    # 3. 返回 Token 文本列表
    return [token.text for token in doc]


PAD = "<pad>"
UNK = "<unk>"

class SummDataset(Dataset):
    """
    Dataset expects JSON list where each element is dict:
      { "id": "...", "sentences": [...], "labels": [...], "highlights": [...] }
    Tokenization is performed on the fly during __getitem__ or in collate.
    """
    def __init__(self, json_path, 
                 max_sent_len=100, 
                 min_freq=1,
                 vocab=None, 
                 build_vocab=True, 
                 save_vocab_path=None,
                 load_vocab_path=None):
        self.data = self._load(json_path)
        self.max_sent_len = max_sent_len
        if vocab:
            self.vocab = vocab
        elif build_vocab:
            self.vocab = self.build_vocab(self.data, min_freq=min_freq)
            if save_vocab_path:
                with open(save_vocab_path, 'wb') as f:
                    pickle.dump(self.vocab, f)
        else:
            if load_vocab_path:
                with open(load_vocab_path, 'rb') as f:
                    self.vocab = pickle.load(f)
                print(f"Load vocab size: {len(self.vocab)}")
            else:
                raise ValueError("load_vocab_path required if build_vocab=False")

    def _load(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    def build_vocab(self, data, min_freq=1):
        counter = Counter()
        for item in data:
            for s in item.get("sentences", []):
                toks = word_tokenize(s.lower())
                counter.update(toks)
        # special tokens
        vocab = {PAD:0, UNK:1}
        idx = 2
        for tok, freq in counter.most_common():
            if freq < min_freq:
                break
            if tok in vocab:
                continue
            vocab[tok] = idx
            idx += 1
        print(f"Built vocab size: {len(vocab)}")
        return vocab

    def __len__(self):
        return len(self.data)

    def sentence_to_ids(self, sentence):
        toks = word_tokenize(sentence.lower())
        toks = toks[:self.max_sent_len]
        ids = [self.vocab.get(t, self.vocab.get(UNK)) for t in toks]
        return ids, len(ids)

    def __getitem__(self, idx):
        item = self.data[idx]
        sents = item.get("sentences", [])
        labels = item.get("labels", [])
        sent_ids = []
        sent_lens = []
        for s in sents:
            ids, l = self.sentence_to_ids(s)
            sent_ids.append(ids)
            sent_lens.append(l)
        return {
            "id": item.get("id"),
            "sentences": sents,
            "sent_ids": sent_ids,
            "sent_lens": sent_lens,
            "labels": labels,
            "highlights": item.get("highlights", [])
        }

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