# src/model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import GloveEmbedding
from .bilstm_encoder import BiLSTMEncoder
from .attention import AdditiveAttention
from .dotattention import DotProductAttention

class ExtractiveSummarizer(nn.Module):
    """
    End-to-end extractive summarizer.
    Given an article represented as a set of sentences (word ids padded),
    produce logits for each sentence (higher -> more likely to be summary sentence).
    """

    def __init__(self, vocab, embed_dim=300, hidden_size=256, glove_path=None, embed_trainable=True):
        super().__init__()
        self.embedding = GloveEmbedding(vocab, embedding_dim=embed_dim, glove_path=glove_path, trainable=embed_trainable)
        self.encoder = BiLSTMEncoder(embed_dim, hidden_size)
        self.attention = AdditiveAttention(hidden_size * 2)
        # self.attention = DotProductAttention(hidden_size * 2)
        self.sigmoid = nn.Sigmoid()
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size * 2, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_size, 1)
        # )

    def forward(self, word_id_tensor, lengths):
        """
        word_id_tensor: LongTensor [num_sent, max_len]
        lengths: LongTensor [num_sent]
        returns:
            logits: FloatTensor [num_sent]
        """
        device = next(self.parameters()).device
        if word_id_tensor.size(0) == 0:
            return torch.zeros(0, device=device)

        embeds = self.embedding(word_id_tensor)  # [num_sent, max_len, embed_dim]
        sent_vecs = self.encoder(embeds, lengths)  # [num_sent, hidden*2]
        # print("sent_vecs:", sent_vecs)

        # attention weights (unused in loss but useful for inference/selection)
        atten_score = self.attention(sent_vecs)  # [num_sent]
        atten_weight = self.sigmoid(atten_score)  # [num_sent], 0~1

        # Optionally combine attn and logits; here we keep logits as classification scores.
        # logits = self.sigmoid(atten_score)
        # logits = self.classifier(atten_weight*sent_vecs).squeeze(-1)  # [num_sent]
        logits = atten_weight

        return logits