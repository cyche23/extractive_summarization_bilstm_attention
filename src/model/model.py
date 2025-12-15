# src/model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import GloveEmbedding
from .bilstm_encoder import BiLSTMEncoder
from .attention import AdditiveAttention
from .mlp_decoder import SequenceLabelingDecoder

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
        # self.decoder = AdditiveAttention(hidden_size * 2)
        self.decoder = SequenceLabelingDecoder(hidden_size * 2)

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

        logits = self.decoder(sent_vecs)  # [num_sent]

        # logits = self.decoder(sent_vecs)

        return logits, []