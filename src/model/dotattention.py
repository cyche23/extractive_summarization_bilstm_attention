# src/model/attention.py
import torch
import torch.nn as nn
import math

class DotProductAttention(nn.Module):
    """
    Dot-product attention with a learnable global query vector.
    Computes importance scores over sentence vectors.
    """

    def __init__(self, hidden_dim=512, scale=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = scale

        # global query vector
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.query.unsqueeze(0))

    def forward(self, sent_vecs):
        """
        sent_vecs: [N_sent, hidden_dim]
        returns:
            scores: [N_sent] (logits, NOT softmaxed)
        """
        if sent_vecs.size(0) == 0:
            return torch.zeros(0, device=sent_vecs.device)

        # [N, H] · [H] -> [N]
        scores = torch.matmul(self.W(sent_vecs), self.query)

        if self.scale:
            scores = scores / math.sqrt(self.hidden_dim)

        return scores  # logits for sentence-level scoring
