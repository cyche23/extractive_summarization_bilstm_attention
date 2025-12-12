import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, sentence_embeddings):
        scores = self.v(torch.tanh(self.W(sentence_embeddings) + self.query))
        weights = torch.softmax(scores, dim=0)
        return weights
