import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim=300, hidden_size=256):
        super().__init__()
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        outputs, _ = self.bilstm(x)
        return outputs[:, -1, :]  # last hidden state
