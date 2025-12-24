# src/model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """
    Dynamic Additive Attention (Context-Aware).
    使用文章的全局均值向量作为 Query。
    关键修正：Forward 返回原始 Logits (未归一化分数)，而非 Softmax 概率。
    """

    def __init__(self, hidden_dim=512):
        super().__init__()
        # W: 变换句子向量 (Local)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # U: 变换文档向量 (Global / Context)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # v: 打分层
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sent_vecs):
        """
        Args:
            sent_vecs: [N_sent, hidden_dim]
        Returns:
            scores: [N_sent] (Raw Logits, 实数范围, 无 Softmax)
        """
        # 边界检查
        if sent_vecs.size(0) == 0:
            return torch.tensor([], device=sent_vecs.device)

        # 1. 构建动态 Query (文档均值)
        # doc_vec: [hidden_dim]
        doc_vec = torch.mean(sent_vecs, dim=0)

        # 2. 维度扩展
        # q_expanded: [N_sent, hidden_dim]
        q_expanded = doc_vec.unsqueeze(0).expand(sent_vecs.size(0), -1)

        # 3. 计算 Energy (非线性变换)
        # projected_sent: [N_sent, hidden_dim]
        projected_sent = self.W(sent_vecs)
        # projected_query: [N_sent, hidden_dim]
        projected_query = self.U(q_expanded)
        
        # energy: [N_sent, hidden_dim]
        energy = torch.tanh(projected_sent + projected_query)

        # 4. 计算 Scores (Raw Logits)
        # scores: [N_sent]
        scores = self.v(energy).squeeze(-1)
        
        return scores
