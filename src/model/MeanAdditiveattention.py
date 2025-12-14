import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Dynamic Additive Attention (Context-Aware).
    使用文章的全局均值向量作为 Query，替代原先的随机静态 Query。
    这使得模型能根据每篇文章特定的主题来计算句子的重要性。
    """

    def __init__(self, hidden_dim=512):
        super().__init__()
        # W: 负责变换句子向量 (Local Features)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # U: 负责变换文档向量 (Global Context / Dynamic Query)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # v: 打分层
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        # [已删除] 删除了原来的静态 Parameter
        # self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, sent_vecs):
        """
        Args:
            sent_vecs: [N_sent, hidden_dim] (假设是一篇文章的所有句子向量)
        Returns:
            weights: [N_sent] (归一化后的注意力权重)
        """
        # 边界检查：防止空文章报错
        if sent_vecs.size(0) == 0:
            return torch.tensor([], device=sent_vecs.device)

        # ==========================================
        # 1. 构建动态 Query (Document Vector)
        # ==========================================
        # 逻辑：整篇文章的主旨 ≈ 所有句子的平均值
        # doc_vec: [hidden_dim]
        doc_vec = torch.mean(sent_vecs, dim=0)

        # ==========================================
        # 2. 维度对齐
        # ==========================================
        # 我们需要把文档向量 doc_vec 复制 N_sent 份，以便和每个句子进行加法运算
        # q_expanded: [N_sent, hidden_dim]
        q_expanded = doc_vec.unsqueeze(0).expand(sent_vecs.size(0), -1)

        # ==========================================
        # 3. 加性注意力计算 (Bahdanau Style)
        # ==========================================
        # 公式: score = v^T * tanh( W(s_i) + U(doc_vec) )

        # projected_sent: [N_sent, hidden_dim]
        projected_sent = self.W(sent_vecs)

        # projected_query: [N_sent, hidden_dim]
        projected_query = self.U(q_expanded)

        # 融合上下文与局部特征
        # energy: [N_sent, hidden_dim]
        energy = torch.tanh(projected_sent + projected_query)

        # 计算原始分数
        # scores: [N_sent, 1] -> [N_sent]
        scores = self.v(energy).squeeze(-1)

        # ==========================================
        # 4. Softmax 归一化
        # ==========================================
        weights = F.softmax(scores, dim=0)

        return weights
