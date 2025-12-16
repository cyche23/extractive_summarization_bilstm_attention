# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class EnhancedSummaryDecoder(nn.Module):
#     """高性能抽取式摘要Decoder，专为单文档输入优化"""
    
#     def __init__(self, input_dim=512, hidden_dim=256, num_heads=4, max_sentences=100):
#         """
#         参数:
#         input_dim: BiLSTM输出的句子嵌入维度 (512)
#         hidden_dim: 内部隐藏层维度 (256)
#         num_heads: 多头注意力头数 (4)
#         max_sentences: 支持的最大句子数 (100)
#         """
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.max_sentences = max_sentences
        
#         # 1. 位置感知层
#         self.position_embedding = nn.Embedding(max_sentences, hidden_dim)
#         position_weights = torch.zeros(max_sentences)
#         position_weights[:3] = 1.2  # 开头3句加权
#         position_weights[-3:] = 1.1  # 结尾3句加权
#         self.register_buffer('position_weights', position_weights)
        
#         # 2. 位置嵌入投影层 (将256维投影到512维)
#         self.pos_proj = nn.Linear(hidden_dim, input_dim)
        
#         # 3. 多头注意力层
#         self.head_dim = hidden_dim // num_heads
#         assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
#         self.q_proj = nn.Linear(input_dim, hidden_dim)
#         self.k_proj = nn.Linear(input_dim, hidden_dim)
#         self.v_proj = nn.Linear(input_dim, hidden_dim)
#         self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        
#         # 4. 修正：门控融合层应接收input_dim + hidden_dim = 768维输入
#         self.sentence_gate = nn.Sequential(
#             nn.Linear(input_dim + hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
        
#         # 5. 最终评分层
#         self.score_layer = nn.Sequential(
#             nn.Linear(hidden_dim + 1, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim // 2, 1)
#         )
        
#         # 6. 辅助层归一化和Dropout
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(0.2)
        
#         self._reset_parameters()
    
#     def _reset_parameters(self):
#         """初始化参数"""
#         nn.init.xavier_uniform_(self.q_proj.weight)
#         nn.init.xavier_uniform_(self.k_proj.weight)
#         nn.init.xavier_uniform_(self.v_proj.weight)
#         nn.init.xavier_uniform_(self.attn_out.weight)
#         nn.init.zeros_(self.q_proj.bias)
#         nn.init.zeros_(self.k_proj.bias)
#         nn.init.zeros_(self.v_proj.bias)
#         nn.init.zeros_(self.attn_out.bias)
#         nn.init.xavier_uniform_(self.pos_proj.weight)
#         nn.init.zeros_(self.pos_proj.bias)
    
#     def positional_encode(self, sentence_embs, sentence_positions=None):
#         """增强位置编码，特别适配新闻文本结构"""
#         num_sentences = sentence_embs.size(0)
        
#         if sentence_positions is None:
#             sentence_positions = torch.arange(num_sentences, device=sentence_embs.device)
        
#         sentence_positions = torch.clamp(sentence_positions, 0, self.max_sentences-1)
        
#         # 获取位置嵌入 [num_sentences, hidden_dim]
#         pos_embs = self.position_embedding(sentence_positions)
        
#         # 将位置嵌入投影到sentence_embs的维度 [num_sentences, input_dim]
#         pos_embs_proj = self.pos_proj(pos_embs)
        
#         # 应用新闻领域先验位置权重
#         weights = self.position_weights[:num_sentences].unsqueeze(-1)
#         pos_embs_weighted = pos_embs_proj * weights
        
#         # 与句子嵌入融合
#         return sentence_embs + pos_embs_weighted
    
#     def multi_head_attention(self, sentence_embs):
#         """轻量级多头注意力机制"""
#         num_sentences = sentence_embs.size(0)
        
#         # 投影Q, K, V
#         Q = self.q_proj(sentence_embs)  # [seq, hidden]
#         K = self.k_proj(sentence_embs)  # [seq, hidden]
#         V = self.v_proj(sentence_embs)  # [seq, hidden]
        
#         # 拆分为多头 [num_heads, seq_len, head_dim]
#         Q = Q.view(num_sentences, self.num_heads, self.head_dim).transpose(0, 1)
#         K = K.view(num_sentences, self.num_heads, self.head_dim).transpose(0, 1)
#         V = V.view(num_sentences, self.num_heads, self.head_dim).transpose(0, 1)
        
#         # 计算注意力分数 [num_heads, seq_len, seq_len]
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
#         # 计算注意力权重
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
        
#         # 应用注意力权重 [num_heads, seq_len, head_dim]
#         attn_output = torch.matmul(attn_weights, V)
        
#         # 重组多头输出 [seq_len, hidden_dim]
#         attn_output = attn_output.transpose(0, 1).contiguous().view(num_sentences, self.hidden_dim)
        
#         # 输出投影
#         attn_output = self.attn_out(attn_output)
#         return attn_output, attn_weights.mean(dim=0)
    
#     def forward(self, sentence_embs, sentence_positions=None, sentence_mask=None):
#         """
#         前向传播 - 专为单文档输入设计
        
#         参数:
#         sentence_embs: [num_sentences, 512] BiLSTM编码的句子嵌入
#         sentence_positions: [num_sentences] 句子在原文中的位置(可选)
#         sentence_mask: [num_sentences] 句子掩码，指示有效句子(1)和填充(0)
        
#         返回:
#         sentence_scores: [num_sentences] 每个句子的重要性分数(0-1)
#         attn_weights: [num_sentences, num_sentences] 注意力权重(用于可视化)
#         """
#         num_sentences = sentence_embs.size(0)
        
#         # 1. 位置增强编码
#         pos_enhanced_embs = self.positional_encode(sentence_embs, sentence_positions)
        
#         # 2. 应用多头注意力
#         attn_output, attn_weights = self.multi_head_attention(pos_enhanced_embs)
        
#         # 3. 修正残差连接：需要投影到相同维度
#         # 将pos_enhanced_embs投影到hidden_dim维度
#         projected_embs = self.q_proj(pos_enhanced_embs)  # [num_sentences, hidden_dim]
#         attn_output = self.layer_norm(attn_output + projected_embs)  # 残差连接
        
#         # 4. 生成位置权重特征
#         if sentence_positions is None:
#             sentence_positions = torch.arange(num_sentences, device=sentence_embs.device)
#         pos_weights = self.position_weights[:num_sentences].unsqueeze(-1)  # [num_sentences, 1]
        
#         # 5. 修正：门控融合使用完整sentence_embs和attn_output
#         # [num_sentences, 512+256=768]
#         gate_input = torch.cat([sentence_embs, attn_output], dim=-1)
#         gate = self.sentence_gate(gate_input)  # [num_sentences, 1]
#         # 动态融合：保留完整sentence_embs，不截取
#         fused_features = gate * sentence_embs + (1 - gate) * self.pos_proj(attn_output)
#         # 投影回hidden_dim维度用于评分
#         fused_features = self.q_proj(fused_features)  # [num_sentences, hidden_dim]
        
#         # 6. 生成最终句子分数
#         score_input = torch.cat([fused_features, pos_weights], dim=-1)
#         sentence_scores = self.score_layer(score_input).squeeze(-1)  # [num_sentences]
        
#         # 7. 应用掩码
#         if sentence_mask is not None:
#             sentence_scores = sentence_scores.masked_fill(~sentence_mask.bool(), -1e9)
        
#         # 8. 归一化为概率
#         # sentence_probs = torch.sigmoid(sentence_scores)
        
#         return sentence_scores, attn_weights

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    标准的正弦位置编码 (Sinusoidal Positional Encoding)。
    因为 Transformer 是位置不敏感的，我们需要手动注入句子在文章中的位置信息。
    这对新闻摘要尤为重要（Lead Bias）。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建一个足够长的 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [max_len, 1, d_model] -> 方便后续直接与 [Seq, Batch, Dim] 相加
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [Seq_Len, Batch_Size, D_Model]
        """
        # 截取对应长度的位置编码并相加
        return x + self.pe[:x.size(0)]

class SelfAttentionDecoder(nn.Module):
    """
    基于 Transformer 自注意力机制的解码器。
    
    架构：
    1. Input Projection: 对齐维度
    2. Positional Encoding: 注入位置信息
    3. Transformer Layer: 多头自注意力 + 前馈网络
    4. Final Classifier: 线性分类头
    
    优势：
    相比 LSTM，它能更好地捕捉句子间的全局依赖和长距离关系。
    """
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        print(f"Initializing SelfAttentionDecoder: D_Model={hidden_dim}, Heads={num_heads}, Layers={num_layers}")
        
        self.d_model = hidden_dim

        # 1. 维度对齐层
        # 如果 Encoder 输出维度与 Transformer 隐藏层维度不同，则投影
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # 3. Transformer 编码层 (核心)
        # 注意：这里使用 TransformerEncoderLayer，因为它允许双向注意力 (Bidirectional Attention)，
        # 我们希望模型在判断第1句时，也能看到最后1句的信息。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'  # GELU 通常优于 ReLU
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), # Tanh 激活，用于最终特征融合
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Xavier 初始化，防止 Transformer 训练初期的梯度问题
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 对分类头的最后一层做特殊处理，使其初始输出较小且接近 0
        # 防止初始 Loss 过大
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, context_vecs):
        """
        Args:
            context_vecs: [num_sentences, input_dim] 
                          (来自 BiLSTM Encoder 的输出)
        Returns:
            logits: [num_sentences]
        """
        # 边界检查
        if context_vecs.size(0) == 0:
            return torch.tensor([], device=context_vecs.device)

        # 1. 维度调整 [Seq_Len, Input_Dim] -> [Seq_Len, Batch=1, Input_Dim]
        # Transformer 需要 3D 输入，这里我们将单篇文章视为 Batch=1 的序列
        src = context_vecs.unsqueeze(1)
        
        # 2. 投影到 d_model
        src = self.input_proj(src) # [Seq, 1, Hidden]
        
        # 3. 缩放 (Transformer 惯例) + 位置编码
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 4. 经过 Transformer 层
        # output: [Seq_Len, Batch=1, Hidden]
        output = self.transformer_encoder(src)
        
        # 5. 变回 2D 并分类
        # [Seq, 1, Hidden] -> [Seq, Hidden]
        output = output.squeeze(1)
        
        # 6. 计算 Logits
        # [Seq, Hidden] -> [Seq, 1] -> [Seq]
        logits = self.classifier(output).squeeze(-1)
        
        return logits