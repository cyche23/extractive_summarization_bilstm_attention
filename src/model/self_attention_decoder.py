# import torch
# import torch.nn as nn
# import math

# class PositionalEncoding(nn.Module):
#     """
#     标准的正弦位置编码 (Sinusoidal Positional Encoding)。
#     因为 Transformer 是位置不敏感的，我们需要手动注入句子在文章中的位置信息。
#     这对新闻摘要尤为重要（Lead Bias）。
#     """
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
        
#         # 创建一个足够长的 PE 矩阵
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         # [max_len, 1, d_model] -> 方便后续直接与 [Seq, Batch, Dim] 相加
#         pe = pe.unsqueeze(1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         x: [Seq_Len, Batch_Size, D_Model]
#         """
#         # 截取对应长度的位置编码并相加
#         return x + self.pe[:x.size(0)]

# class SelfAttentionDecoder(nn.Module):
#     """
#     基于 Transformer 自注意力机制的解码器。
    
#     架构：
#     1. Input Projection: 对齐维度
#     2. Positional Encoding: 注入位置信息
#     3. Transformer Layer: 多头自注意力 + 前馈网络
#     4. Final Classifier: 线性分类头
    
#     优势：
#     相比 LSTM，它能更好地捕捉句子间的全局依赖和长距离关系。
#     """
#     def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
#         super().__init__()
        
#         print(f"Initializing SelfAttentionDecoder: D_Model={hidden_dim}, Heads={num_heads}, Layers={num_layers}")
        
#         self.d_model = hidden_dim

#         # 1. 维度对齐层
#         # 如果 Encoder 输出维度与 Transformer 隐藏层维度不同，则投影
#         if input_dim != hidden_dim:
#             self.input_proj = nn.Linear(input_dim, hidden_dim)
#         else:
#             self.input_proj = nn.Identity()

#         # 2. 位置编码
#         self.pos_encoder = PositionalEncoding(hidden_dim)

#         # 3. Transformer 编码层 (核心)
#         # 注意：这里使用 TransformerEncoderLayer，因为它允许双向注意力 (Bidirectional Attention)，
#         # 我们希望模型在判断第1句时，也能看到最后1句的信息。
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,
#             dropout=dropout,
#             activation='gelu'  # GELU 通常优于 ReLU
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # 4. 分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(), # Tanh 激活，用于最终特征融合
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )

#         self._init_weights()

#     def _init_weights(self):
#         """
#         Xavier 初始化，防止 Transformer 训练初期的梯度问题
#         """
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
        
#         # 对分类头的最后一层做特殊处理，使其初始输出较小且接近 0
#         # 防止初始 Loss 过大
#         for m in self.classifier.modules():
#             if isinstance(m, nn.Linear) and m.out_features == 1:
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

#     def forward(self, context_vecs):
#         """
#         Args:
#             context_vecs: [num_sentences, input_dim] 
#                           (来自 BiLSTM Encoder 的输出)
#         Returns:
#             logits: [num_sentences]
#         """
#         # 边界检查
#         if context_vecs.size(0) == 0:
#             return torch.tensor([], device=context_vecs.device)

#         # 1. 维度调整 [Seq_Len, Input_Dim] -> [Seq_Len, Batch=1, Input_Dim]
#         # Transformer 需要 3D 输入，这里我们将单篇文章视为 Batch=1 的序列
#         src = context_vecs.unsqueeze(1)
        
#         # 2. 投影到 d_model
#         src = self.input_proj(src) # [Seq, 1, Hidden]
        
#         # 3. 缩放 (Transformer 惯例) + 位置编码
#         src = src * math.sqrt(self.d_model)
#         src = self.pos_encoder(src)
        
#         # 4. 经过 Transformer 层
#         # output: [Seq_Len, Batch=1, Hidden]
#         output = self.transformer_encoder(src)
        
#         # 5. 变回 2D 并分类
#         # [Seq, 1, Hidden] -> [Seq, Hidden]
#         output = output.squeeze(1)
        
#         # 6. 计算 Logits
#         # [Seq, Hidden] -> [Seq, 1] -> [Seq]
#         logits = self.classifier(output).squeeze(-1)
        
#         return logits

# src/model/self_attention_decoder.py
import torch
import torch.nn as nn
import math

class SelfAttentionDecoder(nn.Module):
    """
    BiLSTM-Friendly Self-Attention Decoder.
    
    关键修正：
    1. 移除 Positional Encoding：因为 BiLSTM 已经包含序列信息，显式 PE 会导致模型过拟合位置（只选前三句）。
    2. 移除 Scale (sqrt(d)): BiLSTM 输出不需要缩放。
    3. 启用 Pre-Norm (norm_first=True): 梯度流更健康。
    """
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        print(f"Initializing SelfAttentionDecoder (No-PE Version): Input={input_dim}, Hidden={hidden_dim}")
        
        self.d_model = hidden_dim

        # 1. 维度对齐层
        # 如果 Encoder 输出维度(512) 与 Transformer (256) 不同，则投影
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # 2. Transformer Encoder Layer (核心)
        # norm_first=True (Pre-LN) 是现代 Transformer 的标配，收敛更稳
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            norm_first=True  # [关键] 先 Norm 再 Attention，梯度更直通
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # [新增] 分类前加 Norm
            nn.Tanh(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Xavier 初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 针对分类头最后一层，使其初始 Logits 接近 0
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, context_vecs):
        """
        Args:
            context_vecs: [num_sentences, input_dim] (BiLSTM 输出)
        Returns:
            logits: [num_sentences]
        """
        if context_vecs.size(0) == 0:
            return torch.tensor([], device=context_vecs.device)

        # 1. 维度调整 [Seq_Len, Input_Dim] -> [Seq_Len, Batch=1, Input_Dim]
        # BiLSTM 的输出本身就是有序的，所以不需要额外加 Positional Encoding
        src = context_vecs.unsqueeze(1)
        
        # 2. 投影到 d_model
        # [Seq, 1, 512] -> [Seq, 1, 256]
        src = self.input_proj(src) 
        
        # 3. ❌ 移除缩放: src = src * math.sqrt(self.d_model) 
        # BiLSTM 的输出经过了 Tanh，数值范围有限，不应该强行放大
        
        # 4. ❌ 移除位置编码: src = self.pos_encoder(src)
        # 让 Transformer 必须依赖 BiLSTM 传来的语义特征来计算 Attention，
        # 而不是依赖绝对位置索引。
        
        # 5. 经过 Transformer 层 (Self-Attention)
        # output: [Seq_Len, 1, Hidden]
        output = self.transformer_encoder(src)
        
        # 6. 变回 2D 并分类
        output = output.squeeze(1) # [Seq, Hidden]
        
        logits = self.classifier(output).squeeze(-1)
        
        return logits