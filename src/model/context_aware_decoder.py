# # src/model/decoder.py
# import torch
# import torch.nn as nn

# class ContextAwareDecoder(nn.Module):
#     """
#     Context-Aware MLP Decoder (Stabilized Version).
    
#     改进点：
#     1. LayerNorm: 在拼接后立刻进行归一化，解决 concat 导致的特征分布不均。
#     2. Initialization: 修复最后一层 fan_out 初始化导致的权重过大问题。
#     3. LeakyReLU: 防止神经元死亡。
#     """
#     def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
#         super().__init__()
        
#         # 输入维度翻倍 (句子向量 + 文档向量)
#         concat_dim = input_dim * 2
        
#         self.classifier = nn.Sequential(
#             # Layer 1
#             nn.Linear(concat_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),           # [新增] 稳定分布，加速收敛
#             nn.ReLU(), 
#             nn.Dropout(dropout),
            
#             # Layer 2
#             nn.Linear(hidden_dim, 1)
#         )
        
#         self._init_weights()

#     def _init_weights(self):
#         """
#         精细化权重初始化
#         """
#         for name, m in self.classifier.named_modules():
#             if isinstance(m, nn.Linear):
#                 # 判断是否是最后一层 (输出维度为1)
#                 if m.out_features == 1:
#                     # 最后一层使用 Xavier Uniform，保证初始 Logits 接近 0
#                     nn.init.xavier_uniform_(m.weight)
#                 else:
#                     # 中间层使用 Kaiming Normal (配合 Leaky ReLU)
#                     # 使用 mode='fan_in' 更安全
#                     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

#     def forward(self, context_vecs):
#         """
#         Args:
#             context_vecs: [num_sentences, input_dim]
#         Returns:
#             logits: [num_sentences]
#         """
#         if context_vecs.size(0) == 0:
#             return torch.tensor([], device=context_vecs.device)

#         # 1. 计算文档全局上下文 (Mean Pooling)
#         # doc_vec: [input_dim]
#         doc_vec = torch.mean(context_vecs, dim=0)
        
#         # 2. 扩展文档向量
#         # doc_expanded: [num_sentences, input_dim]
#         doc_expanded = doc_vec.unsqueeze(0).expand(context_vecs.size(0), -1)
        
#         # 3. 拼接 (局部特征 + 全局特征)
#         # combined: [num_sentences, input_dim * 2]
#         combined = torch.cat([context_vecs, doc_expanded], dim=-1)
        
#         # 4. MLP 打分
#         logits = self.classifier(combined)
        
#         return logits.squeeze(-1)
    
# src/model/decoder.py
import torch
import torch.nn as nn

class ContextAwareDecoder(nn.Module):
    """
    Context-Aware MLP Decoder (Stabilized Version).
    
    改进点：
    1. LayerNorm: 在拼接后立刻进行归一化，解决 concat 导致的特征分布不均。
    2. Initialization: 修复最后一层 fan_out 初始化导致的权重过大问题。
    3. LeakyReLU: 防止神经元死亡。
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        
        # 输入维度翻倍 (句子向量 + 文档向量)
        concat_dim = hidden_dim * 2

        # W: 变换句子向量 (Local)
        self.W = nn.Linear(input_dim, hidden_dim, bias=True)
        # U: 变换文档向量 (Global / Context)
        self.U = nn.Linear(input_dim, hidden_dim, bias=True)
        
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),           # [新增] 稳定分布，加速收敛
            nn.ReLU(), 
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """
        精细化权重初始化
        """
        for name, m in self.classifier.named_modules():
            if isinstance(m, nn.Linear):
                # 判断是否是最后一层 (输出维度为1)
                if m.out_features == 1:
                    # 最后一层使用 Xavier Uniform，保证初始 Logits 接近 0
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # 中间层使用 Kaiming Normal (配合 Leaky ReLU)
                    # 使用 mode='fan_in' 更安全
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, context_vecs):
        """
        Args:
            context_vecs: [num_sentences, input_dim]
        Returns:
            logits: [num_sentences]
        """
        if context_vecs.size(0) == 0:
            return torch.tensor([], device=context_vecs.device)

        # 1. 计算文档全局上下文 (Mean Pooling)
        # doc_vec: [input_dim]
        doc_vec = torch.mean(context_vecs, dim=0)
        
        # 2. 扩展文档向量
        # doc_expanded: [num_sentences, input_dim]
        doc_expanded = doc_vec.unsqueeze(0).expand(context_vecs.size(0), -1)

        # 新增
        context_vecs = self.W(context_vecs)
        doc_expanded = self.U(doc_expanded)
        
        # 3. 拼接 (局部特征 + 全局特征)
        # combined: [num_sentences, hidden_dim * 2]
        combined = torch.cat([context_vecs, doc_expanded], dim=-1)
        
        # 4. MLP 打分
        logits = self.classifier(combined)
        
        return logits.squeeze(-1)