# src/model/decoder.py
import torch
import torch.nn as nn

class SequenceLabelingDecoder(nn.Module):
    """
    Sequence Labeling Decoder (MLP Classifier).
    
    功能：
    接收包含上下文信息的句子向量，对每一句话进行独立二分类打分。
    
    改进点：
    1. 加入 Dropout：防止过拟合，特别是在小样本（摘要句少）的情况下。
    2. 权重初始化：使用 Xavier/Kaiming 初始化，帮助 Logits 在初始阶段就有更大的方差，避免"数值过小"和"全0陷阱"。
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()

        # 定义 MLP 网络
        self.classifier = nn.Sequential(
            # Layer 1: 降维与特征整合
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),              # 激活函数
            nn.Dropout(dropout),    # [新增] 随机失活，增强鲁棒性
            
            # Layer 2: 输出 Logits
            nn.Linear(hidden_dim, 1)
        )
        
        # [新增] 显式初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化线性层权重，打破对称性，防止初始 Logits 过于趋同。
        """
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # Xavier Uniform 初始化 (适合 Tanh 激活函数)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, context_vecs):
        """
        Args:
            context_vecs: [num_sentences, input_dim] 
                          (来自 Encoder 的输出，通常维度是 512)
        Returns:
            logits: [num_sentences] (实数分数，未经过 Sigmoid/Softmax)
        """
        # 边界检查
        if context_vecs.size(0) == 0:
            return torch.tensor([], device=context_vecs.device)

        # 1. 前向传播
        # [num_sentences, input_dim] -> [num_sentences, 1]
        raw_output = self.classifier(context_vecs)
        
        # 2. 压缩维度
        # [num_sentences, 1] -> [num_sentences]
        logits = raw_output.squeeze(-1)
        
        return logits


# # src/model/decoder.py
# import torch
# import torch.nn as nn

# class SequenceLabelingDecoder(nn.Module):
#     """
#     Context-Aware MLP Decoder.
    
#     架构：
#     1. 输入句子向量 h_i
#     2. 计算文档均值向量 d (Global Context)
#     3. 拼接: [h_i; d]
#     4. MLP: Linear -> ReLU -> Dropout -> Linear
    
#     优势：
#     - ReLU 替代 Tanh，防止梯度消失和数值坍塌。
#     - 显式引入文档上下文，帮助模型打破 Lead Bias (位置偏见)。
#     """
#     def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
#         super().__init__()
        
#         # 因为我们要拼接文档向量，所以输入维度翻倍
#         concat_dim = input_dim * 2
        
#         self.classifier = nn.Sequential(
#             nn.Linear(concat_dim, hidden_dim), # [2*Dim] -> [Hidden]
#             nn.ReLU(),              # ✅ 改用 ReLU，梯度更健康
#             nn.Dropout(dropout),    # 防止过拟合
#             nn.Linear(hidden_dim, 1) # [Hidden] -> [1]
#         )
        
#         self._init_weights()

#     def _init_weights(self):
#         """Kaiming Initialization (专为 ReLU 设计)"""
#         for m in self.classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

#     def forward(self, context_vecs):
#         """
#         Args:
#             context_vecs: [num_sentences, input_dim]
#         Returns:
#             logits: [num_sentences]
#         """
#         # 边界检查
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


# import torch
# import torch.nn as nn

# class SequenceLabelingDecoder(nn.Module):
#     """
#     [Debug Version] 极简线性解码器
#     用于排查梯度流问题。如果这个版本能输出不同的 Logits，说明之前的 MLP 初始化或激活函数有问题。
#     """
#     def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
#         super().__init__()
#         # 暂时移除 Tanh, Dropout 和 Hidden Layer
#         # 直接从 [input_dim] 映射到 [1]
#         self.classifier = nn.Linear(input_dim, 1)

#     def forward(self, context_vecs):
#         # 打印输入的标准差，检查 Encoder 是否输出了相同的值
#         # 如果 std 接近 0，说明 Encoder 挂了（输出了全 0 或全常数）
#         # print(f"Decoder Input Std: {context_vecs.std(dim=0).mean().item():.6f}") 
        
#         logits = self.classifier(context_vecs)
#         return logits.squeeze(-1)