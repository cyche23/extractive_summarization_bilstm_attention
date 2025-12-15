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
