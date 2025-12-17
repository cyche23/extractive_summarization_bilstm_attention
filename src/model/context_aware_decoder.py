# src/model/decoder.py
import torch
import torch.nn as nn

class ContextAwareDecoder(nn.Module):
    """
    Context-Aware MLP Decoder (Fixed Version).
    修复日志：
    1. _init_weights 改为遍历 named_modules，确保覆盖 self.W 和 self.U。
    2. 将 nn.ReLU 替换为 nn.LeakyReLU 以匹配初始化策略并防止神经元死亡。
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        
        # 输入维度翻倍 (句子向量 + 文档向量)
        concat_dim = hidden_dim * 2

        # W: 变换句子向量 (Local)
        # 注意：这里会使用 PyTorch 默认初始化，稍后在 _init_weights 中被覆盖
        self.W = nn.Linear(input_dim, hidden_dim, bias=True)
        # U: 变换文档向量 (Global / Context)
        self.U = nn.Linear(input_dim, hidden_dim, bias=True)
        
        self.classifier = nn.Sequential(
            # Layer 1: Fusion -> Hidden
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),           
            nn.LeakyReLU(negative_slope=0.01),  # ✅ 修复：使用 Leaky 匹配初始化
            nn.Dropout(dropout),
            
            # Layer 2: Hidden -> Logit
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """
        精细化权重初始化，覆盖所有 Linear 层
        """
        # ✅ 修复：遍历 self.named_modules() 而不是 self.classifier
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # 判断是否是最后一层 (输出维度为1)
                if m.out_features == 1:
                    # 最后一层使用 Xavier Uniform，保证初始 Logits 接近 0
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # 中间层 (包括 self.W, self.U, classifier[0]) 使用 Kaiming Normal
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, context_vecs):
        if context_vecs.size(0) == 0:
            return torch.tensor([], device=context_vecs.device)

        # 1. 计算文档全局上下文 (Mean Pooling)
        doc_vec = torch.mean(context_vecs, dim=0)
        
        # 2. 扩展文档向量
        doc_expanded = doc_vec.unsqueeze(0).expand(context_vecs.size(0), -1)

        # 3. 投影 (现在这些层有正确的初始化了)
        context_vecs_proj = self.W(context_vecs)
        doc_expanded_proj = self.U(doc_expanded)
        
        # 4. 拼接
        combined = torch.cat([context_vecs_proj, doc_expanded_proj], dim=-1)
        
        # 5. MLP 打分
        logits = self.classifier(combined)
        
        return logits.squeeze(-1)