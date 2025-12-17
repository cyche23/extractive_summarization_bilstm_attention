# src/model/bilstm_encoder.py
import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    """
    Hierarchical Encoder:
    1. Word-Level BiLSTM: Encode words -> Sentence Vectors
    2. Sentence-Level BiLSTM: Encode sentence vectors -> Context-aware Sentence Representations
    
    Input:
        x: [num_sentences, max_len, embed_dim] (Note: assumes x belongs to ONE document context)
        lengths: list or tensor of lengths (num_sentences,)
    Output:
        context_vecs: [num_sentences, sent_hidden_size*2]
    """
    def __init__(self, embed_dim=300, hidden_size=256, sent_hidden_size=256, dropout=0.2):
        super().__init__()
        
        # === 1. Word-Level LSTM (原有部分) ===
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True, 
            dropout=dropout,
        )
        self.input_dropout = nn.Dropout(dropout)
        
        # === 2. Sentence-Level LSTM (新增部分) ===
        # 输入维度 = Word-LSTM 的输出维度 (hidden_size * 2，因为它是双向的)
        self.sent_lstm = nn.LSTM(
            input_size=hidden_size * 2,  
            hidden_size=sent_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.output_dropout = nn.Dropout(dropout)
        final_dim = (sent_hidden_size if sent_hidden_size else hidden_size) * 2
        self.layer_norm = nn.LayerNorm(final_dim)

    def forward(self, x, lengths):
        """
        x: FloatTensor [num_sentences, max_word_len, embed_dim]
        lengths: LongTensor [num_sentences]
        """
        # ===========================
        # Part 1: Word-Level Encoding (保持原有逻辑)
        # ===========================
        x = self.input_dropout(x)
        
        # move lengths to cpu for packing
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        packed_out, (h_n, c_n) = self.lstm(packed)
        
        # 获取句子向量 [num_sentences, word_hidden*2]
        forward_h = h_n[-2, :, :] 
        backward_h = h_n[-1, :, :]
        sent_vecs = torch.cat([forward_h, backward_h], dim=-1) 
        
        # ===========================
        # Part 2: Sentence-Level Encoding (新增逻辑)
        # ===========================
        # 此时 sent_vecs 的形状是 [num_sentences, word_hidden*2]
        # LSTM 需要 3D 输入 [Batch, Seq_Len, Dim]
        # 我们将这组句子视为"1篇文章"，即 Batch=1，Seq_Len=num_sentences
        
        # 1. 增加 Batch 维度 -> [1, num_sentences, dim]
        sent_seq = sent_vecs.unsqueeze(0)
        
        # 2. 放入 Sentence-LSTM 建模上下文
        # out: [1, num_sentences, sent_hidden*2]
        context_vecs, _ = self.sent_lstm(sent_seq)
        
        # 3. 移除 Batch 维度还原 -> [num_sentences, sent_hidden*2]
        context_vecs = context_vecs.squeeze(0)

        # 4. LayerNorm
        context_vecs = self.layer_norm(context_vecs)
        
        # 5. Dropout
        context_vecs = self.output_dropout(context_vecs)
        
        return context_vecs
