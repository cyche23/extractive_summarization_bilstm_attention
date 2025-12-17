# src/utils.py
import json
import torch
import os
import numpy as np
from rouge_score import rouge_scorer

def save_json(obj, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model

def print_monitor_info(record_idx, record_label, logits_list, count):
    """
    打印评估过程中的监测数据 (Logits, 预测结果, 真实标签)
    """
    if count == 0:
        return

    print("\n" + "="*20 + " [Monitor Info Start] " + "="*20)
    
    print(">>> Record Index (Prediction) and Label:")
    for i in range(count):
        # 这里的 f-string 只是为了让输出更整齐，保留了你原本的逻辑
        print(f"Sample {i}: Pred_Idx={record_idx[i]}, True_Label={record_label[i]}")
    
    print("-" * 50)
    
    print(">>> Logits Data:")
    for i in range(count):
        print(f"Sample {i} Logits:")
        print(logits_list[i])
        
    print("="*20 + " [Monitor Info End] " + "="*20 + "\n")

# ========================================================
# [新增模块] 模型权重与梯度健康监测
# ========================================================
def monitor_model_weights(model):
    """
    监测模型各层的权重分布、梯度情况和异常值。
    重点关注: Decoder (分类头) 是否坍塌，LSTM 梯度是否消失。
    """
    print(f"\n{'='*20} Model Health Monitor {'='*20}")
    print(f"{'Layer Name':<40} | {'Mean':<8} | {'Std':<8} | {'Grad Norm':<10} | {'Status'} | {'Slice (First 5)'}")
    print("-" * 110)

    has_nan = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            grad = param.grad
            
            # 1. 基础统计
            mean_val = data.mean().item()
            std_val = data.std().item()
            min_val = data.min().item()
            max_val = data.max().item()
            
            # 2. 梯度统计
            grad_norm = 0.0
            if grad is not None:
                grad_norm = grad.norm().item()
            
            # 3. 状态检查
            status = "OK"
            if torch.isnan(data).any() or torch.isinf(data).any():
                status = "NaN/Inf DETECTED!"
                has_nan = True
            elif std_val < 1e-6:
                status = "COLLAPSED (Zero Var)"
            elif grad is not None and grad_norm < 1e-9:
                status = "Vanishing Grad"
            
            # 4. 切片采样 (取前5个数值)
            # view(-1) 把 tensor 展平，避免维度不同导致的打印问题
            slice_vals = data.view(-1)[:5].cpu().numpy()
            slice_str = str(np.round(slice_vals, 4))

            # 仅打印关键层（为了避免刷屏，过滤掉太细碎的bias，但保留decoder的所有参数）
            # 或者你可以选择打印所有层
            is_important = "decoder" in name or "sent_lstm" in name or "encoder" in name
            
            if is_important:
                print(f"{name:<40} | {mean_val:8.4f} | {std_val:8.4f} | {grad_norm:10.4f} | {status:<15} | {slice_str}")

    print("-" * 110)
    if has_nan:
        print("[CRITICAL WARNING] Model parameters contain NaN or Inf! Training is likely broken.")
        # 可以选择在这里 sys.exit(1)
    print("\n")

def debug_lead3_data(dataloader):
    print("\n=== DEBUG LEAD-3 DATA ===")
    batch = next(iter(dataloader))
    
    raw_sents = batch["raw_sents"][0] # 取第一个样本
    refs = batch["highlights"][0]     # 取第一个样本的摘要
    
    # Lead-3 预测
    pred_sents = raw_sents[:3]
    
    print(f"[Reference Count]: {len(refs)}")
    print(f"[Ref Example]: {refs[0] if refs else 'EMPTY'}")
    print("-" * 20)
    print(f"[Lead-3 Count]: {len(pred_sents)}")
    print(f"[Lead-3 Example]: {pred_sents[0] if pred_sents else 'EMPTY'}")
    print("-" * 20)
    
    # 打印长度对比
    pred_text = "\n".join(pred_sents)
    ref_text = "\n".join(refs)
    print(f"[Pred Length (char)]: {len(pred_text)}")
    print(f"[Ref Length (char)]: {len(ref_text)}")
    
    # 重新计算一次分数 (开启 Stemmer)
    debug_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeLsum'], use_stemmer=True)
    scores = debug_scorer.score(pred_text, ref_text)
    print(f"[Debug Score with Stemmer] R1: {scores['rouge1'].fmeasure:.4f} | RLsum: {scores['rougeLsum'].fmeasure:.4f}")
    print("=========================\n")

# 在 main() 中调用:
# debug_lead3_data(val_loader)