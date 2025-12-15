# src/utils.py
import json
import torch
import os

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