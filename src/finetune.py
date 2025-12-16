# src/finetune.py
import os
import argparse
import json
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import optuna
from torch.utils.data import DataLoader
import numpy as np

# 从 train.py 导入核心逻辑
from dataset import SummDataset, collate_fn
from model.model import ExtractiveSummarizer
from train import train_epoch, eval_epoch, get_cache_path, setup_seed

# 配置日志等级
optuna.logging.set_verbosity(optuna.logging.INFO)

class HyperparameterTuner:
    def __init__(self, args):
        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 1. 预加载数据
        print("Pre-loading datasets for tuning...")
        train_cache = get_cache_path(args.train_json)
        val_cache = get_cache_path(args.val_json)
        
        self.train_dataset = SummDataset(args.train_json, build_vocab=True, cache_path=train_cache)
        self.val_dataset = SummDataset(args.val_json, vocab=self.train_dataset.vocab, build_vocab=False, cache_path=val_cache)
        
        self.vocab = self.train_dataset.vocab
        print(f"Data loaded. Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

    def objective(self, trial):
        # ==========================================
        # 让每个 Trial 初始化不同，避免陷入完全相同的局部最优
        # ==========================================
        # current_seed = 42 + trial.number
        setup_seed(42)
        
        # A. 定义超参数空间
        lr_enc = trial.suggest_float("lr_encoder", 1e-4, 1e-3, log=True)
        lr_dec = trial.suggest_float("lr_decoder", 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
        
        batch_size = self.args.batch_size

        # B. 初始化加载器 (Shuffle 受 seed 影响)
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # C. 初始化模型
        model = ExtractiveSummarizer(
            self.vocab, 
            embed_dim=300, 
            hidden_size=256, 
            glove_path=self.args.glove_path
        ).to(self.device)
        
        # 强行修改 dropout
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_rate
        
        # D. 优化器
        optimizer = torch.optim.Adam([
            {"params": model.embedding.parameters(), "lr": 1e-4}, # Embedding 几乎冻结
            {"params": model.encoder.parameters(), "lr": lr_enc},
            {"params": model.decoder.parameters(), "lr": lr_dec},
        ], weight_decay=weight_decay)

        best_rouge_l = 0.0
        
        # E. 训练循环
        for epoch in range(1, self.args.trial_epochs + 1):
            # 训练并获取 Loss
            train_loss = train_epoch(model, train_loader, optimizer, self.device)
            
            # 验证
            val_scores = eval_epoch(model, val_loader, self.device, strategy="topk", debug=True)
            current_rl = val_scores['rl_f']
            # current_rl = val_scores['r1_f']
            
            # [关键修改] 打印 Loss，证明模型真的在发生不同的变化
            print(f"[Trial {trial.number} | Ep {epoch}] Loss: {train_loss:.4f} | ROUGE-L: {current_rl:.4f}")
            
            # 向 Optuna 报告
            trial.report(current_rl, epoch)
            
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()
            
            if current_rl > best_rouge_l:
                best_rouge_l = current_rl
        
        return best_rouge_l

    def run(self):
        study = optuna.create_study(direction="maximize", study_name="summarization_finetune")
        
        print(f"Start tuning for {self.args.n_trials} trials...")
        study.optimize(self.objective, n_trials=self.args.n_trials)
        
        print("\n" + "="*30)
        print("Tuning Completed!")
        print("Best ROUGE-L:", study.best_value)
        print("Best Hyperparameters:")
        print(json.dumps(study.best_params, indent=2))
        print("="*30 + "\n")
        
        return study

def plot_results(study, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        # 绘制 LR 关系图
        fig = optuna.visualization.matplotlib.plot_slice(study, params=["lr_encoder", "lr_decoder"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr_vs_score.png"))
        print(f"Saved LR visualization to {output_dir}/lr_vs_score.png")
        
        # 绘制参数重要性
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "param_importance.png"))
    except Exception as e:
        print(f"Plotting error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--glove_path", default=None)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--trial_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="tuning_results")
    
    args = parser.parse_args()
    
    tuner = HyperparameterTuner(args)
    study = tuner.run()
    
    plot_results(study, args.output_dir)
    
    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

if __name__ == "__main__":
    main()