# Extractive Summarization with BiLSTM + Attention

这是一个基于BiLSTM和注意力机制的提取式文本摘要系统，使用GloVe词嵌入进行句子编码，通过注意力机制选择最重要的句子生成摘要，并使用ROUGE-1和ROUGE-L进行评估。

## 项目功能

- **数据预处理**：清洗CNN/DailyMail数据集，句子分割，标签对齐
- **模型训练**：使用BiLSTM编码器和注意力解码器训练摘要模型
- **模型评估**：在测试集上计算ROUGE分数
- **超参数调优**：使用Optuna进行自动超参数搜索
- **推理**：生成提取式摘要

## 文件结构

```
extractive-summarization-bilstm-attention/
│
├── data/
│ ├── labeled_stories/          # 处理后的训练数据
│ ├── labeled_stories_mini/     # 小型数据集用于快速测试
│ ├── cnn_dm_extractive/         # CNN/DM提取式摘要数据
│
├── src/
│ ├── preprocess.py             # 数据预处理脚本
│ ├── train.py                  # 模型训练脚本
│ ├── test.py                   # 模型测试和评估脚本
│ ├── finetune.py               # 超参数调优脚本
│ ├── model/                    # 模型组件
│ │ ├── embedding.py            # GloVe嵌入层
│ │ ├── bilstm_encoder.py       # BiLSTM句子编码器
│ │ ├── attention.py            # 注意力机制
│ │ └── model.py                # 完整模型
│ ├── dataset.py                # 数据加载器
│ ├── utils.py                  # 工具函数
│ └── evaluate.py               # 评估函数
│
├── checkpoints/                 # 训练好的模型权重
├── experiments/                 # 实验结果
├── requirements.txt             # 依赖包
└── README.md
```

## 运行方式

### 1. 数据预处理 (preprocess.py)

预处理原始CNN/DailyMail数据，生成句子级别的训练数据。

```bash
python src/preprocess.py --raw_path /path/to/raw/data --output_dir /path/to/outputdir --shard_size 5000 --seed 42
```

### 2. 模型训练 (train.py)

训练提取式摘要模型。

```bash
python src/train.py --train_json /path/to/json/train --val_json /path/to/json/val --glove_path /path/to/glove.6B.300d.txt --epochs 10 --batch_size 16 --save_path checkpoints/model.pt --patience 5
```

### 3. 模型测试 (test.py)

使用训练好的模型进行测试和评估。

```bash
python src/test.py --test_json /path/to/json/test --vocab_source /path/to/json/train.pkl --model_path checkpoints/model.pt --batch_size 1 --strategy topk
```

### 4. 超参数调优 (finetune.py)

使用Optuna进行超参数搜索。

```bash
python src/finetune.py --train_json /path/to/json/train --val_json /path/to/json/val --glove_path /path/to/glove.6B.300d.txt --n_trials 20 --trial_epochs 5 --batch_size 16 --output_dir /path/to/tuning_results
```

## 依赖

安装所需依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：torch, numpy, nltk, rouge-score, optuna, tqdm 等。


