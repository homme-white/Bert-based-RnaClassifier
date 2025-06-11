import os
import torch
import numpy as np
from collections import Counter
from transformers import BertConfig, BertModel
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # 添加进度条库

# --- 1. 模型类（与训练时一致）---
class KmerBERT(torch.nn.Module):
    def __init__(self, vocab_size, num_classes=2, hidden_size=768, num_layers=12):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=12,
            max_position_embeddings=512
        )
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return {
            "logits": self.classifier(pooled_output)
        }

# --- 2. 预测函数 ---
def predict_sequence(file_path, tokenizer, model, k=6, max_len=512):
    with open(file_path, 'r') as f:
        raw_seq = f.read().replace(" ", "").replace("\n", "").upper()

    # 预处理步骤
    seq_len = len(raw_seq)
    padded_len = ((seq_len + k - 1) // k) * k
    padded_seq = raw_seq + 'P' * (padded_len - seq_len)

    kmers = [padded_seq[i:i+k] for i in range(0, len(padded_seq), k)]

    # 构建input_ids
    input_ids = [tokenizer["[CLS]"]]
    for kmer in kmers:
        input_ids.append(tokenizer.get(kmer, tokenizer["[UNK]"]))
    input_ids.append(tokenizer["[SEP]"])

    # 填充到max_len
    padding_length = max_len - len(input_ids)
    input_ids += [tokenizer["[PAD]"]] * padding_length
    attention_mask = [1]*(len(input_ids)-padding_length) + [0]*padding_length

    # 转换为Tensor并预测
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        pred = torch.argmax(logits, dim=1).item()

    return pred

# --- 3. 主函数执行 ---
if __name__ == "__main__":
    # 配置参数（需替换为实际路径）
    positive_dir = "data/Final/test/nc_pos"  # 正样本文件夹路径
    negative_dir = "data/Final/test/pc_neg"  # 负样本文件夹路径
    tokenizer_path = "tokenizer.pkl"  # 分词器路径
    model_path = "kmer_bert_model.pth"  # 模型路径

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载分词器
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # 加载模型
    vocab_size = len(tokenizer)
    model = KmerBERT(vocab_size, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 收集所有文件路径并显示进度
    positive_files = []
    for root, _, files in os.walk(positive_dir):
        for file in files:
            if file.endswith('.seq'):
                positive_files.append(os.path.join(root, file))

    negative_files = []
    for root, _, files in os.walk(negative_dir):
        for file in files:
            if file.endswith('.seq'):
                negative_files.append(os.path.join(root, file))

    # 初始化标签列表
    true_labels = []
    pred_labels = []

    # 处理正样本（显示进度条）
    print("Processing positive samples...")
    for file_path in tqdm(positive_files, desc="Positive Samples"):
        pred = predict_sequence(file_path, tokenizer, model)
        true_labels.append(1)
        pred_labels.append(pred)

    # 处理负样本（显示进度条）
    print("\nProcessing negative samples...")
    for file_path in tqdm(negative_files, desc="Negative Samples"):
        pred = predict_sequence(file_path, tokenizer, model)
        true_labels.append(0)
        pred_labels.append(pred)

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # 输出结果
    print("\n\nFinal Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
