import os
import torch
import numpy as np
from transformers import BertConfig, BertModel
import pickle


# --- 1. 修改后的模型类（返回所有token的隐藏层输出）---
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
        self.classifier = torch.nn.Linear(hidden_size, num_classes)  # 保留分类层

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # 获取所有token的隐藏层输出
        pooled_output = outputs.pooler_output  # 保留pooler_output用于分类
        return {
            "last_hidden_state": last_hidden_state,
            "logits": self.classifier(pooled_output)
        }


# --- 2. 特征提取函数（提取每个token的上下文特征）---
def extract_features(input_dir, output_dir, tokenizer_path, model_path, k=6, max_len=512):
    # 加载tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # 加载模型
    vocab_size = len(tokenizer)
    model = KmerBERT(vocab_size, num_classes=2)  # 假设训练时num_classes=2
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有.seq文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.seq'):
                file_path = os.path.join(root, file)
                # 读取序列并预处理
                with open(file_path, 'r') as f:
                    raw_seq = f.read().replace(" ", "").replace("\n", "").upper()

                # 预处理步骤（填充到k的倍数）
                seq_len = len(raw_seq)
                padded_len = ((seq_len + k - 1) // k) * k
                padded_seq = raw_seq + 'P' * (padded_len - seq_len)

                # 转换为k-mer
                kmers = [padded_seq[i:i + k] for i in range(0, len(padded_seq), k)]

                # 构建输入
                input_ids = [tokenizer["[CLS]"]]
                for kmer in kmers:
                    input_ids.append(tokenizer.get(kmer, tokenizer["[UNK]"]))
                input_ids.append(tokenizer["[SEP]"])

                # 填充到max_len
                padding_length = max_len - len(input_ids)
                input_ids += [tokenizer["[PAD]"]] * padding_length
                attention_mask = [1] * (len(input_ids) - padding_length) + [0] * padding_length

                # 转换为Tensor
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

                # 获取特征（所有token的隐藏层输出）
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs["last_hidden_state"].squeeze(0).cpu().numpy()

                # 保存特征（形状为 [max_len, 768]）
                output_path = os.path.join(output_dir, f"{file.split('.')[0]}.npy")
                np.save(output_path, last_hidden_state)
                print(f"Saved feature matrix for {file} to {output_path}")


# --- 3. 主函数执行 ---
if __name__ == "__main__":
    input_dir = "data/Final/test/nc_pos/"  # 需替换为你的输入文件夹路径
    output_dir = "processed/Final/test/nc_pos/"  # 特征保存路径
    tokenizer_path = "tokenizer.pkl"  # 训练时保存的tokenizer路径
    model_path = "kmer_bert_model.pth"  # 训练好的模型路径

    extract_features(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        k=6,
        max_len=512
    )
