import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef  # 新增指标
import pickle
import matplotlib.pyplot as plt  # 用于绘制曲线图

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# --- 数据加载与预处理 ---
def load_sequences_and_labels(data_dir):
    sequences = []
    labels = []
    for root, _, files in os.walk(data_dir):
        label = os.path.basename(root)
        for file in files:
            if file.endswith(".seq"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    seq = f.read().replace(" ", "").replace("\n", "").upper()
                    sequences.append(seq)
                    labels.append(label)
    return sequences, labels


def clean_sequences(sequences):
    cleaned = []
    allowed_bases = {'A', 'T', 'C', 'G'}
    for seq in sequences:
        if all(c in allowed_bases for c in seq) and len(seq) >= 6:
            cleaned.append(seq)
    return cleaned


def process_sequences(sequences, k=6, pad_char="P"):
    processed = []
    for seq in sequences:
        seq_len = len(seq)
        padded_len = ((seq_len + k - 1) // k) * k
        padded_seq = seq + pad_char * (padded_len - seq_len)
        processed.append(padded_seq)
    return processed


def build_vocabulary(sequences, k=6, vocab_size=5500):
    all_kmers = []
    for seq in sequences:
        all_kmers.extend([seq[i:i + k] for i in range(0, len(seq), k)])
    vocab = Counter(all_kmers)
    filtered_kmers = [item[0] for item in vocab.most_common(vocab_size)]
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab_with_special = special_tokens + filtered_kmers
    tokenizer = {token: i for i, token in enumerate(vocab_with_special)}
    return tokenizer


# --- 数据集类 ---
class KmerDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=512, k=6):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.k = k

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        kmers = [sequence[i:i + self.k] for i in range(0, len(sequence), self.k)]

        input_ids = [self.tokenizer["[CLS]"]]
        for kmer in kmers:
            input_ids.append(self.tokenizer.get(kmer, self.tokenizer["[UNK]"]))
        input_ids.append(self.tokenizer["[SEP]"])

        padding_length = self.max_len - len(input_ids)
        input_ids += [self.tokenizer["[PAD]"]] * padding_length
        attention_mask = [1] * (len(kmers) + 2) + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# --- BERT模型定义 ---
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
        logits = self.classifier(pooled_output)
        return logits


# --- 训练函数（含AUC、MCC、曲线图）---
def train_model(model, train_loader, val_loader, epochs, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_acc = float("-inf")
    best_metrics = {}

    # 初始化记录指标的列表
    train_losses = []
    train_accuracies = []
    train_f1s = []
    train_precisions = []
    train_recalls = []
    train_aucs = []
    train_mccs = []

    val_losses = []
    val_accuracies = []
    val_f1s = []
    val_precisions = []
    val_recalls = []
    val_aucs = []
    val_mccs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []
        train_probas = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probas = torch.softmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels.cpu().numpy())
            train_probas.append(probas)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average='macro')
        train_precision = precision_score(train_true, train_preds, average='macro')
        train_recall = recall_score(train_true, train_preds, average='macro')

        # 计算AUC和MCC
        train_probas = np.vstack(train_probas)
        if num_classes == 2:
            train_auc = roc_auc_score(train_true, train_probas[:, 1])
            train_mcc = matthews_corrcoef(train_true, train_preds)
        else:
            train_auc = roc_auc_score(train_true, train_probas, multi_class='ovr', average='macro')
            train_mcc = float('nan')

        # 存储指标
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_aucs.append(train_auc)
        train_mccs.append(train_mcc)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train AUC: {train_auc:.4f} | Train MCC: {train_mcc:.4f}"
        )

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        val_probas = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probas = torch.softmax(logits, dim=1).detach().cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
                val_probas.append(probas)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='macro')
        val_precision = precision_score(val_true, val_preds, average='macro')
        val_recall = recall_score(val_true, val_preds, average='macro')

        # 计算AUC和MCC
        val_probas = np.vstack(val_probas)
        if num_classes == 2:
            val_auc = roc_auc_score(val_true, val_probas[:, 1])
            val_mcc = matthews_corrcoef(val_true, val_preds)
        else:
            val_auc = roc_auc_score(val_true, val_probas, multi_class='ovr', average='macro')
            val_mcc = float('nan')

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_aucs.append(val_auc)
        val_mccs.append(val_mcc)

        scheduler.step(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val AUC: {val_auc:.4f} | Val MCC: {val_mcc:.4f}")

        if best_acc < val_acc:
            best_acc = val_acc
            best_val_loss = avg_val_loss
            best_metrics = {
                "loss": avg_val_loss,
                "acc": val_acc,
                "f1": val_f1,
                "precision": val_precision,
                "recall": val_recall,
                "auc": val_auc,
                "mcc": val_mcc
            }
            torch.save(model.state_dict(), "kmer_bert_model.pth")

    print("\nBest Model Metrics:")
    for key, value in best_metrics.items():
        print(f"{key.upper()}: {value:.4f}")

    # 绘制曲线图
    plot_curves(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_f1s, val_f1s,
        train_precisions, val_precisions,
        train_recalls, val_recalls,
        train_aucs, val_aucs,
        train_mccs, val_mccs,
        epochs
    )


# --- 曲线图绘制函数 ---
def plot_curves(
    train_losses, val_losses,
    train_accuracies, val_accuracies,
    train_f1s, val_f1s,
    train_precisions, val_precisions,
    train_recalls, val_recalls,
    train_aucs, val_aucs,
    train_mccs, val_mccs,
    epochs
):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    # Loss
    axes[0].plot(range(1, epochs+1), train_losses, label='Train')
    axes[0].plot(range(1, epochs+1), val_losses, label='Val')
    axes[0].set_title('Loss Curve')
    axes[0].legend()

    # Accuracy
    axes[1].plot(range(1, epochs+1), train_accuracies, label='Train')
    axes[1].plot(range(1, epochs+1), val_accuracies, label='Val')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()

    # F1 Score
    axes[2].plot(range(1, epochs+1), train_f1s, label='Train')
    axes[2].plot(range(1, epochs+1), val_f1s, label='Val')
    axes[2].set_title('F1 Score Curve')
    axes[2].legend()

    # Precision
    axes[3].plot(range(1, epochs+1), train_precisions, label='Train')
    axes[3].plot(range(1, epochs+1), val_precisions, label='Val')
    axes[3].set_title('Precision Curve')
    axes[3].legend()

    # Recall
    axes[4].plot(range(1, epochs+1), train_recalls, label='Train')
    axes[4].plot(range(1, epochs+1), val_recalls, label='Val')
    axes[4].set_title('Recall Curve')
    axes[4].legend()

    # AUC
    axes[5].plot(range(1, epochs+1), train_aucs, label='Train')
    axes[5].plot(range(1, epochs+1), val_aucs, label='Val')
    axes[5].set_title('AUC Curve')
    axes[5].legend()

    # MCC
    axes[6].plot(range(1, epochs+1), train_mccs, label='Train')
    axes[6].plot(range(1, epochs+1), val_mccs, label='Val')
    axes[6].set_title('MCC Curve')
    axes[6].legend()

    # 关闭未使用的子图
    for ax in axes[7:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()


# --- 主函数流程 ---
if __name__ == "__main__":
    data_dir = "data/Final/Bert/train/"
    sequences, labels_str = load_sequences_and_labels(data_dir)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_str)
    num_classes = len(label_encoder.classes_)

    cleaned = clean_sequences(sequences)
    processed = process_sequences(cleaned, k=6, pad_char="P")

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        processed, labels, test_size=0.2, random_state=42
    )

    tokenizer = build_vocabulary(processed, k=6, vocab_size=5500)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    train_dataset = KmerDataset(train_seqs, train_labels, tokenizer, max_len=512, k=6)
    val_dataset = KmerDataset(val_seqs, val_labels, tokenizer, max_len=512, k=6)
    train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=192, shuffle=False)

    vocab_size = len(tokenizer)
    model = KmerBERT(vocab_size, num_classes=num_classes)
    train_model(model, train_loader, val_loader, epochs=25)

