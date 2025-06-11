import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class DNADataset(Dataset):
    def __init__(self, pos_bert_dir, neg_bert_dir,
                 pos_pse_csv_path, neg_pse_csv_path,
                 pos_rnafold_path, neg_rnafold_path):
        # 读取Pse特征和标签
        pos_pse = pd.read_csv(pos_pse_csv_path, header=None).values
        neg_pse = pd.read_csv(neg_pse_csv_path, header=None).values
        self.pse_features = np.concatenate([pos_pse, neg_pse]).astype(np.float32)

        # 读取RNAfold特征
        self.rnafold_features = np.concatenate([
            np.load(pos_rnafold_path),
            np.load(neg_rnafold_path)
        ]).astype(np.float32)

        # 生成标签
        self.labels = np.concatenate([
            np.ones(len(pos_pse)),
            np.zeros(len(neg_pse))
        ]).astype(np.int64)

        # 读取BERT特征路径
        self.bert_files = []
        for folder in [pos_bert_dir, neg_bert_dir]:
            files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
            self.bert_files.extend([os.path.join(folder, f) for f in files])

    def __len__(self):
        return len(self.bert_files)

    def __getitem__(self, idx):
        # 获取BERT特征
        bert_feat = np.load(self.bert_files[idx]).astype(np.float32)

        # 获取Pse特征
        pse_feat = self.pse_features[idx]

        # 获取RNAfold特征
        rnafold_feat = self.rnafold_features[idx]

        # 转换为Tensor
        return (
            torch.from_numpy(bert_feat),
            torch.from_numpy(pse_feat),
            torch.from_numpy(rnafold_feat),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class SEBlock(nn.Module):
    """通道注意力模块（SE Block）"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DynamicFusionGate(nn.Module):
    """动态门控融合模块"""
    def __init__(self, in_channels_a, in_channels_b):
        super().__init__()
        self.a_gate = nn.Conv2d(in_channels_a, 1, kernel_size=1)
        self.b_gate = nn.Conv2d(in_channels_b, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, b):
        a_weight = self.sigmoid(self.a_gate(a))
        b_weight = self.sigmoid(self.b_gate(b))
        fused = a * a_weight + b * b_weight
        return fused

class BERTProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        # 移除了 TransformerEncoder
        self.linear = nn.Linear(768, 256)  # 直接对BERT输出降维
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.se = SEBlock(256)

    def forward(self, x):
        # x 形状: (B, 512, 768) - 假设这是你任务特化BERT的直接输出
        x = self.linear(x)  # (B, 512, 256)
        x = x.permute(0, 2, 1)  # (B, 256, 512)
        x = self.pool(x)  # (B, 256, 64)
        x = x.unsqueeze(-1)  # (B, 256, 64, 1)
        return self.se(x)

class PSEProcessor(nn.Module):
    """PSE特征处理模块"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        self.upsample = nn.Upsample(size=(64, 1))
        self.se = SEBlock(256)

    def forward(self, x):
        # 输入形状：(batch_size, 64)
        x = self.mlp(x)  # (B,256)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B,256,1,1)
        x = self.upsample(x)  # (B,256,64,1)
        return self.se(x)

class RNAfoldProcessor(nn.Module):
    """RNAfold特征处理模块"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 64, kernel_size=5, padding=2)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=64, nhead=4),
            num_layers=3
        )
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.expand = nn.Conv1d(64, 256, kernel_size=1)
        self.se = SEBlock(256)

    def forward(self, x):
        # 输入形状：(batch_size, 3000, 3)
        x = x.permute(0, 2, 1)  # (B,3,3000)
        x = F.relu(self.conv(x))  # (B,64,3000)
        x = self.transformer(x.permute(2, 0, 1)).permute(1, 2, 0)  # (B,64,3000)
        x = self.pool(x).unsqueeze(-1)  # (B,64,64,1)
        x = self.expand(x.squeeze(-1)).unsqueeze(-1)  # (B,256,64,1)
        return self.se(x)

class CrossModalFusion(nn.Module):
    """跨模态融合模块"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1
        )
        self.dynamic_fusion = DynamicFusionGate(256, 256)
        self.se = SEBlock(256)

    def forward(self, bert, pse, rnafold):
        # 调整形状为 (batch_size, channels, seq_len)
        b = bert.view(bert.size(0), 256, 64)
        p = pse.view(pse.size(0), 256, 64)
        r = rnafold.view(rnafold.size(0), 256, 64)

        # 跨模态注意力融合
        # BERT与PSE
        fused_bp = self.cross_attn(
            b.permute(2, 0, 1),  # (seq_len, B, C)
            p.permute(2, 0, 1),
            p.permute(2, 0, 1)
        )[0].permute(1, 2, 0).view(b.shape)  # (B, C, seq_len)

        # BERT与RNAfold
        fused_br = self.cross_attn(
            b.permute(2, 0, 1),
            r.permute(2, 0, 1),
            r.permute(2, 0, 1)
        )[0].permute(1, 2, 0).view(b.shape)

        # 动态门控融合
        fused = self.dynamic_fusion(
            fused_bp.unsqueeze(-1),
            fused_br.unsqueeze(-1)
        )
        return self.se(fused)

class EfficientNetBackbone(nn.Module):
    """EfficientNet主干网络"""
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.efficientnet = models.efficientnet_b1(pretrained=False)
        # 调整输入通道数
        self.efficientnet.features[0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        # 替换分类头
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.efficientnet.classifier(x)

class MultiModalEfficientNet(nn.Module):
    """完整多模态模型"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert_proc = BERTProcessor()
        self.pse_proc = PSEProcessor()
        self.rnafold_proc = RNAfoldProcessor()
        self.fusion = CrossModalFusion()
        self.efficientnet = EfficientNetBackbone(in_channels=256, num_classes=num_classes)

    def forward(self, bert, pse, rnafold):
        # 处理各模态特征
        b = self.bert_proc(bert)
        p = self.pse_proc(pse)
        r = self.rnafold_proc(rnafold)

        # 融合特征
        fused = self.fusion(b, p, r)

        # 输入EfficientNet
        return self.efficientnet(fused)

def plot_confusion_matrix(cm, class_names, epoch, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.savefig(os.path.join(save_path, f'confusion_matrix_epoch_{epoch+1}.png'))
    plt.close()

def plot_metrics(train_metrics, val_metrics, metric_name, save_path):
    """绘制并保存训练和验证指标变化图"""
    plt.figure(figsize=(8, 5))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Val {metric_name}')
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{metric_name.lower()}_curve.png'))
    plt.close()

def main():
    # 数据路径
    POS_BERT_DIR = '../processed/Final/train/nc_pos'
    NEG_BERT_DIR = '../processed/Final/train/pc_neg'
    POS_PSE_CSV = '../processed/Final/F_nc_pos.csv'
    NEG_PSE_CSV = '../processed/Final/F_pc_neg.csv'
    POS_RNAFOLD = '../processed/Final/nc_pos.npy'
    NEG_RNAFOLD = '../processed/Final/pc_neg.npy'

    # 超参数
    batch_size = 256  # 由于特征维度增加，适当减小batch_size
    lr = 3e-4
    epochs = 85
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据集
    dataset = DNADataset(
        POS_BERT_DIR, NEG_BERT_DIR,
        POS_PSE_CSV, NEG_PSE_CSV,
        POS_RNAFOLD, NEG_RNAFOLD
    )

    # 数据集划分（保持原有逻辑）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(40)
    )

    # 修改数据加载器返回格式
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = MultiModalEfficientNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True, min_lr=1e-6)

    # 指标记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_recalls, val_recalls = [], []
    train_precisions, val_precisions = [], []
    train_aucs, val_aucs = [], []
    train_mccs, val_mccs = [], []

    best_val_loss = float('inf')
    best_val_f1 = float('-inf')
    best_val_acc = float('-inf')

    # 创建图像保存路径
    os.makedirs('curves/hybrid', exist_ok=True)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_labels, all_train_probs = [], [], []
        for batch in train_loader:
            bert, pse, rnafold, labels = batch
            bert = bert.to(device)
            pse = pse.to(device)
            rnafold = rnafold.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(bert, pse, rnafold)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bert.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_train_probs.extend(probs)
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels, all_val_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                bert, pse, rnafold, labels = batch
                bert = bert.to(device)
                pse = pse.to(device)
                rnafold = rnafold.to(device)
                labels = labels.to(device)

                outputs = model(bert, pse, rnafold)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * bert.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_val_probs.extend(probs)
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())

        # 指标计算
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds)
        train_recall = recall_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds)
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds)
        val_recall = recall_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        scheduler.step(val_loss)

        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_mccs.append(train_mcc)
        val_mccs.append(val_mcc)

        # 混淆矩阵（可选）
        cm = confusion_matrix(all_val_labels, all_val_preds)
        plot_confusion_matrix(cm, ['Negative', 'Positive'], epoch, 'curves/hybrid')

        # 打印结果
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(f"Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}")
        print(f"Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        print(f"Train MCC: {train_mcc:.4f} | Val MCC: {val_mcc:.4f}")
        print('-'*40)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_hybird.pth')

    # 绘制所有指标曲线
    plot_metrics(train_losses, val_losses, 'Loss', 'curves/hybrid')
    plot_metrics(train_accs, val_accs, 'Accuracy', 'curves/hybrid')
    plot_metrics(train_f1s, val_f1s, 'F1 Score', 'curves/hybrid')
    plot_metrics(train_recalls, val_recalls, 'Recall', 'curves/hybrid')
    plot_metrics(train_precisions, val_precisions, 'Precision', 'curves/hybrid')
    plot_metrics(train_aucs, val_aucs, 'AUC', 'curves/hybrid')
    plot_metrics(train_mccs, val_mccs, 'MCC', 'curves/hybrid')

    # 最终模型评估
    model.load_state_dict(torch.load('best_model_hybird.pth'))
    model.eval()

    val_loss = 0.0
    all_val_preds, all_val_labels, all_val_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            bert, pse, rnafold, labels = batch
            bert = bert.to(device)
            pse = pse.to(device)
            rnafold = rnafold.to(device)
            labels = labels.to(device)

            outputs = model(bert, pse, rnafold)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * bert.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_val_probs.extend(probs)
            all_val_preds.extend(preds)
            all_val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_dataset)
    val_acc = accuracy_score(all_val_labels, all_val_preds)
    val_f1 = f1_score(all_val_labels, all_val_preds)
    val_recall = recall_score(all_val_labels, all_val_preds)
    val_precision = precision_score(all_val_labels, all_val_preds)
    val_auc = roc_auc_score(all_val_labels, all_val_probs)
    val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)

    print("="*60)
    print("Final Evaluation on Best Model:")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print(f"Val Recall: {val_recall:.4f}")
    print(f"Val Precision: {val_precision:.4f}")
    print(f"Val AUC: {val_auc:.4f}")
    print(f"Val MCC: {val_mcc:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
