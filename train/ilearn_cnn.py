import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# -------------------
# 自定义数据集类
# -------------------
class CSVDataSet(Dataset):
    def __init__(self, data, targets):
        self.data = data.astype('float32')
        self.targets = targets.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx]).float()

# -------------------
# 1D MBConv 模块定义
# -------------------
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=6, stride=1, drop_rate=0.2):
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_ratio

        # 扩展层（Pointwise）
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU()  # Swish激活函数
        )

        # 深度可分离卷积（Depthwise）
        self.depthwise = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU()
        )

        # 点卷积（Pointwise）
        self.pointwise = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        # 残差连接（仅当输入输出通道匹配且步长为1时）
        self.residual = (in_channels == out_channels) and (stride == 1)

        # Dropout
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        identity = x

        # 扩展 → 深度卷积 → 点卷积
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.pointwise(x)

        # 应用Dropout
        x = self.dropout(x)

        # 残差连接
        if self.residual:
            x += identity
        return x

# -------------------
# 1D EfficientNetB1 模型
# -------------------
class EfficientNetB1_1D(nn.Module):
    def __init__(self, input_dim=64, num_classes=1):
        super().__init__()

        # 初始层（输入 → 32通道）
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )

        # 主干网络（根据EfficientNetB1的配置调整）
        self.blocks = nn.Sequential(
            # Stage 1: 输入32 → 输出16（扩展率1，步长1）
            MBConvBlock(32, 16, expansion_ratio=1, stride=1),

            # Stage 2: 输入16 → 输出24（扩展率6，步长2）
            MBConvBlock(16, 24, expansion_ratio=6, stride=2),
            MBConvBlock(24, 24, expansion_ratio=6, stride=1),  # 增加深度

            # Stage 3: 输入24 → 输出40（扩展率6，步长2）
            MBConvBlock(24, 40, expansion_ratio=6, stride=2),
            MBConvBlock(40, 40, expansion_ratio=6, stride=1),

            # Stage 4: 输入40 → 输出80（扩展率6，步长2）
            MBConvBlock(40, 80, expansion_ratio=6, stride=2),
            MBConvBlock(80, 80, expansion_ratio=6, stride=1),

            # Stage 5: 输入80 → 输出112（扩展率6，步长1）
            MBConvBlock(80, 112, expansion_ratio=6, stride=1),
            MBConvBlock(112, 112, expansion_ratio=6, stride=1),

            # Stage 6: 输入112 → 输出192（扩展率6，步长2）
            MBConvBlock(112, 192, expansion_ratio=6, stride=2),
            MBConvBlock(192, 192, expansion_ratio=6, stride=1),
            MBConvBlock(192, 192, expansion_ratio=6, stride=1)
        )

        # 全局池化和分类层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(192, num_classes),  # 输出通道数对应最后一层的通道数
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 64)  # 确保输入形状为 (batch, 1, 64)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze()  # 输出形状为 [batch_size]

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = (outputs >= 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        all_probs.extend(outputs.detach().cpu().numpy())  # 保存概率值

    avg_loss = total_loss / len(loader.dataset)

    # 计算指标
    train_f1 = f1_score(all_targets, all_preds)
    train_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    train_precision = precision_score(all_targets, all_preds)
    train_recall = recall_score(all_targets, all_preds)

    # 新增AUC和MCC
    train_auc = roc_auc_score(np.array(all_targets).astype(int), np.array(all_probs))
    train_mcc = matthews_corrcoef(np.array(all_targets).astype(int), np.array(all_preds).astype(int))

    return avg_loss, train_f1, train_acc, train_precision, train_recall, train_auc, train_mcc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())  # 保存概率值

    avg_loss = total_loss / len(loader.dataset)

    # 计算指标
    val_f1 = f1_score(all_targets, all_preds)
    val_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    val_precision = precision_score(all_targets, all_preds)
    val_recall = recall_score(all_targets, all_preds)

    # 新增AUC和MCC
    val_auc = roc_auc_score(np.array(all_targets).astype(int), np.array(all_probs))
    val_mcc = matthews_corrcoef(np.array(all_targets).astype(int), np.array(all_preds).astype(int))

    return avg_loss, val_f1, val_acc, val_precision, val_recall, val_auc, val_mcc

# -------------------
# 主程序
# -------------------
def main():
    # 1. 数据加载与预处理
    positive_data = pd.read_csv('../processed/Final/F_nc_pos.csv', header=None).values
    negative_data = pd.read_csv('../processed/Final/F_pc_neg.csv', header=None).values
    X = np.concatenate([positive_data, negative_data], axis=0)
    y = np.array([1]*24000 + [0]*24000)

    # 打乱数据
    indices = np.random.permutation(X.shape[0])
    X, y = X[indices], y[indices]

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化到0-1范围
    max_val = X_train.max()
    print(max_val)
    X_train = X_train / max_val
    X_val = X_val / max_val

    # 创建数据加载器
    train_dataset = CSVDataSet(X_train, y_train)
    val_dataset = CSVDataSet(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 初始化模型、优化器和损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB1_1D().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # 新增：最佳模型跟踪
    best_val_f1 = 0.0
    best_metrics = {
        'accuracy': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'auc': 0.0,
        'mcc': 0.0
    }

    # 新增：记录训练过程指标
    train_losses, train_f1s, train_accs, train_precisions, train_recalls, train_aucs, train_mccs = [], [], [], [], [], [], []
    val_losses, val_f1s, val_accs, val_precisions, val_recalls, val_aucs, val_mccs = [], [], [], [], [], [], []

    # 训练循环
    num_epochs = 85
    for epoch in range(num_epochs):
        train_loss, train_f1, train_acc, train_precision, train_recall, train_auc, train_mcc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_acc, val_precision, val_recall, val_auc, val_mcc = evaluate(model, val_loader, criterion, device)

        # 更新最佳模型指标
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics['accuracy'] = val_acc
            best_metrics['f1'] = val_f1
            best_metrics['precision'] = val_precision
            best_metrics['recall'] = val_recall
            best_metrics['auc'] = val_auc
            best_metrics['mcc'] = val_mcc
            # 保存模型参数
            torch.save(model.state_dict(), 'best_model_ilearn.pth')

        # 添加到记录列表
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        train_accs.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_aucs.append(train_auc)
        train_mccs.append(train_mcc)

        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        val_accs.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_aucs.append(val_auc)
        val_mccs.append(val_mcc)

        # 打印每轮结果
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train: Loss {train_loss:.4f} | F1 {train_f1:.4f} | Acc {train_acc:.4f} | Precision {train_precision:.4f} | Recall {train_recall:.4f} | AUC {train_auc:.4f} | MCC {train_mcc:.4f}")
        print(f"Val:   Loss {val_loss:.4f} | F1 {val_f1:.4f} | Acc {val_acc:.4f} | Precision {val_precision:.4f} | Recall {val_recall:.4f} | AUC {val_auc:.4f} | MCC {val_mcc:.4f}")
        print('-'*30)

    # 绘制训练过程曲线
    epochs = range(1, num_epochs+1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig('curves/ilearn/loss_curve.png')

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_f1s, label='Train F1')
    plt.plot(epochs, val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Training and Validation F1 Score')
    plt.grid(True)
    plt.savefig('curves/ilearn/f1_curve.png')

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.savefig('curves/ilearn/accuracy_curve.png')

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, val_precisions, label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Training and Validation Precision')
    plt.grid(True)
    plt.savefig('curves/ilearn/precision_curve.png')

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Training and Validation Recall')
    plt.grid(True)
    plt.savefig('curves/ilearn/recall_curve.png')

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_aucs, label='Train AUC')
    plt.plot(epochs, val_aucs, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Training and Validation AUC')
    plt.grid(True)
    plt.savefig('curves/ilearn/auc_curve.png')

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_mccs, label='Train MCC')
    plt.plot(epochs, val_mccs, label='Val MCC')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()
    plt.title('Training and Validation MCC')
    plt.grid(True)
    plt.savefig('curves/ilearn/mcc_curve.png')

    # 训练结束后打印最佳模型参数
    print("\nBest Model Metrics:")
    print(f"max_val: {max_val}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"AUC: {best_metrics['auc']:.4f}")
    print(f"MCC: {best_metrics['mcc']:.4f}")

if __name__ == "__main__":
    main()
