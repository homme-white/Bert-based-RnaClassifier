import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# -------------------
# 数据集类
# -------------------
class RNA1DDataset(Dataset):
    def __init__(self, pos_data, neg_data):
        self.data = np.concatenate([pos_data, neg_data], axis=0)
        self.labels = np.array([1]*len(pos_data) + [0]*len(neg_data))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])  # 形状 (channels=3, seq_len=3000)
        y = torch.LongTensor([self.labels[idx]])
        return x, y

# -------------------
# MBConv模块定义
# -------------------
class MBConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.drop_path = nn.Identity()

    def forward(self, x):
        if self.use_residual:
            return x + self.drop_path(self.conv(x))
        else:
            return self.conv(x)

# -------------------
# EfficientNet-1D模型
# -------------------
class EfficientNet1D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # MBConv配置
        width_mult = 1.0
        depth_mult = 1.1
        dropout_rate = 0.2

        out_channels = self._make_divisible(32 * width_mult)

        # Stem层
        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.SiLU()
        )

        # MBConv模块堆叠
        self.blocks = nn.ModuleList([
            self._make_mbconv(out_channels, 16, 1, 1, depth_mult),
            self._make_mbconv(16, 24, 2, 6, depth_mult),
            self._make_mbconv(24, 40, 2, 6, depth_mult),
            self._make_mbconv(40, 80, 3, 6, depth_mult),
            self._make_mbconv(80, 112, 3, 6, depth_mult),
            self._make_mbconv(112, 192, 4, 6, depth_mult),
            self._make_mbconv(192, 320, 1, 6, depth_mult),
            nn.Dropout(p=0.2)
        ])

        # Head层
        in_channels = self._make_divisible(320 * width_mult)
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm1d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes)
        )

    def _make_divisible(self, v, divisor=8):
        return int(np.ceil(v / divisor) * divisor)

    def _make_mbconv(self, in_channels, out_channels, repeats, expand_ratio, depth_mult):
        layers = []
        num_repeats = int(repeats * depth_mult)
        for i in range(num_repeats):
            stride = 2 if i == 0 else 1
            layers.append(MBConv1D(in_channels if i == 0 else out_channels,
                                out_channels,
                                expand_ratio=expand_ratio,
                                stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

# -------------------
# 主程序
# -------------------
def main():
    # 加载数据
    pos = np.load('../processed/Final/nc_pos.npy').transpose(0, 2, 1)  # 转换为(7000, 3, 3000)
    neg = np.load('../processed/Final/pc_neg.npy').transpose(0, 2, 1)   # 确保原始数据是(channels, seq_len)
    dataset = RNA1DDataset(pos, neg)

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=64, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=64, pin_memory=True)

  # 初始化模型和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet1D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    # 显式计算总步数
    num_epochs = 50
    total_batches_per_epoch = len(train_loader)
    total_steps = total_batches_per_epoch * num_epochs

    # 初始化调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps  # ✅ 显式指定总步数
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # 初始化最佳模型记录
    best_metrics = {
        'epoch': -1,
        'loss': float('inf'),
        'acc': 0,
        'f1': 0,
        'precision': 0,
        'recall': 0,
        'auc': 0,
        'mcc': 0,
        'state_dict': None
    }

    # 记录训练过程
    train_losses, train_accs, train_f1s, train_precisions, train_recalls, train_aucs, train_mccs = [], [], [], [], [], [], []
    val_losses, val_accs, val_f1s, val_precisions, val_recalls, val_aucs, val_mccs = [], [], [], [], [], [], []

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_labels, all_train_probs = [], [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).flatten()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(x)
                loss = criterion(outputs, y)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            train_loss += loss.item() * x.size(0)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())
            all_train_probs.extend(probs.detach()[:,1].cpu().numpy())

        # 计算训练指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds)
        train_recall = recall_score(all_train_labels, all_train_preds)
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels, all_val_probs = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).flatten()

                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(x)
                    loss = criterion(outputs, y)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                val_loss += loss.item() * x.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())
                all_val_probs.extend(probs.detach()[:,1].cpu().numpy())

        # 计算验证指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds)
        val_recall = recall_score(all_val_labels, all_val_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)

        # 更新最佳模型
        if val_loss < best_metrics['loss']:
            best_metrics = {
                'epoch': epoch + 1,
                'loss': val_loss,
                'acc': val_acc,
                'f1': val_f1,
                'precision': val_precision,
                'recall': val_recall,
                'auc': val_auc,
                'mcc': val_mcc,
                'state_dict': model.state_dict()
            }

        # 记录训练过程
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_aucs.append(train_auc)
        train_mccs.append(train_mcc)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_aucs.append(val_auc)
        val_mccs.append(val_mcc)

        # 打印当前epoch结果
        print(f"\nEpoch {epoch+1}/50")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(f"Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}")
        print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        print(f"Train MCC: {train_mcc:.4f} | Val MCC: {val_mcc:.4f}")
        print("-" * 60)

    # 保存最佳模型
    torch.save(best_metrics['state_dict'], 'best_model_fold.pth')

    # 绘制训练过程曲线
    epochs = range(1, 51)

    metrics = [
        ('loss', train_losses, val_losses),
        ('accuracy', train_accs, val_accs),
        ('f1 score', train_f1s, val_f1s),
        ('precision', train_precisions, val_precisions),
        ('recall', train_recalls, val_recalls),
        ('AUC', train_aucs, val_aucs),
        ('MCC', train_mccs, val_mccs)
    ]

    for name, train_vals, val_vals in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_vals, label=f'Train {name}')
        plt.plot(epochs, val_vals, label=f'Val {name}')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title(f'Training and Validation {name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'curves/fold/{name}_curve.png')
        plt.close()

    # 输出最佳模型结果
    print("\n\n=== Best Model Performance ===")
    print(f"At Epoch {best_metrics['epoch']}:")
    print(f"Validation Loss: {best_metrics['loss']:.4f}")
    print(f"Accuracy: {best_metrics['acc']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"AUC: {best_metrics['auc']:.4f}")
    print(f"MCC: {best_metrics['mcc']:.4f}")

if __name__ == "__main__":
    main()
