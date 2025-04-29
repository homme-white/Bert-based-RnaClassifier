import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Pse特征扩展
        self.pse_expand = nn.Sequential(
            nn.Linear(64, 64*16*32),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 32))
        )

        # 修改RNAfold特征处理部分
        self.rnafold_processor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),  # 处理3通道特征
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(512),  # 统一时间维度
            nn.Flatten(),
            nn.Linear(512*64, 64*16*32),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 32))
        )
         # BERT特征降维层
        self.bert_downsample = nn.Linear(768, 384)  # 降维到384通道

        # 注意力融合层（384+64+64=512）
        self.se = SEBlock(512)  # 输入通道为512

        # 修改EfficientNet输入通道为512
        self.efficientnet = models.efficientnet_b1(pretrained=False)
        self.efficientnet.features[0] = nn.Conv2d(
            in_channels=512,  # 从896改为512
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # 分类层（保持不变）
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, bert, pse, rnafold):
        # 处理BERT特征
        bert = self.bert_downsample(bert)  # 先降维到384
        bert = bert.view(-1, 384, 16, 32)  # 调整形状

        # 处理Pse特征（保持不变）
        pse = self.pse_expand(pse)

        # 处理RNAfold特征
        rnafold = rnafold.permute(0, 2, 1)
        rnafold = self.rnafold_processor(rnafold)  # 输出形状 (batch, 64, 16, 32)
        # 无需扩展维度（因为Unflatten已经正确设置为(64,16,32)）

        # 拼接特征
        fused = torch.cat([bert, pse, rnafold], dim=1)  # 384+64+64=512
        fused = self.se(fused)

        # 通过EfficientNet
        x = self.efficientnet.features(fused)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return x
def main():
    # 数据路径
    POS_BERT_DIR = '../processed/new_bert/lnc_train_com'
    NEG_BERT_DIR = '../processed/new_bert/pc_train_com'
    POS_PSE_CSV = 'ilearn_data/lnc_com_train.csv'
    NEG_PSE_CSV = 'ilearn_data/pc_com_train.csv'
    POS_RNAFOLD = 'rnafold/lnc_c.npy'
    NEG_RNAFOLD = 'rnafold/lnc_c.npy'

    # 超参数
    batch_size = 192  # 由于特征维度增加，适当减小batch_size
    lr = 2e-5
    epochs = 50
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
        generator=torch.Generator().manual_seed(42)
    )

    # 修改数据加载器返回格式
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    writer = SummaryWriter()

    # 修改训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for batch in train_loader:
            bert, pse, rnafold, labels = batch  # 新增rnafold
            bert = bert.to(device)
            pse = pse.to(device)
            rnafold = rnafold.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(bert, pse, rnafold)  # 传入三个特征
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bert.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 计算训练指标
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)

        # 验证
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                bert, pse, rnafold, labels = batch  # 新增rnafold
                bert = bert.to(device)
                pse = pse.to(device)
                rnafold = rnafold.to(device)
                labels = labels.to(device)
                outputs = model(bert, pse, rnafold)  # 传入三个特征
                loss = criterion(outputs, labels)

                val_loss += loss.item() * bert.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)

        # 记录指标
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalars('F1', {'train': train_f1, 'val': val_f1}, epoch)
        writer.add_scalars('Recall', {'train': train_recall, 'val': val_recall}, epoch)
        writer.add_scalars('Precision', {'train': train_precision, 'val': val_precision}, epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_hybird_atten.pth')

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(f"Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}")
        print(f"Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}")
        print('-'*40)

    # 训练结束后，加载最佳模型并输出其性能
    print("\n" + "="*60)
    print("Final Evaluation on Best Model:")

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model_hybird_atten.pth'))
    model.eval()

    # 重新计算验证集的指标
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            bert, pse, rnafold, labels = batch  # 新增rnafold
            bert = bert.to(device)
            pse = pse.to(device)
            rnafold = rnafold.to(device)
            labels = labels.to(device)
            outputs = model(bert, pse, rnafold)  # 传入三个特征
            loss = criterion(outputs, labels)

            val_loss += loss.item() * bert.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    val_recall = recall_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds)

    print(f"Best Model Validation Loss: {val_loss:.4f}")
    print(f"Best Model Validation Accuracy: {val_acc:.4f}")
    print(f"Best Model Validation F1 Score: {val_f1:.4f}")
    print(f"Best Model Validation Recall: {val_recall:.4f}")
    print(f"Best Model Validation Precision: {val_precision:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
