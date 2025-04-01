import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# 处理文件函数
def process_file(file_path, label):
    df = pd.read_csv(file_path, header=None)
    data = df.stack().to_frame().T.values.reshape(1, 768, 510)
    return data, label

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        data, _ = process_file(file_path, label)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 收集文件路径和标签
pos_files = glob.glob(os.path.join('../data/train/pos', '*.csv'))
neg_files = glob.glob(os.path.join('../data/train/neg', '*.csv'))

data_files = pos_files + neg_files
labels = [1] * len(pos_files) + [0] * len(neg_files)

# 创建数据集
dataset = CustomDataset(data_files, labels)

# 划分训练集和验证集
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=64)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=64)

# 定义基于 DenseNet40 的模型
class DenseNet40Model(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet40Model, self).__init__()
        # 加载预训练的 DenseNet121 模型（因为没有直接的 DenseNet40，我们可以使用 DenseNet121 并修改）
        densenet = models.densenet121(pretrained=False)

        # 修改第一个卷积层以适应单通道输入
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改分类器部分
        num_features = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_features, num_classes)

        self.densenet = densenet

    def forward(self, x):
        return self.densenet(x)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet40Model(num_classes=2).to(device)
params_to_update = [param for param in model.parameters() if param.requires_grad]
optimizer = optim.Adam(params_to_update, lr=0.00005, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()
nb_epochs = 30

# Warm-up 调度器
warmup_epochs = 5
warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs)

# CosineAnnealingLR 调度器
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# 训练循环
best_val_accuracy = 0
print("Start training...")
for epoch in range(nb_epochs):
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    train_predictions = []
    train_labels = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
        train_predictions.extend(predicted.cpu().numpy())
        train_labels.extend(targets.cpu().numpy())

    train_accuracy = 100 * correct_train / total_train
    train_f1 = f1_score(train_labels, train_predictions, average='weighted')

    # 验证集评估
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(targets.cpu().numpy())

    val_accuracy = 100 * correct_val / total_val
    val_loss /= len(val_loader)
    val_f1 = f1_score(val_labels, val_predictions, average='weighted')
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch + 1}/{nb_epochs}], Loss: {running_loss / len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f}, "
          f"Learning Rate: {current_lr:.6f}")

    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'DenseNet_model_best.pth')
        print("Best model saved.")

print("Training complete.")
