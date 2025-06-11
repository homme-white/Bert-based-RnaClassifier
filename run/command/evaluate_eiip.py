import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score

# -------------------
# 复制模型定义部分
# -------------------
class CSVDataSet(Dataset):
    def __init__(self, data, targets):
        self.data = data.astype('float32')
        self.targets = targets.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx]).float()

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=6, stride=1, drop_rate=0.2):
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_ratio
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU()
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.dropout(x)
        if self.residual:
            x += identity
        return x

class EfficientNetB1_1D(nn.Module):
    def __init__(self, input_dim=64, num_classes=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expansion_ratio=1, stride=1),
            MBConvBlock(16, 24, expansion_ratio=6, stride=2),
            MBConvBlock(24, 24, expansion_ratio=6, stride=1),
            MBConvBlock(24, 40, expansion_ratio=6, stride=2),
            MBConvBlock(40, 40, expansion_ratio=6, stride=1),
            MBConvBlock(40, 80, expansion_ratio=6, stride=2),
            MBConvBlock(80, 80, expansion_ratio=6, stride=1),
            MBConvBlock(80, 112, expansion_ratio=6, stride=1),
            MBConvBlock(112, 112, expansion_ratio=6, stride=1),
            MBConvBlock(112, 192, expansion_ratio=6, stride=2),
            MBConvBlock(192, 192, expansion_ratio=6, stride=1),
            MBConvBlock(192, 192, expansion_ratio=6, stride=1)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(192, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 64)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze()

# -------------------
# 评估函数
# -------------------
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs).squeeze()
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    val_f1 = f1_score(all_targets, all_preds)
    val_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    val_precision = precision_score(all_targets, all_preds)
    val_recall = recall_score(all_targets, all_preds)
    return val_f1, val_acc, val_precision, val_recall

# -------------------
# 主测试流程
# -------------------
def main_test():
    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB1_1D().to(device)
    model.load_state_dict(torch.load('best_model_ilearn.pth'))
    model.eval()

    # 2. 加载测试数据
    # 注意：需要确保测试数据与训练数据有相同的标准化处理
    # 假设测试数据路径为：
    test_positive = pd.read_csv('../processed/Final/F_nc_pos_test.csv', header=None).values
    test_negative = pd.read_csv('../processed/Final/F_pc_neg_test.csv', header=None).values
    X_test = np.concatenate([test_positive, test_negative], axis=0)
    y_test = np.array([1]*len(test_positive) + [0]*len(test_negative))

    # 注意：需要使用训练时的max_val进行标准化
    # 这里需要用户提供训练时的max_val值（例如：从训练代码中保存的值）
    # 示例：假设训练时的max_val为255.0（需替换为实际值）
    max_val = 0.114714286
  # 需要替换为训练时的真实最大值！
    X_test = X_test / max_val

    # 3. 创建数据加载器
    test_dataset = CSVDataSet(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 4. 进行评估
    val_f1, val_acc, val_precision, val_recall = evaluate(model, test_loader, device)

    # 5. 输出结果
    print(f"Test Accuracy: {val_acc:.4f}")
    print(f"Test F1 Score: {val_f1:.4f}")
    print(f"Test Precision: {val_precision:.4f}")
    print(f"Test Recall: {val_recall:.4f}")

if __name__ == "__main__":
    main_test()
