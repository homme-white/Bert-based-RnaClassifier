import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -------------------
# 数据集定义
# -------------------
class RNA1DDataset(Dataset):
    def __init__(self, pos_data, neg_data):
        self.data = np.concatenate([pos_data, neg_data], axis=0)
        self.labels = np.array([1] * len(pos_data) + [0] * len(neg_data))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])  # 输入形状为 (channels=3, seq_len=3000)
        y = torch.LongTensor([self.labels[idx]])
        return x, y

# -------------------
# 模型定义（复制自训练代码）
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

class EfficientNet1D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        width_mult = 1.0
        depth_mult = 1.1
        dropout_rate = 0.3

        out_channels = self._make_divisible(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.SiLU()
        )

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
# 测试函数
# -------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).flatten()

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return acc, f1, precision, recall

# -------------------
# 主测试流程
# -------------------
def main_test():
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet1D(num_classes=2).to(device)
    model.load_state_dict(torch.load('best_model_fold.pth'))
    model.eval()

    # 2. 加载测试数据
    # 确保测试数据路径正确，并且形状与训练数据一致
    pos_test = np.load('rnafold/lnc_c_v.npy').transpose(0, 2, 1)  # 转换为(channels=3, seq_len=3000)
    neg_test = np.load('rnafold/pc_c_v.npy').transpose(0, 2, 1)   # 同上
    test_dataset = RNA1DDataset(pos_test, neg_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

    # 3. 进行评估
    acc, f1, precision, recall = evaluate_model(model, test_loader, device)

    # 4. 输出结果
    print("\n=== Test Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    main_test()
