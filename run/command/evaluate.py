import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torchvision import models
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------
# 复制模型定义部分（必须与训练代码完全一致）
# -------------------

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


# -------------------
# 测试函数
# -------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for bert, pse, rnafold, labels in test_loader:
            bert = bert.to(device)
            pse = pse.to(device)
            rnafold = rnafold.to(device)
            outputs = model(bert, pse, rnafold)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    return acc, f1, precision, recall


# -------------------
# 主测试流程
# -------------------

def evaluate_model():
    # ================= 配置参数 =================
    TEST_POS_BERT_DIR = '../processed/Final/test/nc_pos'  # 测试集正样本BERT特征路径
    TEST_NEG_BERT_DIR = '../processed/Final/test/pc_neg'  # 测试集负样本BERT特征路径
    TEST_POS_PSE_CSV = '../processed/Final/F_nc_pos_test.csv'  # 测试集正样本Pse特征
    TEST_NEG_PSE_CSV = '../processed/Final/F_pc_neg_test.csv'  # 测试集负样本Pse特征
    TEST_POS_RNAFOLD = '../processed/Final/nc_pos_test.npy'  # 测试集正样本RNAfold
    TEST_NEG_RNAFOLD = '../processed/Final/pc_neg_test.npy'  # 测试集负样本RNAfold
    MODEL_PATH = 'best_model_hybird.pth'  # 训练好的模型路径
    BATCH_SIZE = 256  # 根据显存调整

    # ================= 初始化设置 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= 加载测试数据集 =================
    test_dataset = DNADataset(
        pos_bert_dir=TEST_POS_BERT_DIR,
        neg_bert_dir=TEST_NEG_BERT_DIR,
        pos_pse_csv_path=TEST_POS_PSE_CSV,
        neg_pse_csv_path=TEST_NEG_PSE_CSV,
        pos_rnafold_path=TEST_POS_RNAFOLD,
        neg_rnafold_path=TEST_NEG_RNAFOLD
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ================= 初始化模型 =================
    model = MultiModalEfficientNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # 设置为评估模式

    # ================= 测试过程 =================
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            bert, pse, rnafold, labels = batch
            bert = bert.to(device)
            pse = pse.to(device)
            rnafold = rnafold.to(device)

            outputs = model(bert, pse, rnafold)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ================= 计算指标 =================
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)

    # ================= 输出结果 =================
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"- Accuracy:  {accuracy:.4f}")
    print(f"- F1 Score:  {f1:.4f}")
    print(f"- Recall:    {recall:.4f}")
    print(f"- Precision: {precision:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
