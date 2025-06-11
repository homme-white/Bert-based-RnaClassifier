import torch
import torch.nn as nn

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
        return self.classifier(x)
