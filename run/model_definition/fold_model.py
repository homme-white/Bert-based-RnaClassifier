import torch
import torch.nn as nn
import numpy as np

class MBConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)  # ✅ 使用正确的维度计算
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, 1, bias=False),  # ✅ 使用hidden_dim
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=stride,  # ✅ 使用hidden_dim
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),  # ✅ 使用out_channels
            nn.BatchNorm1d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.drop_path = nn.Identity()

    # ✅ 新增 forward 方法
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

        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),  # ✅ 输入通道3→输出通道32
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList([
            self._make_mbconv(32, 16, 1, 1, depth_mult),
            self._make_mbconv(16, 24, 2, 6, depth_mult),
            self._make_mbconv(24, 40, 2, 6, depth_mult),
            self._make_mbconv(40, 80, 3, 6, depth_mult),
            self._make_mbconv(80, 112, 3, 6, depth_mult),
            self._make_mbconv(112, 192, 4, 6, depth_mult),
            self._make_mbconv(192, 320, 1, 6, depth_mult),
        ])

        self.head = nn.Sequential(
            nn.Conv1d(320, 1280, 1, bias=False),
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
            layers.append(MBConv1D(in_channels if i == 0 else out_channels,  # ✅ 正确传递通道数
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
