"""
baseline_cnn.py
---------------
Simple custom CNN — Model 1 (baseline).
Deliberately lightweight so it can train fast and serve as a performance floor.

Architecture:
  3x [Conv2d → BN → ReLU → MaxPool]  →  Adaptive global avg pool  →  FC → 3 classes
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 3, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # fixed spatial size regardless of input
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
