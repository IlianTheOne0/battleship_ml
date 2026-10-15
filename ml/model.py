from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

class GestureCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()

        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.backbone(x)