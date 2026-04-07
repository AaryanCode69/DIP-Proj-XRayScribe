"""Phase 3 module: CNN feature extraction and compression."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.config import MODEL_CFG


class CNNExtractor(nn.Module):
    """Truncated CNN encoder with explicit spatial compression.

    Contract:
    - input:  [B, C, H, W]
    - output: feature_map [B, C_e, Hc, Wc], seq_features [B, S, C_e]
    """

    def __init__(
        self,
        backbone: str = MODEL_CFG.encoder_backbone,
        out_channels: int = MODEL_CFG.encoder_out_channels,
        pooled_h: int = MODEL_CFG.pooled_h,
        pooled_w: int = MODEL_CFG.pooled_w,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone}. Only resnet18 is implemented.")

        weights = None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

        # ResNet18 produces 512-channel feature maps after layer4.
        backbone_channels = 512
        if out_channels != backbone_channels:
            self.projection = nn.Conv2d(backbone_channels, out_channels, kernel_size=1, bias=False)
            feature_channels = out_channels
        else:
            self.projection = nn.Identity()
            feature_channels = backbone_channels

        self.pool = nn.AdaptiveAvgPool2d((pooled_h, pooled_w))
        self.out_channels = feature_channels
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected batched image tensor [B,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError("CNNExtractor expects 1 or 3 input channels.")

        # ResNet stem -> layer4 produces a dense feature map with semantic spatial structure.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feature_map = self.backbone.layer4(x)
        feature_map = self.projection(feature_map)

        # Compression step: [B, C, H, W] -> [B, C, pooled_h, pooled_w].
        feature_map = self.pool(feature_map)

        # Sequence form: flatten spatial grid into S = pooled_h * pooled_w positions.
        seq_features = feature_map.flatten(2).transpose(1, 2).contiguous()
        return feature_map, seq_features
