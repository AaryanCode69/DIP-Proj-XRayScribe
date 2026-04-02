"""Phase 3 module: CNN feature extraction and compression."""

import torch
from torch import nn


class CNNExtractor(nn.Module):
    """Encoder placeholder with explicit output tensor contract.

    Contract:
    - input:  [B, C, H, W]
    - output: feature_map [B, C_e, Hc, Wc], seq_features [B, S, C_e]
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO(Phase 3): replace with truncated backbone + pooling compression.

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Phase 3 task: implement CNN feature extraction.")
