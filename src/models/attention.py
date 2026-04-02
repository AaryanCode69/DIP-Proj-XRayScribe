"""Phase 4 module: attention mechanism over encoder sequence features."""

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Attention placeholder.

    Contract:
    - features: [B, S, C]
    - hidden: [B, H]
    - returns: context [B, C], weights [B, S]
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO(Phase 4): define projection layers.

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Phase 4 task: implement attention module.")
