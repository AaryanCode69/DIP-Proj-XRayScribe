"""Phase 4 module: attention mechanism over encoder sequence features."""

from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Additive attention over encoder sequence features.

    Contract:
    - features: [B, S, C]
    - hidden: [B, H]
    - returns: context [B, C], weights [B, S]
    """

    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int | None = None) -> None:
        super().__init__()
        attention_dim = attention_dim or hidden_dim
        self.feature_proj = nn.Linear(feature_dim, attention_dim, bias=False)
        self.hidden_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected features [B,S,C], got shape={tuple(features.shape)}")
        if hidden.ndim != 2:
            raise ValueError(f"Expected hidden [B,H], got shape={tuple(hidden.shape)}")

        projected_features = self.feature_proj(features)
        projected_hidden = self.hidden_proj(hidden).unsqueeze(1)
        energy = torch.tanh(projected_features + projected_hidden)
        scores = self.score(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return context, weights
