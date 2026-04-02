"""Phase 4 module: attention-based LSTM decoder."""

import torch
from torch import nn


class AttentionLSTMDecoder(nn.Module):
    """Decoder placeholder with train and inference contracts."""

    def __init__(self) -> None:
        super().__init__()
        # TODO(Phase 4): define embeddings, attention, LSTM, and output head.

    def forward(self, seq_features: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training with teacher forcing.

        Returns:
            logits: [B, T, V]
            attn_weights: [B, T, S]
        """
        raise NotImplementedError("Phase 4 task: implement decoder forward pass.")

    def generate(self, seq_features: torch.Tensor, max_len: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive inference placeholder."""
        raise NotImplementedError("Phase 4 task: implement autoregressive generation.")
