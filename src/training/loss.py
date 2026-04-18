"""Loss helpers including class-imbalance handling."""

from __future__ import annotations

import torch
from torch import nn


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute token-level cross entropy while ignoring padding."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,T,V], got shape={tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"Expected targets [B,T], got shape={tuple(targets.shape)}")

    vocab_size = logits.size(-1)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=pad_idx)
    return criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))


def make_token_weights(frequencies: dict[int, int], vocab_size: int, smoothing: float = 1.0) -> torch.Tensor:
    """Create inverse-frequency weights for imbalanced token distributions."""
    weights = torch.full((vocab_size,), smoothing, dtype=torch.float32)
    for token_id, frequency in frequencies.items():
        weights[token_id] = 1.0 / max(float(frequency), 1.0)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights
