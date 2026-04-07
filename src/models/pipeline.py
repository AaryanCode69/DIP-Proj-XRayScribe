"""End-to-end report generation pipeline."""

from __future__ import annotations

import torch
from torch import nn

from src.config import MODEL_CFG
from src.models.decoder import AttentionLSTMDecoder
from src.models.extraction import CNNExtractor


class ReportGenerationPipeline(nn.Module):
    """Wrap the extractor and decoder into a single trainable module."""

    def __init__(self, vocab_size: int, pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2) -> None:
        super().__init__()
        self.extractor = CNNExtractor()
        self.decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            feature_dim=MODEL_CFG.encoder_out_channels,
            embedding_dim=MODEL_CFG.token_embedding_dim,
            hidden_dim=MODEL_CFG.decoder_hidden_dim,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

    def forward(self, images: torch.Tensor, decoder_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map, seq_features = self.extractor(images)
        logits, attention_weights = self.decoder(seq_features, decoder_inputs)
        return logits, attention_weights

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int = MODEL_CFG.max_seq_len) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_features = self.extractor(images)
        return self.decoder.generate(seq_features, max_len=max_len)