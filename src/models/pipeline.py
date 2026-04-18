"""End-to-end report generation pipeline."""

from __future__ import annotations

import torch
from torch import nn

from src.config import MODEL_CFG, ModelConfig
from src.models.decoder import AttentionLSTMDecoder
from src.models.extraction import CNNExtractor


class ReportGenerationPipeline(nn.Module):
    """Wrap the extractor and decoder into a single trainable module."""

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        model_config: ModelConfig = MODEL_CFG,
        extractor: CNNExtractor | None = None,
        decoder: AttentionLSTMDecoder | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.extractor = extractor or CNNExtractor(
            backbone=model_config.encoder_backbone,
            out_channels=model_config.encoder_out_channels,
            pooled_h=model_config.pooled_h,
            pooled_w=model_config.pooled_w,
        )
        self.decoder = decoder or AttentionLSTMDecoder(
            vocab_size=vocab_size,
            feature_dim=self.extractor.out_channels,
            embedding_dim=model_config.token_embedding_dim,
            hidden_dim=model_config.decoder_hidden_dim,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

    def forward(self, images: torch.Tensor, decoder_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map, seq_features = self.extractor(images)
        logits, attention_weights = self.decoder(seq_features, decoder_inputs)
        return logits, attention_weights

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_features = self.extractor(images)
        return self.decoder.generate(seq_features, max_len=max_len or self.model_config.max_seq_len)
