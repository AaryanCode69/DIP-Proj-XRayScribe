"""Phase 4 module: attention-based LSTM decoder."""

from __future__ import annotations

import torch
from torch import nn

from src.config import MODEL_CFG
from src.models.attention import AdditiveAttention


class AttentionLSTMDecoder(nn.Module):
    """Attention + LSTM decoder for report generation."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = MODEL_CFG.encoder_out_channels,
        embedding_dim: int = MODEL_CFG.token_embedding_dim,
        hidden_dim: int = MODEL_CFG.decoder_hidden_dim,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.attention = AdditiveAttention(feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.lstm_cell = nn.LSTMCell(embedding_dim + feature_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, seq_features: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass.

        Args:
            seq_features: Encoder features [B, S, C].
            tokens: Decoder input ids [B, T]. Typically this is the target sequence shifted right,
                containing <bos> and excluding the final token.

        Returns:
            logits: [B, T, V]
            attn_weights: [B, T, S]
        """
        if seq_features.ndim != 3:
            raise ValueError(f"Expected seq_features [B,S,C], got shape={tuple(seq_features.shape)}")
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens [B,T], got shape={tuple(tokens.shape)}")

        batch_size, sequence_length, _ = seq_features.shape
        device = seq_features.device
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        context = seq_features.mean(dim=1)

        logits_list = []
        attention_list = []

        for step in range(tokens.size(1)):
            token_embedding = self.embedding(tokens[:, step])
            lstm_input = torch.cat([token_embedding, context], dim=-1)
            hidden_state, cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))
            context, weights = self.attention(seq_features, hidden_state)
            output = self.output_layer(torch.cat([hidden_state, context], dim=-1))
            logits_list.append(output.unsqueeze(1))
            attention_list.append(weights.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)
        attn_weights = torch.cat(attention_list, dim=1)
        return logits, attn_weights

    @torch.no_grad()
    def generate(self, seq_features: torch.Tensor, max_len: int = MODEL_CFG.max_seq_len) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive inference."""
        if seq_features.ndim != 3:
            raise ValueError(f"Expected seq_features [B,S,C], got shape={tuple(seq_features.shape)}")

        batch_size = seq_features.size(0)
        device = seq_features.device
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        context = seq_features.mean(dim=1)
        current_tokens = torch.full((batch_size,), self.bos_idx, dtype=torch.long, device=device)

        generated_tokens = []
        attention_history = []

        for _ in range(max_len):
            token_embedding = self.embedding(current_tokens)
            lstm_input = torch.cat([token_embedding, context], dim=-1)
            hidden_state, cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))
            context, weights = self.attention(seq_features, hidden_state)
            logits = self.output_layer(torch.cat([hidden_state, context], dim=-1))
            current_tokens = torch.argmax(logits, dim=-1)
            generated_tokens.append(current_tokens.unsqueeze(1))
            attention_history.append(weights.unsqueeze(1))

            if torch.all(current_tokens.eq(self.eos_idx)):
                break

        if generated_tokens:
            token_tensor = torch.cat(generated_tokens, dim=1)
            attention_tensor = torch.cat(attention_history, dim=1)
        else:
            token_tensor = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            attention_tensor = torch.empty(batch_size, 0, seq_features.size(1), device=device)

        return token_tensor, attention_tensor
