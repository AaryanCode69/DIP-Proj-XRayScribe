"""Tests for attention and decoder shapes."""

from __future__ import annotations

import unittest

import torch

from src.models.decoder import AttentionLSTMDecoder


class TestDecoder(unittest.TestCase):
    def test_forward_and_generation_shapes(self) -> None:
        model = AttentionLSTMDecoder(vocab_size=16, feature_dim=64, embedding_dim=32, hidden_dim=48)
        seq_features = torch.randn(2, 9, 64)
        tokens = torch.randint(0, 16, (2, 6))

        logits, attn_weights = model(seq_features, tokens)
        self.assertEqual(tuple(logits.shape), (2, 6, 16))
        self.assertEqual(tuple(attn_weights.shape), (2, 6, 9))

        generated, generated_attn = model.generate(seq_features, max_len=5)
        self.assertEqual(generated.shape[0], 2)
        self.assertEqual(generated_attn.shape[0], 2)


if __name__ == "__main__":
    unittest.main()