"""Tests for the CNN extractor shapes."""

from __future__ import annotations

import unittest

import torch

from src.models.extraction import CNNExtractor


class TestCNNExtractor(unittest.TestCase):
    def test_feature_and_sequence_shapes(self) -> None:
        model = CNNExtractor(out_channels=256, pooled_h=7, pooled_w=7)
        images = torch.randn(2, 1, 128, 128)
        feature_map, seq_features = model(images)

        self.assertEqual(tuple(feature_map.shape), (2, 256, 7, 7))
        self.assertEqual(tuple(seq_features.shape), (2, 49, 256))


if __name__ == "__main__":
    unittest.main()