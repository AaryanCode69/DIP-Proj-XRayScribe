"""Tests for vocabulary and dataset helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data.dataset import ChestXrayReportDataset
from src.data.vocabulary import Vocabulary


class TestVocabularyAndDataset(unittest.TestCase):
    def test_vocabulary_roundtrip(self) -> None:
        vocabulary = Vocabulary.build([
            "heart size is normal .",
            "no pleural effusion .",
        ])
        encoded = vocabulary.encode("heart size is normal .")
        decoded = vocabulary.decode(encoded)
        self.assertIn("heart", decoded)
        self.assertIn("normal", decoded)
        self.assertGreater(len(vocabulary), 4)

    def test_dataset_returns_tensor_and_tokens(self) -> None:
        vocabulary = Vocabulary.build(["no acute disease ."])
        image = np.full((32, 32), 120, dtype=np.uint8)
        dataset = ChestXrayReportDataset(
            records=[{"image_path": "demo.png", "image_array": image, "report": "no acute disease ."}],
            vocabulary=vocabulary,
            image_size=(32, 32),
            preprocess=False,
        )

        sample = dataset[0]
        self.assertIn("image", sample)
        self.assertIn("tokens", sample)
        self.assertEqual(tuple(sample["image"].shape), (1, 32, 32))
        self.assertTrue(torch.is_tensor(sample["tokens"]))


if __name__ == "__main__":
    unittest.main()