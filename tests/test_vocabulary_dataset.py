"""Tests for vocabulary and dataset helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest

import numpy as np
import torch

from src.data.dataset import ChestXrayReportDataset
from src.data.transforms import collate_batch
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

    def test_build_from_csv_accepts_report_fallback_column(self) -> None:
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "report"])
            writer.writeheader()
            writer.writerow({"image_path": "demo.png", "report": "clear lungs and normal heart"})
            handle.flush()

            vocabulary = Vocabulary.build_from_csv(handle.name)

        self.assertIn("clear", vocabulary.token_to_id)
        self.assertIn("heart", vocabulary.token_to_id)

    def test_collate_batch_rejects_mixed_token_presence(self) -> None:
        with self.assertRaises(ValueError):
            collate_batch(
                [
                    {"image": torch.zeros(1, 4, 4), "tokens": torch.tensor([1, 2]), "report": "", "image_path": "a"},
                    {"image": torch.zeros(1, 4, 4), "report": "", "image_path": "b"},
                ],
                pad_idx=0,
            )


if __name__ == "__main__":
    unittest.main()
