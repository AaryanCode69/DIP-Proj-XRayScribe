"""Dataset utilities for IU X-Ray image-report pairs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import load_grayscale_image, resize_image, to_tensor
from src.data.vocabulary import Vocabulary
from src.dip.enhancement import clahe_enhance
from src.dip.segmentation import segment_lung_fields


class XRayDataset(Dataset):
    """Dataset for real IU manifest rows with integrated DIP preprocessing."""

    def __init__(
        self,
        records: Iterable[dict[str, str]] | None = None,
        csv_path: str | Path | None = None,
        split: str | None = None,
        random_seed: int = 42,
        vocabulary: Vocabulary | None = None,
        image_size: tuple[int, int] | None = None,
        tile_size: tuple[int, int] = (32, 32),
        clip_limit: int = 40,
        kernel_size: int = 3,
        opening_iterations: int = 1,
        closing_iterations: int = 1,
        foreground_dark: bool = True,
        preprocess: bool = True,
    ) -> None:
        super().__init__()
        if records is None and csv_path is None:
            records = []

        if csv_path is not None:
            records = self._load_csv(csv_path)

        all_records = list(records or [])
        self.records = self._apply_split(all_records, split=split, random_seed=random_seed)
        self.vocabulary = vocabulary
        self.image_size = image_size
        self.tile_size = tile_size
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size
        self.opening_iterations = opening_iterations
        self.closing_iterations = closing_iterations
        self.foreground_dark = foreground_dark
        self.preprocess = preprocess
        self.split = split

    @staticmethod
    def _load_csv(csv_path: str | Path) -> list[dict[str, str]]:
        path = Path(csv_path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header row: {path}")
            if "image_path" not in reader.fieldnames:
                raise ValueError(f"CSV must include image_path column: {path}")
            report_column = "report_text" if "report_text" in reader.fieldnames else "report"
            if report_column not in reader.fieldnames:
                raise ValueError(f"CSV must include report_text (or report) column: {path}")

            rows = []
            for row in reader:
                rows.append(
                    {
                        "image_path": row["image_path"],
                        "report_text": row.get(report_column, ""),
                        "split": row.get("split", "").strip().lower(),
                    }
                )
            return rows

    @staticmethod
    def _canonical_split(split_value: str) -> str:
        value = split_value.strip().lower()
        if value in {"train", "training"}:
            return "train"
        if value in {"val", "valid", "validation", "dev"}:
            return "val"
        if value in {"test", "testing"}:
            return "test"
        return ""

    @classmethod
    def _apply_split(
        cls,
        records: list[dict[str, str]],
        split: str | None,
        random_seed: int,
    ) -> list[dict[str, str]]:
        if split is None:
            return records

        split = cls._canonical_split(split)
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        has_split_column = any(cls._canonical_split(record.get("split", "")) for record in records)
        if has_split_column:
            return [record for record in records if cls._canonical_split(record.get("split", "")) == split]

        # Deterministic 70/10/20 split when no explicit split column exists.
        total = len(records)
        if total == 0:
            return []
        indices = np.arange(total)
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

        train_end = int(0.7 * total)
        val_end = train_end + int(0.1 * total)
        split_map = {
            "train": set(indices[:train_end].tolist()),
            "val": set(indices[train_end:val_end].tolist()),
            "test": set(indices[val_end:].tolist()),
        }
        selected = split_map[split]
        return [record for idx, record in enumerate(records) if idx in selected]

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[tuple[str, str]],
        **kwargs,
    ) -> "XRayDataset":
        records = [{"image_path": image_path, "report_text": report} for image_path, report in samples]
        return cls(records=records, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image_path = record["image_path"]
        report = record.get("report_text", record.get("report", ""))

        if "image_array" in record and record["image_array"] is not None:
            image = record["image_array"]
        else:
            image = load_grayscale_image(image_path)

        if self.image_size is not None:
            image = resize_image(image, size=self.image_size)

        if self.preprocess:
            # Phase 1: manual CLAHE, implemented via NumPy ops in src/dip/enhancement.py.
            enhanced = clahe_enhance(image, tile_size=self.tile_size, clip_limit=self.clip_limit)
            # Phase 2: manual Otsu + morphology, implemented via NumPy ops in src/dip/segmentation.py.
            _mask, masked = segment_lung_fields(
                enhanced,
                kernel_size=self.kernel_size,
                opening_iterations=self.opening_iterations,
                closing_iterations=self.closing_iterations,
                foreground_dark=self.foreground_dark,
            )
            image_tensor = to_tensor(masked)
        else:
            image_tensor = to_tensor(image)
        item: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "report": report,
            "report_text": report,
            "image_path": image_path,
        }

        if self.vocabulary is not None:
            token_ids = self.vocabulary.encode(report, add_special_tokens=True)
            item["tokens"] = torch.tensor(token_ids, dtype=torch.long)

        return item


# Backward-compatible alias used by existing scripts/tests.
ChestXrayReportDataset = XRayDataset
