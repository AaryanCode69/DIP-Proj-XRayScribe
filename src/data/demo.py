"""Synthetic sample generation for smoke tests and demos."""

from __future__ import annotations

import numpy as np


DEFAULT_DEMO_REPORTS = (
    "heart size is normal . no focal consolidation .",
    "mild bibasal opacity is present .",
    "no pleural effusion or pneumothorax .",
    "low lung volume with mild atelectasis .",
)


def build_demo_records(
    num_samples: int = 4,
    image_size: tuple[int, int] = (128, 128),
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create deterministic in-memory records for training/evaluation smoke tests."""
    rng = np.random.default_rng(seed)
    samples: list[dict[str, object]] = []
    for index in range(num_samples):
        image = rng.integers(0, 256, size=image_size, dtype=np.uint8)
        samples.append(
            {
                "image_path": f"demo_{index}.png",
                "image_array": image,
                "report": DEFAULT_DEMO_REPORTS[index % len(DEFAULT_DEMO_REPORTS)],
            }
        )
    return samples
