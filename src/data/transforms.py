"""Data transforms used by dataset and loaders."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from src.dip.enhancement import clahe_enhance
from src.dip.segmentation import segment_lung_fields


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """Load a grayscale image from disk using basic I/O only."""
    path = Path(image_path)
    if path.suffix.lower() == ".dcm":
        import pydicom

        dataset = pydicom.dcmread(str(path))
        image = dataset.pixel_array.astype(np.float32)
        if hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept"):
            image = image * float(dataset.RescaleSlope) + float(dataset.RescaleIntercept)
        return image

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image.astype(np.float32)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image using OpenCV's basic resizing primitive."""
    height, width = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_image(
    image: np.ndarray,
    size: tuple[int, int] | None = None,
    tile_size: tuple[int, int] = (32, 32),
    clip_limit: int = 40,
    kernel_size: int = 3,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
    foreground_dark: bool = True,
) -> np.ndarray:
    """Apply Phase 1 and Phase 2 preprocessing to a grayscale image."""
    if size is not None:
        image = resize_image(image, size=size)
    enhanced = clahe_enhance(image, tile_size=tile_size, clip_limit=clip_limit)
    _, masked = segment_lung_fields(
        enhanced,
        kernel_size=kernel_size,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations,
        foreground_dark=foreground_dark,
    )
    return masked


def to_tensor(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """Convert HxW image to float tensor [1,H,W]."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={image.shape}")
    tensor = torch.from_numpy(image.astype(np.float32))
    if normalize:
        tensor = tensor / 255.0
    return tensor.unsqueeze(0)


def collate_batch(batch: Iterable[dict], pad_idx: int) -> dict[str, torch.Tensor | list[str]]:
    """Pad token sequences and stack image tensors."""
    batch_list = list(batch)
    images = torch.stack([item["image"] for item in batch_list], dim=0)
    reports = [item.get("report", "") for item in batch_list]
    paths = [item.get("image_path", "") for item in batch_list]

    has_tokens = ["tokens" in item for item in batch_list]
    if any(has_tokens) and not all(has_tokens):
        raise ValueError("Either every batch item must include tokens or none of them may.")

    if all(has_tokens):
        token_sequences = [item["tokens"] for item in batch_list]
        max_len = max(sequence.size(0) for sequence in token_sequences)
        padded = torch.full((len(token_sequences), max_len), pad_idx, dtype=torch.long)
        for index, sequence in enumerate(token_sequences):
            padded[index, : sequence.size(0)] = sequence
    else:
        padded = torch.empty(0, dtype=torch.long)

    return {"image": images, "tokens": padded, "report": reports, "image_path": paths}


def make_collate_fn(pad_idx: int):
    """Build a DataLoader-compatible collate function with a fixed pad index."""
    return partial(collate_batch, pad_idx=pad_idx)
