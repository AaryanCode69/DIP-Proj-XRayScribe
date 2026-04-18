"""Shared helpers for manual DIP preprocessing modules."""

from __future__ import annotations

import numpy as np


def ensure_grayscale_image(image: np.ndarray) -> None:
    """Require a 2D grayscale image."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={image.shape}")


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert arbitrary grayscale input to uint8 in [0, 255]."""
    ensure_grayscale_image(image)

    if image.dtype == np.uint8:
        return image.copy()

    arr = image.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())

    if min_val >= 0.0 and max_val <= 1.0:
        return np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    if min_val >= 0.0 and max_val <= 255.0:
        return np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.uint8)

    scaled = (arr - min_val) / (max_val - min_val)
    return np.clip(np.rint(scaled * 255.0), 0.0, 255.0).astype(np.uint8)
