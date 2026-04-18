"""Phase 2 module: manual Otsu thresholding + morphology (NumPy only)."""

from __future__ import annotations

import numpy as np

from src.dip.common import ensure_grayscale_image, normalize_to_uint8


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around the shared uint8 normalization helper."""
    return normalize_to_uint8(image)


def _validate_kernel_size(kernel_size: int) -> None:
    """Require odd, positive kernel size for centered morphology."""
    if kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")


def otsu_threshold(image: np.ndarray) -> int:
    """Compute Otsu threshold from scratch for a grayscale image."""
    ensure_grayscale_image(image)

    image_uint8 = normalize_to_uint8(image)
    hist = np.bincount(image_uint8.ravel(), minlength=256).astype(np.float64)

    total_pixels = image_uint8.size
    if total_pixels == 0:
        raise ValueError("Input image has zero pixels.")

    probabilities = hist / total_pixels
    bins = np.arange(256, dtype=np.float64)

    weight_bg = np.cumsum(probabilities)
    weight_fg = 1.0 - weight_bg
    mean_bg = np.cumsum(probabilities * bins)
    global_mean = mean_bg[-1]

    # Between-class variance with safe denominator handling.
    denominator = weight_bg * weight_fg
    numerator = (global_mean * weight_bg - mean_bg) ** 2
    sigma_b2 = np.zeros_like(numerator)
    valid = denominator > 0.0
    sigma_b2[valid] = numerator[valid] / denominator[valid]

    return int(np.argmax(sigma_b2))


def _binary_from_threshold(image: np.ndarray, threshold: int, foreground_dark: bool) -> np.ndarray:
    """Create binary mask from threshold using either dark or bright foreground."""
    if foreground_dark:
        return (image <= threshold).astype(np.uint8)
    return (image > threshold).astype(np.uint8)


def binary_erosion(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Binary erosion using explicit sliding-window logic."""
    _validate_kernel_size(kernel_size)
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer.")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape={mask.shape}")

    return _binary_morphology(mask, kernel_size=kernel_size, iterations=iterations, reducer=np.all)


def binary_dilation(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Binary dilation using explicit sliding-window logic."""
    _validate_kernel_size(kernel_size)
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer.")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape={mask.shape}")

    return _binary_morphology(mask, kernel_size=kernel_size, iterations=iterations, reducer=np.any)


def binary_opening(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Binary opening = erosion followed by dilation."""
    eroded = binary_erosion(mask, kernel_size=kernel_size, iterations=iterations)
    return binary_dilation(eroded, kernel_size=kernel_size, iterations=iterations)


def binary_closing(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Binary closing = dilation followed by erosion."""
    dilated = binary_dilation(mask, kernel_size=kernel_size, iterations=iterations)
    return binary_erosion(dilated, kernel_size=kernel_size, iterations=iterations)


def segment_lung_fields(
    enhanced: np.ndarray,
    kernel_size: int = 3,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
    foreground_dark: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment lung fields via Otsu thresholding and custom morphology.

    Args:
        enhanced: Enhanced grayscale image [H, W].
        kernel_size: Odd morphology kernel size.
        opening_iterations: Opening iterations for external noise removal.
        closing_iterations: Closing iterations for internal gap filling.
        foreground_dark: Whether lung region is represented by darker intensities.

    Returns:
        mask: Binary segmentation mask [H, W] (0/1).
        masked_img: Enhanced image multiplied by mask [H, W], dtype uint8.
    """
    ensure_grayscale_image(enhanced)
    _validate_kernel_size(kernel_size)
    if opening_iterations <= 0 or closing_iterations <= 0:
        raise ValueError("opening_iterations and closing_iterations must be positive integers.")

    enhanced_uint8 = normalize_to_uint8(enhanced)
    threshold = otsu_threshold(enhanced_uint8)

    mask = _binary_from_threshold(enhanced_uint8, threshold=threshold, foreground_dark=foreground_dark)
    mask = binary_opening(mask, kernel_size=kernel_size, iterations=opening_iterations)
    mask = binary_closing(mask, kernel_size=kernel_size, iterations=closing_iterations)

    masked_img = (enhanced_uint8 * mask).astype(np.uint8)
    return mask.astype(np.uint8), masked_img


def _binary_morphology(
    mask: np.ndarray,
    kernel_size: int,
    iterations: int,
    reducer,
) -> np.ndarray:
    """Apply a sliding-window binary morphology reducer."""
    current = mask.astype(bool)
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(current, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
        current = reducer(windows, axis=(-2, -1))

    return current.astype(np.uint8)
