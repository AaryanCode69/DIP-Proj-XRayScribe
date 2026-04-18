"""Phase 1 module: manual CLAHE enhancement (NumPy only)."""

from __future__ import annotations

import math

import numpy as np

from src.dip.common import ensure_grayscale_image, normalize_to_uint8


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper around the shared uint8 normalization helper."""
    return normalize_to_uint8(image)


def _pad_to_tile_shape(image: np.ndarray, tile_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad image so each spatial dimension is divisible by tile size."""
    tile_h, tile_w = tile_size
    h, w = image.shape
    pad_h = (tile_h - (h % tile_h)) % tile_h
    pad_w = (tile_w - (w % tile_w)) % tile_w
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, (h, w)


def _tile_histogram(tile: np.ndarray) -> np.ndarray:
    """Compute 256-bin histogram for a uint8 tile."""
    return np.bincount(tile.ravel(), minlength=256).astype(np.int64)


def _clip_histogram(hist: np.ndarray, clip_limit: int, tile_pixels: int) -> np.ndarray:
    """Clip histogram and redistribute overflow counts."""
    if clip_limit <= 0:
        raise ValueError("clip_limit must be a positive integer.")

    n_bins = hist.size
    # Ensure clipping is mathematically feasible for this tile.
    effective_clip = max(clip_limit, math.ceil(tile_pixels / n_bins))
    clipped = np.minimum(hist, effective_clip).astype(np.int64)
    overflow = int((hist - clipped).clip(min=0).sum())

    if overflow == 0:
        return clipped

    # Standard CLAHE-style redistribution:
    # add the same amount to all bins, then spread the remainder.
    add_all = overflow // n_bins
    residual = overflow % n_bins

    if add_all > 0:
        clipped += add_all
    if residual > 0:
        clipped[:residual] += 1

    return clipped


def _build_lut(hist: np.ndarray, tile_pixels: int) -> np.ndarray:
    """Build LUT from clipped histogram via CDF mapping."""
    cdf = hist.cumsum(dtype=np.float64)
    nonzero = np.flatnonzero(cdf > 0)
    if nonzero.size == 0:
        return np.arange(256, dtype=np.uint8)

    cdf_min = cdf[nonzero[0]]
    denom = tile_pixels - cdf_min
    if denom <= 0:
        return np.arange(256, dtype=np.uint8)

    lut = np.clip(np.rint((cdf - cdf_min) * 255.0 / denom), 0.0, 255.0).astype(np.uint8)
    return lut


def _lut_grid(padded: np.ndarray, tile_size: tuple[int, int], clip_limit: int) -> np.ndarray:
    """Compute one LUT per tile."""
    tile_h, tile_w = tile_size
    h, w = padded.shape
    n_tiles_y = h // tile_h
    n_tiles_x = w // tile_w
    luts = np.zeros((n_tiles_y, n_tiles_x, 256), dtype=np.uint8)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * tile_h
            y1 = y0 + tile_h
            x0 = tx * tile_w
            x1 = x0 + tile_w
            tile = padded[y0:y1, x0:x1]
            tile_pixels = tile.size
            hist = _tile_histogram(tile)
            clipped_hist = _clip_histogram(hist, clip_limit=clip_limit, tile_pixels=tile_pixels)
            luts[ty, tx] = _build_lut(clipped_hist, tile_pixels=tile_pixels)

    return luts


def _bilinear_apply(padded: np.ndarray, luts: np.ndarray, tile_size: tuple[int, int]) -> np.ndarray:
    """Apply per-tile LUTs and bilinearly blend neighboring tiles."""
    tile_h, tile_w = tile_size
    out = np.empty_like(padded, dtype=np.uint8)
    n_tiles_y, n_tiles_x, _ = luts.shape

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * tile_h
            y1 = y0 + tile_h
            x0 = tx * tile_w
            x1 = x0 + tile_w
            region = padded[y0:y1, x0:x1]

            ty1 = min(ty + 1, n_tiles_y - 1)
            tx1 = min(tx + 1, n_tiles_x - 1)

            lut00 = luts[ty, tx]
            lut01 = luts[ty, tx1]
            lut10 = luts[ty1, tx]
            lut11 = luts[ty1, tx1]

            mapped00 = lut00[region].astype(np.float32)
            mapped01 = lut01[region].astype(np.float32)
            mapped10 = lut10[region].astype(np.float32)
            mapped11 = lut11[region].astype(np.float32)

            h_reg, w_reg = region.shape
            alpha = (np.arange(h_reg, dtype=np.float32) / max(h_reg, 1)).reshape(-1, 1)
            beta = (np.arange(w_reg, dtype=np.float32) / max(w_reg, 1)).reshape(1, -1)

            top = (1.0 - beta) * mapped00 + beta * mapped01
            bottom = (1.0 - beta) * mapped10 + beta * mapped11
            blended = (1.0 - alpha) * top + alpha * bottom
            out[y0:y1, x0:x1] = np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)

    return out


def clahe_enhance(
    image: np.ndarray,
    tile_size: tuple[int, int] = (32, 32),
    clip_limit: int = 40,
) -> np.ndarray:
    """Enhance a grayscale image using custom CLAHE.

    Args:
        image: Grayscale image with shape [H, W].
        tile_size: Tile height/width used for local histogram equalization.
        clip_limit: Absolute clip value for each histogram bin.

    Returns:
        Enhanced image with the same shape as input and dtype `uint8`.
    """
    ensure_grayscale_image(image)

    tile_h, tile_w = tile_size
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"tile_size must contain positive integers, got {tile_size}")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be a positive integer.")

    image_uint8 = normalize_to_uint8(image)
    padded, (orig_h, orig_w) = _pad_to_tile_shape(image_uint8, tile_size=tile_size)
    luts = _lut_grid(padded, tile_size=tile_size, clip_limit=clip_limit)
    enhanced_padded = _bilinear_apply(padded, luts=luts, tile_size=tile_size)
    return enhanced_padded[:orig_h, :orig_w]
