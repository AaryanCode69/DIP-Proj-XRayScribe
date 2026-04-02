"""Phase 1 module: manual CLAHE enhancement.

Implementation intentionally deferred to Phase 1.
"""

import numpy as np


def clahe_enhance(
    image: np.ndarray,
    tile_size: tuple[int, int] = (32, 32),
    clip_limit: int = 40,
) -> np.ndarray:
    """Enhance a grayscale image using CLAHE implemented from scratch.

    Expected contract:
    - input: image [H, W]
    - output: enhanced image [H, W]
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={image.shape}")

    # TODO(Phase 1): implement tile histogram clipping, CDF remap, and interpolation.
    raise NotImplementedError("Phase 1 task: implement manual CLAHE.")
