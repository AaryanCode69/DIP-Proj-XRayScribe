"""Phase 2 module: manual Otsu thresholding + morphology.

Implementation intentionally deferred to Phase 2.
"""

import numpy as np


def segment_lung_fields(
    enhanced: np.ndarray,
    kernel_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment lung region from enhanced grayscale X-ray.

    Expected contract:
    - input: enhanced [H, W]
    - output: (mask [H, W], masked [H, W])
    """
    if enhanced.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={enhanced.shape}")

    # TODO(Phase 2): implement Otsu histogram threshold + custom morphology.
    raise NotImplementedError("Phase 2 task: implement segmentation pipeline.")
