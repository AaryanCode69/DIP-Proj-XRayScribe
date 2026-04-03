"""Unit tests for Phase 2 manual segmentation implementation."""

import unittest

import numpy as np

from src.dip.segmentation import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    otsu_threshold,
    segment_lung_fields,
)


class TestSegmentationPhase2(unittest.TestCase):
    def test_otsu_threshold_in_valid_range(self) -> None:
        image = np.random.default_rng(1).integers(0, 256, size=(64, 64), dtype=np.uint8)
        threshold = otsu_threshold(image)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 255)

    def test_otsu_detects_bimodal_separation(self) -> None:
        image = np.zeros((40, 40), dtype=np.uint8)
        image[:, :20] = 25
        image[:, 20:] = 210
        threshold = otsu_threshold(image)
        # Otsu may land on the first valley boundary in a two-level synthetic image.
        self.assertGreaterEqual(threshold, 25)
        self.assertLess(threshold, 210)

    def test_erosion_and_dilation_are_monotonic(self) -> None:
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2:7, 2:7] = 1

        eroded = binary_erosion(mask, kernel_size=3, iterations=1)
        dilated = binary_dilation(mask, kernel_size=3, iterations=1)

        self.assertLessEqual(int(eroded.sum()), int(mask.sum()))
        self.assertGreaterEqual(int(dilated.sum()), int(mask.sum()))

    def test_opening_removes_single_pixel_noise(self) -> None:
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 1
        opened = binary_opening(mask, kernel_size=3, iterations=1)
        self.assertEqual(int(opened.sum()), 0)

    def test_segment_outputs_shape_binary_and_masking_rule(self) -> None:
        image = np.full((32, 32), 220, dtype=np.uint8)
        image[8:24, 8:24] = 30  # synthetic dark foreground region

        mask, masked = segment_lung_fields(
            image,
            kernel_size=3,
            opening_iterations=1,
            closing_iterations=1,
            foreground_dark=True,
        )

        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(masked.shape, image.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(masked.dtype, np.uint8)

        unique_vals = set(np.unique(mask).tolist())
        self.assertTrue(unique_vals.issubset({0, 1}))

        expected_masked = (image * mask).astype(np.uint8)
        self.assertTrue(np.array_equal(masked, expected_masked))

    def test_segment_invalid_inputs_raise(self) -> None:
        image_3d = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            segment_lung_fields(image_3d)

        image_2d = np.zeros((8, 8), dtype=np.uint8)
        with self.assertRaises(ValueError):
            segment_lung_fields(image_2d, kernel_size=4)


if __name__ == "__main__":
    unittest.main()
