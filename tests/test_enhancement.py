"""Unit tests for Phase 1 manual CLAHE implementation."""

import unittest

import numpy as np

from src.dip.enhancement import _clip_histogram, _tile_histogram, clahe_enhance


class TestEnhancementCLAHE(unittest.TestCase):
    def test_tile_histogram_sum_matches_pixel_count(self) -> None:
        tile = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
        hist = _tile_histogram(tile)
        self.assertEqual(int(hist.sum()), tile.size)
        self.assertEqual(int(hist[0]), 1)
        self.assertEqual(int(hist[1]), 2)
        self.assertEqual(int(hist[2]), 3)

    def test_output_shape_is_preserved_for_non_divisible_image(self) -> None:
        image = np.random.default_rng(42).integers(0, 256, size=(53, 71), dtype=np.uint8)
        enhanced = clahe_enhance(image, tile_size=(16, 16), clip_limit=40)
        self.assertEqual(enhanced.shape, image.shape)

    def test_clip_histogram_preserves_total_counts(self) -> None:
        hist = np.zeros(256, dtype=np.int64)
        hist[0] = 256
        clipped = _clip_histogram(hist, clip_limit=20, tile_pixels=256)
        self.assertEqual(int(clipped.sum()), 256)

    def test_output_is_uint8_and_within_range(self) -> None:
        image = np.random.default_rng(7).random((64, 64), dtype=np.float32)
        enhanced = clahe_enhance(image, tile_size=(8, 8), clip_limit=20)
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreaterEqual(int(enhanced.min()), 0)
        self.assertLessEqual(int(enhanced.max()), 255)

    def test_uniform_image_remains_spatially_uniform(self) -> None:
        image = np.full((48, 48), 128, dtype=np.uint8)
        enhanced = clahe_enhance(image, tile_size=(8, 8), clip_limit=40)
        self.assertEqual(int(enhanced.std()), 0)

    def test_non_2d_input_raises(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            clahe_enhance(image)

    def test_invalid_tile_size_raises(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8)
        with self.assertRaises(ValueError):
            clahe_enhance(image, tile_size=(0, 8), clip_limit=10)


if __name__ == "__main__":
    unittest.main()
