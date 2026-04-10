"""Process a sample chest X-ray through the DIP preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

if __package__ in {None, ""}:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import load_grayscale_image, preprocess_image


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_IMAGE = REPO_ROOT / "data" / "raw" / "sample_01.png"
OUTPUT_IMAGE = REPO_ROOT / "data" / "processed" / "sample_01.png"


def process_sample_image() -> Path:
	"""Run the Phase 1-2 DIP pipeline on the sample image and save the result."""
	if not INPUT_IMAGE.exists():
		raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

	image = load_grayscale_image(INPUT_IMAGE)
	processed_image = preprocess_image(image)

	OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
	success = cv2.imwrite(str(OUTPUT_IMAGE), processed_image)
	if not success:
		raise RuntimeError(f"Failed to write output image: {OUTPUT_IMAGE}")

	return OUTPUT_IMAGE


def main() -> None:
	"""Execute the sample preprocessing job and print a confirmation message."""
	output_path = process_sample_image()
	print(f"Saved processed image to {output_path}")


if __name__ == "__main__":
	main()