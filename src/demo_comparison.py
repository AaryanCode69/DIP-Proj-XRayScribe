"""Compare GPT-4o findings for raw versus DIP-processed chest X-rays."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = "Analyze this chest X-ray and provide a clinical finding in under 15 words."
MODEL_NAME = "gpt-4o"
REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (
	REPO_ROOT / "dataset",
	REPO_ROOT / "data",
)
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def load_environment() -> None:
	"""Load environment variables from a local .env file."""
	load_dotenv()


def encode_image_to_base64(image_path: str | Path) -> tuple[str, str]:
	"""Encode a local image to base64 and return its MIME type."""
	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"Image file not found: {path}")

	mime_type, _ = mimetypes.guess_type(path.name)
	mime_type = mime_type or "image/png"
	image_bytes = path.read_bytes()
	return base64.b64encode(image_bytes).decode("utf-8"), mime_type


def _find_first_image(folder: Path) -> Path | None:
	"""Return the first image found in a folder, or None if it is empty."""
	if not folder.exists():
		return None

	for pattern in IMAGE_EXTENSIONS:
		matches = sorted(folder.glob(pattern))
		if matches:
			return matches[0]
	return None


def _collect_images(folder: Path) -> dict[str, Path]:
	"""Collect images in a folder keyed by stem."""
	images: dict[str, Path] = {}
	if not folder.exists():
		return images

	for pattern in IMAGE_EXTENSIONS:
		for image_path in sorted(folder.glob(pattern)):
			images.setdefault(image_path.stem, image_path)
	return images


def _resolve_image_pair(
	raw_image_path: str | None,
	processed_image_path: str | None,
) -> tuple[Path, Path]:
	"""Resolve a raw/processed image pair from explicit paths or common folders."""
	if raw_image_path and processed_image_path:
		return Path(raw_image_path), Path(processed_image_path)

	if raw_image_path and not processed_image_path:
		raw_image = Path(raw_image_path)
		processed_image = _find_first_image(SEARCH_ROOTS[0] / "processed") or _find_first_image(SEARCH_ROOTS[1] / "processed")
		if processed_image is not None:
			return raw_image, processed_image

	if processed_image_path and not raw_image_path:
		processed_image = Path(processed_image_path)
		raw_image = _find_first_image(SEARCH_ROOTS[0] / "raw") or _find_first_image(SEARCH_ROOTS[1] / "raw")
		if raw_image is not None:
			return raw_image, processed_image

	raw_images = {
		**_collect_images(SEARCH_ROOTS[0] / "raw"),
		**_collect_images(SEARCH_ROOTS[1] / "raw"),
		**_collect_images(SEARCH_ROOTS[0] / "interim"),
		**_collect_images(SEARCH_ROOTS[1] / "interim"),
	}
	processed_images = {
		**_collect_images(SEARCH_ROOTS[0] / "processed"),
		**_collect_images(SEARCH_ROOTS[1] / "processed"),
	}

	matching_stems = sorted(set(raw_images) & set(processed_images))
	if matching_stems:
		stem = matching_stems[0]
		return raw_images[stem], processed_images[stem]

	raw_image = Path(raw_image_path) if raw_image_path else next(iter(raw_images.values()), None)
	processed_image = Path(processed_image_path) if processed_image_path else next(iter(processed_images.values()), None)

	if raw_image is None:
		raise FileNotFoundError(
			"No raw image found. Put an image in dataset/raw or data/raw, or pass --raw-image-path."
		)
	if processed_image is None:
		raise FileNotFoundError(
			"No processed image found. Put an image in dataset/processed or data/processed, or pass --processed-image-path."
		)

	return raw_image, processed_image


def analyze_image_with_gpt4o(image_path: str | Path, client: OpenAI) -> str:
	"""Send one image to GPT-4o and return the short clinical finding."""
	image_b64, mime_type = encode_image_to_base64(image_path)

	response = client.chat.completions.create(
		model=MODEL_NAME,
		max_tokens=40,
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role": "user",
				"content": [
					{"type": "text", "text": SYSTEM_PROMPT},
					{
						"type": "image_url",
						"image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
					},
				],
			},
		],
	)

	content = response.choices[0].message.content or ""
	return content.strip()


def _print_side_by_side(raw_text: str, processed_text: str) -> None:
	"""Print the two findings in a readable terminal comparison."""
	left_width = 44
	right_width = 44
	separator = " | "

	print("=" * (left_width + right_width + len(separator)))
	print(f"{'RAW IMAGE REPORT'.ljust(left_width)}{separator}{'DIP-PROCESSED IMAGE REPORT'.ljust(right_width)}")
	print("-" * (left_width + right_width + len(separator)))

	raw_lines = textwrap.wrap(raw_text or "<no response>", width=left_width)
	processed_lines = textwrap.wrap(processed_text or "<no response>", width=right_width)
	total_lines = max(len(raw_lines), len(processed_lines))

	for index in range(total_lines):
		raw_line = raw_lines[index] if index < len(raw_lines) else ""
		processed_line = processed_lines[index] if index < len(processed_lines) else ""
		print(f"{raw_line.ljust(left_width)}{separator}{processed_line.ljust(right_width)}")

	print("=" * (left_width + right_width + len(separator)))


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare GPT-4o findings for raw and processed chest X-rays.")
	parser.add_argument("--raw-image-path", default=None, help="Optional raw image path.")
	parser.add_argument("--processed-image-path", default=None, help="Optional processed image path.")
	args = parser.parse_args()

	load_environment()
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

	raw_image_path, processed_image_path = _resolve_image_pair(args.raw_image_path, args.processed_image_path)
	client = OpenAI(api_key=api_key)

	raw_report = analyze_image_with_gpt4o(raw_image_path, client)
	processed_report = analyze_image_with_gpt4o(processed_image_path, client)

	print(f"Raw image: {raw_image_path}")
	print(f"Processed image: {processed_image_path}")
	_print_side_by_side(raw_report, processed_report)


if __name__ == "__main__":
	main()
