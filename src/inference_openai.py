"""OpenAI vision inference for DIP-enhanced chest X-rays."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def load_environment() -> None:
	"""Load variables from a local .env file into process environment."""
	load_dotenv()


def encode_image_to_base64(image_path: str | Path) -> tuple[str, str]:
	"""Read a local image and return (base64_data, mime_type)."""
	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"Image file not found: {path}")

	mime_type, _ = mimetypes.guess_type(path.name)
	mime_type = mime_type or "image/png"

	image_bytes = path.read_bytes()
	base64_data = base64.b64encode(image_bytes).decode("utf-8")
	return base64_data, mime_type


def analyze_xray_with_gpt4o(image_path: str | Path) -> str:
	"""Send a local X-ray image to gpt-4o and return a short clinical finding."""
	load_environment()
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

	client = OpenAI(api_key=api_key)
	image_b64, mime_type = encode_image_to_base64(image_path)

	response = client.chat.completions.create(
		model="gpt-4o",
		max_tokens=40,
		messages=[
			{
				"role": "system",
				"content": "You are a Radiologist.",
			},
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": (
							"Analyze this DIP-enhanced X-ray (CLAHE + Segmented) "
							"and provide a clinical finding in under 15 words."
						),
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:{mime_type};base64,{image_b64}",
						},
					},
				],
			},
		],
	)

	return (response.choices[0].message.content or "").strip()


def generate_clinical_report(image_path: str | Path) -> str:
	"""Compatibility wrapper for callers that expect a report-generation API."""
	return analyze_xray_with_gpt4o(image_path)


def _resolve_sample_image(explicit_path: str | None) -> Path:
	"""Pick an explicit image path, or discover one in common processed folders."""
	if explicit_path:
		return Path(explicit_path)

	search_dirs = [Path("dataset/processed"), Path("data/processed")]
	patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")

	for folder in search_dirs:
		if not folder.exists():
			continue
		for pattern in patterns:
			matches = sorted(folder.glob(pattern))
			if matches:
				return matches[0]

	raise FileNotFoundError(
		"No sample image found. Add an image to dataset/processed or data/processed, "
		"or pass --image-path."
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Analyze a DIP-enhanced X-ray using gpt-4o.")
	parser.add_argument(
		"--image-path",
		default=None,
		help="Optional image path. If omitted, the script picks the first image in dataset/processed or data/processed.",
	)
	args = parser.parse_args()

	sample_image = _resolve_sample_image(args.image_path)
	finding = analyze_xray_with_gpt4o(sample_image)

	print(f"Image: {sample_image}")
	print(f"Clinical finding: {finding}")


if __name__ == "__main__":
	main()
