"""Generate side-by-side raw vs CLAHE-enhanced previews for Phase 1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dip.enhancement import clahe_enhance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 preview: raw vs manual CLAHE.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Directory containing images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures/phase1_preview"),
        help="Directory to save preview figures.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of images to preview.")
    parser.add_argument("--tile-height", type=int, default=32, help="CLAHE tile height.")
    parser.add_argument("--tile-width", type=int, default=32, help="CLAHE tile width.")
    parser.add_argument("--clip-limit", type=int, default=40, help="CLAHE histogram clip limit.")
    return parser.parse_args()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale using NumPy ops only."""
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        # cv2 reads color as BGR; convert to luminance with explicit weights.
        bgr = image[..., :3].astype(np.float32)
        gray = 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
        return gray

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize arbitrary grayscale to uint8 [0, 255] for plotting."""
    if image.dtype == np.uint8:
        return image

    arr = image.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.uint8)

    scaled = (arr - min_val) / (max_val - min_val)
    return np.clip(np.rint(scaled * 255.0), 0.0, 255.0).astype(np.uint8)


def load_image(path: Path) -> np.ndarray:
    """Load one image from disk (supports DICOM and common image formats)."""
    if path.suffix.lower() == ".dcm":
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("cv2 failed to read image")
    return arr


def collect_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def save_preview(raw: np.ndarray, enhanced: np.ndarray, output_file: Path, source_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Raw")
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Enhanced (CLAHE)")
    axes[1].axis("off")

    fig.suptitle(source_name)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer")
    if args.tile_height <= 0 or args.tile_width <= 0:
        raise ValueError("--tile-height and --tile-width must be positive integers")
    if args.clip_limit <= 0:
        raise ValueError("--clip-limit must be a positive integer")

    image_paths = collect_image_paths(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found under: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_paths = image_paths[: args.num_samples]
    print(f"Found {len(image_paths)} total images. Saving {len(selected_paths)} previews...")

    saved = 0
    for idx, path in enumerate(selected_paths, start=1):
        try:
            loaded = load_image(path)
            gray = to_grayscale(loaded)
            raw_uint8 = normalize_for_display(gray)
            enhanced = clahe_enhance(
                raw_uint8,
                tile_size=(args.tile_height, args.tile_width),
                clip_limit=args.clip_limit,
            )

            stem = path.stem.replace(" ", "_")
            output_file = args.output_dir / f"{idx:02d}_{stem}_preview.png"
            save_preview(raw_uint8, enhanced, output_file, source_name=path.name)
            print(f"[OK] {path.name} -> {output_file}")
            saved += 1
        except Exception as exc:  # pragma: no cover - best-effort preview utility
            print(f"[SKIP] {path}: {exc}")

    print(f"Done. Saved {saved} preview file(s) to: {args.output_dir}")


if __name__ == "__main__":
    main()
