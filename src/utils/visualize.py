"""Visualization utility for the full chest X-ray DIP pipeline."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.transforms import to_tensor
from src.data.vocabulary import Vocabulary
from src.dip.enhancement import clahe_enhance
from src.dip.segmentation import segment_lung_fields as segment_lungs
from src.models.pipeline import ReportGenerationPipeline as Pipeline


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_existing_path(path: str | Path, *, fallback_dirs: tuple[Path, ...] = ()) -> Path:
    """Resolve a path from several common project locations."""
    candidate = Path(path)
    search_paths = [candidate]

    if not candidate.is_absolute():
        search_paths.append(Path.cwd() / candidate)
        search_paths.append(_REPO_ROOT / candidate)
        for fallback_dir in fallback_dirs:
            search_paths.append(fallback_dir / candidate.name)

    for search_path in search_paths:
        if search_path.exists():
            return search_path

    raise FileNotFoundError(f"Unable to locate file: {path}")


def _load_vocabulary(checkpoint: dict, vocab_csv_path: str | Path | None) -> Vocabulary:
    """Resolve the vocabulary used for decoding.

    Preference order:
    1. Vocabulary built from the provided CSV manifest.
    2. Vocabulary saved inside the checkpoint, if available.
    3. Minimal fallback vocabulary.
    """
    if vocab_csv_path:
        return Vocabulary.build_from_csv(vocab_csv_path, column_name="report_text", min_freq=2)

    if "vocabulary" in checkpoint and checkpoint["vocabulary"]:
        return Vocabulary(checkpoint["vocabulary"])

    return Vocabulary.build(["no acute cardiopulmonary abnormality"])


def _load_checkpoint_model(checkpoint_path: str | Path, vocabulary: Vocabulary) -> Pipeline:
    """Instantiate the pipeline and load the serialized weights."""
    resolved_checkpoint_path = _resolve_existing_path(
        checkpoint_path,
        fallback_dirs=(_REPO_ROOT / "artifacts" / "checkpoints",),
    )
    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    model = Pipeline(
        vocab_size=len(vocabulary),
        pad_idx=vocabulary.pad_idx,
        bos_idx=vocabulary.bos_idx,
        eos_idx=vocabulary.eos_idx,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _prepare_image(image_path: str | Path) -> np.ndarray:
    """Load a grayscale image using cv2 only, per project constraints."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return image.astype(np.float32)


def _generate_report(model: Pipeline, image_tensor: torch.Tensor, vocabulary: Vocabulary) -> str:
    """Run greedy decoding and map token ids back to text."""
    with torch.no_grad():
        generated_tokens, _ = model.generate(image_tensor.unsqueeze(0), max_len=80)
    token_ids = generated_tokens.squeeze(0).tolist()
    report = vocabulary.decode(token_ids)
    return report.strip()


def visualize_pipeline(
    image_path: str | Path,
    checkpoint_path: str | Path,
    vocab_csv_path: str | Path | None = None,
) -> tuple[plt.Figure, str]:
    """Visualize the full DIP pipeline from raw image to generated report."""
    resolved_checkpoint_path = _resolve_existing_path(
        checkpoint_path,
        fallback_dirs=(_REPO_ROOT / "artifacts" / "checkpoints",),
    )
    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    candidate_vocabulary = _load_vocabulary(checkpoint, vocab_csv_path)
    checkpoint_vocabulary = Vocabulary(checkpoint["vocabulary"]) if "vocabulary" in checkpoint and checkpoint["vocabulary"] else None

    def _build_model(vocabulary: Vocabulary) -> Pipeline:
        model = Pipeline(
            vocab_size=len(vocabulary),
            pad_idx=vocabulary.pad_idx,
            bos_idx=vocabulary.bos_idx,
            eos_idx=vocabulary.eos_idx,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    try:
        vocabulary = candidate_vocabulary
        model = _build_model(vocabulary)
    except RuntimeError:
        if checkpoint_vocabulary is None:
            raise
        vocabulary = checkpoint_vocabulary
        model = _build_model(vocabulary)

    original = _prepare_image(image_path)
    enhanced = clahe_enhance(original)
    mask, masked = segment_lungs(enhanced)
    input_tensor = to_tensor(masked)
    generated_report = _generate_report(model, input_tensor, vocabulary)

    figure, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")

    axes[1].imshow(enhanced, cmap="gray")
    axes[1].set_title("Enhanced Image (CLAHE)")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Lung Mask")

    axes[3].imshow(masked, cmap="gray")
    axes[3].set_title("Final Masked Image (CNN Input)")

    for axis in axes:
        axis.axis("off")

    figure.suptitle("Full DIP Pipeline Visualization", fontsize=16, y=0.98)
    wrapped_report = textwrap.fill(f"Generated Report: {generated_report}", width=120)
    figure.text(
        0.5,
        0.01,
        wrapped_report,
        ha="center",
        va="bottom",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black", "pad": 6},
    )
    figure.tight_layout(rect=[0.0, 0.08, 1.0, 0.92])
    return figure, generated_report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the full chest X-ray DIP pipeline.")
    parser.add_argument("image_path", help="Path to the chest X-ray image.")
    parser.add_argument("checkpoint_path", help="Path to the trained model checkpoint.")
    parser.add_argument(
        "--vocab-csv-path",
        default=None,
        help="Optional CSV manifest used to build a vocabulary when the checkpoint does not include one.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to save the visualization figure (PNG, PDF, etc.).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    figure, report = visualize_pipeline(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint_path,
        vocab_csv_path=args.vocab_csv_path,
    )

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    print("Generated report:\n" + report)


if __name__ == "__main__":
    main()