"""CLI entry point for inference/evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the chest X-ray report generation pipeline.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional checkpoint path.")
    args = parser.parse_args()
    outputs = evaluate(checkpoint_path=args.checkpoint_path)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
