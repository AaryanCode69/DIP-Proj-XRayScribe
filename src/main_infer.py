"""CLI entry point for inference/evaluation."""

from __future__ import annotations

import argparse

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
