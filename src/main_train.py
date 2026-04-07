"""CLI entry point for training."""

from __future__ import annotations

import argparse

from src.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the chest X-ray report generation pipeline.")
    parser.add_argument("--csv-path", default=None, help="Optional CSV manifest with image_path and report columns.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the default epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the default batch size.")
    args = parser.parse_args()
    train(csv_path=args.csv_path, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
