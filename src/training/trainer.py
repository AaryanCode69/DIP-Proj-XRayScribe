"""Training loop for the end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.data.demo import build_demo_records
from src.data.dataset import XRayDataset
from src.data.transforms import make_collate_fn
from src.data.vocabulary import Vocabulary
from src.models.pipeline import ReportGenerationPipeline
from src.training.loss import make_token_weights, sequence_cross_entropy
from src.utils.reproducibility import set_seed


@dataclass
class TrainingResult:
    model: ReportGenerationPipeline
    vocabulary: Vocabulary
    last_loss: float
    val_loss: float


def _demo_samples(num_samples: int = 4, image_size: tuple[int, int] = (128, 128)) -> list[dict[str, object]]:
    """Backward-compatible wrapper for older imports/tests."""
    return build_demo_records(num_samples=num_samples, image_size=image_size, seed=TRAIN_CFG.random_seed)


def _build_loader(dataset: XRayDataset, vocabulary: Vocabulary, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(vocabulary.pad_idx),
    )


def _compute_average_loss(
    model: ReportGenerationPipeline,
    loader: DataLoader,
    pad_idx: int,
    class_weights: torch.Tensor,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"]
            if tokens.numel() == 0 or tokens.size(1) < 2:
                continue
            images = batch["image"]
            decoder_inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits, _ = model(images, decoder_inputs)
            loss = sequence_cross_entropy(logits, targets, pad_idx=pad_idx, class_weights=class_weights)
            losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        return float("nan")
    return float(sum(losses) / len(losses))


def _build_training_datasets(
    csv_path: str | None,
    use_demo_data: bool,
) -> tuple[XRayDataset, XRayDataset, Vocabulary]:
    image_size = MODEL_CFG.input_image_size

    if csv_path is not None and Path(csv_path).exists():
        vocabulary = Vocabulary.build_from_csv(
            csv_path=csv_path,
            column_name="report_text",
            fallback_column_names=("report",),
            min_freq=2,
        )
        train_dataset = XRayDataset(
            csv_path=csv_path,
            split="train",
            random_seed=TRAIN_CFG.random_seed,
            vocabulary=vocabulary,
            image_size=image_size,
        )
        val_dataset = XRayDataset(
            csv_path=csv_path,
            split="val",
            random_seed=TRAIN_CFG.random_seed,
            vocabulary=vocabulary,
            image_size=image_size,
        )
        return train_dataset, val_dataset, vocabulary

    if not use_demo_data:
        raise ValueError("Either csv_path must be provided or use_demo_data must remain True.")

    records = build_demo_records(seed=TRAIN_CFG.random_seed)
    vocabulary = Vocabulary.build(record["report"] for record in records)
    train_dataset = XRayDataset(records=records, vocabulary=vocabulary, image_size=image_size)
    val_dataset = XRayDataset(records=records[: max(1, len(records) // 4)], vocabulary=vocabulary, image_size=image_size)
    return train_dataset, val_dataset, vocabulary


def _count_token_frequencies(dataset: XRayDataset, vocabulary: Vocabulary) -> dict[int, int]:
    frequencies: dict[int, int] = {}
    for record in dataset.records:
        report_text = str(record.get("report_text", record.get("report", "")))
        for token_id in vocabulary.encode(report_text, add_special_tokens=True):
            frequencies[token_id] = frequencies.get(token_id, 0) + 1
    return frequencies


def train(
    csv_path: str | None = DATA_CFG.manifest_csv,
    output_dir: str | Path = "artifacts",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    use_demo_data: bool = True,
) -> TrainingResult:
    """Train the pipeline on a manifest or a small synthetic demo set."""
    set_seed(TRAIN_CFG.random_seed)
    epochs = epochs or (1 if use_demo_data and csv_path is None else TRAIN_CFG.epochs)
    batch_size = batch_size or TRAIN_CFG.batch_size
    learning_rate = learning_rate or TRAIN_CFG.learning_rate

    train_dataset, val_dataset, vocabulary = _build_training_datasets(csv_path=csv_path, use_demo_data=use_demo_data)

    train_loader = _build_loader(train_dataset, vocabulary=vocabulary, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_dataset, vocabulary=vocabulary, batch_size=batch_size, shuffle=False)

    model = ReportGenerationPipeline(vocab_size=len(vocabulary), pad_idx=vocabulary.pad_idx, bos_idx=vocabulary.bos_idx, eos_idx=vocabulary.eos_idx)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    class_weights = make_token_weights(_count_token_frequencies(train_dataset, vocabulary), vocab_size=len(vocabulary))

    model.train()
    last_loss = 0.0
    val_loss = float("nan")
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"]
            tokens = batch["tokens"]
            if tokens.numel() == 0 or tokens.size(1) < 2:
                continue

            decoder_inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images, decoder_inputs)
            loss = sequence_cross_entropy(logits, targets, pad_idx=vocabulary.pad_idx, class_weights=class_weights)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            epoch_losses.append(last_loss)

        val_loss = _compute_average_loss(model, val_loader, pad_idx=vocabulary.pad_idx, class_weights=class_weights)
        train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
        print(f"Epoch {epoch + 1}/{epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary.token_to_id,
            "config": {
                "image_size": MODEL_CFG.input_image_size,
                "vocab_size": len(vocabulary),
            },
        },
        checkpoint_path / "report_generation_demo.pt",
    )

    return TrainingResult(model=model, vocabulary=vocabulary, last_loss=last_loss, val_loss=val_loss)
