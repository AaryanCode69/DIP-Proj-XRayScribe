"""Training loop for the end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.data.dataset import XRayDataset
from src.data.transforms import collate_batch
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
    rng = np.random.default_rng(TRAIN_CFG.random_seed)
    samples: list[dict[str, object]] = []
    reports = [
        "heart size is normal . no focal consolidation .",
        "mild bibasal opacity is present .",
        "no pleural effusion or pneumothorax .",
        "low lung volume with mild atelectasis .",
    ]
    for index in range(num_samples):
        image = rng.integers(0, 256, size=image_size, dtype=np.uint8)
        samples.append({"image_path": f"demo_{index}.png", "image_array": image, "report": reports[index % len(reports)]})
    return samples


def _build_loader(dataset: XRayDataset, vocabulary: Vocabulary, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, pad_idx=vocabulary.pad_idx),
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

    if csv_path is not None and Path(csv_path).exists():
        vocabulary = Vocabulary.build_from_csv(csv_path=csv_path, column_name="report_text", min_freq=2)
        train_dataset = XRayDataset(
            csv_path=csv_path,
            split="train",
            random_seed=TRAIN_CFG.random_seed,
            vocabulary=vocabulary,
            image_size=(MODEL_CFG.pooled_h * 32, MODEL_CFG.pooled_w * 32),
        )
        val_dataset = XRayDataset(
            csv_path=csv_path,
            split="val",
            random_seed=TRAIN_CFG.random_seed,
            vocabulary=vocabulary,
            image_size=(MODEL_CFG.pooled_h * 32, MODEL_CFG.pooled_w * 32),
        )
    elif use_demo_data:
        records = _demo_samples()
        vocabulary = Vocabulary.build(record["report"] for record in records)
        train_dataset = XRayDataset(records=records, vocabulary=vocabulary, image_size=(MODEL_CFG.pooled_h * 32, MODEL_CFG.pooled_w * 32))
        val_dataset = XRayDataset(records=records[: max(1, len(records) // 4)], vocabulary=vocabulary, image_size=(MODEL_CFG.pooled_h * 32, MODEL_CFG.pooled_w * 32))
    else:
        raise ValueError("Either csv_path must be provided or use_demo_data must remain True.")

    train_loader = _build_loader(train_dataset, vocabulary=vocabulary, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_dataset, vocabulary=vocabulary, batch_size=batch_size, shuffle=False)

    model = ReportGenerationPipeline(vocab_size=len(vocabulary), pad_idx=vocabulary.pad_idx, bos_idx=vocabulary.bos_idx, eos_idx=vocabulary.eos_idx)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    token_frequencies: dict[int, int] = {}
    for record in train_dataset.records:
        report_text = record.get("report_text", record.get("report", ""))
        encoded = vocabulary.encode(report_text, add_special_tokens=True)
        for token_id in encoded:
            token_frequencies[token_id] = token_frequencies.get(token_id, 0) + 1
    class_weights = make_token_weights(token_frequencies, vocab_size=len(vocabulary))

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
                "image_size": (MODEL_CFG.pooled_h * 32, MODEL_CFG.pooled_w * 32),
                "vocab_size": len(vocabulary),
            },
        },
        checkpoint_path / "report_generation_demo.pt",
    )

    return TrainingResult(model=model, vocabulary=vocabulary, last_loss=last_loss, val_loss=val_loss)
