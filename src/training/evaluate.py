"""Evaluation and inference utilities for generated reports."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import MODEL_CFG
from src.data.demo import build_demo_records
from src.data.dataset import XRayDataset
from src.data.vocabulary import Vocabulary, tokenize
from src.models.pipeline import ReportGenerationPipeline


def _load_checkpoint(checkpoint_path: str | Path) -> tuple[ReportGenerationPipeline, Vocabulary]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocabulary = Vocabulary(checkpoint["vocabulary"])
    model = ReportGenerationPipeline(vocab_size=len(vocabulary), pad_idx=vocabulary.pad_idx, bos_idx=vocabulary.bos_idx, eos_idx=vocabulary.eos_idx)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary


def evaluate(checkpoint_path: str | None = None, use_demo_data: bool = True) -> list[dict[str, object]]:
    """Run a lightweight evaluation/inference pass."""
    image_size = MODEL_CFG.input_image_size
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model, vocabulary = _load_checkpoint(checkpoint_path)
    else:
        demo_records = build_demo_records(seed=42)
        demo_dataset = XRayDataset(records=demo_records, image_size=image_size)
        vocabulary = Vocabulary.build(record["report"] for record in demo_dataset.records)
        demo_dataset.vocabulary = vocabulary
        model = ReportGenerationPipeline(vocab_size=len(vocabulary), pad_idx=vocabulary.pad_idx, bos_idx=vocabulary.bos_idx, eos_idx=vocabulary.eos_idx)
        model.eval()

    dataset = XRayDataset(
        records=build_demo_records(seed=42) if use_demo_data else [],
        vocabulary=vocabulary,
        image_size=image_size,
    )
    if len(dataset) == 0:
        return []

    outputs: list[dict[str, object]] = []
    with torch.no_grad():
        for sample in dataset:
            image = sample["image"].unsqueeze(0)
            tokens, attention = model.generate(image, max_len=MODEL_CFG.max_seq_len)
            generated_text = vocabulary.decode(tokens.squeeze(0).tolist())
            outputs.append(
                {
                    "image_path": sample["image_path"],
                    "report": generated_text,
                    "ground_truth": sample.get("report_text", sample.get("report", "")),
                    "attention_shape": tuple(attention.shape),
                }
            )
    return outputs


def calculate_bleu4(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU-4 score for generated reports."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have equal length")
    if not hypotheses:
        return 0.0

    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as exc:
        raise ImportError("nltk is required for BLEU-4 evaluation. Install with: pip install nltk") from exc

    reference_tokens = [[tokenize(reference)] for reference in references]
    hypothesis_tokens = [tokenize(hypothesis) for hypothesis in hypotheses]
    smoothing = SmoothingFunction().method1
    score = corpus_bleu(reference_tokens, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    return float(score)


def evaluate_test_bleu4(checkpoint_path: str | Path, csv_path: str | Path) -> float:
    """Generate reports for test split and compute BLEU-4 against ground truth."""
    model, vocabulary = _load_checkpoint(checkpoint_path)
    dataset = XRayDataset(
        csv_path=csv_path,
        split="test",
        random_seed=42,
        vocabulary=vocabulary,
        image_size=MODEL_CFG.input_image_size,
    )
    if len(dataset) == 0:
        return 0.0

    hypotheses: list[str] = []
    references: list[str] = []
    with torch.no_grad():
        for sample in dataset:
            image = sample["image"].unsqueeze(0)
            generated_tokens, _ = model.generate(image, max_len=MODEL_CFG.max_seq_len)
            hypotheses.append(vocabulary.decode(generated_tokens.squeeze(0).tolist()))
            references.append(str(sample.get("report_text", sample.get("report", ""))))
    return calculate_bleu4(hypotheses=hypotheses, references=references)
