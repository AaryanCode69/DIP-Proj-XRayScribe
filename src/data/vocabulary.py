"""Vocabulary/tokenization helpers for report generation."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def tokenize(text: str) -> list[str]:
    """Deterministic tokenizer for radiology text.

    Tokenization policy for IU reports:
    - lowercase conversion
    - punctuation removal
    - keep only alphanumeric word tokens
    """
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Vocabulary:
    """Token-ID mapping for decoder training/inference."""

    token_to_id: dict[str, int]
    id_to_token: list[str]

    pad_token: str = "[PAD]"
    bos_token: str = "[START]"
    eos_token: str = "[END]"
    unk_token: str = "[UNK]"

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        if token_to_id is None:
            token_to_id = {token: index for index, token in enumerate(special_tokens)}
        else:
            token_to_id = dict(token_to_id)
            for token in special_tokens:
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
        self.token_to_id = dict(token_to_id)
        self.id_to_token = [""] * len(self.token_to_id)
        for token, index in self.token_to_id.items():
            self.id_to_token[index] = token

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        special_tokens = ["[PAD]", "[START]", "[END]", "[UNK]"]
        token_to_id = {token: index for index, token in enumerate(special_tokens)}

        sorted_tokens = [token for token, freq in counter.items() if freq >= min_freq]
        sorted_tokens.sort(key=lambda token: (-counter[token], token))

        if max_size is not None:
            sorted_tokens = sorted_tokens[:max(0, max_size - len(special_tokens))]

        for token in sorted_tokens:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id=token_to_id)

    @classmethod
    def build_from_csv(
        cls,
        csv_path: str | Path,
        column_name: str = "report_text",
        min_freq: int = 1,
    ) -> "Vocabulary":
        """Build vocabulary directly from a manifest CSV text column."""
        path = Path(csv_path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column_name not in reader.fieldnames:
                raise ValueError(f"Column '{column_name}' not found in CSV: {path}")
            texts = [row.get(column_name, "") for row in reader]
        return cls.build(texts=texts, min_freq=min_freq)

    def _resolve_idx(self, primary: str, fallback: str | None = None) -> int:
        if primary in self.token_to_id:
            return self.token_to_id[primary]
        if fallback is not None and fallback in self.token_to_id:
            return self.token_to_id[fallback]
        raise KeyError(f"Missing required token: {primary}")

    @property
    def pad_idx(self) -> int:
        return self._resolve_idx(self.pad_token, "<pad>")

    @property
    def bos_idx(self) -> int:
        return self._resolve_idx(self.bos_token, "<bos>")

    @property
    def eos_idx(self) -> int:
        return self._resolve_idx(self.eos_token, "<eos>")

    @property
    def unk_idx(self) -> int:
        return self._resolve_idx(self.unk_token, "<unk>")

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> list[int]:
        tokens = tokenize(text)
        ids = [self.token_to_id.get(token, self.unk_idx) for token in tokens]
        if add_special_tokens:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        if max_length is not None:
            ids = ids[:max_length]
            if add_special_tokens and ids and ids[-1] != self.eos_idx:
                ids[-1] = self.eos_idx
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for index in ids:
            if index < 0 or index >= len(self.id_to_token):
                token = self.unk_token
            else:
                token = self.id_to_token[index]
            if skip_special_tokens and token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            tokens.append(token)
        return " ".join(tokens).replace(" ,", ",").replace(" .", ".")
