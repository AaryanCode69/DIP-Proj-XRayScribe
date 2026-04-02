"""Vocabulary/tokenization placeholders for report generation."""


class Vocabulary:
    """Token-ID mapping scaffold for decoder training/inference."""

    def __init__(self) -> None:
        # TODO(Phase 4/5): build from report corpus.
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def __len__(self) -> int:
        raise NotImplementedError
