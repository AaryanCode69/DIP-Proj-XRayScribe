"""Project-wide configuration dataclasses for quick bootstrap.

Phase 0 goal: centralize default values so upcoming phases can import from one place.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DIPConfig:
    image_height: int = 512
    image_width: int = 512
    tile_height: int = 32
    tile_width: int = 32
    clip_limit: int = 40
    morphology_kernel_size: int = 3


@dataclass(frozen=True)
class ModelConfig:
    encoder_backbone: str = "resnet18"
    encoder_out_channels: int = 512
    pooled_h: int = 7
    pooled_w: int = 7
    decoder_hidden_dim: int = 512
    token_embedding_dim: int = 256
    max_seq_len: int = 80


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    random_seed: int = 42


DIP_CFG = DIPConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
