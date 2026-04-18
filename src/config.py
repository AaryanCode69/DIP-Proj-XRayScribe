"""Project-wide configuration dataclasses."""

from dataclasses import dataclass


DATA_PATH = "data/raw"
MANIFEST_CSV = "data/processed/iu_xray_manifest.csv"
BATCH_SIZE = 16
LEARNING_RATE = 3e-4


@dataclass(frozen=True)
class DIPConfig:
    image_height: int = 512
    image_width: int = 512
    tile_height: int = 32
    tile_width: int = 32
    clip_limit: int = 40
    morphology_kernel_size: int = 3

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_height, self.image_width)

    @property
    def tile_size(self) -> tuple[int, int]:
        return (self.tile_height, self.tile_width)


@dataclass(frozen=True)
class ModelConfig:
    encoder_backbone: str = "resnet18"
    encoder_out_channels: int = 512
    pooled_h: int = 7
    pooled_w: int = 7
    decoder_hidden_dim: int = 512
    token_embedding_dim: int = 256
    max_seq_len: int = 80

    @property
    def input_image_size(self) -> tuple[int, int]:
        """Default model input size matching the extractor downsampling schedule."""
        return (self.pooled_h * 32, self.pooled_w * 32)


@dataclass(frozen=True)
class DataConfig:
    data_path: str = DATA_PATH
    manifest_csv: str = MANIFEST_CSV


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = 1e-5
    epochs: int = 30
    random_seed: int = 42


DIP_CFG = DIPConfig()
MODEL_CFG = ModelConfig()
DATA_CFG = DataConfig()
TRAIN_CFG = TrainConfig()
