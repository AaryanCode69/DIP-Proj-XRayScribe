"""Phase 0 smoke test: verify package imports and baseline utilities."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DIP_CFG, MODEL_CFG, TRAIN_CFG
from src.utils.reproducibility import set_seed

# Module imports only (no execution of TODO logic).
from src.dip import enhancement, segmentation  # noqa: F401
from src.models import extraction, attention, decoder  # noqa: F401
from src.data import dataset, transforms, vocabulary  # noqa: F401
from src.training import loss, trainer, evaluate  # noqa: F401


def main() -> None:
    set_seed(TRAIN_CFG.random_seed)
    print("Smoke import test passed.")
    print(f"DIP config: {DIP_CFG}")
    print(f"Model config: {MODEL_CFG}")
    print(f"Train config: {TRAIN_CFG}")


if __name__ == "__main__":
    main()
