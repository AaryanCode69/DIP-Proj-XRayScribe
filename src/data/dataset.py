"""Dataset placeholders for IU X-Ray image-report pairs."""

from torch.utils.data import Dataset


class ChestXrayReportDataset(Dataset):
    """Minimal dataset scaffold for Phase 5 integration."""

    def __init__(self) -> None:
        super().__init__()
        # TODO(Phase 5): load image paths, report text, and transforms.

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError
