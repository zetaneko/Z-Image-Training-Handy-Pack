"""
PyTorch Dataset wrapper for zitpack archives.

Provides ZitpackDataset, a torch.utils.data.Dataset that reads image/caption
pairs from one or more .zitpack archive files. Returns data in the same format
as DiffSynth-Studio's UnifiedDataset: {"image": PIL.Image, "prompt": str}.
"""

from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

from .reader import DatasetArchive


class ZitpackDataset:
    """
    PyTorch-compatible Dataset that reads from .zitpack archives.

    Supports multiple archive chunk files with unified indexing and repeat
    multiplier for epoch expansion.

    Usage:
        from dataset_archive import ZitpackDataset

        dataset = ZitpackDataset(["chunk_000.zitpack", "chunk_001.zitpack"], repeat=10)
        sample = dataset[0]  # {"image": PIL.Image, "prompt": str}
        print(len(dataset))  # total_entries * repeat
    """

    def __init__(
        self,
        archive_paths: list[Union[str, Path]],
        repeat: int = 1,
    ):
        """
        Args:
            archive_paths: List of paths to .zitpack archive files.
            repeat: Number of times to repeat the dataset (for epoch expansion).
        """
        if not archive_paths:
            raise ValueError("archive_paths must not be empty")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")

        self.repeat = repeat
        self._archives: list[DatasetArchive] = []
        self._global_index: list[tuple[int, int]] = []  # (archive_idx, entry_idx)

        for path in archive_paths:
            archive = DatasetArchive(path)
            archive_idx = len(self._archives)
            self._archives.append(archive)
            for entry_idx in range(len(archive)):
                self._global_index.append((archive_idx, entry_idx))

    def __getitem__(self, data_id: int) -> dict:
        """
        Get an image/caption pair by index.

        Args:
            data_id: Index into the (repeated) dataset.

        Returns:
            Dict with "image" (PIL.Image in RGB) and "prompt" (str).
        """
        real_idx = data_id % len(self._global_index)
        archive_idx, entry_idx = self._global_index[real_idx]

        image_bytes, caption = self._archives[archive_idx][entry_idx]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        return {"image": image, "prompt": caption}

    def __len__(self) -> int:
        return len(self._global_index) * self.repeat

    def close(self):
        """Close all archive file handles."""
        for archive in self._archives:
            archive.close()
        self._archives.clear()
        self._global_index.clear()

    @property
    def total_entries(self) -> int:
        """Number of unique entries (before repeat)."""
        return len(self._global_index)

    @property
    def archive_count(self) -> int:
        """Number of loaded archive files."""
        return len(self._archives)
