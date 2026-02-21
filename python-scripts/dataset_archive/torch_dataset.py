"""
PyTorch Dataset wrapper for zitpack archives.

Provides ZitpackDataset, a torch.utils.data.Dataset that reads image/caption
pairs from one or more .zitpack archive files. Returns data in the same format
as DiffSynth-Studio's UnifiedDataset: {"image": PIL.Image, "prompt": str}.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .reader import DatasetArchive


def parse_zitpack_repeats(
    archive_paths: list,
    repeats_str: Optional[str],
) -> list[int]:
    """
    Match a repeats string like "anime:3,portrait:2" to a list of archive paths.

    A file matches a name if its stem equals the name or starts with name + "_".
    Files without a match get repeat=1.

    Args:
        archive_paths: List of paths to .zitpack files.
        repeats_str: Comma-separated "name:count" pairs, or None.

    Returns:
        List of repeat counts, one per archive path.

    Example:
        files = ["anime_chunk_001.zitpack", "anime_chunk_002.zitpack", "portrait_chunk_001.zitpack"]
        parse_zitpack_repeats(files, "anime:3,portrait:2")
        # -> [3, 3, 2]
    """
    repeat_map: dict[str, int] = {}
    if repeats_str:
        for part in repeats_str.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            name, _, count = part.partition(":")
            repeat_map[name.strip()] = int(count.strip())

    result = []
    for path in archive_paths:
        stem = Path(path).stem
        matched = 1
        for name, count in repeat_map.items():
            if stem == name or stem.startswith(name + "_"):
                matched = count
                break
        result.append(matched)
    return result


class ZitpackDataset:
    """
    PyTorch-compatible Dataset that reads from .zitpack archives.

    Supports multiple archive chunk files with unified indexing, a global
    repeat multiplier for epoch expansion, and per-archive repeat multipliers
    so that some datasets appear more frequently than others during training.

    Usage:
        from dataset_archive import ZitpackDataset, parse_zitpack_repeats

        files = sorted(Path("./archives").glob("*.zitpack"))
        repeats = parse_zitpack_repeats(files, "anime:3,portrait:2")
        dataset = ZitpackDataset(files, repeat=10, archive_repeats=repeats)
        sample = dataset[0]  # {"image": PIL.Image, "prompt": str}
        print(len(dataset))  # weighted_entries * repeat
    """

    def __init__(
        self,
        archive_paths: list[Union[str, Path]],
        repeat: int = 1,
        archive_repeats: Optional[list[int]] = None,
    ):
        """
        Args:
            archive_paths: List of paths to .zitpack archive files.
            repeat: Global repeat multiplier applied on top of per-archive repeats.
            archive_repeats: Per-archive repeat counts (parallel list to archive_paths).
                             When None, all archives default to repeat=1.
        """
        if not archive_paths:
            raise ValueError("archive_paths must not be empty")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        if archive_repeats is not None and len(archive_repeats) != len(archive_paths):
            raise ValueError(
                f"archive_repeats length ({len(archive_repeats)}) must match "
                f"archive_paths length ({len(archive_paths)})"
            )

        self.repeat = repeat
        self._archives: list[DatasetArchive] = []
        self._global_index: list[tuple[int, int]] = []  # (archive_idx, entry_idx)
        self._unique_count: int = 0  # unique entries before any repeats

        for i, path in enumerate(archive_paths):
            archive = DatasetArchive(path)
            archive_idx = len(self._archives)
            self._archives.append(archive)
            arc_repeat = archive_repeats[i] if archive_repeats is not None else 1
            for entry_idx in range(len(archive)):
                self._unique_count += 1
                for _ in range(arc_repeat):
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
        """Number of unique entries across all archives (before any repeats)."""
        return self._unique_count

    @property
    def weighted_entries(self) -> int:
        """Number of entries after per-archive repeats (before global repeat)."""
        return len(self._global_index)

    @property
    def archive_count(self) -> int:
        """Number of loaded archive files."""
        return len(self._archives)
