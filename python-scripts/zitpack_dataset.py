#!/usr/bin/env python3
"""
PyTorch Dataset for training directly from zitpack archives.

Memory-efficient: only indices are kept in RAM, images are read on-demand.

Usage:
    from zitpack_dataset import ZitpackDataset
    from torch.utils.data import DataLoader

    dataset = ZitpackDataset("./archives/dataset_chunk_*.zitpack")
    loader = DataLoader(dataset, batch_size=4, num_workers=4)

    for batch in loader:
        images = batch['image']  # PIL Images or tensors if transform provided
        prompts = batch['prompt']  # List of caption strings
        filenames = batch['filename']  # List of filenames
"""

import glob
import io
import sys
from pathlib import Path
from typing import Callable, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_archive import DatasetArchive

# Only import torch/PIL if available
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # Fallback for type hints

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ZitpackDataset(Dataset):
    """
    PyTorch Dataset that reads directly from zitpack archives.

    Features:
    - Memory-efficient: only indices in RAM, images read on-demand
    - Multi-archive: seamlessly combines multiple chunk files
    - Thread-safe: each worker gets its own file handles
    - Transforms: optional image transforms (e.g., torchvision.transforms)

    Args:
        archive_paths: Path pattern (glob) or list of archive paths
        transform: Optional transform to apply to images
        return_pil: If True, return PIL Images; if False, return raw bytes
        verify_checksums: If True, verify CRC32 on every read (slower)

    Returns dict with keys:
        - 'image': PIL Image (or bytes if return_pil=False, or tensor if transform returns tensor)
        - 'prompt': Caption string
        - 'filename': Original filename
        - 'archive': Archive filename (useful for debugging)
        - 'index': Global index in dataset
    """

    def __init__(
        self,
        archive_paths: Union[str, Path, list[Union[str, Path]]],
        transform: Optional[Callable] = None,
        return_pil: bool = True,
        verify_checksums: bool = False,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ZitpackDataset. Install with: pip install torch")

        self.transform = transform
        self.return_pil = return_pil
        self.verify_checksums = verify_checksums

        # Expand paths
        self._archive_paths = self._expand_paths(archive_paths)

        if not self._archive_paths:
            raise ValueError(f"No archives found matching: {archive_paths}")

        # Build global index: list of (archive_idx, local_idx)
        self._global_index: list[tuple[int, int]] = []
        self._archive_lengths: list[int] = []

        for archive_idx, path in enumerate(self._archive_paths):
            with DatasetArchive(path) as archive:
                length = len(archive)
                self._archive_lengths.append(length)

                for local_idx in range(length):
                    self._global_index.append((archive_idx, local_idx))

        # Worker-local archive handles (populated lazily per worker)
        self._archives: dict[int, DatasetArchive] = {}

    def _expand_paths(self, paths: Union[str, Path, list[Union[str, Path]]]) -> list[Path]:
        """Expand path patterns to list of archive paths."""
        if isinstance(paths, (str, Path)):
            paths = [paths]

        result = []
        for p in paths:
            p_str = str(p)
            if '*' in p_str or '?' in p_str or '[' in p_str:
                # Glob pattern
                matched = sorted(glob.glob(p_str))
                result.extend(Path(m) for m in matched if m.endswith('.zitpack'))
            else:
                # Literal path
                path = Path(p)
                if path.exists() and path.suffix.lower() == '.zitpack':
                    result.append(path)

        return result

    def _get_archive(self, archive_idx: int) -> DatasetArchive:
        """Get archive handle for current worker (lazy init)."""
        if archive_idx not in self._archives:
            path = self._archive_paths[archive_idx]
            self._archives[archive_idx] = DatasetArchive(
                path, verify_checksums=self.verify_checksums
            )
        return self._archives[archive_idx]

    def __len__(self) -> int:
        """Total number of entries across all archives."""
        return len(self._global_index)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single entry by global index.

        Returns:
            Dict with 'image', 'prompt', 'filename', 'archive', 'index'
        """
        if idx < 0 or idx >= len(self._global_index):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        archive_idx, local_idx = self._global_index[idx]
        archive = self._get_archive(archive_idx)

        image_bytes, caption = archive[local_idx]
        info = archive.get_entry_info(local_idx)

        # Process image
        if self.return_pil and HAS_PIL:
            image = Image.open(io.BytesIO(image_bytes))
            if self.transform is not None:
                image = self.transform(image)
        elif self.transform is not None:
            # Transform on bytes (unusual but allowed)
            image = self.transform(image_bytes)
        else:
            image = image_bytes

        return {
            'image': image,
            'prompt': caption,
            'filename': info.filename,
            'archive': self._archive_paths[archive_idx].name,
            'index': idx,
        }

    def get_entry_info(self, idx: int) -> dict:
        """
        Get metadata about an entry without loading the image.

        Returns:
            Dict with 'filename', 'image_size', 'caption_size', 'width', 'height'
        """
        archive_idx, local_idx = self._global_index[idx]
        archive = self._get_archive(archive_idx)
        info = archive.get_entry_info(local_idx)

        return {
            'filename': info.filename,
            'image_size': info.image_size,
            'caption_size': info.caption_size,
            'width': info.width,
            'height': info.height,
            'archive': self._archive_paths[archive_idx].name,
        }

    def close(self) -> None:
        """Close all open archive handles."""
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    @property
    def archive_paths(self) -> list[Path]:
        """List of archive paths being used."""
        return list(self._archive_paths)

    @property
    def num_archives(self) -> int:
        """Number of archives in the dataset."""
        return len(self._archive_paths)


def collate_fn(batch: list[dict]) -> dict:
    """
    Default collate function for ZitpackDataset.

    Handles mixed data types (PIL Images, tensors, strings).
    Use this with DataLoader if you need custom batching.
    """
    if not batch:
        return {}

    result = {
        'image': [],
        'prompt': [],
        'filename': [],
        'archive': [],
        'index': [],
    }

    for item in batch:
        for key in result:
            result[key].append(item.get(key))

    # Stack tensors if possible
    if HAS_TORCH and result['image'] and isinstance(result['image'][0], torch.Tensor):
        result['image'] = torch.stack(result['image'])

    return result


# Example usage and testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test ZitpackDataset')
    parser.add_argument('archives', type=str,
                        help='Archive path or glob pattern')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of DataLoader workers')

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    from torch.utils.data import DataLoader

    print(f"Loading dataset from: {args.archives}")
    dataset = ZitpackDataset(args.archives)

    print(f"Total entries: {len(dataset)}")
    print(f"Archives: {dataset.num_archives}")

    if len(dataset) == 0:
        print("Dataset is empty!")
        sys.exit(1)

    # Test single access
    print("\nFirst entry:")
    item = dataset[0]
    print(f"  Filename: {item['filename']}")
    print(f"  Prompt: {item['prompt'][:100]}..." if len(item['prompt']) > 100 else f"  Prompt: {item['prompt']}")
    print(f"  Image type: {type(item['image'])}")

    # Test DataLoader
    print(f"\nTesting DataLoader (batch_size={args.batch_size}, workers={args.num_workers})...")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    for i, batch in enumerate(loader):
        if i >= 2:
            break
        print(f"  Batch {i}: {len(batch['image'])} items")

    print("\nDataset test passed!")
    dataset.close()
