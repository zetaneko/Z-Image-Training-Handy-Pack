"""
Dataset Archive (zitpack) - Memory-efficient chunked archive format for image training datasets.

Instead of managing millions of individual image + caption files, datasets are packed into
512MB self-contained archives with embedded indices for efficient seek-based access.

Key Features:
- Reduce millions of files to hundreds of archive files
- Minimal RAM: Only index in memory (~700KB per 10,000 images)
- Seek-based access: Read specific entries without loading entire archive
- Self-contained: Each chunk has its own index, portable across systems
- Training-ready: Direct PyTorch Dataset integration

Usage:
    # Creating archives
    from dataset_archive import DatasetArchiveWriter

    writer = DatasetArchiveWriter("output.zitpack")
    writer.add_entry(image_path, caption_path)
    writer.finalize()

    # Reading archives
    from dataset_archive import DatasetArchive

    with DatasetArchive("data.zitpack") as archive:
        image_bytes, caption = archive[0]           # By index
        image_bytes, caption = archive["img.png"]   # By filename

        for image, caption, filename in archive:
            ...  # Iterate all entries
"""

from .exceptions import (
    ZitpackError,
    InvalidArchiveError,
    CorruptedDataError,
    EntryNotFoundError,
)

from .format import (
    MAGIC,
    VERSION_MAJOR,
    VERSION_MINOR,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    TARGET_CHUNK_SIZE,
    HARD_CHUNK_LIMIT,
    Flags,
    ImageFormat,
    Header,
    IndexEntry,
)

from .reader import DatasetArchive, ArchiveEntry, SearchResult
from .writer import DatasetArchiveWriter
from .torch_dataset import ZitpackDataset, parse_zitpack_repeats


__all__ = [
    # Exceptions
    'ZitpackError',
    'InvalidArchiveError',
    'CorruptedDataError',
    'EntryNotFoundError',

    # Format constants
    'MAGIC',
    'VERSION_MAJOR',
    'VERSION_MINOR',
    'HEADER_SIZE',
    'INDEX_ENTRY_SIZE',
    'TARGET_CHUNK_SIZE',
    'HARD_CHUNK_LIMIT',
    'Flags',
    'ImageFormat',
    'Header',
    'IndexEntry',

    # Main classes
    'DatasetArchive',
    'ArchiveEntry',
    'SearchResult',
    'DatasetArchiveWriter',
    'ZitpackDataset',
    'parse_zitpack_repeats',
]

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}"
