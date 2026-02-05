"""
DatasetArchiveWriter for creating zitpack archive files.
"""

import zlib
from io import BytesIO
from pathlib import Path
from typing import Optional

from .format import (
    MAGIC, VERSION_MAJOR, VERSION_MINOR, HEADER_SIZE, INDEX_ENTRY_SIZE,
    DATA_ALIGNMENT, Flags, ImageFormat, Header, IndexEntry,
    pack_index_entry, align_offset,
)


class DatasetArchiveWriter:
    """
    Creates new zitpack archive files.

    Usage:
        writer = DatasetArchiveWriter(Path("output.zitpack"), chunk_index=0, total_chunks=1)
        writer.add_entry(image_path, caption_path)
        writer.add_entry_bytes(image_bytes, "caption text", "filename.png")
        writer.finalize()
    """

    def __init__(
        self,
        output_path: Path,
        chunk_index: int = 0,
        total_chunks: int = 1,
        compute_checksums: bool = True,
    ):
        """
        Initialize a new archive writer.

        Args:
            output_path: Path to write the archive file
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks in the dataset
            compute_checksums: Whether to compute CRC32 checksums
        """
        self.output_path = Path(output_path)
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.compute_checksums = compute_checksums

        # Pending entries: list of (image_bytes, caption_str, filename, width, height, format, precomputed_crc)
        self._entries: list[tuple[bytes, str, str, int, int, ImageFormat, Optional[int]]] = []
        self._finalized = False

    def add_entry(
        self,
        image_path: Path,
        caption_path: Optional[Path] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Add an image+caption entry from files.

        Args:
            image_path: Path to the image file
            caption_path: Path to the caption .txt file (optional if caption provided)
            caption: Caption text (optional if caption_path provided)
        """
        if self._finalized:
            raise RuntimeError("Cannot add entries after finalize()")

        image_path = Path(image_path)
        image_bytes = image_path.read_bytes()

        # Get caption text
        if caption is not None:
            caption_text = caption
        elif caption_path is not None:
            caption_text = Path(caption_path).read_text(encoding='utf-8')
        else:
            # Try to find .txt file with same name
            txt_path = image_path.with_suffix('.txt')
            if txt_path.exists():
                caption_text = txt_path.read_text(encoding='utf-8')
            else:
                caption_text = ""

        # Get image dimensions
        width, height = self._get_image_dimensions(image_bytes)

        # Get image format
        img_format = ImageFormat.from_extension(image_path.suffix)

        self._entries.append((
            image_bytes,
            caption_text,
            image_path.name,
            width,
            height,
            img_format,
            None,  # No precomputed CRC
        ))

    def add_entry_bytes(
        self,
        image_bytes: bytes,
        caption: str,
        filename: str,
        width: int = 0,
        height: int = 0,
        image_format: Optional[ImageFormat] = None,
        precomputed_crc32: Optional[int] = None,
    ) -> None:
        """
        Add an image+caption entry from bytes.

        Args:
            image_bytes: Raw image data
            caption: Caption text
            filename: Original filename (for reference)
            width: Image width (0 if unknown)
            height: Image height (0 if unknown)
            image_format: Image format enum (auto-detected from filename if None)
            precomputed_crc32: Pre-computed CRC32 (avoids recomputation when copying)
        """
        if self._finalized:
            raise RuntimeError("Cannot add entries after finalize()")

        if image_format is None:
            ext = Path(filename).suffix
            image_format = ImageFormat.from_extension(ext)

        # Try to get dimensions if not provided
        if width == 0 or height == 0:
            w, h = self._get_image_dimensions(image_bytes)
            width = width or w
            height = height or h

        self._entries.append((
            image_bytes,
            caption,
            filename,
            width,
            height,
            image_format,
            precomputed_crc32,
        ))

    def _get_image_dimensions(self, image_bytes: bytes) -> tuple[int, int]:
        """Try to get image dimensions without full decode."""
        try:
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(image_bytes)) as img:
                return img.size
        except Exception:
            return (0, 0)

    def finalize(self) -> None:
        """
        Write all entries to the archive file.

        This builds the complete archive with:
        1. Header (64 bytes)
        2. Index entries (40 bytes each)
        3. String table (filenames)
        4. Data section (image+caption data, aligned)
        """
        if self._finalized:
            raise RuntimeError("Archive already finalized")
        self._finalized = True

        if not self._entries:
            raise ValueError("Cannot create empty archive")

        entry_count = len(self._entries)

        # Build string table
        string_table = BytesIO()
        string_offsets: list[tuple[int, int]] = []  # (offset, length) for each entry

        for _, _, filename, _, _, _, _ in self._entries:
            encoded = filename.encode('utf-8')
            offset = string_table.tell()
            string_table.write(encoded)
            string_offsets.append((offset, len(encoded)))

        string_table_bytes = string_table.getvalue()
        string_table_size = len(string_table_bytes)

        # Calculate section offsets
        index_offset = HEADER_SIZE
        index_size = entry_count * INDEX_ENTRY_SIZE

        string_table_offset = index_offset + index_size

        # Data starts after string table, aligned
        data_offset = align_offset(string_table_offset + string_table_size)

        # Build data section and index entries
        data_buffer = BytesIO()
        index_entries: list[IndexEntry] = []

        for i, (image_bytes, caption, filename, width, height, img_format, precomputed_crc) in enumerate(self._entries):
            caption_bytes = caption.encode('utf-8')

            # Current position in data buffer (relative to data section start)
            image_data_offset = data_offset + data_buffer.tell()

            # Write image data
            data_buffer.write(image_bytes)
            image_size = len(image_bytes)

            # Align for caption
            padding = align_offset(data_buffer.tell()) - data_buffer.tell()
            data_buffer.write(b'\x00' * padding)

            # Write caption
            caption_data_offset = data_offset + data_buffer.tell()
            data_buffer.write(caption_bytes)
            caption_size = len(caption_bytes)

            # Align for next entry
            padding = align_offset(data_buffer.tell()) - data_buffer.tell()
            data_buffer.write(b'\x00' * padding)

            # Use precomputed checksum if available, otherwise compute
            crc32 = 0
            if self.compute_checksums:
                if precomputed_crc is not None:
                    crc32 = precomputed_crc
                else:
                    crc32 = zlib.crc32(image_bytes) & 0xFFFFFFFF

            # Get string table reference
            str_offset, str_length = string_offsets[i]

            index_entries.append(IndexEntry(
                image_data_offset=image_data_offset,
                image_size=image_size,
                caption_data_offset=caption_data_offset,
                caption_size=caption_size,
                filename_string_offset=str_offset,
                filename_length=str_length,
                crc32=crc32,
                image_format=img_format,
                width=width,
                height=height,
            ))

        data_bytes = data_buffer.getvalue()
        data_size = len(data_bytes)

        # Build header
        flags = Flags.HAS_CHECKSUM if self.compute_checksums else Flags.NONE

        header = Header(
            magic=MAGIC,
            version_major=VERSION_MAJOR,
            version_minor=VERSION_MINOR,
            flags=flags,
            index_offset=index_offset,
            index_size=index_size,
            entry_count=entry_count,
            chunk_index=self.chunk_index,
            total_chunks=self.total_chunks,
            data_offset=data_offset,
            data_size=data_size,
        )

        # Write complete file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'wb') as f:
            # Header
            f.write(header.pack())

            # Index entries
            for entry in index_entries:
                f.write(pack_index_entry(entry))

            # String table
            f.write(string_table_bytes)

            # Padding to data section
            current_pos = f.tell()
            padding = data_offset - current_pos
            if padding > 0:
                f.write(b'\x00' * padding)

            # Data section
            f.write(data_bytes)

    @property
    def current_size(self) -> int:
        """Estimate current archive size in bytes."""
        if not self._entries:
            return HEADER_SIZE

        # Header + index
        size = HEADER_SIZE + len(self._entries) * INDEX_ENTRY_SIZE

        # String table
        for _, _, filename, _, _, _, _ in self._entries:
            size += len(filename.encode('utf-8'))

        # Data (with alignment overhead estimate)
        for image_bytes, caption, _, _, _, _, _ in self._entries:
            size += len(image_bytes)
            size += len(caption.encode('utf-8'))
            size += DATA_ALIGNMENT * 2  # Alignment overhead per entry

        return size

    @property
    def entry_count(self) -> int:
        """Number of entries added so far."""
        return len(self._entries)

    def copy_entry_from_archive(self, archive: 'DatasetArchive', index: int) -> None:
        """
        Copy an entry from an existing archive without re-processing.

        This is efficient because it:
        - Reads raw image bytes directly (no decode/encode)
        - Preserves the existing CRC32 checksum
        - Preserves dimensions and format metadata

        Args:
            archive: Source DatasetArchive to copy from
            index: Index of entry to copy
        """
        if self._finalized:
            raise RuntimeError("Cannot add entries after finalize()")

        # Get entry info (metadata only, no data read yet)
        info = archive.get_entry_info(index)

        # Read raw image bytes and caption
        image_bytes = archive.get_image(index)
        caption = archive.get_caption(index)

        self._entries.append((
            image_bytes,
            caption,
            info.filename,
            info.width,
            info.height,
            info.image_format,
            info.crc32 if info.crc32 != 0 else None,  # Preserve existing CRC
        ))


# Import at module level for type hints (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .reader import DatasetArchive
