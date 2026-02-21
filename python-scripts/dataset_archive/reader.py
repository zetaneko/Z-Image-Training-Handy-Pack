"""
DatasetArchive reader for memory-efficient archive access.
"""

import fnmatch
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional, Union

from .exceptions import InvalidArchiveError, CorruptedDataError, EntryNotFoundError
from .format import (
    MAGIC, VERSION_MAJOR, HEADER_SIZE, INDEX_ENTRY_SIZE,
    Flags, ImageFormat, Header, IndexEntry, unpack_index_entry,
)


class ArchiveEntry:
    """Information about a single entry in the archive."""

    def __init__(
        self,
        index: int,
        filename: str,
        image_size: int,
        caption_size: int,
        width: int,
        height: int,
        image_format: ImageFormat,
        crc32: int,
    ):
        self.index = index
        self.filename = filename
        self.image_size = image_size
        self.caption_size = caption_size
        self.width = width
        self.height = height
        self.image_format = image_format
        self.crc32 = crc32

    def __repr__(self) -> str:
        return (
            f"ArchiveEntry(index={self.index}, filename={self.filename!r}, "
            f"image_size={self.image_size}, caption_size={self.caption_size}, "
            f"width={self.width}, height={self.height})"
        )


@dataclass
class SearchResult:
    """Result from a search operation."""
    index: int
    filename: str
    caption: str
    match_in: str  # 'filename', 'caption', or 'both'
    matched_text: str  # The portion that matched (for highlighting)
    archive_path: Optional[Path] = None  # Set when searching multiple archives

    def __repr__(self) -> str:
        return f"SearchResult(index={self.index}, filename={self.filename!r}, match_in={self.match_in!r})"


class DatasetArchive:
    """
    Memory-efficient archive reader.

    Only the index is loaded into RAM. Image and caption data are read
    on-demand via seek operations.

    Usage:
        archive = DatasetArchive("dataset.zitpack")

        # Access by index
        image_bytes, caption = archive[0]

        # Access by filename
        image_bytes, caption = archive["image001.png"]

        # Iterate all entries
        for image_bytes, caption, filename in archive:
            ...

        # Get entry info without loading data
        info = archive.get_entry_info(0)
        print(info.width, info.height)
    """

    def __init__(self, path: Union[str, Path], verify_checksums: bool = False):
        """
        Open an archive for reading.

        Args:
            path: Path to the .zitpack file
            verify_checksums: If True, verify CRC32 on every read (slower)
        """
        self.path = Path(path)
        self.verify_checksums = verify_checksums

        self._file = open(self.path, 'rb')
        self._header: Header
        self._index_entries: list[IndexEntry] = []
        self._filenames: list[str] = []
        self._filename_to_idx: dict[str, int] = {}

        self._load_header()
        self._load_index()

    def _load_header(self) -> None:
        """Read and validate the header."""
        header_bytes = self._file.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            raise InvalidArchiveError(f"File too small for header: {self.path}")

        self._header = Header.unpack(header_bytes)

        # Validate magic
        if self._header.magic != MAGIC:
            raise InvalidArchiveError(
                f"Invalid magic bytes: expected {MAGIC!r}, got {self._header.magic!r}"
            )

        # Check version compatibility
        if self._header.version_major > VERSION_MAJOR:
            raise InvalidArchiveError(
                f"Unsupported version: {self._header.version_major}.{self._header.version_minor} "
                f"(reader supports up to {VERSION_MAJOR}.x)"
            )

    def _load_index(self) -> None:
        """Load the index and string table into memory."""
        # Read index entries
        self._file.seek(self._header.index_offset)
        index_bytes = self._file.read(self._header.index_size)

        if len(index_bytes) < self._header.index_size:
            raise InvalidArchiveError("Truncated index section")

        for i in range(self._header.entry_count):
            offset = i * INDEX_ENTRY_SIZE
            entry_bytes = index_bytes[offset:offset + INDEX_ENTRY_SIZE]
            entry = unpack_index_entry(entry_bytes)
            self._index_entries.append(entry)

        # Read string table
        string_table_offset = self._header.index_offset + self._header.index_size
        string_table_size = self._header.data_offset - string_table_offset

        self._file.seek(string_table_offset)
        string_table = self._file.read(string_table_size)

        # Extract filenames
        for i, entry in enumerate(self._index_entries):
            start = entry.filename_string_offset
            end = start + entry.filename_length
            filename = string_table[start:end].decode('utf-8')
            self._filenames.append(filename)
            self._filename_to_idx[filename] = i

    def close(self) -> None:
        """Close the archive file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> 'DatasetArchive':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        """Return the number of entries in the archive."""
        return self._header.entry_count

    def _resolve_key(self, key: Union[int, str]) -> int:
        """Convert a key (index or filename) to an index."""
        if isinstance(key, int):
            if key < 0 or key >= len(self._index_entries):
                raise EntryNotFoundError(f"Index out of range: {key}")
            return key
        elif isinstance(key, str):
            if key not in self._filename_to_idx:
                raise EntryNotFoundError(f"Entry not found: {key}")
            return self._filename_to_idx[key]
        else:
            raise TypeError(f"Key must be int or str, got {type(key)}")

    def __getitem__(self, key: Union[int, str]) -> tuple[bytes, str]:
        """
        Get image bytes and caption by index or filename.

        Args:
            key: Integer index or filename string

        Returns:
            Tuple of (image_bytes, caption_string)
        """
        idx = self._resolve_key(key)
        return self.get_image(idx), self.get_caption(idx)

    def __iter__(self) -> Iterator[tuple[bytes, str, str]]:
        """
        Iterate over all entries.

        Yields:
            Tuple of (image_bytes, caption_string, filename)
        """
        for i in range(len(self)):
            image_bytes = self.get_image(i)
            caption = self.get_caption(i)
            filename = self._filenames[i]
            yield image_bytes, caption, filename

    def get_image(self, key: Union[int, str]) -> bytes:
        """
        Get image bytes by index or filename.

        Args:
            key: Integer index or filename string

        Returns:
            Raw image bytes
        """
        idx = self._resolve_key(key)
        entry = self._index_entries[idx]

        self._file.seek(entry.image_data_offset)
        image_bytes = self._file.read(entry.image_size)

        if len(image_bytes) < entry.image_size:
            raise CorruptedDataError(f"Truncated image data at index {idx}")

        # Verify checksum if enabled
        if self.verify_checksums and (self._header.flags & Flags.HAS_CHECKSUM):
            computed_crc = zlib.crc32(image_bytes) & 0xFFFFFFFF
            if computed_crc != entry.crc32:
                raise CorruptedDataError(
                    f"CRC32 mismatch for entry {idx}: expected {entry.crc32:08x}, "
                    f"got {computed_crc:08x}"
                )

        return image_bytes

    def get_caption(self, key: Union[int, str]) -> str:
        """
        Get caption text by index or filename.

        Args:
            key: Integer index or filename string

        Returns:
            Caption string
        """
        idx = self._resolve_key(key)
        entry = self._index_entries[idx]

        if entry.caption_size == 0:
            return ""

        self._file.seek(entry.caption_data_offset)
        caption_bytes = self._file.read(entry.caption_size)

        if len(caption_bytes) < entry.caption_size:
            raise CorruptedDataError(f"Truncated caption data at index {idx}")

        return caption_bytes.decode('utf-8', errors='replace')

    def get_entry_info(self, key: Union[int, str]) -> ArchiveEntry:
        """
        Get metadata about an entry without loading the data.

        Args:
            key: Integer index or filename string

        Returns:
            ArchiveEntry with metadata
        """
        idx = self._resolve_key(key)
        entry = self._index_entries[idx]

        return ArchiveEntry(
            index=idx,
            filename=self._filenames[idx],
            image_size=entry.image_size,
            caption_size=entry.caption_size,
            width=entry.width,
            height=entry.height,
            image_format=ImageFormat(entry.image_format),
            crc32=entry.crc32,
        )

    def list_entries(self) -> list[str]:
        """
        Get list of all filenames in the archive.

        Returns:
            List of filename strings
        """
        return list(self._filenames)

    @property
    def header(self) -> Header:
        """Access the archive header."""
        return self._header

    @property
    def chunk_index(self) -> int:
        """Index of this chunk (0-based)."""
        return self._header.chunk_index

    @property
    def total_chunks(self) -> int:
        """Total number of chunks in the dataset."""
        return self._header.total_chunks

    def search(
        self,
        query: str,
        search_in: Literal['filename', 'caption', 'both'] = 'both',
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 0,
    ) -> list[SearchResult]:
        """
        Search for entries matching a query.

        Args:
            query: Search term (substring, glob pattern, or regex)
            search_in: Where to search - 'filename', 'caption', or 'both'
            regex: If True, treat query as regex; if False, substring match
            case_sensitive: If True, match case exactly
            limit: Maximum results (0 = unlimited)

        Returns:
            List of SearchResult objects
        """
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        # Compile pattern
        if regex:
            try:
                pattern = re.compile(query, flags)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        else:
            # Escape for literal substring matching
            escaped = re.escape(query)
            pattern = re.compile(escaped, flags)

        for i, filename in enumerate(self._filenames):
            if limit > 0 and len(results) >= limit:
                break

            filename_match = None
            caption_match = None
            matched_text = ""

            # Check filename
            if search_in in ('filename', 'both'):
                match = pattern.search(filename)
                if match:
                    filename_match = match.group()

            # Check caption (only load if needed)
            if search_in in ('caption', 'both'):
                caption = self.get_caption(i)
                match = pattern.search(caption)
                if match:
                    caption_match = match.group()
            else:
                caption = ""

            # Determine match type
            if filename_match and caption_match:
                match_in = 'both'
                matched_text = filename_match
            elif filename_match:
                match_in = 'filename'
                matched_text = filename_match
            elif caption_match:
                match_in = 'caption'
                matched_text = caption_match
                if not caption:
                    caption = self.get_caption(i)
            else:
                continue  # No match

            results.append(SearchResult(
                index=i,
                filename=filename,
                caption=caption if caption else self.get_caption(i),
                match_in=match_in,
                matched_text=matched_text,
                archive_path=self.path,
            ))

        return results

    def search_filename(
        self,
        pattern: str,
        glob: bool = True,
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 0,
    ) -> list[SearchResult]:
        """
        Search entries by filename pattern.

        Args:
            pattern: Filename pattern to match
            glob: If True (default), use glob/wildcard matching (*.png, image_???.jpg)
            regex: If True, use regex matching (overrides glob)
            case_sensitive: If True, match case exactly
            limit: Maximum results (0 = unlimited)

        Returns:
            List of SearchResult objects
        """
        results = []

        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
            match_fn = lambda fn: compiled.search(fn)
        elif glob:
            # Use fnmatch for glob patterns
            if case_sensitive:
                match_fn = lambda fn: fnmatch.fnmatch(fn, pattern)
            else:
                match_fn = lambda fn: fnmatch.fnmatch(fn.lower(), pattern.lower())
        else:
            # Simple substring
            if case_sensitive:
                match_fn = lambda fn: pattern in fn
            else:
                pattern_lower = pattern.lower()
                match_fn = lambda fn: pattern_lower in fn.lower()

        for i, filename in enumerate(self._filenames):
            if limit > 0 and len(results) >= limit:
                break

            if match_fn(filename):
                results.append(SearchResult(
                    index=i,
                    filename=filename,
                    caption=self.get_caption(i),
                    match_in='filename',
                    matched_text=filename,
                    archive_path=self.path,
                ))

        return results

    def search_caption(
        self,
        query: str,
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 0,
        context_chars: int = 50,
    ) -> list[SearchResult]:
        """
        Search entries by caption content.

        Args:
            query: Text to search for in captions
            regex: If True, treat query as regex
            case_sensitive: If True, match case exactly
            limit: Maximum results (0 = unlimited)
            context_chars: Characters of context around match for matched_text

        Returns:
            List of SearchResult objects
        """
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        if regex:
            try:
                pattern = re.compile(query, flags)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        else:
            pattern = re.compile(re.escape(query), flags)

        for i in range(len(self._filenames)):
            if limit > 0 and len(results) >= limit:
                break

            caption = self.get_caption(i)
            match = pattern.search(caption)

            if match:
                # Extract context around match
                start = max(0, match.start() - context_chars)
                end = min(len(caption), match.end() + context_chars)
                context = caption[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(caption):
                    context = context + "..."

                results.append(SearchResult(
                    index=i,
                    filename=self._filenames[i],
                    caption=caption,
                    match_in='caption',
                    matched_text=context,
                    archive_path=self.path,
                ))

        return results

    def search_advanced(
        self,
        filename_pattern: Optional[str] = None,
        caption_query: Optional[str] = None,
        operator: Literal['and', 'or'] = 'and',
        regex: bool = False,
        case_sensitive: bool = False,
        min_width: int = 0,
        min_height: int = 0,
        max_width: int = 0,
        max_height: int = 0,
        has_caption: Optional[bool] = None,
        limit: int = 0,
    ) -> list[SearchResult]:
        """
        Advanced search with multiple criteria.

        Args:
            filename_pattern: Glob pattern for filename (e.g., "*.png")
            caption_query: Text to search in captions
            operator: 'and' requires all criteria match; 'or' requires any
            regex: If True, treat patterns as regex
            case_sensitive: If True, match case exactly
            min_width: Minimum image width (0 = no limit)
            min_height: Minimum image height (0 = no limit)
            max_width: Maximum image width (0 = no limit)
            max_height: Maximum image height (0 = no limit)
            has_caption: If True, only entries with captions; False, only without
            limit: Maximum results (0 = unlimited)

        Returns:
            List of SearchResult objects
        """
        results = []

        # Compile patterns
        filename_matcher = None
        caption_matcher = None

        if filename_pattern:
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                filename_matcher = re.compile(filename_pattern, flags)
            else:
                if case_sensitive:
                    filename_matcher = lambda fn: fnmatch.fnmatch(fn, filename_pattern)
                else:
                    filename_matcher = lambda fn: fnmatch.fnmatch(fn.lower(), filename_pattern.lower())

        if caption_query:
            flags = 0 if case_sensitive else re.IGNORECASE
            if regex:
                caption_matcher = re.compile(caption_query, flags)
            else:
                caption_matcher = re.compile(re.escape(caption_query), flags)

        for i, filename in enumerate(self._filenames):
            if limit > 0 and len(results) >= limit:
                break

            entry = self._index_entries[i]
            matches = []
            match_locations = []

            # Check filename
            if filename_matcher:
                if callable(filename_matcher):
                    fn_match = filename_matcher(filename)
                else:
                    fn_match = filename_matcher.search(filename)
                if fn_match:
                    matches.append(True)
                    match_locations.append('filename')
                else:
                    matches.append(False)

            # Check caption
            caption = None
            if caption_matcher or has_caption is not None:
                caption = self.get_caption(i)

                if has_caption is not None:
                    has_cap = len(caption) > 0
                    if has_caption != has_cap:
                        if operator == 'and':
                            continue
                        matches.append(False)
                    else:
                        matches.append(True)

                if caption_matcher:
                    cap_match = caption_matcher.search(caption)
                    if cap_match:
                        matches.append(True)
                        match_locations.append('caption')
                    else:
                        matches.append(False)

            # Check dimensions
            if min_width > 0 or min_height > 0 or max_width > 0 or max_height > 0:
                width_ok = (min_width == 0 or entry.width >= min_width) and \
                           (max_width == 0 or entry.width <= max_width)
                height_ok = (min_height == 0 or entry.height >= min_height) and \
                            (max_height == 0 or entry.height <= max_height)
                matches.append(width_ok and height_ok)

            # Apply operator
            if not matches:
                continue  # No criteria specified

            if operator == 'and':
                if not all(matches):
                    continue
            else:  # 'or'
                if not any(matches):
                    continue

            # Determine match location
            if 'filename' in match_locations and 'caption' in match_locations:
                match_in = 'both'
            elif 'filename' in match_locations:
                match_in = 'filename'
            elif 'caption' in match_locations:
                match_in = 'caption'
            else:
                match_in = 'filter'  # Matched by dimension/has_caption filter

            results.append(SearchResult(
                index=i,
                filename=filename,
                caption=caption if caption is not None else self.get_caption(i),
                match_in=match_in,
                matched_text=filename if match_in == 'filename' else '',
                archive_path=self.path,
            ))

        return results

    def verify_all(self) -> list[tuple[int, str, str]]:
        """
        Verify CRC32 checksums for all entries.

        Returns:
            List of (index, filename, error_message) for failed entries.
            Empty list if all entries pass.
        """
        if not (self._header.flags & Flags.HAS_CHECKSUM):
            return []  # No checksums to verify

        errors = []
        for i, entry in enumerate(self._index_entries):
            self._file.seek(entry.image_data_offset)
            image_bytes = self._file.read(entry.image_size)

            if len(image_bytes) < entry.image_size:
                errors.append((i, self._filenames[i], "Truncated data"))
                continue

            computed_crc = zlib.crc32(image_bytes) & 0xFFFFFFFF
            if computed_crc != entry.crc32:
                errors.append((
                    i,
                    self._filenames[i],
                    f"CRC mismatch: expected {entry.crc32:08x}, got {computed_crc:08x}"
                ))

        return errors
