"""
Binary format constants and struct definitions for zitpack archives.

File Structure:
    [HEADER: 64 bytes fixed]
    [INDEX SECTION: variable, loaded at open]
    [STRING TABLE: filenames, referenced by offset+length]
    [DATA SECTION: bulk data, seek-accessed]
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple


# Magic bytes identifying a zitpack file
MAGIC = b'ZITPACK\x00'

# Current format version
VERSION_MAJOR = 1
VERSION_MINOR = 0

# Header size in bytes (fixed)
HEADER_SIZE = 64

# Index entry size in bytes (fixed for O(1) access)
INDEX_ENTRY_SIZE = 40

# Chunk size limits
TARGET_CHUNK_SIZE = 512 * 1024 * 1024   # 512 MB soft target
HARD_CHUNK_LIMIT = 600 * 1024 * 1024    # 600 MB hard limit

# Data alignment
DATA_ALIGNMENT = 8  # 8-byte aligned


class Flags(IntEnum):
    """Header flag bits."""
    NONE = 0
    HAS_CHECKSUM = 1 << 2  # Bit 2: CRC32 checksums present


class ImageFormat(IntEnum):
    """Image format identifiers stored in index entries."""
    UNKNOWN = 0
    PNG = 1
    JPEG = 2
    WEBP = 3
    GIF = 4
    BMP = 5
    TIFF = 6

    @classmethod
    def from_extension(cls, ext: str) -> 'ImageFormat':
        """Get format from file extension (with or without dot)."""
        ext = ext.lower().lstrip('.')
        mapping = {
            'png': cls.PNG,
            'jpg': cls.JPEG,
            'jpeg': cls.JPEG,
            'webp': cls.WEBP,
            'gif': cls.GIF,
            'bmp': cls.BMP,
            'tiff': cls.TIFF,
            'tif': cls.TIFF,
        }
        return mapping.get(ext, cls.UNKNOWN)

    def to_extension(self) -> str:
        """Get file extension for this format."""
        mapping = {
            self.PNG: '.png',
            self.JPEG: '.jpg',
            self.WEBP: '.webp',
            self.GIF: '.gif',
            self.BMP: '.bmp',
            self.TIFF: '.tiff',
        }
        return mapping.get(self, '.bin')


# Struct format for header (64 bytes)
# 8s = magic, H = uint16, I = uint32, Q = uint64
HEADER_STRUCT = struct.Struct('<8s HH I Q Q I HH Q Q Q')

# Struct format for index entry (40 bytes)
# Q = uint64, I = uint32, H = uint16
INDEX_ENTRY_STRUCT = struct.Struct('<Q I Q I I H I H H H')


@dataclass
class Header:
    """Parsed archive header."""
    magic: bytes
    version_major: int
    version_minor: int
    flags: int
    index_offset: int
    index_size: int
    entry_count: int
    chunk_index: int
    total_chunks: int
    data_offset: int
    data_size: int
    reserved: int = 0

    def pack(self) -> bytes:
        """Pack header to bytes."""
        return HEADER_STRUCT.pack(
            self.magic,
            self.version_major,
            self.version_minor,
            self.flags,
            self.index_offset,
            self.index_size,
            self.entry_count,
            self.chunk_index,
            self.total_chunks,
            self.data_offset,
            self.data_size,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'Header':
        """Unpack header from bytes."""
        values = HEADER_STRUCT.unpack(data)
        return cls(
            magic=values[0],
            version_major=values[1],
            version_minor=values[2],
            flags=values[3],
            index_offset=values[4],
            index_size=values[5],
            entry_count=values[6],
            chunk_index=values[7],
            total_chunks=values[8],
            data_offset=values[9],
            data_size=values[10],
            reserved=values[11],
        )


class IndexEntry(NamedTuple):
    """Parsed index entry for a single image+caption pair."""
    image_data_offset: int      # Absolute offset in file
    image_size: int             # Image data size in bytes
    caption_data_offset: int    # Absolute offset in file
    caption_size: int           # Caption size in bytes
    filename_string_offset: int # Offset in string table
    filename_length: int        # Filename length in bytes
    crc32: int                  # CRC32 checksum (0 if not computed)
    image_format: int           # ImageFormat enum value
    width: int                  # Image width in pixels
    height: int                 # Image height in pixels


def pack_index_entry(entry: IndexEntry) -> bytes:
    """Pack an index entry to bytes."""
    return INDEX_ENTRY_STRUCT.pack(
        entry.image_data_offset,
        entry.image_size,
        entry.caption_data_offset,
        entry.caption_size,
        entry.filename_string_offset,
        entry.filename_length,
        entry.crc32,
        entry.image_format,
        entry.width,
        entry.height,
    )


def unpack_index_entry(data: bytes) -> IndexEntry:
    """Unpack an index entry from bytes."""
    values = INDEX_ENTRY_STRUCT.unpack(data)
    return IndexEntry(
        image_data_offset=values[0],
        image_size=values[1],
        caption_data_offset=values[2],
        caption_size=values[3],
        filename_string_offset=values[4],
        filename_length=values[5],
        crc32=values[6],
        image_format=values[7],
        width=values[8],
        height=values[9],
    )


def align_offset(offset: int, alignment: int = DATA_ALIGNMENT) -> int:
    """Round offset up to the next alignment boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)
