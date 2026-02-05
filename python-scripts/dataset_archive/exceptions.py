"""
Exception classes for the zitpack dataset archive system.
"""


class ZitpackError(Exception):
    """Base exception for all zitpack-related errors."""
    pass


class InvalidArchiveError(ZitpackError):
    """Raised when an archive file has invalid format or structure."""
    pass


class CorruptedDataError(ZitpackError):
    """Raised when data integrity check fails (CRC mismatch)."""
    pass


class EntryNotFoundError(ZitpackError):
    """Raised when a requested entry does not exist in the archive."""
    pass
