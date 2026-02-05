#!/usr/bin/env python3
"""
Inspect zitpack archive contents.

Usage:
  python archive_info.py ./data.zitpack              # Summary
  python archive_info.py ./data.zitpack --list       # List all entries
  python archive_info.py ./data.zitpack --list -c    # List with captions
  python archive_info.py ./data.zitpack --entry img.png  # Entry details
  python archive_info.py ./data.zitpack --verify     # Verify checksums

Example:
  python archive_info.py ./archives/dataset_chunk_000.zitpack --list -c
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_archive import DatasetArchive, ImageFormat, Flags


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def show_summary(archive_path: Path) -> None:
    """Show archive summary information."""
    with DatasetArchive(archive_path) as archive:
        header = archive.header
        file_size = archive_path.stat().st_size

        print(f"Archive: {archive_path.name}")
        print(f"File size: {format_size(file_size)}")
        print()
        print(f"Version: {header.version_major}.{header.version_minor}")
        print(f"Entries: {header.entry_count}")
        print(f"Chunk: {header.chunk_index + 1} of {header.total_chunks}")
        print()
        print(f"Index offset: {header.index_offset}")
        print(f"Index size: {format_size(header.index_size)}")
        print(f"Data offset: {header.data_offset}")
        print(f"Data size: {format_size(header.data_size)}")
        print()
        print(f"Checksums: {'yes' if header.flags & Flags.HAS_CHECKSUM else 'no'}")

        # Calculate some statistics
        if header.entry_count > 0:
            total_image_size = 0
            total_caption_size = 0
            widths = []
            heights = []
            formats = {}

            for i in range(header.entry_count):
                info = archive.get_entry_info(i)
                total_image_size += info.image_size
                total_caption_size += info.caption_size
                if info.width > 0:
                    widths.append(info.width)
                if info.height > 0:
                    heights.append(info.height)

                fmt_name = info.image_format.name
                formats[fmt_name] = formats.get(fmt_name, 0) + 1

            print()
            print("Statistics:")
            print(f"  Total image data: {format_size(total_image_size)}")
            print(f"  Total caption data: {format_size(total_caption_size)}")
            print(f"  Avg image size: {format_size(total_image_size // header.entry_count)}")

            if widths and heights:
                print(f"  Image dimensions: {min(widths)}-{max(widths)} x {min(heights)}-{max(heights)}")

            if formats:
                print(f"  Formats: {', '.join(f'{k}={v}' for k, v in sorted(formats.items()))}")


def list_entries(archive_path: Path, verbose: bool = False, show_captions: bool = False) -> None:
    """List all entries in the archive."""
    with DatasetArchive(archive_path) as archive:
        print(f"Entries in {archive_path.name}:")
        print()

        if show_captions:
            # Caption mode: show filename and caption text
            for i in range(len(archive)):
                info = archive.get_entry_info(i)
                caption = archive.get_caption(i)

                print(f"[{i}] {info.filename}")
                if caption:
                    # Truncate long captions
                    if len(caption) > 200:
                        print(f"    {caption[:200]}...")
                    else:
                        print(f"    {caption}")
                else:
                    print("    (no caption)")
                print()
        elif verbose:
            print(f"{'#':>5}  {'Filename':<40}  {'Image Size':>12}  {'Caption':>8}  {'Dimensions':>12}  {'Format':<6}")
            print("-" * 95)

            for i in range(len(archive)):
                info = archive.get_entry_info(i)
                dims = f"{info.width}x{info.height}" if info.width > 0 else "-"
                print(
                    f"{i:>5}  {info.filename:<40}  "
                    f"{format_size(info.image_size):>12}  "
                    f"{info.caption_size:>8}  "
                    f"{dims:>12}  "
                    f"{info.image_format.name:<6}"
                )
        else:
            print(f"{'#':>5}  {'Filename':<60}")
            print("-" * 68)

            for i in range(len(archive)):
                info = archive.get_entry_info(i)
                print(f"{i:>5}  {info.filename:<60}")

        print()
        print(f"Total: {len(archive)} entries")


def show_entry(archive_path: Path, entry_key: str) -> None:
    """Show detailed information about a specific entry."""
    with DatasetArchive(archive_path) as archive:
        # Try to resolve as filename first, then as index
        try:
            if entry_key.isdigit():
                idx = int(entry_key)
            else:
                idx = entry_key

            info = archive.get_entry_info(idx)
        except Exception as e:
            print(f"Error: Entry not found: {entry_key}")
            return

        print(f"Entry: {info.filename}")
        print()
        print(f"Index: {info.index}")
        print(f"Image size: {format_size(info.image_size)}")
        print(f"Caption size: {info.caption_size} bytes")
        print(f"Dimensions: {info.width} x {info.height}")
        print(f"Format: {info.image_format.name}")
        print(f"CRC32: {info.crc32:08x}")
        print()

        # Show caption preview
        caption = archive.get_caption(info.index)
        if caption:
            print("Caption:")
            # Truncate long captions
            if len(caption) > 500:
                print(f"  {caption[:500]}...")
            else:
                print(f"  {caption}")
        else:
            print("Caption: (empty)")


def verify_archive(archive_path: Path) -> bool:
    """Verify all checksums in the archive."""
    print(f"Verifying {archive_path.name}...")

    with DatasetArchive(archive_path) as archive:
        if not (archive.header.flags & Flags.HAS_CHECKSUM):
            print("  No checksums stored in this archive.")
            return True

        errors = archive.verify_all()

        if not errors:
            print(f"  All {len(archive)} entries verified successfully.")
            return True

        print(f"  {len(errors)} error(s) found:")
        for idx, filename, error in errors:
            print(f"    [{idx}] {filename}: {error}")

        return False


def main():
    parser = argparse.ArgumentParser(
        description='Inspect zitpack archive contents.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python archive_info.py ./data.zitpack              # Summary
  python archive_info.py ./data.zitpack --list       # List entries
  python archive_info.py ./data.zitpack --list -v    # Verbose list (sizes, dimensions)
  python archive_info.py ./data.zitpack --list -c    # List with captions
  python archive_info.py ./data.zitpack --entry 0    # Entry by index
  python archive_info.py ./data.zitpack --entry img.png  # Entry by name
  python archive_info.py ./data.zitpack --verify     # Verify checksums
        """
    )
    parser.add_argument('archive', type=Path,
                        help='Path to .zitpack archive')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all entries')
    parser.add_argument('--entry', '-e', type=str,
                        help='Show details for specific entry (index or filename)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify CRC32 checksums')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show more details (with --list)')
    parser.add_argument('--captions', '-c', action='store_true',
                        help='Show captions for each entry (with --list)')

    args = parser.parse_args()

    # Validate archive path
    if not args.archive.exists():
        print(f"Error: Archive not found: {args.archive}")
        return 1
    if not args.archive.is_file():
        print(f"Error: Not a file: {args.archive}")
        return 1

    try:
        if args.verify:
            success = verify_archive(args.archive)
            return 0 if success else 1
        elif args.entry:
            show_entry(args.archive, args.entry)
        elif args.list:
            list_entries(args.archive, verbose=args.verbose, show_captions=args.captions)
        else:
            show_summary(args.archive)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
