#!/usr/bin/env python3
"""
Edit captions in zitpack archives.

Updates are done via rebuild - the archive is rewritten with modified captions.
Original archive is preserved as .bak until verification completes.

Usage:
  python archive_edit.py --archive ./data.zitpack --entry img.png --caption "new caption"
  python archive_edit.py --archive ./data.zitpack --csv ./updates.csv

Example:
  python archive_edit.py --archive ./data.zitpack --entry photo001.jpg --caption "a photo of a cat"
  python archive_edit.py --archive ./data.zitpack --csv captions.csv

CSV format:
  filename,caption
  image001.png,"a beautiful sunset"
  image002.jpg,"portrait of a person"
"""

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_archive import DatasetArchive, DatasetArchiveWriter, Flags


def print_progress(current: int, total: int, prefix: str = "Processing"):
    """Print progress that works in both terminal and GUI modes."""
    if total == 0:
        return
    percent = (current / total) * 100
    message = f"{prefix}: {current}/{total} ({percent:.0f}%)"

    if sys.stdout.isatty():
        print(f"\r{message}", end="", flush=True)
    else:
        if current == 1 or current == total or current % max(1, total // 20) == 0:
            print(message, flush=True)


def load_updates_from_csv(csv_path: Path) -> dict[str, str]:
    """Load caption updates from CSV file."""
    updates = {}

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        # Check for required columns
        if 'filename' not in reader.fieldnames or 'caption' not in reader.fieldnames:
            raise ValueError("CSV must have 'filename' and 'caption' columns")

        for row in reader:
            filename = row['filename'].strip()
            caption = row['caption'].strip()
            if filename:
                updates[filename] = caption

    return updates


def rebuild_archive(
    archive_path: Path,
    updates: dict[str, str],
) -> tuple[int, list[str]]:
    """
    Rebuild archive with updated captions.

    Args:
        archive_path: Path to the archive to modify
        updates: Dict mapping filename to new caption

    Returns:
        Tuple of (update_count, list of not_found filenames)
    """
    # Create temporary file for new archive
    temp_fd, temp_path = tempfile.mkstemp(suffix='.zitpack', dir=archive_path.parent)
    temp_path = Path(temp_path)

    try:
        update_count = 0
        not_found = []

        with DatasetArchive(archive_path) as archive:
            # Check which updates are valid
            archive_filenames = set(archive.list_entries())
            for filename in updates:
                if filename not in archive_filenames:
                    not_found.append(filename)

            # Determine if checksums should be computed
            compute_checksums = bool(archive.header.flags & Flags.HAS_CHECKSUM)

            # Create writer
            writer = DatasetArchiveWriter(
                temp_path,
                chunk_index=archive.chunk_index,
                total_chunks=archive.total_chunks,
                compute_checksums=compute_checksums,
            )

            total = len(archive)

            # Copy entries, applying updates
            for i, (image_bytes, old_caption, filename) in enumerate(archive):
                print_progress(i + 1, total, "Rebuilding")

                # Check if this entry has an update
                if filename in updates:
                    new_caption = updates[filename]
                    update_count += 1
                else:
                    new_caption = old_caption

                # Get original entry info
                info = archive.get_entry_info(i)

                # Add to new archive
                writer.add_entry_bytes(
                    image_bytes,
                    new_caption,
                    filename,
                    width=info.width,
                    height=info.height,
                    image_format=info.image_format,
                )

            writer.finalize()

        # Clear progress line
        if sys.stdout.isatty():
            print()

        # Atomic replace: backup original, then rename temp to original
        backup_path = archive_path.with_suffix('.zitpack.bak')

        # Remove old backup if exists
        if backup_path.exists():
            backup_path.unlink()

        # Backup original
        shutil.move(str(archive_path), str(backup_path))

        # Move new archive into place
        shutil.move(str(temp_path), str(archive_path))

        # Verify new archive
        with DatasetArchive(archive_path) as archive:
            if len(archive) != total:
                # Restore from backup
                shutil.move(str(backup_path), str(archive_path))
                raise RuntimeError("Verification failed: entry count mismatch")

        # Remove backup
        backup_path.unlink()

        return update_count, not_found

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Edit captions in zitpack archives.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Update single caption
  python archive_edit.py --archive ./data.zitpack --entry img.png --caption "new caption"

  # Batch update from CSV
  python archive_edit.py --archive ./data.zitpack --csv updates.csv

CSV format:
  filename,caption
  image001.png,"a beautiful sunset"
  image002.jpg,"portrait of a person"

Note:
  The archive is rebuilt with updated captions. Original is kept as .bak
  during the operation and removed after successful verification.
        """
    )
    parser.add_argument('--archive', '-a', type=Path, required=True,
                        help='Path to .zitpack archive to modify')
    parser.add_argument('--entry', '-e', type=str,
                        help='Filename of entry to update')
    parser.add_argument('--caption', '-c', type=str,
                        help='New caption text (use with --entry)')
    parser.add_argument('--csv', type=Path,
                        help='CSV file with updates (columns: filename, caption)')

    args = parser.parse_args()

    # Validate arguments
    if not args.archive.exists():
        print(f"Error: Archive not found: {args.archive}")
        return 1

    if args.entry and args.caption is None:
        print("Error: --caption is required with --entry")
        return 1

    if args.caption and not args.entry:
        print("Error: --entry is required with --caption")
        return 1

    if not args.entry and not args.csv:
        print("Error: Either --entry/--caption or --csv is required")
        return 1

    if args.entry and args.csv:
        print("Error: Cannot use both --entry and --csv")
        return 1

    # Build updates dict
    if args.csv:
        if not args.csv.exists():
            print(f"Error: CSV file not found: {args.csv}")
            return 1

        try:
            updates = load_updates_from_csv(args.csv)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return 1

        if not updates:
            print("No updates found in CSV file")
            return 0

        print(f"Loaded {len(updates)} updates from {args.csv}")
    else:
        updates = {args.entry: args.caption}
        print(f"Updating caption for: {args.entry}")

    print(f"Archive: {args.archive}")
    print()

    try:
        update_count, not_found = rebuild_archive(args.archive, updates)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"Updated {update_count} caption(s)")

    if not_found:
        print(f"\nWarning: {len(not_found)} entries not found:")
        for filename in not_found[:10]:
            print(f"  {filename}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")

    return 0


if __name__ == '__main__':
    sys.exit(main())
