#!/usr/bin/env python3
"""
Extract zitpack archives back to individual image and caption files.

Usage:
  python unpack_dataset.py --input "./packed/dataset_chunk_*.zitpack" --output ./extracted
  python unpack_dataset.py --input ./archive.zitpack --output ./extracted

Example:
  python unpack_dataset.py --input "./archives/mymodel_chunk_*.zitpack" --output ./restored
"""

import argparse
import glob
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_archive import DatasetArchive, ImageFormat


def print_progress(current: int, total: int, prefix: str = "Processing"):
    """Print progress that works in both terminal and GUI modes."""
    if total == 0:
        return
    percent = (current / total) * 100
    message = f"{prefix}: {current}/{total} ({percent:.0f}%)"

    if sys.stdout.isatty():
        # Terminal: use carriage return for in-place update
        print(f"\r{message}", end="", flush=True)
    else:
        # GUI/pipe: print full lines (less frequent to avoid spam)
        if current == 1 or current == total or current % max(1, total // 20) == 0:
            print(message, flush=True)


def expand_glob_pattern(pattern: str) -> list[Path]:
    """Expand a glob pattern to list of paths."""
    # Handle both literal paths and glob patterns
    if '*' in pattern or '?' in pattern or '[' in pattern:
        paths = [Path(p) for p in sorted(glob.glob(pattern))]
    else:
        paths = [Path(pattern)]

    return [p for p in paths if p.exists() and p.suffix.lower() == '.zitpack']


def unpack_archive(
    archive_path: Path,
    output_dir: Path,
    verify: bool = False,
) -> tuple[int, int]:
    """
    Extract all entries from an archive.

    Args:
        archive_path: Path to the .zitpack file
        output_dir: Directory to extract files to
        verify: Whether to verify CRC32 checksums

    Returns:
        Tuple of (images_extracted, captions_extracted)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images_extracted = 0
    captions_extracted = 0

    with DatasetArchive(archive_path, verify_checksums=verify) as archive:
        total = len(archive)

        for i, (image_bytes, caption, filename) in enumerate(archive):
            print_progress(i + 1, total, f"Extracting {archive_path.name}")

            # Get entry info for format
            info = archive.get_entry_info(i)

            # Write image file
            image_path = output_dir / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)
            images_extracted += 1

            # Write caption file if present
            if caption:
                caption_path = image_path.with_suffix('.txt')
                caption_path.write_text(caption, encoding='utf-8')
                captions_extracted += 1

    # Clear progress line
    if sys.stdout.isatty():
        print()

    return images_extracted, captions_extracted


def main():
    parser = argparse.ArgumentParser(
        description='Extract zitpack archives to individual files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python unpack_dataset.py --input ./archive.zitpack --output ./extracted
  python unpack_dataset.py --input "./packed/dataset_chunk_*.zitpack" --output ./restored
  python unpack_dataset.py --input ./data.zitpack --output ./files --verify

Note:
  Use quotes around glob patterns to prevent shell expansion.
        """
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input archive(s) - path or glob pattern')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for extracted files')
    parser.add_argument('--verify', '-v', action='store_true',
                        help='Verify CRC32 checksums during extraction')

    args = parser.parse_args()

    # Expand input pattern
    archive_paths = expand_glob_pattern(args.input)

    if not archive_paths:
        print(f"Error: No zitpack files found matching: {args.input}")
        return 1

    print(f"Found {len(archive_paths)} archive(s) to extract")
    print(f"Output directory: {args.output}")
    if args.verify:
        print("Checksum verification: enabled")
    print()

    total_images = 0
    total_captions = 0
    errors = []

    for archive_path in archive_paths:
        try:
            images, captions = unpack_archive(
                archive_path,
                args.output,
                verify=args.verify,
            )
            total_images += images
            total_captions += captions
            print(f"  {archive_path.name}: {images} images, {captions} captions")
        except Exception as e:
            errors.append((archive_path, str(e)))
            print(f"  {archive_path.name}: ERROR - {e}")

    print()
    print(f"Total extracted: {total_images} images, {total_captions} captions")

    if errors:
        print(f"\n{len(errors)} archive(s) had errors:")
        for path, error in errors:
            print(f"  {path.name}: {error}")
        return 1

    print(f"\nExtracted to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
