#!/usr/bin/env python3
"""
Pack a folder of images+captions into chunked zitpack archives.

Creates memory-efficient archives for training datasets. Each archive chunk
is ~512MB and self-contained with its own index.

Captions are built by concatenating multiple caption files per image into a
single combined caption (joined with ", "). The order is:
  1. all.txt (shared file in the same folder as the image, if present)
  2. Per-image caption files by suffix, e.g.: .score.txt, .autocaption.txt, .quality.txt

For .autocaption.txt specifically: if the file doesn't exist, the original
.txt caption is used as a fallback. The .txt caption is never used when
.autocaption.txt exists.

Usage:
  python pack_dataset.py --input ./images --output ./packed/dataset
  python pack_dataset.py --input ./images --output ./packed/dataset --chunk-size 256
  python pack_dataset.py --input ./images --output ./packed/dataset --caption-suffixes ".score.txt,.autocaption.txt,.quality.txt"

Incremental Update:
  python pack_dataset.py --input ./images --output ./packed/dataset --update
  # Only packs new files not already in existing archives

Example:
  python pack_dataset.py --input ./training_data --output ./archives/mymodel
  # Creates: mymodel_chunk_000.zitpack, mymodel_chunk_001.zitpack, ...
"""

import argparse
import glob
import struct
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_archive import DatasetArchive, DatasetArchiveWriter, TARGET_CHUNK_SIZE, HARD_CHUNK_LIMIT

# Image extensions to look for
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}

# Default caption suffixes to concatenate (in order)
DEFAULT_CAPTION_SUFFIXES = ['.score.txt', '.autocaption.txt', '.quality.txt']


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


def collect_image_files(input_dir: Path) -> list[Path]:
    """Collect all image files from input directory."""
    image_files = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(p)
    return image_files


def validate_image(image_path: Path) -> bool:
    """Verify an image can be fully decoded by PIL. Returns True if valid."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.load()  # force full decode to catch truncated/corrupt data
        return True
    except Exception:
        return False


def build_combined_caption(image_path: Path, caption_suffixes: list[str]) -> str:
    """
    Build a combined caption from multiple caption files.

    Order:
      1. all.txt in the same folder (shared across all images in that folder)
      2. Per-image files in suffix order (e.g. .score.txt, .autocaption.txt, .quality.txt)

    For .autocaption.txt: falls back to .txt if autocaption doesn't exist.
    The .txt caption is never used when .autocaption.txt exists.

    Returns:
        Combined caption string, parts joined with ", "
    """
    parts = []

    # 1. Check for all.txt in the same folder
    all_txt = image_path.parent / 'all.txt'
    if all_txt.exists():
        text = all_txt.read_text(encoding='utf-8').strip()
        if text:
            parts.append(text)

    # 2. Per-image caption files in suffix order
    stem = image_path.stem
    folder = image_path.parent

    for suffix in caption_suffixes:
        caption_file = folder / (stem + suffix)

        if suffix == '.autocaption.txt':
            # Special handling: use .autocaption.txt if exists, else fall back to .txt
            if caption_file.exists():
                text = caption_file.read_text(encoding='utf-8').strip()
                if text:
                    parts.append(text)
            else:
                # Fallback to original .txt
                fallback = image_path.with_suffix('.txt')
                if fallback.exists():
                    text = fallback.read_text(encoding='utf-8').strip()
                    if text:
                        parts.append(text)
        else:
            if caption_file.exists():
                text = caption_file.read_text(encoding='utf-8').strip()
                if text:
                    parts.append(text)

    return ', '.join(parts)


def find_existing_archives(output_base: Path) -> list[Path]:
    """Find existing archive chunks matching the output pattern."""
    pattern = f"{output_base}_chunk_*.zitpack"
    return sorted(Path(p) for p in glob.glob(str(pattern)))


def get_existing_filenames(archive_paths: list[Path]) -> set[str]:
    """Get set of all filenames already in existing archives."""
    existing = set()
    for archive_path in archive_paths:
        try:
            with DatasetArchive(archive_path) as archive:
                existing.update(archive.list_entries())
        except Exception as e:
            print(f"Warning: Could not read {archive_path.name}: {e}")
    return existing


def get_last_chunk_info(archive_path: Path) -> tuple[int, int, int]:
    """
    Get info about last chunk for appending.

    Returns:
        Tuple of (chunk_index, entry_count, current_size_bytes)
    """
    with DatasetArchive(archive_path) as archive:
        return (
            archive.chunk_index,
            len(archive),
            archive_path.stat().st_size,
        )


def pack_dataset(
    input_dir: Path,
    output_base: Path,
    chunk_size_mb: int = 512,
    compute_checksums: bool = True,
    update_mode: bool = False,
    caption_suffixes: list[str] | None = None,
) -> list[Path]:
    """
    Pack a dataset folder into chunked zitpack archives.

    Args:
        input_dir: Directory containing images and captions
        output_base: Base path for output (e.g., ./archives/dataset)
        chunk_size_mb: Target chunk size in MB
        compute_checksums: Whether to compute CRC32 checksums
        update_mode: If True, only pack new files not in existing archives
        caption_suffixes: Ordered list of caption file suffixes to concatenate.
                         None uses DEFAULT_CAPTION_SUFFIXES.

    Returns:
        List of created/modified archive paths
    """
    if caption_suffixes is None:
        caption_suffixes = DEFAULT_CAPTION_SUFFIXES
    # Collect all image files
    print(f"Scanning {input_dir}...")
    all_image_files = collect_image_files(input_dir)

    if not all_image_files:
        print("No image files found.")
        return []

    # Check for existing archives in update mode
    existing_archives = find_existing_archives(output_base) if update_mode else []
    existing_filenames = set()
    start_chunk = 0
    last_chunk_path = None
    last_chunk_size = 0

    if update_mode and existing_archives:
        print(f"Found {len(existing_archives)} existing archive(s)")

        # Get already-packed filenames
        print("Scanning existing archives...")
        existing_filenames = get_existing_filenames(existing_archives)
        print(f"Found {len(existing_filenames)} existing entries")

        # Get last chunk info for potential appending
        last_chunk_path = existing_archives[-1]
        chunk_idx, entry_count, last_chunk_size = get_last_chunk_info(last_chunk_path)
        start_chunk = chunk_idx

        # Filter to only new files
        image_files = [
            p for p in all_image_files
            if p.name not in existing_filenames
        ]

        if not image_files:
            print("No new files to pack.")
            return existing_archives

        print(f"Found {len(image_files)} new images to pack")
    else:
        image_files = all_image_files
        print(f"Found {len(image_files)} images")

    # Validate images through PIL to skip corrupted files
    print("Validating images...")
    valid_files = []
    skipped = 0
    for i, image_path in enumerate(image_files):
        print_progress(i + 1, len(image_files), "Validating")
        if validate_image(image_path):
            valid_files.append(image_path)
        else:
            skipped += 1
            print(f"\n  Skipping corrupt image: {image_path.name}")

    if sys.stdout.isatty():
        print()

    if skipped:
        print(f"Skipped {skipped} corrupt image(s)")

    image_files = valid_files
    if not image_files:
        print("No valid images to pack.")
        return existing_archives if update_mode else []

    total_images = len(image_files)

    # Calculate target chunk size in bytes
    target_size = chunk_size_mb * 1024 * 1024

    # First pass: estimate total size and chunk count
    total_size = 0
    for image_path in image_files:
        total_size += image_path.stat().st_size
        # Estimate caption size from all potential caption files
        stem = image_path.stem
        folder = image_path.parent
        all_txt = folder / 'all.txt'
        if all_txt.exists():
            total_size += all_txt.stat().st_size
        for suffix in caption_suffixes:
            cap_file = folder / (stem + suffix)
            if cap_file.exists():
                total_size += cap_file.stat().st_size
            elif suffix == '.autocaption.txt':
                fallback = image_path.with_suffix('.txt')
                if fallback.exists():
                    total_size += fallback.stat().st_size

    # Account for last chunk remaining space
    remaining_in_last = 0
    if update_mode and last_chunk_path and last_chunk_size < target_size:
        remaining_in_last = target_size - last_chunk_size

    estimated_new_size = max(0, total_size - remaining_in_last)
    estimated_new_chunks = max(0, (estimated_new_size + target_size - 1) // target_size) if estimated_new_size > 0 else 0

    if update_mode and last_chunk_path:
        print(f"New data size: {total_size / (1024*1024):.1f} MB")
        if remaining_in_last > 0:
            print(f"Space in last chunk: {remaining_in_last / (1024*1024):.1f} MB")
    else:
        estimated_chunks = max(1, (total_size + target_size - 1) // target_size)
        print(f"Estimated size: {total_size / (1024*1024):.1f} MB ({estimated_chunks} chunks)")

    # Create output directory
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Pack into chunks
    created_archives: list[Path] = []
    modified_archives: list[Path] = []
    current_chunk = start_chunk
    current_writer: DatasetArchiveWriter | None = None
    file_index = 0

    # If updating and last chunk has room, rebuild it with new entries
    if update_mode and last_chunk_path and remaining_in_last > 0:
        print(f"Appending to {last_chunk_path.name}...")

        # Create writer and copy existing entries
        temp_path = last_chunk_path.with_suffix('.zitpack.tmp')
        current_writer = DatasetArchiveWriter(
            temp_path,
            chunk_index=current_chunk,
            total_chunks=0,
            compute_checksums=compute_checksums,
        )

        # Copy existing entries efficiently
        with DatasetArchive(last_chunk_path) as old_archive:
            for i in range(len(old_archive)):
                current_writer.copy_entry_from_archive(old_archive, i)

        # Continue with this writer for new entries
        # (will be finalized below when full or done)

    for i, image_path in enumerate(image_files):
        print_progress(i + 1, total_images, "Packing")

        # Start new chunk if needed
        if current_writer is None:
            chunk_path = Path(f"{output_base}_chunk_{current_chunk:03d}.zitpack")
            current_writer = DatasetArchiveWriter(
                chunk_path,
                chunk_index=current_chunk,
                total_chunks=0,  # Will update at end
                compute_checksums=compute_checksums,
            )

        # Build combined caption from multiple sources
        combined_caption = build_combined_caption(image_path, caption_suffixes)
        current_writer.add_entry(image_path, caption=combined_caption)
        file_index += 1

        # Check if chunk should be finalized
        if current_writer.current_size >= target_size:
            # Finalize current chunk
            current_writer.finalize()
            final_path = current_writer.output_path

            # Handle temp file for updated chunks
            if final_path.suffix == '.tmp':
                actual_path = final_path.with_suffix('')  # Remove .tmp
                if actual_path.exists():
                    actual_path.unlink()  # Remove old version
                final_path.rename(actual_path)
                final_path = actual_path
                modified_archives.append(final_path)
            else:
                created_archives.append(final_path)

            current_writer = None
            current_chunk += 1

    # Finalize last chunk if it has entries
    if current_writer is not None and current_writer.entry_count > 0:
        current_writer.finalize()
        final_path = current_writer.output_path

        # Handle temp file for updated chunks
        if final_path.suffix == '.tmp':
            actual_path = final_path.with_suffix('')
            if actual_path.exists():
                actual_path.unlink()
            final_path.rename(actual_path)
            final_path = actual_path
            modified_archives.append(final_path)
        else:
            created_archives.append(final_path)

    # Clear progress line
    if sys.stdout.isatty():
        print()

    # Get all archives for updating total_chunks
    all_archives = find_existing_archives(output_base)
    total_chunks = len(all_archives)

    # Update total_chunks in all archives
    for archive_path in all_archives:
        _update_total_chunks(archive_path, total_chunks)

    return created_archives + modified_archives


def _update_total_chunks(archive_path: Path, total_chunks: int) -> None:
    """Update the total_chunks field in an archive header."""
    with open(archive_path, 'r+b') as f:
        # Read current header
        header_bytes = bytearray(f.read(64))

        # Update total_chunks at offset 38 (2 bytes, little-endian)
        struct.pack_into('<H', header_bytes, 38, total_chunks)

        # Write back
        f.seek(0)
        f.write(header_bytes)


def main():
    parser = argparse.ArgumentParser(
        description='Pack images and captions into zitpack archives.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python pack_dataset.py --input ./training_data --output ./archives/mymodel
  python pack_dataset.py --input ./images --output ./packed/dataset --chunk-size 256

Incremental Update:
  python pack_dataset.py --input ./images --output ./packed/dataset --update
  # Only packs new files not already in existing archives
  # Efficiently appends to last chunk if space available

Output:
  Creates: <output>_chunk_000.zitpack, <output>_chunk_001.zitpack, ...
        """
    )
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input folder containing images and captions')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output base path (e.g., ./archives/dataset)')
    parser.add_argument('--chunk-size', '-c', type=int, default=512,
                        help='Target chunk size in MB (default: 512)')
    parser.add_argument('--no-checksums', action='store_true',
                        help='Disable CRC32 checksum computation')
    parser.add_argument('--update', '-u', action='store_true',
                        help='Incremental update: only pack new files not in existing archives')
    parser.add_argument('--caption-suffixes', '-s', type=str,
                        default=','.join(DEFAULT_CAPTION_SUFFIXES),
                        help='Comma-separated caption file suffixes in order '
                             '(default: %(default)s). '
                             'all.txt in each folder is always included first. '
                             '.autocaption.txt falls back to .txt if not found.')

    args = parser.parse_args()

    # Parse caption suffixes
    caption_suffixes = [s.strip() for s in args.caption_suffixes.split(',') if s.strip()]

    # Validate input
    if not args.input.exists():
        print(f"Error: Input folder not found: {args.input}")
        return 1
    if not args.input.is_dir():
        print(f"Error: Input is not a directory: {args.input}")
        return 1

    # Validate chunk size
    if args.chunk_size < 1:
        print("Error: Chunk size must be at least 1 MB")
        return 1
    if args.chunk_size > 2048:
        print("Warning: Large chunk sizes may affect performance")

    print(f"Input folder    : {args.input}")
    print(f"Output base     : {args.output}")
    print(f"Chunk size      : {args.chunk_size} MB")
    print(f"Checksums       : {'disabled' if args.no_checksums else 'enabled'}")
    print(f"Update mode     : {'enabled' if args.update else 'disabled'}")
    print(f"Caption suffixes: {', '.join(caption_suffixes)}")
    print(f"  (all.txt in each folder is always included first)")
    print()

    result = pack_dataset(
        args.input,
        args.output,
        chunk_size_mb=args.chunk_size,
        compute_checksums=not args.no_checksums,
        update_mode=args.update,
        caption_suffixes=caption_suffixes,
    )

    if result:
        # Count new vs modified
        all_archives = find_existing_archives(args.output)
        print(f"\nTotal archives: {len(all_archives)}")

        total_size = 0
        for archive_path in all_archives:
            size = archive_path.stat().st_size
            total_size += size
            print(f"  {archive_path.name}: {size / (1024*1024):.1f} MB")
        print(f"\nTotal size: {total_size / (1024*1024):.1f} MB")

        if args.update:
            print(f"\nUpdated/created {len(result)} archive(s)")
    else:
        if not args.update:
            print("\nNo archives created.")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
