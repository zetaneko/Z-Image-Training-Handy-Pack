#!/usr/bin/env python3
"""
Pack a folder of images+captions into chunked zitpack archives.

Creates memory-efficient archives for training datasets. Each archive chunk
is ~512MB and self-contained with its own index.

Usage:
  python pack_dataset.py --input ./images --output ./packed/dataset
  python pack_dataset.py --input ./images --output ./packed/dataset --chunk-size 256

Example:
  python pack_dataset.py --input ./training_data --output ./archives/mymodel
  # Creates: mymodel_chunk_000.zitpack, mymodel_chunk_001.zitpack, ...
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_archive import DatasetArchiveWriter, TARGET_CHUNK_SIZE, HARD_CHUNK_LIMIT

# Image extensions to look for
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}


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


def pack_dataset(
    input_dir: Path,
    output_base: Path,
    chunk_size_mb: int = 512,
    compute_checksums: bool = True,
) -> list[Path]:
    """
    Pack a dataset folder into chunked zitpack archives.

    Args:
        input_dir: Directory containing images and captions
        output_base: Base path for output (e.g., ./archives/dataset)
        chunk_size_mb: Target chunk size in MB
        compute_checksums: Whether to compute CRC32 checksums

    Returns:
        List of created archive paths
    """
    # Collect all image files
    print(f"Scanning {input_dir}...")
    image_files = collect_image_files(input_dir)
    total_images = len(image_files)

    if total_images == 0:
        print("No image files found.")
        return []

    print(f"Found {total_images} images")

    # Calculate target chunk size in bytes
    target_size = chunk_size_mb * 1024 * 1024
    hard_limit = int(target_size * 1.17)  # Allow ~17% overage for last image

    # First pass: estimate total size and chunk count
    total_size = 0
    for image_path in image_files:
        total_size += image_path.stat().st_size
        caption_path = image_path.with_suffix('.txt')
        if caption_path.exists():
            total_size += caption_path.stat().st_size

    estimated_chunks = max(1, (total_size + target_size - 1) // target_size)
    print(f"Estimated size: {total_size / (1024*1024):.1f} MB ({estimated_chunks} chunks)")

    # Create output directory
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Pack into chunks
    created_archives: list[Path] = []
    current_chunk = 0
    current_writer: DatasetArchiveWriter | None = None
    processed = 0

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

        # Add entry
        caption_path = image_path.with_suffix('.txt')
        current_writer.add_entry(image_path, caption_path if caption_path.exists() else None)
        processed += 1

        # Check if chunk should be finalized
        if current_writer.current_size >= target_size:
            # Finalize current chunk
            current_writer.finalize()
            created_archives.append(current_writer.output_path)
            current_writer = None
            current_chunk += 1

    # Finalize last chunk if it has entries
    if current_writer is not None and current_writer.entry_count > 0:
        current_writer.finalize()
        created_archives.append(current_writer.output_path)

    # Clear progress line
    if sys.stdout.isatty():
        print()

    # Update total_chunks in all archives (requires rewriting headers)
    total_chunks = len(created_archives)
    for archive_path in created_archives:
        _update_total_chunks(archive_path, total_chunks)

    return created_archives


def _update_total_chunks(archive_path: Path, total_chunks: int) -> None:
    """Update the total_chunks field in an archive header."""
    import struct

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

    args = parser.parse_args()

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

    print(f"Input folder : {args.input}")
    print(f"Output base  : {args.output}")
    print(f"Chunk size   : {args.chunk_size} MB")
    print(f"Checksums    : {'disabled' if args.no_checksums else 'enabled'}")
    print()

    created = pack_dataset(
        args.input,
        args.output,
        chunk_size_mb=args.chunk_size,
        compute_checksums=not args.no_checksums,
    )

    if created:
        print(f"\nCreated {len(created)} archive(s):")
        total_size = 0
        for archive_path in created:
            size = archive_path.stat().st_size
            total_size += size
            print(f"  {archive_path.name}: {size / (1024*1024):.1f} MB")
        print(f"\nTotal size: {total_size / (1024*1024):.1f} MB")
    else:
        print("\nNo archives created.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
