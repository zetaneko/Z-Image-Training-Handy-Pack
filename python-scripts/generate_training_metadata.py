#!/usr/bin/env python3
"""
Generates a single metadata.csv containing image filenames and their prompts.
Supports delta mode - appends new entries without duplicating existing ones.
This is used for specifying image prompts in training data set for
DiffSynth-Studio Z-Image training which is different to the most common
set up of having the image and associated .txt caption file.

Usage:
  python generate_training_metadata.py --input <folder> [--output <csv>]
  python generate_training_metadata.py  # Process folders in script directory

Example:
  python generate_training_metadata.py --input ./dataset
  python generate_training_metadata.py --input ./dataset --output ./metadata.csv
"""

import argparse
import csv
import sys
from pathlib import Path

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


def load_existing_metadata(csv_path: Path) -> set[str]:
    """Load existing image filenames from metadata.csv if it exists."""
    existing = set()
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row['image'])
    return existing


def find_image_prompt_pairs(folder: Path, show_progress: bool = False) -> list[tuple[str, str]]:
    """Find all image files and their corresponding .txt prompts in a folder."""
    pairs = []

    # Collect image files first to get total count
    image_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    total = len(image_files)

    for i, file_path in enumerate(image_files, 1):
        if show_progress:
            print_progress(i, total, f"Scanning {folder.name}")

        # Look for corresponding .txt file
        txt_path = file_path.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                # Replace newlines with spaces for CSV compatibility
                prompt = prompt.replace('\n', ' ').replace('\r', ' ')
            pairs.append((file_path.name, prompt))

    # Clear progress line in terminal mode
    if show_progress and sys.stdout.isatty() and total > 0:
        print()

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Generate metadata.csv for DiffSynth-Studio Z-Image training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_training_metadata.py --input ./dataset
  python generate_training_metadata.py --input ./dataset --output ./metadata.csv
  python generate_training_metadata.py  # Process folders in script directory
        """
    )
    parser.add_argument('--input', '-i', type=Path, default=None,
                        help='Input folder to scan (default: script directory)')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output metadata.csv path (default: <input>/metadata.csv)')

    args = parser.parse_args()

    # Determine base directory
    if args.input:
        if not args.input.exists():
            print(f"Error: Folder not found: {args.input}")
            return 1
        if not args.input.is_dir():
            print(f"Error: Not a directory: {args.input}")
            return 1
        base_dir = args.input
    else:
        base_dir = Path(__file__).parent

    # Determine output path
    csv_path = args.output if args.output else base_dir / 'metadata.csv'

    # Load existing entries to avoid duplicates
    existing_images = load_existing_metadata(csv_path)
    print(f"Found {len(existing_images)} existing entries in metadata.csv")

    # Collect all new pairs
    new_pairs = []

    for item in base_dir.iterdir():
        # Skip if not a directory, or if it's 'original' or hidden
        if not item.is_dir() or item.name == 'original' or item.name.startswith('.'):
            continue

        folder_pairs = find_image_prompt_pairs(item, show_progress=True)

        # Filter out already existing entries
        for image_name, prompt in folder_pairs:
            if image_name not in existing_images:
                new_pairs.append((image_name, prompt))
                existing_images.add(image_name)  # Track to avoid duplicates within this run

        if folder_pairs:
            new_count = sum(1 for img, _ in folder_pairs if img in [p[0] for p in new_pairs])
            print(f"{item.name}: found {len(folder_pairs)} images, {new_count} new")

    # Write to CSV
    if new_pairs:
        # Check if file exists to determine if we need header
        file_exists = csv_path.exists()

        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            # Write header only if file is new
            if not file_exists:
                writer.writerow(['image', 'prompt'])

            for image_name, prompt in new_pairs:
                writer.writerow([image_name, prompt])

        print(f"\nAdded {len(new_pairs)} new entries to metadata.csv")
    else:
        # Create empty file with header if it doesn't exist
        if not csv_path.exists():
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'prompt'])
            print("\nCreated empty metadata.csv with header")
        else:
            print("\nNo new entries to add")

    # Print total count
    total = len(load_existing_metadata(csv_path))
    print(f"Total entries in metadata.csv: {total}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
