"""
Create metadata.csv from a folder of images with corresponding .txt prompt files.

Usage:
    python create_metadata.py /path/to/image/folder
    python create_metadata.py /path/to/image/folder --output metadata.csv
    python create_metadata.py /path/to/image/folder --recursive

Expected structure:
    folder/
    ├── cat001.jpg
    ├── cat001.txt      # Contains the prompt for cat001.jpg
    ├── dog002.png
    ├── dog002.txt
    └── subfolder/      # With --recursive
        ├── bird.jpg
        └── bird.txt

Output CSV format:
    image,prompt
    cat001.jpg,"A fluffy cat sitting on a couch"
    dog002.png,"A golden retriever playing fetch"
"""

import argparse
import csv
import os
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def find_image_prompt_pairs(folder: Path, recursive: bool = False) -> list[tuple[Path, str]]:
    """Find all image files that have corresponding .txt prompt files."""
    pairs = []

    if recursive:
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(folder.rglob(f'*{ext}'))
            image_files.extend(folder.rglob(f'*{ext.upper()}'))
    else:
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))

    for image_path in sorted(set(image_files)):
        txt_path = image_path.with_suffix('.txt')
        if txt_path.exists():
            prompt = txt_path.read_text(encoding='utf-8').strip()
            if prompt:
                pairs.append((image_path, prompt))
            else:
                print(f"Warning: Empty prompt file, skipping: {txt_path}")
        else:
            print(f"Warning: No prompt file found for: {image_path}")

    return pairs


def create_metadata_csv(folder: str, output: str = None, recursive: bool = False):
    """Create metadata.csv from image/prompt pairs."""
    folder_path = Path(folder).resolve()

    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return

    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return

    pairs = find_image_prompt_pairs(folder_path, recursive)

    if not pairs:
        print("No image/prompt pairs found.")
        return

    output_path = Path(output) if output else folder_path / 'metadata.csv'

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'prompt'])

        for image_path, prompt in pairs:
            relative_path = image_path.relative_to(folder_path)
            writer.writerow([str(relative_path), prompt])

    print(f"Created {output_path}")
    print(f"Total entries: {len(pairs)}")


def main():
    parser = argparse.ArgumentParser(
        description='Create metadata.csv from images with corresponding .txt prompt files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('folder', type=str, help='Path to folder containing images and .txt files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV path (default: <folder>/metadata.csv)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search subfolders recursively')

    args = parser.parse_args()
    create_metadata_csv(args.folder, args.output, args.recursive)


if __name__ == '__main__':
    main()
