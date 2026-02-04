#!/usr/bin/env python3
"""
Validates, normalizes, and optimizes images for training.
Converts to RGB, resizes to minimum dimension, and re-encodes as PNG.
Quarantines broken images and copies matching caption files.

Usage:
  python scan_and_fix_images.py --input <folder> [--output <folder>] [--quarantine <folder>]

Example:
  python scan_and_fix_images.py --input ./Raw/Screencaps --output ./Fixed --quarantine ./bad
"""

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFile


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
        # Update every 5% or on first/last item
        if current == 1 or current == total or current % max(1, total // 20) == 0:
            print(message, flush=True)

# Enable best-effort load for slightly truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}
DEFAULT_MIN_DIM = 1024


def resize_for_training(im: Image.Image, target_min: int) -> Image.Image:
    """
    Resize image so the smallest dimension equals target_min.
    Only downscales; images already at or below target are unchanged.
    """
    w, h = im.size
    min_dim = min(w, h)

    if min_dim <= target_min:
        return im

    scale = target_min / min_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    return im.resize((new_w, new_h), Image.LANCZOS)


def process_images(input_dir: Path, output_dir: Path, quarantine_dir: Path, min_dim: int):
    """Process all images in input directory."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    bad = []
    checked = 0
    fixed = 0
    resized = 0
    copied = 0

    # First, collect all image files to get total count
    image_files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    total = len(image_files)

    if total == 0:
        print("No image files found.")
        return

    print(f"Found {total} images to process...")

    for i, p in enumerate(image_files, 1):
        print_progress(i, total, "Processing")

        checked += 1
        rel = p.relative_to(input_dir)
        out = output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(p) as im:
                im.load()  # force decode
                orig_size = im.size
                im = im.convert("RGB")  # normalize mode (no alpha)
                im = resize_for_training(im, min_dim)
                if im.size != orig_size:
                    resized += 1
                out = out.with_suffix(".png")
                im.save(out, format="PNG")
                fixed += 1
        except Exception as e:
            bad.append((str(p), repr(e)))
            # quarantine the broken file + its caption if present
            qimg = quarantine_dir / rel
            qimg.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(qimg))
            cap = p.with_suffix(".txt")
            if cap.exists():
                qcap = quarantine_dir / cap.relative_to(input_dir)
                qcap.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cap), str(qcap))
            continue

        # copy matching caption (if any) to the fixed folder
        cap_src = p.with_suffix(".txt")
        if cap_src.exists():
            cap_dst = out.with_suffix(".txt")
            shutil.copy2(str(cap_src), str(cap_dst))
            copied += 1

    # Clear progress line in terminal mode
    if sys.stdout.isatty():
        print()

    # Print summary
    print(f"Checked images : {checked}")
    print(f"Re-encoded OK  : {fixed}")
    print(f"Resized (>{min_dim}): {resized}")
    print(f"Copied captions: {copied}")

    if bad:
        print("\nBroken files:")
        for path, err in bad:
            print(f" - {path}  ::  {err}")
        print(f"\nQuarantined {len(bad)} items under: {quarantine_dir}")
    else:
        print("\nNo broken files detected.")

    print(f"\nClean dataset at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate, normalize, and optimize images for training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scan_and_fix_images.py --input ./Raw/Screencaps
  python scan_and_fix_images.py --input ./Raw --output ./Fixed --quarantine ./bad --min-size 512
        """
    )
    parser.add_argument('--input', '-i', required=True, type=Path,
                        help='Input folder containing images to process')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output folder for fixed images (default: <input_parent>/Fixed)')
    parser.add_argument('--quarantine', '-q', type=Path, default=None,
                        help='Folder for broken/quarantined files (default: <input_parent>/bad)')
    parser.add_argument('--min-size', '-m', type=int, default=DEFAULT_MIN_DIM,
                        help=f'Minimum dimension size for resizing (default: {DEFAULT_MIN_DIM})')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input folder not found: {args.input}")
        return 1
    if not args.input.is_dir():
        print(f"Error: Input is not a directory: {args.input}")
        return 1

    # Set defaults for output and quarantine
    input_parent = args.input.parent
    output_dir = args.output if args.output else input_parent / "Fixed"
    quarantine_dir = args.quarantine if args.quarantine else input_parent / "bad"

    print(f"Input folder  : {args.input}")
    print(f"Output folder : {output_dir}")
    print(f"Quarantine    : {quarantine_dir}")
    print(f"Min dimension : {args.min_size}")
    print()

    process_images(args.input, output_dir, quarantine_dir, args.min_size)
    return 0


if __name__ == '__main__':
    exit(main())
