#!/usr/bin/env python3
"""
Analyze image quality and generate natural-language quality captions.

Computes sharpness, brightness, contrast, and overall quality for each image,
maps each metric to one of 10 precise levels, and writes flowing captions
to .quality.txt files alongside the images.

Uses the image-quality-analysis package for overall quality scoring and blur
detection, plus OpenCV for brightness/contrast measurement.

Usage:
  python quality_caption.py --input ./training_data
  python quality_caption.py --input ./images --overwrite --no-recursive

Example:
  # Basic usage - analyze all images in subdirectories
  python quality_caption.py --input ./training_data

  # Preview without writing files
  python quality_caption.py --input ./images --dry-run

Output:
  Creates .quality.txt files alongside images (e.g., cat.png -> cat.quality.txt)
"""

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from image_quality_checker import image_quality_score, analyze_blur

# Image extensions to process
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}


# ---------------------------------------------------------------------------
# 10-level bucket definitions: (threshold, descriptor)
# Value is classified into the first bucket where value < threshold.
# ---------------------------------------------------------------------------

SHARPNESS_BUCKETS = [
    (5, "extremely blurry"),
    (10, "very blurry"),
    (15, "noticeably blurry"),
    (20, "moderately blurry"),
    (25, "slightly blurry"),
    (30, "slightly soft"),
    (35, "reasonably sharp"),
    (42, "sharp and in focus"),
    (50, "very sharp"),
    (60, "slightly oversharpened"),
    (75, "noticeably oversharpened"),
    (float('inf'), "heavily oversharpened"),
]

BRIGHTNESS_BUCKETS = [
    (25, "extremely dark"),
    (50, "very dark"),
    (75, "dark"),
    (100, "somewhat dim"),
    (115, "slightly dim"),
    (140, "well balanced"),
    (165, "bright"),
    (190, "quite bright"),
    (220, "very bright"),
    (float('inf'), "extremely bright"),
]

CONTRAST_BUCKETS = [
    (8, "essentially flat"),
    (16, "very low contrast"),
    (24, "low contrast"),
    (32, "somewhat muted"),
    (40, "moderate contrast"),
    (48, "balanced contrast"),
    (56, "good contrast"),
    (64, "strong contrast"),
    (72, "very high contrast"),
    (float('inf'), "extremely high contrast"),
]

QUALITY_BUCKETS = [
    (0.1, "very poor"),
    (0.2, "poor"),
    (0.3, "below average"),
    (0.4, "slightly below average"),
    (0.5, "average"),
    (0.6, "above average"),
    (0.7, "good"),
    (0.8, "very good"),
    (0.9, "excellent"),
    (float('inf'), "outstanding"),
]


# ---------------------------------------------------------------------------
# Sentence templates for natural-sounding captions
# ---------------------------------------------------------------------------

SHARPNESS_TEMPLATES = [
    "This image is {sharpness}.",
    "The image appears {sharpness}.",
    "A {sharpness} photograph.",
    "The focus in this image is {sharpness}.",
    "This photograph is {sharpness} in detail.",
]

BRIGHTNESS_CONTRAST_TEMPLATES = [
    "The lighting is {brightness} with {contrast}.",
    "It features {brightness} lighting and {contrast}.",
    "The exposure is {brightness}, and the tonal range shows {contrast}.",
    "Brightness is {brightness} while the contrast is {contrast}.",
    "The scene has {brightness} illumination with {contrast}.",
]

QUALITY_TEMPLATES = [
    "The overall image quality is {quality}.",
    "Overall, the quality is {quality}.",
    "In terms of overall quality, this image is {quality}.",
    "The composite quality assessment is {quality}.",
    "As a whole, the image quality rates as {quality}.",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


def collect_images(input_dir: Path, recursive: bool = True) -> list[Path]:
    """Collect all image files from directory."""
    images = []
    io_errors = []

    if recursive:
        for p in sorted(input_dir.rglob("*")):
            try:
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(p)
            except OSError as e:
                io_errors.append((p, e))
    else:
        for p in sorted(input_dir.iterdir()):
            try:
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(p)
            except OSError as e:
                io_errors.append((p, e))

    if io_errors:
        print(f"\nWarning: Skipped {len(io_errors)} file(s) due to I/O errors:", file=sys.stderr)
        for p, e in io_errors[:5]:
            print(f"  {p}: {e}", file=sys.stderr)
        if len(io_errors) > 5:
            print(f"  ... and {len(io_errors) - 5} more", file=sys.stderr)

    return images


def classify_metric(value: float, buckets: list[tuple[float, str]]) -> str:
    """Map a raw metric value to its natural-language descriptor."""
    for threshold, descriptor in buckets:
        if value < threshold:
            return descriptor
    return buckets[-1][1]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_sharpness(image_path: Path) -> float:
    """
    Compute sharpness as inverted blur ratio (0-100).
    100 = perfectly sharp, 0 = completely blurry.
    """
    result = analyze_blur(str(image_path))
    blur_ratio = result.get('blur_ratio', 50.0)
    return max(0.0, min(100.0, 100.0 - blur_ratio))


def compute_brightness_contrast(image_path: Path) -> tuple[float, float]:
    """
    Compute brightness (mean grayscale, 0-255) and contrast (std dev, 0-80+)
    in a single image read.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return brightness, contrast


def compute_overall_quality(image_path: Path) -> float:
    """
    Compute overall quality score (0.0-1.0).
    Returns -1.0 if the image is detected as corrupted or single-color.
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    result = image_quality_score(image_bytes)
    if result is True:
        return -1.0
    if isinstance(result, (int, float)):
        return float(max(0.0, min(1.0, result)))
    return -1.0


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def generate_quality_caption(
    sharpness_desc: str,
    brightness_desc: str,
    contrast_desc: str,
    quality_desc: str,
) -> str:
    """Generate a natural-sounding quality caption from metric descriptors."""
    # Deterministic template selection based on descriptor combination
    seed_str = f"{sharpness_desc}|{brightness_desc}|{contrast_desc}|{quality_desc}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

    sharp_tmpl = SHARPNESS_TEMPLATES[seed % len(SHARPNESS_TEMPLATES)]
    bc_tmpl = BRIGHTNESS_CONTRAST_TEMPLATES[(seed >> 8) % len(BRIGHTNESS_CONTRAST_TEMPLATES)]
    qual_tmpl = QUALITY_TEMPLATES[(seed >> 16) % len(QUALITY_TEMPLATES)]

    sentence1 = sharp_tmpl.format(sharpness=sharpness_desc)
    sentence2 = bc_tmpl.format(brightness=brightness_desc, contrast=contrast_desc)

    if quality_desc.startswith("unable to assess"):
        sentence3 = ("The overall quality could not be assessed; the image may be "
                     "corrupted or consists of a single dominant color.")
    else:
        sentence3 = qual_tmpl.format(quality=quality_desc)

    return f"{sentence1} {sentence2} {sentence3}"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_images(
    input_dir: Path,
    recursive: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Process all images in a directory, computing quality metrics
    and writing .quality.txt files.

    Returns:
        Tuple of (processed, skipped, errors)
    """
    images = collect_images(input_dir, recursive)
    total = len(images)

    if total == 0:
        print("No images found.")
        return 0, 0, 0

    print(f"Found {total} images to process")

    processed = 0
    skipped = 0
    errors = 0

    for i, image_path in enumerate(images):
        print_progress(i + 1, total, "Analyzing")

        try:
            # Check if output already exists
            output_path = image_path.with_suffix('.quality.txt')
            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            # Compute metrics
            sharpness = compute_sharpness(image_path)
            brightness, contrast = compute_brightness_contrast(image_path)
            quality = compute_overall_quality(image_path)

            # Classify into 10-level buckets
            sharpness_desc = classify_metric(sharpness, SHARPNESS_BUCKETS)
            brightness_desc = classify_metric(brightness, BRIGHTNESS_BUCKETS)
            contrast_desc = classify_metric(contrast, CONTRAST_BUCKETS)
            if quality < 0:
                quality_desc = "unable to assess"
            else:
                quality_desc = classify_metric(quality, QUALITY_BUCKETS)

            if dry_run:
                print(f"\n[DRY RUN] Would process: {image_path.name}")
                print(f"  Sharpness : {sharpness:.1f}/100 -> {sharpness_desc}")
                print(f"  Brightness: {brightness:.1f}/255 -> {brightness_desc}")
                print(f"  Contrast  : {contrast:.1f}      -> {contrast_desc}")
                print(f"  Quality   : {quality:.3f}/1.0  -> {quality_desc}")
                processed += 1
                continue

            # Generate and write caption
            caption = generate_quality_caption(
                sharpness_desc, brightness_desc, contrast_desc, quality_desc
            )
            output_path.write_text(caption, encoding='utf-8')
            processed += 1

        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}", file=sys.stderr)
            errors += 1
            continue

    # Clear progress line
    if sys.stdout.isatty():
        print()

    return processed, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description='Analyze image quality and generate natural-language quality captions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - analyze all images in subdirectories
  python quality_caption.py --input ./training_data

  # Non-recursive, overwrite existing
  python quality_caption.py --input ./images --no-recursive --overwrite

  # Preview metrics without writing files
  python quality_caption.py --input ./images --dry-run

Output:
  Creates .quality.txt files alongside images (e.g., cat.png -> cat.quality.txt)

Metrics (10 levels each):
  Sharpness  - from "extremely blurry" to "heavily oversharpened" (~35-42 = in focus)
  Brightness - from "extremely dark" to "extremely bright"
  Contrast   - from "essentially flat" to "extremely high contrast"
  Quality    - from "very poor" to "outstanding"
        """
    )

    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input folder containing images (traverses subdirectories)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not traverse subdirectories')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing .quality.txt files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview metrics without writing files')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input folder not found: {args.input}")
        return 1
    if not args.input.is_dir():
        print(f"Error: Input is not a directory: {args.input}")
        return 1

    print(f"Input folder : {args.input}")
    print(f"Recursive    : {'yes' if not args.no_recursive else 'no'}")
    print(f"Overwrite    : {'yes' if args.overwrite else 'no'}")
    if args.dry_run:
        print(f"Mode         : DRY RUN")
    print()

    processed, skipped, errors = process_images(
        input_dir=args.input,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print()
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped} (already have .quality.txt)")
    print(f"Errors:    {errors}")

    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
