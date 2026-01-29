#!/usr/bin/env python3
"""
Adds a prefix to all image and txt filenames in a specified folder.
Useful for merging multiple source training folders into one bucket.

Usage:
  python add_prefix.py --input <folder> --prefix <prefix> [--dry-run]
  python add_prefix.py  # Interactive mode

Example:
  python add_prefix.py --input ./cat --prefix cat_
  python add_prefix.py --input ./cat --prefix cat_ --dry-run

  Before: 0001.png, 0001.txt
  After:  cat_0001.png, cat_0001.txt
"""

import argparse
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}
TEXT_EXTENSIONS = {'.txt'}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS


def add_prefix_to_folder(folder: Path, prefix: str, dry_run: bool = False) -> int:
    """Add prefix to all image and txt files in folder. Returns count of renamed files."""
    renamed = 0

    # Get all files to rename (excluding 'original' subfolder)
    files_to_rename = []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ALL_EXTENSIONS:
            # Skip if already has the prefix
            if file_path.name.startswith(prefix):
                continue
            files_to_rename.append(file_path)

    # Rename files
    for file_path in files_to_rename:
        new_name = prefix + file_path.name
        new_path = file_path.parent / new_name

        if dry_run:
            print(f"  {file_path.name} -> {new_name}")
        else:
            file_path.rename(new_path)
        renamed += 1

    return renamed


def interactive_mode(script_dir: Path):
    """Run in interactive mode, prompting for folder and prefix."""
    print("Add Prefix to Filenames")
    print("-" * 40)

    # List available folders
    folders = [f.name for f in script_dir.iterdir()
               if f.is_dir() and f.name != 'original' and not f.name.startswith('.')]
    print(f"Available folders: {', '.join(sorted(folders))}")
    print()

    folder_name = input("Enter folder name: ").strip()
    prefix = input("Enter prefix (e.g., 'cat_'): ").strip()

    # Validate folder
    folder_path = script_dir / folder_name
    if not folder_path.exists():
        print(f"Error: Folder '{folder_name}' not found")
        return 1
    if not folder_path.is_dir():
        print(f"Error: '{folder_name}' is not a directory")
        return 1

    # Validate prefix
    if not prefix:
        print("Error: Prefix cannot be empty")
        return 1

    # Show preview (dry run)
    print(f"\nPreview of changes in '{folder_name}':")
    count = add_prefix_to_folder(folder_path, prefix, dry_run=True)

    if count == 0:
        print("  No files to rename (already prefixed or no matching files)")
        return 0

    print(f"\n{count} files will be renamed.")

    # Confirm
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return 0

    # Do the actual rename
    renamed = add_prefix_to_folder(folder_path, prefix, dry_run=False)
    print(f"\nDone! Renamed {renamed} files.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Add a prefix to all image and txt filenames in a folder.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python add_prefix.py --input ./cat --prefix cat_
  python add_prefix.py --input ./cat --prefix cat_ --dry-run
  python add_prefix.py  # Interactive mode
        """
    )
    parser.add_argument('--input', '-i', type=Path, default=None,
                        help='Input folder containing files to rename')
    parser.add_argument('--prefix', '-p', type=str, default=None,
                        help='Prefix to add to filenames')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Preview changes without actually renaming')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')

    args = parser.parse_args()

    # If no arguments provided, run interactive mode
    if args.input is None or args.prefix is None:
        if args.input is not None or args.prefix is not None:
            parser.error("Both --input and --prefix are required when using CLI mode")
        script_dir = Path(__file__).parent
        return interactive_mode(script_dir)

    # CLI mode
    if not args.input.exists():
        print(f"Error: Folder not found: {args.input}")
        return 1
    if not args.input.is_dir():
        print(f"Error: Not a directory: {args.input}")
        return 1
    if not args.prefix:
        print("Error: Prefix cannot be empty")
        return 1

    # Show preview
    print(f"Preview of changes in '{args.input}':")
    count = add_prefix_to_folder(args.input, args.prefix, dry_run=True)

    if count == 0:
        print("  No files to rename (already prefixed or no matching files)")
        return 0

    print(f"\n{count} files will be renamed.")

    if args.dry_run:
        print("\nDry run - no files were renamed.")
        return 0

    # Confirm unless --yes flag
    if not args.yes:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return 0

    # Do the actual rename
    renamed = add_prefix_to_folder(args.input, args.prefix, dry_run=False)
    print(f"\nDone! Renamed {renamed} files.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
