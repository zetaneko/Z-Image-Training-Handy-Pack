#!/usr/bin/env python3
"""
Auto-caption images using LM Studio's OpenAI-compatible REST API.

Traverses subdirectories, reads existing captions from .txt files,
sends images to a vision model, and writes new captions to .autocaption.txt files.

Usage:
  python auto_caption.py --input ./Danbooru
  python auto_caption.py --input ./images --use-existing-caption
  python auto_caption.py --input ./images --api-url http://localhost:1234/v1

Example:
  # Basic usage - caption all images in subdirectories
  python auto_caption.py --input ./training_data

  # Include existing tags in the prompt for the model to extend
  python auto_caption.py --input ./Danbooru --use-existing-caption

  # Custom system prompt
  python auto_caption.py --input ./images --system-prompt "Describe this anime image in detail"
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# Image extensions to process
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """Write a descriptive caption for this image in a formal tone. Focus on the main subjects, their appearance, actions, setting, and mood. Be detailed but concise."""

DEFAULT_SYSTEM_PROMPT_WITH_TAGS = """Write a descriptive caption for this image in a formal tone. These are the tags previously made for this image to extend on:

{existing_caption}

Use these tags as a starting point to write a natural, flowing description. Focus on the main subjects, their appearance, actions, setting, and mood. Be detailed but concise."""


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


def encode_image_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_image_mime_type(image_path: Path) -> str:
    """Get MIME type for an image based on extension."""
    ext = image_path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
    }
    return mime_types.get(ext, 'image/png')


def get_existing_caption(image_path: Path) -> Optional[str]:
    """Read existing caption from .txt file if it exists."""
    txt_path = image_path.with_suffix('.txt')
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding='utf-8').strip()
        except Exception:
            return None
    return None


def call_lm_studio_api(
    api_url: str,
    image_path: Path,
    system_prompt: str,
    user_prompt: str = "Please caption this image.",
    model: str = "",
    max_tokens: int = 500,
    temperature: float = 0.7,
    timeout: int = 120,
) -> Optional[str]:
    """
    Call LM Studio API with an image for captioning.

    Args:
        api_url: Base URL for LM Studio API (e.g., http://localhost:1234/v1)
        image_path: Path to the image file
        system_prompt: System prompt for the model
        user_prompt: User prompt to send with the image
        model: Model name (empty string uses the loaded model)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        timeout: Request timeout in seconds

    Returns:
        Generated caption or None on error
    """
    # Encode image
    image_base64 = encode_image_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    # Build request payload (OpenAI-compatible format)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    # Add model if specified
    if model:
        payload["model"] = model

    # Make request
    url = f"{api_url.rstrip('/')}/chat/completions"
    data = json.dumps(payload).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json',
        },
        method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))

            # Extract caption from response
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                content = message.get('content', '')
                return content.strip()

            return None
    except urllib.error.URLError as e:
        print(f"\nAPI Error: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"\nJSON Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return None


def collect_images(input_dir: Path, recursive: bool = True) -> list[Path]:
    """Collect all image files from directory."""
    images = []

    if recursive:
        for p in sorted(input_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(p)
    else:
        for p in sorted(input_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(p)

    return images


def process_images(
    input_dir: Path,
    api_url: str,
    system_prompt: str,
    use_existing_caption: bool = False,
    user_prompt: str = "Please caption this image.",
    model: str = "",
    max_tokens: int = 500,
    temperature: float = 0.7,
    recursive: bool = True,
    overwrite: bool = False,
    delay: float = 0.0,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Process all images in a directory.

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
        print_progress(i + 1, total, "Captioning")

        # Check if output already exists
        output_path = image_path.with_suffix('.autocaption.txt')
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        # Get existing caption if needed
        existing_caption = None
        if use_existing_caption:
            existing_caption = get_existing_caption(image_path)

        # Build system prompt
        if existing_caption and use_existing_caption:
            final_system_prompt = system_prompt.format(existing_caption=existing_caption)
        else:
            final_system_prompt = system_prompt

        if dry_run:
            print(f"\n[DRY RUN] Would process: {image_path.name}")
            if existing_caption:
                print(f"  Existing caption: {existing_caption[:100]}...")
            processed += 1
            continue

        # Call API
        caption = call_lm_studio_api(
            api_url=api_url,
            image_path=image_path,
            system_prompt=final_system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if caption:
            # Write caption to .autocaption.txt
            output_path.write_text(caption, encoding='utf-8')
            processed += 1
        else:
            errors += 1
            print(f"\nFailed to caption: {image_path.name}")

        # Delay between requests if specified
        if delay > 0 and i < total - 1:
            time.sleep(delay)

    # Clear progress line
    if sys.stdout.isatty():
        print()

    return processed, skipped, errors


def test_api_connection(api_url: str) -> bool:
    """Test if the API is reachable."""
    try:
        url = f"{api_url.rstrip('/')}/models"
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Auto-caption images using LM Studio vision model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - caption all images in subdirectories
  python auto_caption.py --input ./training_data

  # Include existing .txt captions in the prompt
  python auto_caption.py --input ./Danbooru --use-existing-caption

  # Custom system prompt
  python auto_caption.py --input ./images --system-prompt "Describe this anime character"

  # Custom API endpoint
  python auto_caption.py --input ./images --api-url http://192.168.1.100:1234/v1

  # Preview without making changes
  python auto_caption.py --input ./images --dry-run

Output:
  Creates .autocaption.txt files alongside images (e.g., cat.png -> cat.autocaption.txt)
        """
    )

    # Required arguments
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input folder containing images (traverses subdirectories)')

    # API settings
    parser.add_argument('--api-url', '-a', type=str, default='http://localhost:1234/v1',
                        help='LM Studio API URL (default: http://localhost:1234/v1)')
    parser.add_argument('--model', '-m', type=str, default='',
                        help='Model name (default: use currently loaded model)')

    # Prompt settings
    parser.add_argument('--system-prompt', '-s', type=str, default=None,
                        help='Custom system prompt (default: built-in descriptive prompt)')
    parser.add_argument('--user-prompt', '-u', type=str, default='Please caption this image.',
                        help='User prompt sent with image')
    parser.add_argument('--use-existing-caption', '-e', action='store_true',
                        help='Include existing .txt caption in system prompt for extension')

    # Generation settings
    parser.add_argument('--max-tokens', type=int, default=500,
                        help='Maximum tokens in response (default: 500)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')

    # Processing options
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not traverse subdirectories')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing .autocaption.txt files')
    parser.add_argument('--delay', '-d', type=float, default=0.0,
                        help='Delay between API calls in seconds (default: 0)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what would be processed without making API calls')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input folder not found: {args.input}")
        return 1
    if not args.input.is_dir():
        print(f"Error: Input is not a directory: {args.input}")
        return 1

    # Determine system prompt
    if args.system_prompt:
        system_prompt = args.system_prompt
        # Add placeholder for existing caption if using that feature
        if args.use_existing_caption and '{existing_caption}' not in system_prompt:
            system_prompt += "\n\nExisting tags to extend: {existing_caption}"
    elif args.use_existing_caption:
        system_prompt = DEFAULT_SYSTEM_PROMPT_WITH_TAGS
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    print(f"Input folder    : {args.input}")
    print(f"API URL         : {args.api_url}")
    print(f"Use existing    : {'yes' if args.use_existing_caption else 'no'}")
    print(f"Recursive       : {'yes' if not args.no_recursive else 'no'}")
    print(f"Overwrite       : {'yes' if args.overwrite else 'no'}")
    if args.dry_run:
        print(f"Mode            : DRY RUN")
    print()

    # Test API connection (skip for dry run)
    if not args.dry_run:
        print("Testing API connection...", end=" ")
        if test_api_connection(args.api_url):
            print("OK")
        else:
            print("FAILED")
            print(f"\nCould not connect to LM Studio at {args.api_url}")
            print("Make sure LM Studio is running with a vision model loaded.")
            print("Enable the local server in LM Studio: Developer -> Local Server")
            return 1

    print()

    # Process images
    processed, skipped, errors = process_images(
        input_dir=args.input,
        api_url=args.api_url,
        system_prompt=system_prompt,
        use_existing_caption=args.use_existing_caption,
        user_prompt=args.user_prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
        delay=args.delay,
        dry_run=args.dry_run,
    )

    print()
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped} (already have .autocaption.txt)")
    print(f"Errors:    {errors}")

    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
