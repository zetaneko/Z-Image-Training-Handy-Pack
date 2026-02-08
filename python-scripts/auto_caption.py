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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# Image extensions to process
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}

# User-Agent for API requests (Python-urllib default is blocked by Cloudflare)
_USER_AGENT = 'auto-caption/1.0'

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
    api_key: str = "",
) -> Optional[str]:
    """
    Call an OpenAI-compatible vision API with an image for captioning.

    Args:
        api_url: Base URL (e.g., http://localhost:1234/v1)
        image_path: Path to the image file
        system_prompt: System prompt for the model
        user_prompt: User prompt to send with the image
        model: Model name (empty string uses the loaded model)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        api_key: Optional Bearer token for authenticated endpoints (e.g. VLLM)

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

    headers = {'Content-Type': 'application/json', 'User-Agent': _USER_AGENT}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
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


def get_folder_name(image_path: Path, input_dir: Path, level: int) -> str:
    """
    Extract folder name at the given depth level relative to input_dir.

    For input_dir=/media/images and image_path=/media/images/gallery1/part1/cat.jpg:
      level 1 -> 'gallery1'
      level 2 -> 'part1'

    Returns empty string if the image is not deep enough for the requested level.
    """
    try:
        relative = image_path.relative_to(input_dir)
    except ValueError:
        return ""
    # parts = ('gallery1', 'part1', 'cat.jpg') - last element is the filename
    folder_parts = relative.parts[:-1]  # exclude filename
    if level < 1 or level > len(folder_parts):
        return ""
    return folder_parts[level - 1]


def _process_single_image(
    image_path: Path,
    input_dir: Path,
    api_url: str,
    system_prompt: str,
    use_existing_caption: bool,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    overwrite: bool,
    dry_run: bool,
    folder_name_level: int,
    api_key: str = "",
) -> str:
    """
    Process a single image. Returns 'processed', 'skipped', or 'error'.
    Thread-safe: only reads shared state and writes to its own output file.
    """
    try:
        # Check if output already exists
        output_path = image_path.with_suffix('.autocaption.txt')
        if output_path.exists() and not overwrite:
            return 'skipped'

        # Get existing caption if needed
        existing_caption = None
        if use_existing_caption:
            existing_caption = get_existing_caption(image_path)

        # Get folder name if requested
        folder_name = ""
        if folder_name_level > 0:
            folder_name = get_folder_name(image_path, input_dir, folder_name_level)

        # Build system prompt with substitutions
        format_kwargs = {
            'existing_caption': existing_caption or '',
            'folder_name': folder_name,
        }
        try:
            final_system_prompt = system_prompt.format(**format_kwargs)
        except (KeyError, IndexError):
            final_system_prompt = system_prompt

        if dry_run:
            parts = [f"[DRY RUN] Would process: {image_path.name}"]
            if folder_name:
                parts.append(f"  Folder name (level {folder_name_level}): {folder_name}")
            if existing_caption:
                parts.append(f"  Existing caption: {existing_caption[:100]}...")
            print('\n'.join(parts), flush=True)
            return 'processed'

        # Call API
        caption = call_lm_studio_api(
            api_url=api_url,
            image_path=image_path,
            system_prompt=final_system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
        )

        if caption:
            output_path.write_text(caption, encoding='utf-8')
            return 'processed'
        else:
            print(f"Failed to caption: {image_path.name}", flush=True)
            return 'error'
    except OSError as e:
        print(f"I/O error processing {image_path.name}: {e}", file=sys.stderr, flush=True)
        return 'error'


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
    folder_name_level: int = 0,
    threads: int = 1,
    api_key: str = "",
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
    if threads > 1:
        print(f"Using {threads} threads")

    processed = 0
    skipped = 0
    errors = 0
    completed = 0
    lock = threading.Lock()

    def on_result(result: str):
        nonlocal processed, skipped, errors, completed
        with lock:
            if result == 'processed':
                processed += 1
            elif result == 'skipped':
                skipped += 1
            else:
                errors += 1
            completed += 1
            print_progress(completed, total, "Captioning")

    common_kwargs = dict(
        input_dir=input_dir,
        api_url=api_url,
        system_prompt=system_prompt,
        use_existing_caption=use_existing_caption,
        user_prompt=user_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        overwrite=overwrite,
        dry_run=dry_run,
        folder_name_level=folder_name_level,
        api_key=api_key,
    )

    if threads <= 1:
        # Single-threaded path (preserves original delay behaviour)
        for i, image_path in enumerate(images):
            result = _process_single_image(image_path=image_path, **common_kwargs)
            on_result(result)
            if delay > 0 and i < total - 1:
                time.sleep(delay)
    else:
        # Multi-threaded path
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(_process_single_image, image_path=img, **common_kwargs): img
                for img in images
            }
            for future in as_completed(futures):
                result = future.result()
                on_result(result)

    # Clear progress line
    if sys.stdout.isatty():
        print()

    return processed, skipped, errors


def normalize_api_url(api_url: str) -> str:
    """Normalize API URL to fix common mistakes.

    Fixes http:// with port 443 to https:// (e.g. RunPod proxy URLs).
    """
    url = api_url.strip().rstrip('/')
    # http://host:443 or http://host:443/path -> https://host or https://host/path
    if url.startswith('http://') and ':443' in url:
        url = url.replace('http://', 'https://', 1)
        # Remove redundant :443 from https URLs
        url = url.replace(':443', '', 1)
    return url


def test_api_connection(api_url: str, api_key: str = "") -> tuple[bool, str]:
    """Test if the API is reachable.

    Returns:
        Tuple of (success, message) with details about the result.
    """
    url = f"{api_url.rstrip('/')}/models"
    headers = {'User-Agent': _USER_AGENT}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(url, headers=headers, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return True, "OK"
            return False, f"Unexpected status: {response.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"Connection error: {e.reason}"
    except Exception as e:
        return False, str(e)


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
    parser.add_argument('--api-key', '-k', type=str, default='',
                        help='API key for authenticated endpoints (e.g. VLLM). '
                             'Sent as Bearer token in Authorization header.')

    # Prompt settings
    parser.add_argument('--system-prompt', '-s', type=str, default=None,
                        help='Custom system prompt (default: built-in descriptive prompt)')
    parser.add_argument('--user-prompt', '-u', type=str, default='Please caption this image.',
                        help='User prompt sent with image')
    parser.add_argument('--use-existing-caption', '-e', action='store_true',
                        help='Include existing .txt caption in system prompt for extension')
    parser.add_argument('--folder-name-level', type=int, default=0,
                        help='Include subfolder name in system prompt at this depth level '
                             'relative to --input (0=disabled, 1=first subfolder, 2=second, etc.). '
                             'Use {folder_name} placeholder in system prompt.')

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
    parser.add_argument('--threads', '-t', type=int, default=1,
                        help='Number of concurrent API requests (default: 1). '
                             'Increase to saturate network throughput to remote LM Studio instances.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what would be processed without making API calls')

    args = parser.parse_args()

    # Normalize API URL (fixes http://:443 -> https://)
    args.api_url = normalize_api_url(args.api_url)

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
        # Add placeholder for folder name if using that feature
        if args.folder_name_level > 0 and '{folder_name}' not in system_prompt:
            system_prompt += "\n\nThis image is from the category: {folder_name}"
    elif args.use_existing_caption:
        system_prompt = DEFAULT_SYSTEM_PROMPT_WITH_TAGS
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Add folder name placeholder to default prompts if needed
    if args.folder_name_level > 0 and '{folder_name}' not in system_prompt:
        system_prompt += "\n\nThis image is from the category: {folder_name}"

    print(f"Input folder    : {args.input}")
    print(f"API URL         : {args.api_url}")
    print(f"API Key         : {'set' if args.api_key else 'none'}")
    print(f"Use existing    : {'yes' if args.use_existing_caption else 'no'}")
    print(f"Folder name lvl : {args.folder_name_level if args.folder_name_level > 0 else 'disabled'}")
    print(f"Threads         : {args.threads}")
    print(f"Recursive       : {'yes' if not args.no_recursive else 'no'}")
    print(f"Overwrite       : {'yes' if args.overwrite else 'no'}")
    if args.dry_run:
        print(f"Mode            : DRY RUN")
    print()

    # Test API connection (skip for dry run)
    if not args.dry_run:
        print("Testing API connection...", end=" ")
        ok, msg = test_api_connection(args.api_url, api_key=args.api_key)
        if ok:
            print("OK")
        else:
            print("FAILED")
            print(f"\nCould not connect to API at {args.api_url}")
            print(f"Error: {msg}")
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
        folder_name_level=args.folder_name_level,
        threads=args.threads,
        api_key=args.api_key,
    )

    print()
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped} (already have .autocaption.txt)")
    print(f"Errors:    {errors}")

    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
