#!/usr/bin/env python3
"""
Search zitpack archives by filename, caption, or both.
Supports multi-threaded searching across multiple archive chunks.

Basic Usage:
  python archive_search.py ./data.zitpack "sunset"              # Search captions
  python archive_search.py ./data.zitpack --filename "*.png"    # Glob filename
  python archive_search.py "./dataset_*.zitpack" "cat"          # Search multiple chunks

Multi-Archive Examples:
  # Search all chunks in parallel (auto-detects CPU cores)
  python archive_search.py "./dataset_chunk_*.zitpack" "portrait"

  # Specify thread count
  python archive_search.py "./dataset_*.zitpack" "sunset" --threads 8

  # Group results by archive
  python archive_search.py "./dataset_*.zitpack" "cat" --group-by-archive

Advanced Usage:
  python archive_search.py ./data.zitpack "portrait" --regex
  python archive_search.py ./data.zitpack --filename "img_*" --caption "woman"
  python archive_search.py ./data.zitpack --no-caption          # Find uncaptioned images
  python archive_search.py ./data.zitpack --min-width 1024      # By dimensions
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_archive import DatasetArchive, SearchResult


@dataclass
class SearchParams:
    """Parameters for a search operation."""
    caption_query: Optional[str] = None
    filename_pattern: Optional[str] = None
    regex: bool = False
    case_sensitive: bool = False
    operator: str = 'and'
    min_width: int = 0
    min_height: int = 0
    max_width: int = 0
    max_height: int = 0
    has_caption: Optional[bool] = None
    limit: int = 0


def expand_archive_paths(pattern: str) -> list[Path]:
    """Expand a glob pattern to list of archive paths."""
    if '*' in pattern or '?' in pattern or '[' in pattern:
        paths = [Path(p) for p in sorted(glob.glob(pattern))]
    else:
        paths = [Path(pattern)]
    return [p for p in paths if p.exists() and p.suffix.lower() == '.zitpack']


def format_result(
    result: SearchResult,
    show_caption: bool = True,
    show_archive: bool = False,
    caption_length: int = 100,
) -> str:
    """Format a search result for display."""
    lines = []

    # Header line
    if show_archive and result.archive_path:
        header = f"[{result.archive_path.name}:{result.index}] {result.filename}"
    else:
        header = f"[{result.index}] {result.filename}"

    if result.match_in == 'caption':
        header += "  (caption match)"
    elif result.match_in == 'both':
        header += "  (filename + caption)"

    lines.append(header)

    # Caption
    if show_caption and result.caption:
        caption = result.caption
        if len(caption) > caption_length:
            caption = caption[:caption_length] + "..."
        lines.append(f"    {caption}")

    return "\n".join(lines)


def search_single_archive(
    archive_path: Path,
    params: SearchParams,
) -> list[SearchResult]:
    """Search a single archive and return results."""
    results = []

    try:
        with DatasetArchive(archive_path) as archive:
            # Determine search type
            if params.filename_pattern and params.caption_query:
                # Advanced search with both criteria
                results = archive.search_advanced(
                    filename_pattern=params.filename_pattern,
                    caption_query=params.caption_query,
                    operator=params.operator,
                    regex=params.regex,
                    case_sensitive=params.case_sensitive,
                    min_width=params.min_width,
                    min_height=params.min_height,
                    max_width=params.max_width,
                    max_height=params.max_height,
                    has_caption=params.has_caption,
                    limit=params.limit,
                )
            elif params.filename_pattern:
                # Filename search only
                results = archive.search_filename(
                    params.filename_pattern,
                    glob=not params.regex,
                    regex=params.regex,
                    case_sensitive=params.case_sensitive,
                    limit=params.limit,
                )
                # Apply additional filters
                if params.has_caption is not None or params.min_width or params.min_height or params.max_width or params.max_height:
                    results = _filter_results(
                        archive, results,
                        has_caption=params.has_caption,
                        min_width=params.min_width, min_height=params.min_height,
                        max_width=params.max_width, max_height=params.max_height,
                    )
            elif params.caption_query:
                # Caption search only
                results = archive.search_caption(
                    params.caption_query,
                    regex=params.regex,
                    case_sensitive=params.case_sensitive,
                    limit=params.limit,
                )
                # Apply additional filters
                if params.min_width or params.min_height or params.max_width or params.max_height:
                    results = _filter_results(
                        archive, results,
                        min_width=params.min_width, min_height=params.min_height,
                        max_width=params.max_width, max_height=params.max_height,
                    )
            elif params.has_caption is not None or params.min_width or params.min_height or params.max_width or params.max_height:
                # Filter-only search
                results = archive.search_advanced(
                    has_caption=params.has_caption,
                    min_width=params.min_width,
                    min_height=params.min_height,
                    max_width=params.max_width,
                    max_height=params.max_height,
                    limit=params.limit,
                )
    except Exception as e:
        print(f"Warning: Error searching {archive_path.name}: {e}", file=sys.stderr)
        return []

    return results


def search_archives_parallel(
    archive_paths: list[Path],
    params: SearchParams,
    max_threads: int = 0,
    show_progress: bool = True,
) -> tuple[list[SearchResult], dict[str, int]]:
    """
    Search across multiple archives in parallel.

    Args:
        archive_paths: List of archive paths to search
        params: Search parameters
        max_threads: Max threads (0 = auto based on CPU count)
        show_progress: Show progress indicator

    Returns:
        Tuple of (all_results, per_archive_counts)
    """
    if not archive_paths:
        return [], {}

    # Determine thread count
    if max_threads <= 0:
        max_threads = min(len(archive_paths), os.cpu_count() or 4)

    all_results: list[SearchResult] = []
    per_archive_counts: dict[str, int] = {}
    completed_count = 0
    total_archives = len(archive_paths)

    # For single archive, don't bother with threading
    if len(archive_paths) == 1:
        results = search_single_archive(archive_paths[0], params)
        per_archive_counts[archive_paths[0].name] = len(results)
        return results, per_archive_counts

    # Multi-threaded search
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all search tasks
        future_to_path = {
            executor.submit(search_single_archive, path, params): path
            for path in archive_paths
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            archive_path = future_to_path[future]
            completed_count += 1

            if show_progress and sys.stderr.isatty():
                print(f"\rSearching: {completed_count}/{total_archives} archives...", end="", file=sys.stderr)

            try:
                results = future.result()
                all_results.extend(results)
                per_archive_counts[archive_path.name] = len(results)

                # Check global limit
                if params.limit > 0 and len(all_results) >= params.limit:
                    # Cancel remaining futures
                    for f in future_to_path:
                        f.cancel()
                    break
            except Exception as e:
                print(f"\nError processing {archive_path.name}: {e}", file=sys.stderr)
                per_archive_counts[archive_path.name] = 0

    if show_progress and sys.stderr.isatty():
        print(file=sys.stderr)  # Clear progress line

    # Apply global limit
    if params.limit > 0 and len(all_results) > params.limit:
        all_results = all_results[:params.limit]

    return all_results, per_archive_counts


def _filter_results(
    archive: DatasetArchive,
    results: list[SearchResult],
    has_caption: Optional[bool] = None,
    min_width: int = 0,
    min_height: int = 0,
    max_width: int = 0,
    max_height: int = 0,
) -> list[SearchResult]:
    """Apply additional filters to search results."""
    filtered = []

    for result in results:
        info = archive.get_entry_info(result.index)

        # Check caption presence
        if has_caption is not None:
            caption = archive.get_caption(result.index)
            if has_caption and not caption:
                continue
            if not has_caption and caption:
                continue

        # Check dimensions
        if min_width > 0 and info.width < min_width:
            continue
        if min_height > 0 and info.height < min_height:
            continue
        if max_width > 0 and info.width > max_width:
            continue
        if max_height > 0 and info.height > max_height:
            continue

        filtered.append(result)

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description='Search zitpack archives by filename, caption, or both.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple caption search
  python archive_search.py ./data.zitpack "sunset"

  # Search multiple archive chunks (parallel)
  python archive_search.py "./dataset_chunk_*.zitpack" "portrait"

  # Specify thread count for large datasets
  python archive_search.py "./dataset_*.zitpack" "cat" --threads 8

  # Group results by archive
  python archive_search.py "./dataset_*.zitpack" "sunset" --group-by-archive

  # Filename glob pattern
  python archive_search.py ./data.zitpack --filename "*.png"

  # Combined search (AND)
  python archive_search.py ./data.zitpack --filename "img_*" --caption "portrait"

  # Combined search (OR)
  python archive_search.py ./data.zitpack --filename "sunset*" --caption "sunset" --or

  # Regex search
  python archive_search.py ./data.zitpack "(cat|dog|bird)" --regex

  # Find uncaptioned images
  python archive_search.py ./data.zitpack --no-caption

  # Filter by dimensions
  python archive_search.py ./data.zitpack --min-width 1024 --min-height 768

Output Formats:
  --output filenames    Just print filenames (for piping)
  --output json         JSON output (for scripts)
  --output full         Full details with captions (default)
        """
    )

    # Positional arguments
    parser.add_argument('archives', type=str,
                        help='Archive path or glob pattern (e.g., "./dataset_*.zitpack")')
    parser.add_argument('query', type=str, nargs='?', default=None,
                        help='Caption search query (optional if using --filename)')

    # Search options
    parser.add_argument('--filename', '-f', type=str, default=None,
                        help='Filename pattern (glob by default, e.g., "*.png", "img_???.jpg")')
    parser.add_argument('--caption', '-c', type=str, default=None,
                        help='Caption search term (alternative to positional query)')
    parser.add_argument('--regex', '-r', action='store_true',
                        help='Treat patterns as regular expressions')
    parser.add_argument('--case-sensitive', '-s', action='store_true',
                        help='Case-sensitive matching')
    parser.add_argument('--or', dest='use_or', action='store_true',
                        help='Use OR logic (match any criteria); default is AND')

    # Filter options
    parser.add_argument('--has-caption', action='store_true',
                        help='Only show entries with captions')
    parser.add_argument('--no-caption', action='store_true',
                        help='Only show entries without captions')
    parser.add_argument('--min-width', type=int, default=0,
                        help='Minimum image width')
    parser.add_argument('--min-height', type=int, default=0,
                        help='Minimum image height')
    parser.add_argument('--max-width', type=int, default=0,
                        help='Maximum image width')
    parser.add_argument('--max-height', type=int, default=0,
                        help='Maximum image height')

    # Threading options
    parser.add_argument('--threads', '-t', type=int, default=0,
                        help='Number of threads for parallel search (0 = auto)')

    # Output options
    parser.add_argument('--limit', '-n', type=int, default=0,
                        help='Maximum number of results (0 = unlimited)')
    parser.add_argument('--output', '-o', choices=['full', 'filenames', 'json'],
                        default='full', help='Output format')
    parser.add_argument('--no-caption-text', action='store_true',
                        help='Hide caption text in output')
    parser.add_argument('--group-by-archive', '-g', action='store_true',
                        help='Group results by archive file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Expand archive paths
    archive_paths = expand_archive_paths(args.archives)
    if not archive_paths:
        print(f"Error: No archives found matching: {args.archives}")
        return 1

    # Resolve caption query
    caption_query = args.caption or args.query

    # Validate we have something to search for
    if not caption_query and not args.filename and not args.has_caption and not args.no_caption \
       and not args.min_width and not args.min_height and not args.max_width and not args.max_height:
        print("Error: Specify a search query, --filename pattern, or filter options")
        return 1

    # Handle caption filter flags
    has_caption = None
    if args.has_caption:
        has_caption = True
    elif args.no_caption:
        has_caption = False

    # Build search params
    params = SearchParams(
        caption_query=caption_query,
        filename_pattern=args.filename,
        regex=args.regex,
        case_sensitive=args.case_sensitive,
        operator='or' if args.use_or else 'and',
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        has_caption=has_caption,
        limit=args.limit,
    )

    # Show search info
    if not args.quiet and len(archive_paths) > 1:
        thread_count = args.threads if args.threads > 0 else min(len(archive_paths), os.cpu_count() or 4)
        print(f"Searching {len(archive_paths)} archives with {thread_count} threads...", file=sys.stderr)

    # Perform parallel search
    results, per_archive_counts = search_archives_parallel(
        archive_paths,
        params,
        max_threads=args.threads,
        show_progress=not args.quiet,
    )

    # Output results
    show_archive = len(archive_paths) > 1

    if args.output == 'filenames':
        for result in results:
            if show_archive:
                print(f"{result.archive_path.name}:{result.filename}")
            else:
                print(result.filename)

    elif args.output == 'json':
        import json
        output = {
            'total_results': len(results),
            'archives_searched': len(archive_paths),
            'results': [],
        }

        if show_archive:
            output['per_archive_counts'] = per_archive_counts

        for result in results:
            entry = {
                'index': result.index,
                'filename': result.filename,
                'caption': result.caption,
                'match_in': result.match_in,
            }
            if show_archive:
                entry['archive'] = result.archive_path.name
            output['results'].append(entry)

        print(json.dumps(output, indent=2))

    else:  # full
        if not results:
            print("No matches found.")
        else:
            if args.group_by_archive and show_archive:
                # Group results by archive
                from collections import defaultdict
                grouped = defaultdict(list)
                for result in results:
                    grouped[result.archive_path.name].append(result)

                print(f"Found {len(results)} match(es) across {len(grouped)} archive(s):\n")

                for archive_name in sorted(grouped.keys()):
                    archive_results = grouped[archive_name]
                    print(f"=== {archive_name} ({len(archive_results)} matches) ===\n")

                    for result in archive_results:
                        print(format_result(
                            result,
                            show_caption=not args.no_caption_text,
                            show_archive=False,  # Already shown in header
                        ))
                        print()
            else:
                # Flat list
                print(f"Found {len(results)} match(es)", end="")
                if show_archive:
                    print(f" across {len(per_archive_counts)} archive(s)", end="")
                print(":\n")

                for result in results:
                    print(format_result(
                        result,
                        show_caption=not args.no_caption_text,
                        show_archive=show_archive,
                    ))
                    print()

            # Show per-archive summary for multi-archive search
            if show_archive and not args.group_by_archive and not args.quiet:
                print("---")
                print("Results per archive:")
                for name, count in sorted(per_archive_counts.items()):
                    if count > 0:
                        print(f"  {name}: {count}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
