#!/usr/bin/env python3
"""
Replaces tags in .txt files with captions from tagconversions.csv.
Backs up original files to an 'original' subfolder in each directory.
Intelligently combines common Danbooru tags into natural language descriptions.

Useful if you have a dataset originally written for a Danbooru-tag based model
and need a quick first-pass to adapt it to caption format.

Usage:
  python replace_booru_tags.py --input <folder> [--conversions <csv>]
  python replace_booru_tags.py  # Process all folders in script directory

Example:
  python replace_booru_tags.py --input ./dataset
  python replace_booru_tags.py --input ./dataset --conversions ./my_tags.csv
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path


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


def load_tag_conversions(csv_path: Path) -> dict[str, str]:
    """Load tag to caption mappings from CSV file."""
    conversions = {}
    if not csv_path.exists():
        return conversions
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row['Tag'].strip()
            caption = row['Caption'].strip()
            if tag:
                conversions[tag] = caption
    return conversions


# =============================================================================
# Danbooru Tag Definitions
# =============================================================================

# Tags to skip entirely (handled specially or redundant)
SKIP_TAGS = {'1girl', 'solo', 'female', 'woman', 'solo focus', 'breasts'}

# Hair
HAIR_LENGTHS = ['very short', 'short', 'medium', 'long', 'very long', 'absurdly long']
HAIR_COLORS = [
    'brown', 'black', 'blonde', 'green', 'yellow', 'red', 'blue', 'purple',
    'white', 'grey', 'gray', 'pink', 'aqua', 'orange', 'silver', 'light brown',
    'dark brown', 'light blue', 'dark blue', 'light green', 'light purple',
    'platinum blonde', 'strawberry blonde', 'auburn', 'multicolored', 'two-tone'
]
HAIR_STYLES = [
    'ponytail', 'twintails', 'twin braids', 'braid', 'braided', 'bun', 'hair bun',
    'double bun', 'side ponytail', 'low ponytail', 'high ponytail', 'bob cut',
    'pixie cut', 'hime cut', 'straight hair', 'wavy hair', 'curly hair', 'messy hair'
]

# Eyes
EYE_COLORS = [
    'blue', 'brown', 'green', 'red', 'yellow', 'purple', 'pink', 'orange',
    'aqua', 'grey', 'gray', 'black', 'amber', 'golden', 'violet'
]

# Body
BODY_TYPES = ['thin', 'slim', 'slender', 'skinny', 'petite', 'chubby', 'plump',
              'curvy', 'muscular', 'athletic', 'toned', 'thick', 'voluptuous']
BREAST_SIZES = ['flat chest', 'small breasts', 'medium breasts', 'large breasts',
                'huge breasts', 'gigantic breasts']
SKIN_TONES = ['pale', 'fair', 'tan', 'tanned', 'dark-skinned', 'dark skin', 'olive']

# Ethnicity
ETHNICITIES = ['chinese', 'japanese', 'korean', 'asian', 'k-pop', 'caucasian',
               'african', 'latina', 'indian', 'southeast asian', 'european']

# Locations
LOCATIONS = ['indoors', 'outdoors', 'bedroom', 'bathroom', 'kitchen', 'living room',
             'office', 'gym', 'beach', 'pool', 'forest', 'park', 'street', 'car',
             'studio', 'hospital', 'hotel', 'shower', 'bathtub']

# Poses
POSES = ['standing', 'sitting', 'lying', 'kneeling', 'squatting', 'bending over',
         'on back', 'on side', 'on stomach', 'leaning', 'crouching']

# Gaze/Looking
GAZE_TAGS = {
    'looking at viewer': 'looking at the viewer',
    'looking away': 'looking away',
    'looking down': 'looking down',
    'looking up': 'looking up',
    'looking back': 'looking back',
    'looking at phone': 'looking at her phone',
    'looking at mirror': 'looking at the mirror',
    'eyes closed': 'her eyes are closed',
    'closed eyes': 'her eyes are closed',
}

# Expressions
EXPRESSIONS = {
    'smile': 'smiling',
    'smiling': 'smiling',
    'grin': 'grinning',
    'frown': 'frowning',
    'crying': 'crying',
    'angry': 'looking angry',
    'surprised': 'looking surprised',
    'embarrassed': 'looking embarrassed',
    'nervous': 'looking nervous',
    'parted lips': 'with parted lips',
    'closed mouth': 'with her mouth closed',
    'open mouth': 'with her mouth open',
    'tongue out': 'with her tongue out',
}

# Clothing items (for combining)
TOPS = ['shirt', 'blouse', 't-shirt', 'tank top', 'crop top', 'sweater', 'hoodie',
        'jacket', 'coat', 'cardigan', 'vest', 'tube top', 'bikini top', 'bra',
        'sports bra', 'camisole', 'bustier']
BOTTOMS = ['pants', 'jeans', 'shorts', 'skirt', 'miniskirt', 'dress', 'leggings',
           'sweatpants', 'bikini bottom', 'panties', 'underwear', 'thong']
FULL_BODY_CLOTHING = ['dress', 'gown', 'jumpsuit', 'bodysuit', 'swimsuit', 'bikini',
                      'one-piece swimsuit', 'lingerie', 'negligee', 'robe', 'kimono']
FOOTWEAR = ['shoes', 'heels', 'high heels', 'boots', 'sandals', 'sneakers', 'slippers',
            'barefoot', 'socks', 'thighhighs', 'stockings', 'pantyhose', 'knee highs']
ACCESSORIES = ['glasses', 'sunglasses', 'hat', 'cap', 'earrings', 'necklace',
               'bracelet', 'ring', 'watch', 'mask', 'choker', 'hair ribbon',
               'hair bow', 'headband', 'scrunchie', 'nail polish']

# Colors for clothing
COLORS = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
          'orange', 'brown', 'grey', 'gray', 'beige', 'navy', 'cream', 'striped']

# Actions
ACTIONS = {
    'holding phone': 'holding a phone',
    'holding': 'holding something',
    'selfie': 'taking a selfie',
    'phone': None,  # Skip, redundant with holding phone
    'cellphone': None,
    'smartphone': None,
    'holding hands': 'holding hands with someone',
    'hand on own stomach': 'with her hand on her stomach',
    'hand on hip': 'with her hand on her hip',
    'hands on hips': 'with her hands on her hips',
    'arms crossed': 'with her arms crossed',
    'arms at sides': 'with her arms at her sides',
    'v': 'making a peace sign',
    'peace sign': 'making a peace sign',
}

# Visible body parts that need phrasing
BODY_PARTS_VISIBLE = {
    'navel': 'her navel is visible',
    'midriff': 'her midriff is exposed',
    'cleavage': 'showing cleavage',
    'underboob': 'showing underboob',
    'sideboob': 'showing sideboob',
    'nipples': 'her nipples are visible',
    'pubic hair': 'her pubic hair is visible',
    'female pubic hair': None,  # redundant
    'pussy': 'her pussy is visible',
    'ass': 'her butt is visible',
    'feet': 'her feet are visible',
    'toes': 'her toes are visible',
    'toenails': 'her toenails are visible',
    'lips': None,  # Skip, too generic
    'thighs': 'her thighs are visible',
    'armpits': 'her armpits are visible',
    'back': 'her back is visible',
    'belly': 'her belly is visible',
    'stomach': 'her stomach is visible',
}

# Photo composition
COMPOSITION = {
    'full body': 'full body shot',
    'upper body': 'upper body shot',
    'lower body': 'lower body shot',
    'portrait': 'portrait shot',
    'close-up': 'close-up shot',
    'from above': 'shot from above',
    'from below': 'shot from below',
    'from side': 'shot from the side',
    'from behind': 'shot from behind',
    'dutch angle': 'at a tilted angle',
    'pov': 'from her point of view',
    'mirror': 'in a mirror',
    'reflection': 'showing her reflection',
    'selfie': 'taking a selfie',
}

# Background/atmosphere
BACKGROUND = {
    'blurry': 'slightly blurry',
    'blurry background': 'with a blurred background',
    'depth of field': 'with depth of field',
    'bokeh': 'with bokeh effect',
    'day': 'during the day',
    'night': 'at night',
    'sunset': 'at sunset',
    'sunrise': 'at sunrise',
    'bright': 'brightly lit',
    'dark': 'dimly lit',
    'natural lighting': 'with natural lighting',
    'out of frame': None,  # Skip
}


# =============================================================================
# Tag Extraction Functions
# =============================================================================

# Pattern for character tags like "jeanne d'arc alter \(fate\)"
CHARACTER_PATTERN = re.compile(r'^(.+?)\s*\\\((.+?)\\\)$')


class TagProcessor:
    def __init__(self, tags: list[str], conversions: dict[str, str]):
        self.original_tags = tags
        self.remaining_tags = set(tags)
        self.conversions = conversions
        self.phrases = {
            'character': [],     # character name and series
            'appearance': [],    # hair, eyes, body
            'clothing': [],      # what she's wearing
            'ethnicity': [],     # ethnicity/nationality
            'location': [],      # where she is
            'pose': [],          # body position
            'action': [],        # what she's doing
            'expression': [],    # facial expression
            'body_visible': [],  # visible body parts
            'composition': [],   # photo composition
            'other': [],         # everything else
        }

    def remove_tag(self, tag: str):
        self.remaining_tags.discard(tag)

    def extract_character(self):
        """Extract character tags like 'jeanne d'arc alter \\(fate\\)'."""
        for tag in list(self.remaining_tags):
            match = CHARACTER_PATTERN.match(tag)
            if match:
                character_name = match.group(1).strip()
                series_name = match.group(2).strip()
                self.phrases['character'].append(
                    f'the character {character_name} from {series_name}'
                )
                self.remove_tag(tag)

    def extract_hair(self):
        """Extract and combine hair attributes."""
        length = None
        color = None
        style = None

        for tag in list(self.remaining_tags):
            # Length
            for hl in sorted(HAIR_LENGTHS, key=len, reverse=True):
                if tag == f'{hl} hair':
                    length = hl
                    self.remove_tag(tag)
                    break

            # Color
            for hc in sorted(HAIR_COLORS, key=len, reverse=True):
                if tag == f'{hc} hair':
                    color = hc
                    self.remove_tag(tag)
                    break

            # Style
            if tag in HAIR_STYLES:
                style = tag
                self.remove_tag(tag)

        if length or color:
            parts = [p for p in [length, color] if p]
            self.phrases['appearance'].append(f'{" ".join(parts)} hair')

        if style:
            if style in ['braid', 'braided']:
                self.phrases['appearance'].append('her hair is braided')
            elif style.endswith(' hair'):
                self.phrases['appearance'].append(f'her hair is {style.replace(" hair", "")}')
            else:
                self.phrases['appearance'].append(f'her hair in a {style}')

    def extract_eyes(self):
        """Extract eye color."""
        for tag in list(self.remaining_tags):
            for ec in sorted(EYE_COLORS, key=len, reverse=True):
                if tag == f'{ec} eyes':
                    self.phrases['appearance'].append(f'{ec} eyes')
                    self.remove_tag(tag)
                    return

    def extract_body(self):
        """Extract body type, breast size, skin tone."""
        # Body type
        for tag in list(self.remaining_tags):
            if tag in BODY_TYPES:
                self.phrases['appearance'].append(f'a {tag} figure')
                self.remove_tag(tag)
                break

        # Breast size
        for tag in list(self.remaining_tags):
            if tag in BREAST_SIZES:
                if tag == 'flat chest':
                    self.phrases['appearance'].append('a flat chest')
                else:
                    self.phrases['appearance'].append(tag)
                self.remove_tag(tag)
                break

        # Skin tone
        for tag in list(self.remaining_tags):
            for st in SKIN_TONES:
                if tag == st or tag == f'{st} skin':
                    if st in ['tan', 'tanned']:
                        self.phrases['appearance'].append('tanned skin')
                    elif st == 'dark-skinned' or st == 'dark skin':
                        self.phrases['appearance'].append('dark skin')
                    else:
                        self.phrases['appearance'].append(f'{st} skin')
                    self.remove_tag(tag)
                    return

    def extract_ethnicity(self):
        """Extract ethnicity/nationality."""
        for tag in list(self.remaining_tags):
            if tag in ETHNICITIES:
                if tag == 'k-pop':
                    self.phrases['ethnicity'].append('korean')
                else:
                    self.phrases['ethnicity'].append(tag)
                self.remove_tag(tag)

    def extract_location(self):
        """Extract location."""
        for tag in list(self.remaining_tags):
            if tag in LOCATIONS:
                self.phrases['location'].append(tag)
                self.remove_tag(tag)
                return

    def extract_pose(self):
        """Extract pose/position."""
        for tag in list(self.remaining_tags):
            if tag in POSES:
                self.phrases['pose'].append(tag)
                self.remove_tag(tag)
                return

    def extract_gaze(self):
        """Extract gaze/looking direction."""
        for tag in list(self.remaining_tags):
            if tag in GAZE_TAGS:
                phrase = GAZE_TAGS[tag]
                if phrase:
                    self.phrases['action'].append(phrase)
                self.remove_tag(tag)

    def extract_expression(self):
        """Extract facial expression."""
        for tag in list(self.remaining_tags):
            if tag in EXPRESSIONS:
                phrase = EXPRESSIONS[tag]
                if phrase:
                    self.phrases['expression'].append(phrase)
                self.remove_tag(tag)

    def extract_clothing(self):
        """Extract and intelligently combine clothing items."""
        found_tops = []
        found_bottoms = []
        found_full = []
        found_footwear = []
        found_accessories = []

        for tag in list(self.remaining_tags):
            # Check for color + item combinations (e.g., "grey shirt", "white bra")
            tag_lower = tag.lower()

            # Extract item with optional color
            color_prefix = None
            item = tag
            for c in COLORS:
                if tag_lower.startswith(f'{c} '):
                    color_prefix = c
                    item = tag[len(c)+1:]
                    break

            item_lower = item.lower()

            # Categorize
            matched = False
            for top in TOPS:
                if item_lower == top or item_lower == f'{top}s':
                    desc = f'{color_prefix} {item}' if color_prefix else item
                    found_tops.append(desc)
                    self.remove_tag(tag)
                    matched = True
                    break

            if not matched:
                for bottom in BOTTOMS:
                    if item_lower == bottom or item_lower == f'{bottom}s':
                        desc = f'{color_prefix} {item}' if color_prefix else item
                        found_bottoms.append(desc)
                        self.remove_tag(tag)
                        matched = True
                        break

            if not matched:
                for full in FULL_BODY_CLOTHING:
                    if item_lower == full:
                        desc = f'{color_prefix} {item}' if color_prefix else item
                        found_full.append(desc)
                        self.remove_tag(tag)
                        matched = True
                        break

            if not matched:
                for foot in FOOTWEAR:
                    if item_lower == foot or item_lower == f'{foot}s':
                        desc = f'{color_prefix} {item}' if color_prefix else item
                        found_footwear.append(desc)
                        self.remove_tag(tag)
                        matched = True
                        break

            if not matched:
                for acc in ACCESSORIES:
                    if item_lower == acc or item_lower == f'{acc}s':
                        desc = f'{color_prefix} {item}' if color_prefix else item
                        found_accessories.append(desc)
                        self.remove_tag(tag)
                        matched = True
                        break

        # Build clothing phrase
        clothing_items = []
        if found_full:
            clothing_items.extend(found_full)
        else:
            clothing_items.extend(found_tops)
            clothing_items.extend(found_bottoms)

        if clothing_items:
            if len(clothing_items) == 1:
                self.phrases['clothing'].append(f'wearing a {clothing_items[0]}')
            else:
                items_str = ', '.join(clothing_items[:-1]) + ' and ' + clothing_items[-1]
                self.phrases['clothing'].append(f'wearing {items_str}')

        if found_footwear:
            if 'barefoot' in [f.lower() for f in found_footwear]:
                self.phrases['clothing'].append('barefoot')
            else:
                self.phrases['clothing'].append(f'wearing {" and ".join(found_footwear)}')

        if found_accessories:
            self.phrases['clothing'].append(f'with {", ".join(found_accessories)}')

        # Handle special clothing states
        for tag in list(self.remaining_tags):
            if tag == 'bottomless':
                self.phrases['clothing'].append('not wearing bottoms')
                self.remove_tag(tag)
            elif tag == 'topless':
                self.phrases['clothing'].append('topless')
                self.remove_tag(tag)
            elif tag == 'nude' or tag == 'naked':
                self.phrases['clothing'].append('nude')
                self.remove_tag(tag)

    def extract_actions(self):
        """Extract actions."""
        for tag in list(self.remaining_tags):
            if tag in ACTIONS:
                phrase = ACTIONS[tag]
                if phrase:
                    self.phrases['action'].append(phrase)
                self.remove_tag(tag)

    def extract_body_visible(self):
        """Extract visible body parts."""
        for tag in list(self.remaining_tags):
            if tag in BODY_PARTS_VISIBLE:
                phrase = BODY_PARTS_VISIBLE[tag]
                if phrase:
                    self.phrases['body_visible'].append(phrase)
                self.remove_tag(tag)

    def extract_composition(self):
        """Extract photo composition tags."""
        for tag in list(self.remaining_tags):
            if tag in COMPOSITION:
                phrase = COMPOSITION[tag]
                if phrase:
                    self.phrases['composition'].append(phrase)
                self.remove_tag(tag)

            if tag in BACKGROUND:
                phrase = BACKGROUND[tag]
                if phrase:
                    self.phrases['composition'].append(phrase)
                self.remove_tag(tag)

    def process_remaining(self):
        """Process remaining tags through CSV conversions."""
        for tag in list(self.remaining_tags):
            if tag in SKIP_TAGS:
                self.remove_tag(tag)
                continue

            if tag in self.conversions:
                self.phrases['other'].append(self.conversions[tag])
                self.remove_tag(tag)
            elif tag.strip():
                # Keep unknown tags as-is
                self.phrases['other'].append(tag)
                self.remove_tag(tag)

    def build_caption(self) -> str:
        """Build the final natural language caption."""
        parts = []

        # Start with "a woman"
        opener = 'a woman'

        # Add character info right after "a woman" (e.g., "a woman as the character X from Y")
        if self.phrases['character']:
            opener += ' as ' + ', '.join(self.phrases['character'])

        # Add appearance (hair, eyes, body) - flows directly after "a woman"
        if self.phrases['appearance']:
            opener += ' with ' + ', '.join(self.phrases['appearance'])

        # Add ethnicity
        if self.phrases['ethnicity']:
            ethnicities = ' and '.join(self.phrases['ethnicity'])
            opener += f' who is {ethnicities}'

        parts.append(opener)

        # Add pose
        if self.phrases['pose']:
            parts.append(f'she is {self.phrases["pose"][0]}')

        # Add location
        if self.phrases['location']:
            parts.append(f'she is {self.phrases["location"][0]}')

        # Add clothing
        parts.extend(self.phrases['clothing'])

        # Add expression
        if self.phrases['expression']:
            parts.append(f'she is {", ".join(self.phrases["expression"])}')

        # Add actions
        parts.extend([f'she is {a}' for a in self.phrases['action']])

        # Add visible body parts
        parts.extend(self.phrases['body_visible'])

        # Add composition
        parts.extend(self.phrases['composition'])

        # Add other converted tags
        parts.extend(self.phrases['other'])

        return ', '.join(parts)

    def process(self) -> str:
        """Run all extraction steps and build caption."""
        self.extract_character()  # Extract character tags first
        self.extract_hair()
        self.extract_eyes()
        self.extract_body()
        self.extract_ethnicity()
        self.extract_location()
        self.extract_pose()
        self.extract_gaze()
        self.extract_expression()
        self.extract_clothing()
        self.extract_actions()
        self.extract_body_visible()
        self.extract_composition()
        self.process_remaining()
        return self.build_caption()


def process_txt_file(txt_path: Path, conversions: dict[str, str]) -> str:
    """Process a single txt file and return the converted content."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    tags = [t.strip() for t in content.split(',') if t.strip()]
    processor = TagProcessor(tags, conversions)
    return processor.process()


def backup_and_replace(folder: Path, conversions: dict[str, str], show_progress: bool = False) -> tuple[int, int]:
    """Process all .txt files in a folder."""
    original_folder = folder / 'original'
    processed = 0
    skipped = 0

    txt_files = list(folder.glob('*.txt'))
    if not txt_files:
        return 0, 0

    original_folder.mkdir(exist_ok=True)
    total = len(txt_files)

    for i, txt_path in enumerate(txt_files, 1):
        if show_progress:
            print_progress(i, total, f"Converting tags in {folder.name}")

        backup_path = original_folder / txt_path.name

        if backup_path.exists():
            skipped += 1
            continue

        shutil.copy2(txt_path, backup_path)
        new_content = process_txt_file(txt_path, conversions)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        processed += 1

    # Clear progress line in terminal mode
    if show_progress and sys.stdout.isatty() and total > 0:
        print()

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Convert Danbooru tags to natural language captions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python replace_booru_tags.py --input ./dataset
  python replace_booru_tags.py --input ./dataset --conversions ./my_tags.csv
  python replace_booru_tags.py  # Process all folders in script directory
        """
    )
    parser.add_argument('--input', '-i', type=Path, default=None,
                        help='Input folder to process (default: all folders in script directory)')
    parser.add_argument('--conversions', '-c', type=Path, default=None,
                        help='Path to tagconversions.csv (default: script_dir/tagconversions.csv)')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    csv_path = args.conversions if args.conversions else script_dir / 'tagconversions.csv'

    conversions = load_tag_conversions(csv_path)
    if csv_path.exists():
        print(f"Loaded {len(conversions)} tag conversions from {csv_path}")
    else:
        print(f"No conversions file found at {csv_path}, using built-in conversions only")

    # Determine folders to process
    if args.input:
        if not args.input.exists():
            print(f"Error: Folder not found: {args.input}")
            return 1
        if not args.input.is_dir():
            print(f"Error: Not a directory: {args.input}")
            return 1
        folders_to_process = [args.input]
    else:
        folders_to_process = [
            item for item in script_dir.iterdir()
            if item.is_dir() and item.name != 'original' and not item.name.startswith('.')
        ]

    total_processed = 0
    total_skipped = 0

    for folder in folders_to_process:
        processed, skipped = backup_and_replace(folder, conversions, show_progress=True)
        if processed > 0 or skipped > 0:
            print(f"{folder.name}: {processed} processed, {skipped} skipped")
            total_processed += processed
            total_skipped += skipped

    print(f"\nTotal: {total_processed} files processed, {total_skipped} files skipped")
    return 0


if __name__ == '__main__':
    sys.exit(main())
