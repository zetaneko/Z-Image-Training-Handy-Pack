# Z-Image-Training-Handy-Pack

A collection of scripts, guides and information to help with fine-tuning or LoRA training with Z-Image (DiffSynth-Studio). I've also added a few scripts that will help adapt datasets already structured for most existing toolsets, and also with any prompts that were written Danbooru tag-style to do a quick-first pass conversion without LLM to preserve all intent.

## DiffSynth-Studio Attribution

This repository includes a modified version of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by the ModelScope Community, specifically focusing on Z-Image training and inference capabilities.

**Original Project:** https://github.com/modelscope/DiffSynth-Studio
**License:** Apache License 2.0
**Copyright:** ModelScope Community

**What's Included:**
- Z-Image model training and inference pipelines
- Low-VRAM layer group training implementation for full fine-tuning
- Core DiffSynth components (attention, data loading, model management, VRAM optimization)

**Modifications:**
- Streamlined to focus exclusively on Z-Image workflows
- Added enhanced low-VRAM training via layer group splitting with CPU-offloaded optimizer
- Integrated with dataset preparation utilities in this repository

The original DiffSynth-Studio is a comprehensive diffusion model framework supporting multiple models (FLUX, Qwen-Image, Wan-Video, etc.). This repository contains only the Z-Image-specific components.

For the full DiffSynth-Studio framework, visit the [official repository](https://github.com/modelscope/DiffSynth-Studio).

## Getting Started with Z-Image Training

For official documentation on setting up and running Z-Image training, see the [DiffSynth-Studio Z-Image documentation](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md).

## Quick Start

**Linux/macOS:**
```bash
./run.sh
```

**Windows:**
```batch
run.bat
```

These launcher scripts will:
1. Create a Python virtual environment (if one doesn't exist)
2. Install all required dependencies (Pillow, PyTorch, safetensors)
3. Launch the GUI

## GUI

The GUI provides a tabbed interface with file browsers for each script and displays output in real-time.

To run manually (if you have dependencies installed):
```bash
python python-scripts/gui.py
```

## Python Scripts (CLI)

All scripts are in `python-scripts/` and use consistent CLI arguments. Run with `--help` for details.

### add_prefix.py

When building training datasets, you often collect images from multiple sources. These typically have generic numbered filenames like `0001.png`, `0002.png`, etc. If you try to merge these folders into one training bucket, you'll get filename collisions. This script adds a prefix to all files in a folder, letting you combine datasets while keeping filenames unique.

```bash
python add_prefix.py --input ./cat --prefix cat_
python add_prefix.py --input ./cat --prefix cat_ --dry-run  # Preview only
python add_prefix.py --input ./cat --prefix cat_ --yes      # Skip confirmation
python add_prefix.py  # Interactive mode
```

### scan_and_fix_images.py

Training datasets scraped from various sources often have issues: corrupted/truncated images, inconsistent formats (JPEG, WebP, PNG with alpha channels), and unnecessarily large resolutions that waste VRAM during training. This script normalizes your entire dataset—converting everything to clean RGB PNGs, downscaling oversized images to a sensible training resolution, and quarantining any broken files so they don't crash your training run.

```bash
python scan_and_fix_images.py --input ./Raw/Screencaps
python scan_and_fix_images.py --input ./Raw --output ./Fixed --quarantine ./bad
python scan_and_fix_images.py --input ./Raw --min-size 512  # Custom min dimension (default: 1024)
```

**Requires:** Pillow (`pip install pillow`)

### replace_booru_tags.py

Many existing image datasets (especially anime-style) use Danbooru tag format—comma-separated tags like `1girl, long brown hair, blue eyes, standing, outdoors`. However, Z-Image and other caption-based models work better with natural language descriptions. This script intelligently converts tag-based captions to natural language, combining related tags (e.g., "long brown hair" + "ponytail" becomes "long brown hair in a ponytail") and structuring them into readable sentences. Original files are backed up to an `original/` subfolder.

```bash
python replace_booru_tags.py --input ./dataset
python replace_booru_tags.py --input ./dataset --conversions ./my_tags.csv  # Custom tag mappings
python replace_booru_tags.py  # Process all folders in script directory
```

### generate_training_metadata.py

Most diffusion model training setups expect image/caption pairs as separate files (`image.png` + `image.txt`). DiffSynth-Studio Z-Image training uses a different approach—it reads captions from a single `metadata.csv` file that maps image filenames to their prompts. This script scans your dataset folders and generates that CSV file. It supports delta mode, so you can add new images to your dataset and re-run the script without duplicating existing entries.

```bash
python generate_training_metadata.py --input ./dataset
python generate_training_metadata.py --input ./dataset --output ./metadata.csv
python generate_training_metadata.py  # Process folders in script directory
```

### fix-diffsynth-model-output.py

After fine-tuning with DiffSynth-Studio, the output checkpoint uses a different internal format than standard diffusion models—attention layers are stored unfused (`to_q`, `to_k`, `to_v` separate instead of combined `qkv`), and some layer names use DiffSynth-specific prefixes. This means you can't directly load your fine-tuned model in ComfyUI or other inference tools. This script converts the checkpoint back to standard format by fusing the attention keys and merging your fine-tuned weights with the original base model.

```bash
python fix-diffsynth-model-output.py \
  --original /path/to/zImageBase_base.safetensors \
  --input /path/to/step-2000.safetensors \
  --output /path/to/step-2000-fixed.safetensors
```

**Requires:** PyTorch, safetensors (`pip install torch safetensors`)

## Typical Workflow

1. **Organize datasets** — Use `add_prefix.py` to merge multiple source folders without filename collisions
2. **Fix images** — Run `scan_and_fix_images.py` to normalize formats and remove corrupted files
3. **Convert captions** — Run `replace_booru_tags.py` if your dataset uses Danbooru tags instead of natural language
4. **Generate metadata** — Run `generate_training_metadata.py` to create the CSV file DiffSynth-Studio expects
5. **Train model** — Use your prepared dataset with DiffSynth-Studio Z-Image training
6. **Fix checkpoint** — Run `fix-diffsynth-model-output.py` so you can use your model in ComfyUI
