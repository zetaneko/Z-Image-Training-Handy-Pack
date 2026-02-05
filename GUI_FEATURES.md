# GUI Features and Settings

## Overview

The Z-Image Training Handy Pack GUI has been enhanced with comprehensive features including layer group training support, automatic settings persistence, unified venv management, HuggingFace cache compatibility, process control, and intelligent progress bar handling.

## Recent Improvements

### HuggingFace Cache Compatibility
The model loader now automatically detects and uses existing HuggingFace cache directories:

**Standard HF Cache Structure:**
```
~/.cache/huggingface/hub/
  models--{org}--{model}/
    snapshots/
      {commit_hash}/
        {model files}
```

**How it works:**
- Automatically converts model IDs like `Tongyi-MAI/Z-Image` to `models--Tongyi-MAI--Z-Image`
- Finds the latest snapshot in the cache
- Uses cached files instead of re-downloading
- Falls back to standard download if not found in cache

**Usage:**
```bash
# In GUI, set Model Base Path to:
~/.cache/huggingface/hub

# Or via environment variable:
export DIFFSYNTH_MODEL_BASE_PATH=~/.cache/huggingface/hub
```

The system will automatically find models like:
- `/home/user/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/{hash}/`

### Process Control (Stop Button)
Each tab now includes a **Stop** button to terminate running processes:

- **Available during execution**: Stop button activates when a script starts
- **Clean termination**: Sends SIGTERM to gracefully stop the process
- **Status feedback**: Shows "Process terminated by user" message
- **Auto-disable**: Stop button disables after process completes

**Usage:**
1. Click "Start Training" or "Run"
2. Stop button becomes active
3. Click "Stop Training" or "Stop" to terminate
4. Process stops and buttons reset

### Intelligent Progress Bar Handling
The output console now properly handles progress bars that update on a single line:

**Before:**
- Progress bars created hundreds of duplicate lines
- Output was cluttered and hard to read
- Couldn't see actual progress

**After:**
- Progress bars update on the same line (like in terminal)
- Clean, readable output
- Real-time progress visualization

**Technical details:**
- Detects carriage return (`\r`) characters
- Updates current line instead of appending
- Preserves newlines for actual log messages
- Character-by-character streaming for accuracy

## Key Features

### 1. Layer Group Training Tab

A new "Layer Group Training" tab provides a graphical interface for low-VRAM full fine-tuning:

**Features:**
- **Scrollable interface** - All settings in one comprehensive view
- **Dataset configuration** - Path, metadata, repeat count, max pixels
- **Model settings** - Custom model base path, model ID specifications
- **Layer group tuning** - Configure groups and batch size for VRAM optimization
- **Training parameters** - Learning rate, epochs, save steps, gradient accumulation
- **Output configuration** - Output path, random seed, checkpoint resumption
- **Real-time feedback** - Stream training output directly in the GUI

**VRAM Recommendations (built into UI):**
- 12GB: 6-10 layer groups, 2-4 images per batch, 131072-262144 pixels
- 16GB: 4-6 layer groups, 4-8 images per batch, 262144-524288 pixels
- 24GB: 2-4 layer groups, 8-16 images per batch, 524288-1048576 pixels

### 2. Settings Persistence

All tab settings are automatically saved and restored:

**When settings are saved:**
- Automatically when you click "Run" on any tab
- Manually via File menu → "Save All Settings"
- On application exit (for the current tab)

**Settings location:**
- Linux: `~/.config/z-image-training-handy-pack/settings.json`
- Windows: `%APPDATA%\z-image-training-handy-pack\settings.json`
- macOS: `~/Library/Application Support/z-image-training-handy-pack/settings.json`

**What's saved:**
- All file paths and folder selections
- All configuration values (learning rates, batch sizes, etc.)
- Checkbox states (dry run, continue training, etc.)
- Text field contents (prefixes, model paths, etc.)

### 3. Unified Virtual Environment

**Single venv for everything:**
- Dataset preparation scripts use the project venv
- Layer group training uses the project venv
- GUI runs from the project venv
- All dependencies managed in one place

**Auto-installation:**
- `run.sh` (Linux/macOS) and `run.bat` (Windows) automatically:
  1. Create venv if it doesn't exist
  2. Install/upgrade pip
  3. Install core dependencies (Pillow, PyTorch, safetensors)
  4. Install training dependencies (einops, tqdm, psutil, accelerate, etc.)
  5. Install DiffSynth-Studio in editable mode
  6. Launch the GUI

**Dependencies:**
- **Core:** pillow, torch, safetensors
- **Training:** einops, tqdm, psutil, accelerate, transformers, diffusers
- **DiffSynth:** Installed from `./DiffSynth-Studio-ZImage-LowVRAM/` in editable mode

### 4. Menu System

**File Menu:**
- "Save All Settings" - Manually save current tab settings
- "Exit" - Close the application

**Help Menu:**
- "About" - Information about the application
- "Settings Location" - Shows where settings.json is stored

## Using the Layer Group Training Tab

### Quick Start

1. **Launch the GUI:**
   ```bash
   ./run.sh  # Linux/macOS
   run.bat   # Windows
   ```

2. **Select the "Layer Group Training" tab**

3. **Configure your training:**
   - **Dataset Folder**: Path to your training images
   - **Metadata CSV**: Path to metadata.csv (or leave empty for auto-detection)
   - **Model Base Path**: Optional custom location for model downloads
   - **Layer Groups**: Higher = less VRAM, more swaps (default: 6)
   - **Images Per Batch**: Higher = fewer swaps, more memory (default: 128)

4. **Click "Start Training"**

Settings are automatically saved when you click Run, so next time you open the GUI, all your paths and configuration will be restored.

### Continuing Training

To resume from a checkpoint:
1. Check "Continue from checkpoint" option
2. Ensure Output Path points to the directory with your checkpoint
3. Click "Start Training"

The script will automatically find and resume from `training_state_latest.pt` in the output directory.

### Model Storage Configuration

You can specify a custom model base path in three ways:

1. **Via GUI** (Layer Group Training tab):
   - Set "Model Base Path" field
   - Example: `/data/ai-models` or `$HOME/.cache/huggingface/hub`

2. **Via Environment Variable:**
   ```bash
   export DIFFSYNTH_MODEL_BASE_PATH="/data/ai-models"
   ./run.sh
   ```

3. **Default:**
   - Models download to `./models/` in project directory

**Path Resolution:**

The system intelligently handles both path formats:

- **HuggingFace cache format**: `~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/`
- **Simple format**: `{base_path}/{model-id}/` (e.g., `./models/Tongyi-MAI/Z-Image/`)

When you set the base path to a HuggingFace cache directory, the system:
1. Detects the HF cache structure
2. Resolves model IDs to the snapshot directory
3. Uses cached files automatically
4. Only downloads if files aren't found

**Example:**
```
Model Base Path: ~/.cache/huggingface/hub
Model ID: Tongyi-MAI/Z-Image

Resolved to: ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb.../
```

## Settings JSON Structure

The `settings.json` file stores all configuration per tab:

```json
{
  "add_prefix": {
    "input_path": "/path/to/dataset",
    "prefix": "cat_",
    "dry_run": false,
    "yes": true
  },
  "layer_group_training": {
    "dataset_path": "/path/to/dataset",
    "metadata_path": "/path/to/metadata.csv",
    "dataset_repeat": 25,
    "max_pixels": 262144,
    "model_base_path": "/data/models",
    "model_paths": "Tongyi-MAI/Z-Image:transformer/*.safetensors,...",
    "num_groups": 6,
    "images_per_batch": 128,
    "learning_rate": "1e-4",
    "num_epochs": 1,
    "save_steps": 10,
    "output_path": "./models/train/Z-Image_layer_groups",
    "seed": 42,
    "continue_training": false
  }
}
```

## Troubleshooting

### Settings not saving
- Check that the config directory exists and is writable
- Use Help → "Settings Location" to verify the path
- Look for error messages in the terminal/console

### Dependencies missing
- Re-run `./run.sh` or `run.bat` to reinstall
- Manually activate venv and run: `pip install -e ./DiffSynth-Studio-ZImage-LowVRAM`

### Training script not found
- Ensure `DiffSynth-Studio-ZImage-LowVRAM/examples/z_image/model_training/train_layer_groups.py` exists
- Check that you're running from the project root directory

### Model download location
- Set "Model Base Path" in GUI or use `DIFFSYNTH_MODEL_BASE_PATH` environment variable
- Default location: `./models/{model-id}/`

## Tips

1. **Save frequently**: Use File → "Save All Settings" before closing if you want to ensure settings are persisted

2. **Test with small values**: Start with low epochs (1), small batch size, and frequent saves when testing

3. **Monitor VRAM**: Watch GPU memory usage and adjust layer groups/batch size accordingly

4. **Backup settings**: Copy `settings.json` to preserve your configuration

5. **Share configurations**: Settings file is human-readable JSON - easy to share with others
