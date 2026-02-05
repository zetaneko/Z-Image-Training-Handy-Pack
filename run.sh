#!/bin/bash
# Z-Image Training Handy Pack Launcher
# Creates venv if needed, installs dependencies, and launches GUI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install/upgrade dependencies
echo "Checking dependencies..."
pip install --quiet --upgrade pip

# Core dependencies (for dataset tools and GUI)
pip install --quiet pillow torch safetensors

# Training dependencies (for layer group training)
pip install --quiet einops tqdm psutil accelerate transformers diffusers

# Install DiffSynth-Studio in editable mode (if not already installed)
if ! python -c "import diffsynth" 2>/dev/null; then
    echo "Installing DiffSynth-Studio..."
    pip install --quiet -e "$SCRIPT_DIR/DiffSynth-Studio-ZImage-LowVRAM"
fi

echo "✓ Dependencies installed"

# Launch GUI
echo "Launching GUI..."
python "$SCRIPT_DIR/python-scripts/gui.py"
