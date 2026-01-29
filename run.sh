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
pip install --quiet pillow torch safetensors

# Launch GUI
echo "Launching GUI..."
python "$SCRIPT_DIR/python-scripts/gui.py"
