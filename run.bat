@echo off
REM Z-Image Training Handy Pack Launcher
REM Creates venv if needed, installs dependencies, and launches GUI

setlocal

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%venv

REM Create venv if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating Python virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate venv
call "%VENV_DIR%\Scripts\activate.bat"

REM Install/upgrade dependencies
echo Checking dependencies...
pip install --quiet --upgrade pip

REM Core dependencies (for dataset tools and GUI)
pip install --quiet pillow torch safetensors

REM Training dependencies (for layer group training)
pip install --quiet einops tqdm psutil accelerate transformers diffusers

REM Install DiffSynth-Studio in editable mode
echo Installing DiffSynth-Studio...
pip install --quiet -e "%SCRIPT_DIR%DiffSynth-Studio-ZImage-LowVRAM"

echo Dependencies installed

REM Launch GUI
echo Launching GUI...
python "%SCRIPT_DIR%python-scripts\gui.py"

endlocal
