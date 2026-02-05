#!/usr/bin/env python3
"""
GUI for Z-Image Training Handy Pack scripts.
Provides a tabbed interface to run each utility script with file browsers and output display.

Usage:
  python gui.py
"""

import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import json
import os
from typing import Dict, Any


def get_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA', Path.home()))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:  # Linux and others
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))

    config_dir = base / 'z-image-training-handy-pack'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


class SettingsManager:
    """Manages persistent settings storage."""

    def __init__(self):
        self.config_dir = get_config_dir()
        self.settings_file = self.config_dir / 'settings.json'
        self.settings = self.load()

    def load(self) -> Dict[str, Any]:
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load settings: {e}")
        return {}

    def save(self, settings: Dict[str, Any]):
        """Save settings to file."""
        try:
            self.settings = settings
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    def get_tab_settings(self, tab_name: str) -> Dict[str, Any]:
        """Get settings for a specific tab."""
        return self.settings.get(tab_name, {})

    def set_tab_settings(self, tab_name: str, settings: Dict[str, Any]):
        """Set settings for a specific tab."""
        self.settings[tab_name] = settings
        self.save(self.settings)


class ScriptRunner:
    """Handles running scripts in background threads and capturing output."""

    def __init__(self, output_widget: scrolledtext.ScrolledText, run_button: ttk.Button, stop_button: ttk.Button = None):
        self.output_widget = output_widget
        self.run_button = run_button
        self.stop_button = stop_button
        self.process = None
        self.is_running = False
        self.current_line_start = None  # Track position for progress bar updates

    def run(self, cmd: list[str]):
        """Run a command in a background thread."""
        self.run_button.config(state='disabled')
        if self.stop_button:
            self.stop_button.config(state='normal')
        self.is_running = True
        self.clear_output()
        self.append_output(f"$ {' '.join(cmd)}\n\n")

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def stop(self):
        """Stop the currently running process."""
        if self.process and self.is_running:
            try:
                self.process.terminate()
                self.append_output("\n✗ Process terminated by user\n")
            except Exception as e:
                self.append_output(f"\n✗ Error stopping process: {e}\n")

    def _run_process(self, cmd: list[str]):
        """Execute the process and stream output."""
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            buffer = ""
            for char in iter(lambda: self.process.stdout.read(1), ''):
                buffer += char

                # Handle carriage return (progress bars)
                if char == '\r':
                    # Progress bar update - replace current line
                    if '\n' in buffer[:-1]:
                        # Buffer has newlines, process them first
                        lines = buffer[:-1].split('\n')
                        for line in lines[:-1]:
                            self.append_output(line + '\n')
                        # Last partial line before \r
                        self.update_current_line(lines[-1])
                        buffer = ""
                    else:
                        # Just update the current line
                        self.update_current_line(buffer[:-1])
                        buffer = ""

                # Handle newline
                elif char == '\n':
                    self.append_output(buffer)
                    buffer = ""
                    self.current_line_start = None

            # Output any remaining buffer
            if buffer:
                self.append_output(buffer)

            self.process.wait()

            if self.process.returncode == 0:
                self.append_output("\n✓ Completed successfully\n")
            elif self.process.returncode == -15:  # SIGTERM
                pass  # Already showed "terminated by user"
            else:
                self.append_output(f"\n✗ Exited with code {self.process.returncode}\n")
        except Exception as e:
            self.append_output(f"\n✗ Error: {e}\n")
        finally:
            self.is_running = False
            self.process = None
            self.current_line_start = None
            self.run_button.config(state='normal')
            if self.stop_button:
                self.stop_button.config(state='disabled')

    def append_output(self, text: str):
        """Thread-safe append to output widget."""
        self.output_widget.after(0, self._append_output, text)

    def _append_output(self, text: str):
        self.output_widget.config(state='normal')
        if self.current_line_start:
            # If we have a tracked line position, we already inserted text there
            # Just update the end marker
            self.current_line_start = None
        self.output_widget.insert(tk.END, text)
        self.output_widget.see(tk.END)
        self.output_widget.config(state='disabled')

    def update_current_line(self, text: str):
        """Thread-safe update of current line (for progress bars)."""
        self.output_widget.after(0, self._update_current_line, text)

    def _update_current_line(self, text: str):
        """Update the current line in place (for progress bar updates)."""
        self.output_widget.config(state='normal')

        if self.current_line_start:
            # Delete the old line content
            self.output_widget.delete(self.current_line_start, tk.END)
        else:
            # Mark the start of this line
            self.current_line_start = self.output_widget.index(tk.END + "-1c linestart")

        # Insert new content
        self.output_widget.insert(tk.END, text)
        self.output_widget.see(tk.END)
        self.output_widget.config(state='disabled')

    def clear_output(self):
        """Clear the output widget."""
        self.output_widget.config(state='normal')
        self.output_widget.delete(1.0, tk.END)
        self.output_widget.config(state='disabled')
        self.current_line_start = None


class PathSelector(ttk.Frame):
    """A widget combining an entry field with a browse button."""

    def __init__(self, parent, label: str, mode: str = 'folder', **kwargs):
        super().__init__(parent, **kwargs)
        self.mode = mode

        ttk.Label(self, text=label, width=20, anchor='e').pack(side='left', padx=(0, 5))

        self.var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.var, width=50)
        self.entry.pack(side='left', fill='x', expand=True, padx=(0, 5))

        ttk.Button(self, text="Browse...", command=self._browse, width=10).pack(side='left')

    def _browse(self):
        if self.mode == 'folder':
            path = filedialog.askdirectory()
        elif self.mode == 'open':
            path = filedialog.askopenfilename()
        elif self.mode == 'save':
            path = filedialog.asksaveasfilename()
        else:
            path = None

        if path:
            self.var.set(path)

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str):
        self.var.set(value)


class AddPrefixTab(ttk.Frame):
    """Tab for add_prefix.py script."""

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText, settings_manager: SettingsManager):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'add_prefix.py'
        self.output_widget = output_widget
        self.settings_manager = settings_manager
        self.tab_name = 'add_prefix'

        # Description
        desc = ttk.Label(self, text="Add a prefix to all image and text filenames in a folder.\n"
                                    "Useful for merging multiple datasets without filename collisions.",
                        wraplength=600, justify='left')
        desc.pack(anchor='w', pady=(0, 15))

        # Input folder
        self.input_path = PathSelector(self, "Input Folder:", mode='folder')
        self.input_path.pack(fill='x', pady=5)

        # Prefix
        prefix_frame = ttk.Frame(self)
        prefix_frame.pack(fill='x', pady=5)
        ttk.Label(prefix_frame, text="Prefix:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.prefix_var = tk.StringVar()
        ttk.Entry(prefix_frame, textvariable=self.prefix_var, width=30).pack(side='left')
        ttk.Label(prefix_frame, text="(e.g., 'cat_')").pack(side='left', padx=(10, 0))

        # Options
        options_frame = ttk.Frame(self)
        options_frame.pack(fill='x', pady=10)
        ttk.Label(options_frame, text="Options:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.dry_run_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Dry run (preview only)", variable=self.dry_run_var).pack(side='left')
        self.yes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Skip confirmation", variable=self.yes_var).pack(side='left', padx=(20, 0))

        # Buttons frame
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15)

        self.run_btn = ttk.Button(btn_frame, text="Run", command=self._run)
        self.run_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        self.runner = ScriptRunner(output_widget, self.run_btn, self.stop_btn)

        # Load saved settings
        self.load_settings()

    def _stop(self):
        """Stop the running script."""
        self.runner.stop()

    def _run(self):
        input_path = self.input_path.get()
        prefix = self.prefix_var.get().strip()

        if not input_path:
            self.runner.clear_output()
            self.runner.append_output("Error: Please select an input folder\n")
            return
        if not prefix:
            self.runner.clear_output()
            self.runner.append_output("Error: Please enter a prefix\n")
            return

        cmd = [sys.executable, str(self.script_path), '--input', input_path, '--prefix', prefix]
        if self.dry_run_var.get():
            cmd.append('--dry-run')
        if self.yes_var.get():
            cmd.append('--yes')

        # Save settings before running
        self.save_settings()
        self.runner.run(cmd)

    def get_settings(self) -> Dict[str, Any]:
        """Get current tab settings."""
        return {
            'input_path': self.input_path.get(),
            'prefix': self.prefix_var.get(),
            'dry_run': self.dry_run_var.get(),
            'yes': self.yes_var.get()
        }

    def set_settings(self, settings: Dict[str, Any]):
        """Set tab settings."""
        self.input_path.set(settings.get('input_path', ''))
        self.prefix_var.set(settings.get('prefix', ''))
        self.dry_run_var.set(settings.get('dry_run', False))
        self.yes_var.set(settings.get('yes', True))

    def save_settings(self):
        """Save current settings."""
        self.settings_manager.set_tab_settings(self.tab_name, self.get_settings())

    def load_settings(self):
        """Load saved settings."""
        settings = self.settings_manager.get_tab_settings(self.tab_name)
        if settings:
            self.set_settings(settings)


class ScanFixImagesTab(ttk.Frame):
    """Tab for scan_and_fix_images.py script."""

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText, settings_manager: SettingsManager):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'scan_and_fix_images.py'
        self.output_widget = output_widget
        self.settings_manager = settings_manager
        self.tab_name = 'scan_fix_images'

        # Description
        desc = ttk.Label(self, text="Validate, normalize, and optimize images for training.\n"
                                    "Converts to RGB PNG, resizes to min dimension, quarantines broken files.",
                        wraplength=600, justify='left')
        desc.pack(anchor='w', pady=(0, 15))

        # Input folder
        self.input_path = PathSelector(self, "Input Folder:", mode='folder')
        self.input_path.pack(fill='x', pady=5)

        # Output folder (optional)
        self.output_path = PathSelector(self, "Output Folder:", mode='folder')
        self.output_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(Leave empty for default: <input_parent>/Fixed)").pack(anchor='e', padx=(0, 80))

        # Quarantine folder (optional)
        self.quarantine_path = PathSelector(self, "Quarantine Folder:", mode='folder')
        self.quarantine_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(Leave empty for default: <input_parent>/bad)").pack(anchor='e', padx=(0, 80))

        # Min size
        size_frame = ttk.Frame(self)
        size_frame.pack(fill='x', pady=5)
        ttk.Label(size_frame, text="Min Dimension:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.min_size_var = tk.IntVar(value=1024)
        ttk.Spinbox(size_frame, from_=256, to=4096, increment=64, textvariable=self.min_size_var, width=10).pack(side='left')
        ttk.Label(size_frame, text="pixels").pack(side='left', padx=(5, 0))

        # Run button
        self.run_btn = ttk.Button(self, text="Run", command=self._run)
        self.run_btn.pack(pady=15)

        self.runner = ScriptRunner(output_widget, self.run_btn)

    def _run(self):
        input_path = self.input_path.get()

        if not input_path:
            self.runner.clear_output()
            self.runner.append_output("Error: Please select an input folder\n")
            return

        cmd = [sys.executable, str(self.script_path), '--input', input_path]

        output_path = self.output_path.get()
        if output_path:
            cmd.extend(['--output', output_path])

        quarantine_path = self.quarantine_path.get()
        if quarantine_path:
            cmd.extend(['--quarantine', quarantine_path])

        cmd.extend(['--min-size', str(self.min_size_var.get())])

        self.runner.run(cmd)


class ReplaceBooruTagsTab(ttk.Frame):
    """Tab for replace_booru_tags.py script."""

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText, settings_manager: SettingsManager):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'replace_booru_tags.py'
        self.output_widget = output_widget
        self.settings_manager = settings_manager
        self.tab_name = 'replace_booru_tags'

        # Description
        desc = ttk.Label(self, text="Convert Danbooru-style tags to natural language captions.\n"
                                    "Intelligently combines tags and backs up originals to 'original/' subfolder.",
                        wraplength=600, justify='left')
        desc.pack(anchor='w', pady=(0, 15))

        # Input folder
        self.input_path = PathSelector(self, "Input Folder:", mode='folder')
        self.input_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(Leave empty to process all folders in script directory)").pack(anchor='e', padx=(0, 80))

        # Conversions CSV (optional)
        self.conversions_path = PathSelector(self, "Conversions CSV:", mode='open')
        self.conversions_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(Optional: custom tag mappings file)").pack(anchor='e', padx=(0, 80))

        # Run button
        self.run_btn = ttk.Button(self, text="Run", command=self._run)
        self.run_btn.pack(pady=15)

        self.runner = ScriptRunner(output_widget, self.run_btn)

    def _run(self):
        cmd = [sys.executable, str(self.script_path)]

        input_path = self.input_path.get()
        if input_path:
            cmd.extend(['--input', input_path])

        conversions_path = self.conversions_path.get()
        if conversions_path:
            cmd.extend(['--conversions', conversions_path])

        self.runner.run(cmd)


class GenerateMetadataTab(ttk.Frame):
    """Tab for generate_training_metadata.py script."""

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText, settings_manager: SettingsManager):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'generate_training_metadata.py'
        self.output_widget = output_widget
        self.settings_manager = settings_manager
        self.tab_name = 'generate_metadata'

        # Description
        desc = ttk.Label(self, text="Generate metadata.csv for DiffSynth-Studio Z-Image training.\n"
                                    "Maps image filenames to prompts. Supports delta mode (won't duplicate entries).",
                        wraplength=600, justify='left')
        desc.pack(anchor='w', pady=(0, 15))

        # Input folder
        self.input_path = PathSelector(self, "Input Folder:", mode='folder')
        self.input_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(Leave empty to use script directory)").pack(anchor='e', padx=(0, 80))

        # Output CSV (optional)
        self.output_path = PathSelector(self, "Output CSV:", mode='save')
        self.output_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(Leave empty for default: <input>/metadata.csv)").pack(anchor='e', padx=(0, 80))

        # Run button
        self.run_btn = ttk.Button(self, text="Run", command=self._run)
        self.run_btn.pack(pady=15)

        self.runner = ScriptRunner(output_widget, self.run_btn)

    def _run(self):
        cmd = [sys.executable, str(self.script_path)]

        input_path = self.input_path.get()
        if input_path:
            cmd.extend(['--input', input_path])

        output_path = self.output_path.get()
        if output_path:
            cmd.extend(['--output', output_path])

        self.runner.run(cmd)


class FixDiffSynthTab(ttk.Frame):
    """Tab for fix-diffsynth-model-output.py script."""

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText, settings_manager: SettingsManager):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'fix-diffsynth-model-output.py'
        self.output_widget = output_widget
        self.settings_manager = settings_manager
        self.tab_name = 'fix_diffsynth'

        # Description
        desc = ttk.Label(self, text="Convert DiffSynth-Studio checkpoints to standard format for ComfyUI.\n"
                                    "Fuses attention keys and merges with base model weights.",
                        wraplength=600, justify='left')
        desc.pack(anchor='w', pady=(0, 15))

        # Original model
        self.original_path = PathSelector(self, "Original Base Model:", mode='open')
        self.original_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(e.g., zImageBase_base.safetensors)").pack(anchor='e', padx=(0, 80))

        # Input checkpoint
        self.input_path = PathSelector(self, "Fine-tuned Checkpoint:", mode='open')
        self.input_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(e.g., step-2000.safetensors)").pack(anchor='e', padx=(0, 80))

        # Output file
        self.output_path = PathSelector(self, "Output File:", mode='save')
        self.output_path.pack(fill='x', pady=5)
        ttk.Label(self, text="(e.g., step-2000-fixed.safetensors)").pack(anchor='e', padx=(0, 80))

        # Run button
        self.run_btn = ttk.Button(self, text="Run", command=self._run)
        self.run_btn.pack(pady=15)

        self.runner = ScriptRunner(output_widget, self.run_btn)

    def _run(self):
        original_path = self.original_path.get()
        input_path = self.input_path.get()
        output_path = self.output_path.get()

        if not original_path:
            self.runner.clear_output()
            self.runner.append_output("Error: Please select the original base model\n")
            return
        if not input_path:
            self.runner.clear_output()
            self.runner.append_output("Error: Please select the fine-tuned checkpoint\n")
            return
        if not output_path:
            self.runner.clear_output()
            self.runner.append_output("Error: Please specify an output file\n")
            return

        cmd = [sys.executable, str(self.script_path),
               '--original', original_path,
               '--input', input_path,
               '--output', output_path]

        self.runner.run(cmd)


class LayerGroupTrainingTab(ttk.Frame):
    """Tab for low-VRAM layer group training."""

    def __init__(self, parent, project_root: Path, output_widget: scrolledtext.ScrolledText, settings_manager: SettingsManager):
        super().__init__(parent, padding=10)
        self.project_root = project_root
        self.script_path = project_root / 'DiffSynth-Studio-ZImage-LowVRAM' / 'examples' / 'z_image' / 'model_training' / 'train_layer_groups.py'
        self.output_widget = output_widget
        self.settings_manager = settings_manager
        self.tab_name = 'layer_group_training'

        # Create canvas with scrollbar for scrollable content
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Description
        desc = ttk.Label(scrollable_frame, text="Low-VRAM full fine-tuning using layer group offloading.\n"
                                    "Splits the transformer into groups, processes batches, and uses CPU-offloaded optimizer.\n"
                                    "Suitable for 12-16GB VRAM GPUs.",
                        wraplength=600, justify='left', font=('TkDefaultFont', 9, 'bold'))
        desc.pack(anchor='w', pady=(0, 15))

        # Dataset settings
        dataset_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Settings", padding=10)
        dataset_frame.pack(fill='x', pady=5)

        self.dataset_path = PathSelector(dataset_frame, "Dataset Folder:", mode='folder')
        self.dataset_path.pack(fill='x', pady=3)

        self.metadata_path = PathSelector(dataset_frame, "Metadata CSV:", mode='open')
        self.metadata_path.pack(fill='x', pady=3)
        ttk.Label(dataset_frame, text="(Leave empty for <dataset>/metadata.csv)", font=('TkDefaultFont', 8)).pack(anchor='e')

        repeat_frame = ttk.Frame(dataset_frame)
        repeat_frame.pack(fill='x', pady=3)
        ttk.Label(repeat_frame, text="Dataset Repeat:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.dataset_repeat_var = tk.IntVar(value=25)
        ttk.Spinbox(repeat_frame, from_=1, to=1000, textvariable=self.dataset_repeat_var, width=10).pack(side='left')

        pixels_frame = ttk.Frame(dataset_frame)
        pixels_frame.pack(fill='x', pady=3)
        ttk.Label(pixels_frame, text="Max Pixels:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.max_pixels_var = tk.IntVar(value=262144)
        ttk.Combobox(pixels_frame, textvariable=self.max_pixels_var, values=[131072, 262144, 524288, 1048576], width=15).pack(side='left')
        ttk.Label(pixels_frame, text="(131072=362px, 262144=512px, 524288=724px)", font=('TkDefaultFont', 8)).pack(side='left', padx=(10, 0))

        # Model settings
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Settings", padding=10)
        model_frame.pack(fill='x', pady=5)

        self.model_base_path = PathSelector(model_frame, "Model Base Path:", mode='folder')
        self.model_base_path.pack(fill='x', pady=3)
        ttk.Label(model_frame, text="(Optional: custom model storage location)", font=('TkDefaultFont', 8)).pack(anchor='e')

        model_paths_frame = ttk.Frame(model_frame)
        model_paths_frame.pack(fill='x', pady=3)
        ttk.Label(model_paths_frame, text="Model Paths:", width=20, anchor='e').pack(side='left', padx=(0, 5), anchor='n')
        self.model_paths_var = tk.StringVar(value="Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image:text_encoder/*.safetensors,Tongyi-MAI/Z-Image:vae/diffusion_pytorch_model.safetensors")
        model_paths_text = tk.Text(model_paths_frame, height=3, width=50, wrap='word')
        model_paths_text.insert('1.0', self.model_paths_var.get())
        model_paths_text.pack(side='left', fill='x', expand=True)
        self.model_paths_text = model_paths_text

        # Layer group settings
        layergroup_frame = ttk.LabelFrame(scrollable_frame, text="Layer Group Settings (VRAM Tuning)", padding=10)
        layergroup_frame.pack(fill='x', pady=5)

        groups_frame = ttk.Frame(layergroup_frame)
        groups_frame.pack(fill='x', pady=3)
        ttk.Label(groups_frame, text="Num Layer Groups:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.num_groups_var = tk.IntVar(value=6)
        ttk.Spinbox(groups_frame, from_=2, to=15, textvariable=self.num_groups_var, width=10).pack(side='left')
        ttk.Label(groups_frame, text="(More = less VRAM, more swaps)", font=('TkDefaultFont', 8)).pack(side='left', padx=(10, 0))

        batch_frame = ttk.Frame(layergroup_frame)
        batch_frame.pack(fill='x', pady=3)
        ttk.Label(batch_frame, text="Images Per Group Batch:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.images_per_batch_var = tk.IntVar(value=128)
        ttk.Spinbox(batch_frame, from_=1, to=256, textvariable=self.images_per_batch_var, width=10).pack(side='left')
        ttk.Label(batch_frame, text="(Higher = fewer swaps, more memory)", font=('TkDefaultFont', 8)).pack(side='left', padx=(10, 0))

        # Training settings
        training_frame = ttk.LabelFrame(scrollable_frame, text="Training Settings", padding=10)
        training_frame.pack(fill='x', pady=5)

        lr_frame = ttk.Frame(training_frame)
        lr_frame.pack(fill='x', pady=3)
        ttk.Label(lr_frame, text="Learning Rate:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.lr_var = tk.StringVar(value="1e-4")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=15).pack(side='left')

        epochs_frame = ttk.Frame(training_frame)
        epochs_frame.pack(fill='x', pady=3)
        ttk.Label(epochs_frame, text="Num Epochs:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.epochs_var = tk.IntVar(value=1)
        ttk.Spinbox(epochs_frame, from_=1, to=100, textvariable=self.epochs_var, width=10).pack(side='left')

        steps_frame = ttk.Frame(training_frame)
        steps_frame.pack(fill='x', pady=3)
        ttk.Label(steps_frame, text="Save Steps:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.save_steps_var = tk.IntVar(value=10)
        ttk.Spinbox(steps_frame, from_=1, to=10000, increment=10, textvariable=self.save_steps_var, width=10).pack(side='left')

        grad_accum_frame = ttk.Frame(training_frame)
        grad_accum_frame.pack(fill='x', pady=3)
        ttk.Label(grad_accum_frame, text="Gradient Accumulation:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.grad_accum_var = tk.IntVar(value=1)
        ttk.Spinbox(grad_accum_frame, from_=1, to=32, textvariable=self.grad_accum_var, width=10).pack(side='left')

        # Output settings
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Settings", padding=10)
        output_frame.pack(fill='x', pady=5)

        self.output_path = PathSelector(output_frame, "Output Path:", mode='folder')
        self.output_path.pack(fill='x', pady=3)
        self.output_path.set("./models/train/Z-Image_layer_groups")

        seed_frame = ttk.Frame(output_frame)
        seed_frame.pack(fill='x', pady=3)
        ttk.Label(seed_frame, text="Random Seed:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(seed_frame, from_=0, to=999999, textvariable=self.seed_var, width=10).pack(side='left')

        # Options
        options_frame = ttk.Frame(output_frame)
        options_frame.pack(fill='x', pady=5)
        ttk.Label(options_frame, text="Options:", width=20, anchor='e').pack(side='left', padx=(0, 5))
        self.continue_training_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Continue from checkpoint", variable=self.continue_training_var).pack(side='left')

        # Buttons frame
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(pady=15)

        self.run_btn = ttk.Button(btn_frame, text="Start Training", command=self._run, width=15)
        self.run_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Training", command=self._stop, state='disabled', width=15)
        self.stop_btn.pack(side='left', padx=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # Linux scroll down

        self.runner = ScriptRunner(output_widget, self.run_btn, self.stop_btn)

        # Load saved settings
        self.load_settings()

    def _stop(self):
        """Stop the running training."""
        self.runner.stop()

    def _run(self):
        dataset_path = self.dataset_path.get()
        if not dataset_path:
            self.runner.clear_output()
            self.runner.append_output("Error: Please select a dataset folder\n")
            return

        # Build command
        cmd = [sys.executable, str(self.script_path),
               '--dataset_base_path', dataset_path]

        # Metadata path
        metadata_path = self.metadata_path.get()
        if metadata_path:
            cmd.extend(['--dataset_metadata_path', metadata_path])

        # Dataset settings
        cmd.extend([
            '--dataset_repeat', str(self.dataset_repeat_var.get()),
            '--max_pixels', str(self.max_pixels_var.get())
        ])

        # Model paths
        model_paths = self.model_paths_text.get('1.0', 'end').strip()
        if model_paths:
            cmd.extend(['--model_id_with_origin_paths', model_paths])

        # Model base path
        model_base_path = self.model_base_path.get()
        if model_base_path:
            cmd.extend(['--model_base_path', model_base_path])

        # Layer group settings
        cmd.extend([
            '--num_layer_groups', str(self.num_groups_var.get()),
            '--images_per_group_batch', str(self.images_per_batch_var.get()),
            '--trainable_models', 'dit'
        ])

        # Training settings
        cmd.extend([
            '--learning_rate', self.lr_var.get(),
            '--num_epochs', str(self.epochs_var.get()),
            '--save_steps', str(self.save_steps_var.get()),
            '--gradient_accumulation_steps', str(self.grad_accum_var.get())
        ])

        # Output settings
        output_path = self.output_path.get()
        if output_path:
            cmd.extend(['--output_path', output_path])

        cmd.extend([
            '--seed', str(self.seed_var.get()),
            '--remove_prefix_in_ckpt', 'pipe.dit.',
            '--use_gradient_checkpointing',
            '--use_gradient_checkpointing_offload'
        ])

        # Continue training
        if self.continue_training_var.get():
            cmd.append('--continue_training')

        # Save settings before running
        self.save_settings()
        self.runner.run(cmd)

    def get_settings(self) -> Dict[str, Any]:
        """Get current tab settings."""
        return {
            'dataset_path': self.dataset_path.get(),
            'metadata_path': self.metadata_path.get(),
            'dataset_repeat': self.dataset_repeat_var.get(),
            'max_pixels': self.max_pixels_var.get(),
            'model_base_path': self.model_base_path.get(),
            'model_paths': self.model_paths_text.get('1.0', 'end').strip(),
            'num_groups': self.num_groups_var.get(),
            'images_per_batch': self.images_per_batch_var.get(),
            'learning_rate': self.lr_var.get(),
            'num_epochs': self.epochs_var.get(),
            'save_steps': self.save_steps_var.get(),
            'grad_accum': self.grad_accum_var.get(),
            'output_path': self.output_path.get(),
            'seed': self.seed_var.get(),
            'continue_training': self.continue_training_var.get()
        }

    def set_settings(self, settings: Dict[str, Any]):
        """Set tab settings."""
        self.dataset_path.set(settings.get('dataset_path', ''))
        self.metadata_path.set(settings.get('metadata_path', ''))
        self.dataset_repeat_var.set(settings.get('dataset_repeat', 25))
        self.max_pixels_var.set(settings.get('max_pixels', 262144))
        self.model_base_path.set(settings.get('model_base_path', ''))

        model_paths = settings.get('model_paths', 'Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image:text_encoder/*.safetensors,Tongyi-MAI/Z-Image:vae/diffusion_pytorch_model.safetensors')
        self.model_paths_text.delete('1.0', 'end')
        self.model_paths_text.insert('1.0', model_paths)

        self.num_groups_var.set(settings.get('num_groups', 6))
        self.images_per_batch_var.set(settings.get('images_per_batch', 128))
        self.lr_var.set(settings.get('learning_rate', '1e-4'))
        self.epochs_var.set(settings.get('num_epochs', 1))
        self.save_steps_var.set(settings.get('save_steps', 10))
        self.grad_accum_var.set(settings.get('grad_accum', 1))
        self.output_path.set(settings.get('output_path', './models/train/Z-Image_layer_groups'))
        self.seed_var.set(settings.get('seed', 42))
        self.continue_training_var.set(settings.get('continue_training', False))

    def save_settings(self):
        """Save current settings."""
        self.settings_manager.set_tab_settings(self.tab_name, self.get_settings())

    def load_settings(self):
        """Load saved settings."""
        settings = self.settings_manager.get_tab_settings(self.tab_name)
        if settings:
            self.set_settings(settings)


class App(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Z-Image Training Handy Pack")
        self.geometry("800x700")
        self.minsize(700, 600)

        # Get directories
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent

        # Create settings manager
        self.settings_manager = SettingsManager()

        # Create menu bar
        self._create_menu()

        # Create main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Create output console (shared by all tabs)
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding=5)
        output_frame.pack(fill='both', expand=True, pady=(10, 0))

        self.output_text = scrolledtext.ScrolledText(output_frame, height=12, state='disabled',
                                                      font=('Consolas', 9) if sys.platform == 'win32' else ('Monaco', 10))
        self.output_text.pack(fill='both', expand=True)

        # Add tabs
        self.notebook.add(AddPrefixTab(self.notebook, self.script_dir, self.output_text, self.settings_manager),
                         text="Add Prefix")
        self.notebook.add(ScanFixImagesTab(self.notebook, self.script_dir, self.output_text, self.settings_manager),
                         text="Scan & Fix Images")
        self.notebook.add(ReplaceBooruTagsTab(self.notebook, self.script_dir, self.output_text, self.settings_manager),
                         text="Replace Booru Tags")
        self.notebook.add(GenerateMetadataTab(self.notebook, self.script_dir, self.output_text, self.settings_manager),
                         text="Generate Metadata")
        self.notebook.add(FixDiffSynthTab(self.notebook, self.script_dir, self.output_text, self.settings_manager),
                         text="Fix DiffSynth Output")
        self.notebook.add(LayerGroupTrainingTab(self.notebook, self.project_root, self.output_text, self.settings_manager),
                         text="Layer Group Training")

        # Clear button for output
        ttk.Button(output_frame, text="Clear Output",
                   command=lambda: self._clear_output()).pack(anchor='e', pady=(5, 0))

    def _create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save All Settings", command=self._save_all_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Settings Location", command=self._show_settings_location)

    def _save_all_settings(self):
        """Save settings from all tabs."""
        # Get current tab and save its settings
        current_tab_index = self.notebook.index(self.notebook.select())
        current_tab = self.notebook.nametowidget(self.notebook.tabs()[current_tab_index])
        if hasattr(current_tab, 'save_settings'):
            current_tab.save_settings()
        messagebox.showinfo("Settings Saved", "All settings have been saved.")

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About",
                           "Z-Image Training Handy Pack GUI\n\n"
                           "A collection of utilities for preparing datasets\n"
                           "and training with Z-Image (DiffSynth-Studio).\n\n"
                           "Settings are automatically saved when you run scripts.")

    def _show_settings_location(self):
        """Show where settings are stored."""
        config_dir = get_config_dir()
        settings_file = config_dir / 'settings.json'
        messagebox.showinfo("Settings Location",
                           f"Settings are stored at:\n{settings_file}\n\n"
                           f"Config directory:\n{config_dir}")

    def _clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
