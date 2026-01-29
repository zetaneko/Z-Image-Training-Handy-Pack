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
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path


class ScriptRunner:
    """Handles running scripts in background threads and capturing output."""

    def __init__(self, output_widget: scrolledtext.ScrolledText, run_button: ttk.Button):
        self.output_widget = output_widget
        self.run_button = run_button
        self.process = None

    def run(self, cmd: list[str]):
        """Run a command in a background thread."""
        self.run_button.config(state='disabled')
        self.clear_output()
        self.append_output(f"$ {' '.join(cmd)}\n\n")

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _run_process(self, cmd: list[str]):
        """Execute the process and stream output."""
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in self.process.stdout:
                self.append_output(line)

            self.process.wait()

            if self.process.returncode == 0:
                self.append_output("\n✓ Completed successfully\n")
            else:
                self.append_output(f"\n✗ Exited with code {self.process.returncode}\n")
        except Exception as e:
            self.append_output(f"\n✗ Error: {e}\n")
        finally:
            self.run_button.config(state='normal')

    def append_output(self, text: str):
        """Thread-safe append to output widget."""
        self.output_widget.after(0, self._append_output, text)

    def _append_output(self, text: str):
        self.output_widget.config(state='normal')
        self.output_widget.insert(tk.END, text)
        self.output_widget.see(tk.END)
        self.output_widget.config(state='disabled')

    def clear_output(self):
        """Clear the output widget."""
        self.output_widget.config(state='normal')
        self.output_widget.delete(1.0, tk.END)
        self.output_widget.config(state='disabled')


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

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'add_prefix.py'
        self.output_widget = output_widget

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

        # Run button
        self.run_btn = ttk.Button(self, text="Run", command=self._run)
        self.run_btn.pack(pady=15)

        self.runner = ScriptRunner(output_widget, self.run_btn)

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

        self.runner.run(cmd)


class ScanFixImagesTab(ttk.Frame):
    """Tab for scan_and_fix_images.py script."""

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'scan_and_fix_images.py'
        self.output_widget = output_widget

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

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'replace_booru_tags.py'
        self.output_widget = output_widget

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

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'generate_training_metadata.py'
        self.output_widget = output_widget

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

    def __init__(self, parent, script_dir: Path, output_widget: scrolledtext.ScrolledText):
        super().__init__(parent, padding=10)
        self.script_path = script_dir / 'fix-diffsynth-model-output.py'
        self.output_widget = output_widget

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


class App(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Z-Image Training Handy Pack")
        self.geometry("750x650")
        self.minsize(650, 550)

        # Get script directory
        self.script_dir = Path(__file__).parent

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
        self.notebook.add(AddPrefixTab(self.notebook, self.script_dir, self.output_text),
                         text="Add Prefix")
        self.notebook.add(ScanFixImagesTab(self.notebook, self.script_dir, self.output_text),
                         text="Scan & Fix Images")
        self.notebook.add(ReplaceBooruTagsTab(self.notebook, self.script_dir, self.output_text),
                         text="Replace Booru Tags")
        self.notebook.add(GenerateMetadataTab(self.notebook, self.script_dir, self.output_text),
                         text="Generate Metadata")
        self.notebook.add(FixDiffSynthTab(self.notebook, self.script_dir, self.output_text),
                         text="Fix DiffSynth Output")

        # Clear button for output
        ttk.Button(output_frame, text="Clear Output",
                   command=lambda: self._clear_output()).pack(anchor='e', pady=(5, 0))

    def _clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
