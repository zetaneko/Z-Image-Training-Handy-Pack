#!/usr/bin/env python3
"""
Caption Inspector Tab for Z-Image Training Handy Pack GUI.
Browse, search, edit, and batch-modify image captions at scale.
Handles datasets of 30,000+ images with efficient lazy loading.
"""

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}


@dataclass
class ImageEntry:
    image_path: Path
    caption_path: Optional[Path]
    caption_text: str
    caption_type: str  # 'autocaption' | 'original' | 'none'
    relative_path: str
    item_id: str = ''


def parse_search_query(query: str):
    """Parse search query into required and excluded terms.

    Supports:
      word          - caption must contain 'word'
      "a phrase"    - caption must contain 'a phrase' (exact)
      -word         - caption must NOT contain 'word'
      -"a phrase"   - caption must NOT contain 'a phrase'

    Returns (required: list[str], excluded: list[str]), all lowercased.
    """
    required = []
    excluded = []
    # Match: optional minus, then either "quoted phrase" or bare word
    for m in re.finditer(r'(-?)(?:"([^"]*)"|([\S]+))', query):
        negate = m.group(1) == '-'
        term = (m.group(2) if m.group(2) is not None else m.group(3)).lower()
        if not term:
            continue
        if negate:
            excluded.append(term)
        else:
            required.append(term)
    return required, excluded


def caption_matches_query(caption_lower: str, required: list, excluded: list) -> bool:
    """Check if a lowercased caption matches all required terms and no excluded terms."""
    return (all(term in caption_lower for term in required) and
            not any(term in caption_lower for term in excluded))


def find_caption_file(image_path: Path):
    """Find caption file for an image. Priority: .autocaption.txt > .txt"""
    autocaption = image_path.with_suffix('.autocaption.txt')
    if autocaption.exists():
        return autocaption, 'autocaption'
    txt = image_path.with_suffix('.txt')
    if txt.exists():
        return txt, 'original'
    return None, 'none'


def read_caption(caption_path: Optional[Path]) -> str:
    if caption_path is None:
        return ''
    try:
        return caption_path.read_text(encoding='utf-8').strip()
    except Exception:
        return ''


class CaptionInspectorTab(ttk.Frame):
    """Tab for browsing, searching, editing, and batch-modifying image captions."""

    def __init__(self, parent, script_dir: Path, output_widget, settings_manager):
        super().__init__(parent, padding=5)
        self.script_dir = script_dir
        self.output_widget = output_widget
        self.settings_manager = settings_manager
        self.tab_name = 'caption_inspector'

        # Data
        self.entries: Dict[str, ImageEntry] = {}  # item_id -> ImageEntry
        self.all_item_ids: List[str] = []  # all items in insertion order
        self.detached_ids: Set[str] = set()  # currently hidden by filter
        self.root_folder: Optional[Path] = None
        self.is_loading = False
        self.cancel_loading = False
        self.current_preview_image = None  # keep reference to prevent GC

        self._build_ui()
        self.load_settings()

    def _build_ui(self):
        # === Top bar: folder selection ===
        top_frame = ttk.Frame(self)
        top_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(top_frame, text="Folder:").pack(side='left', padx=(0, 5))
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(top_frame, textvariable=self.folder_var)
        self.folder_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(top_frame, text="Browse...", command=self._browse_folder, width=10).pack(side='left', padx=(0, 5))
        self.load_btn = ttk.Button(top_frame, text="Load", command=self._load_folder, width=8)
        self.load_btn.pack(side='left', padx=(0, 5))
        self.stop_load_btn = ttk.Button(top_frame, text="Stop", command=self._stop_loading, width=8, state='disabled')
        self.stop_load_btn.pack(side='left')

        # === Search & Bulk Operations ===
        ops_frame = ttk.LabelFrame(self, text="Search & Bulk Operations", padding=5)
        ops_frame.pack(fill='x', pady=(0, 5))

        # Search row
        search_row = ttk.Frame(ops_frame)
        search_row.pack(fill='x', pady=(0, 4))

        ttk.Label(search_row, text="Search:").pack(side='left', padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_row, textvariable=self.search_var, width=30)
        search_entry.pack(side='left', padx=(0, 5))
        search_entry.bind('<Return>', lambda e: self._apply_filter())
        ttk.Button(search_row, text="Filter", command=self._apply_filter, width=8).pack(side='left', padx=(0, 3))
        ttk.Button(search_row, text="Clear", command=self._clear_filter, width=8).pack(side='left', padx=(0, 10))
        self.status_var = tk.StringVar(value="No folder loaded")
        ttk.Label(search_row, textvariable=self.status_var).pack(side='right')

        # Search syntax hint
        hint_row = ttk.Frame(ops_frame)
        hint_row.pack(fill='x', pady=(0, 4))
        ttk.Label(hint_row, text='words = AND | "exact phrase" | -exclude | -"exclude phrase"',
                  font=('TkDefaultFont', 8), foreground='gray').pack(side='left', padx=(52, 0))

        # Find & Replace row
        replace_row = ttk.Frame(ops_frame)
        replace_row.pack(fill='x', pady=(0, 4))

        ttk.Label(replace_row, text="Find:").pack(side='left', padx=(0, 5))
        self.find_var = tk.StringVar()
        ttk.Entry(replace_row, textvariable=self.find_var, width=20).pack(side='left', padx=(0, 5))
        ttk.Label(replace_row, text="Replace:").pack(side='left', padx=(0, 5))
        self.replace_var = tk.StringVar()
        ttk.Entry(replace_row, textvariable=self.replace_var, width=20).pack(side='left', padx=(0, 5))
        ttk.Button(replace_row, text="Replace", command=self._do_replace, width=8).pack(side='left')

        # Prepend row
        prepend_row = ttk.Frame(ops_frame)
        prepend_row.pack(fill='x', pady=(0, 4))

        ttk.Label(prepend_row, text="Prepend:").pack(side='left', padx=(0, 5))
        self.prepend_var = tk.StringVar()
        ttk.Entry(prepend_row, textvariable=self.prepend_var, width=30).pack(side='left', padx=(0, 5))
        ttk.Button(prepend_row, text="Prepend", command=self._do_prepend, width=8).pack(side='left')

        # Apply-to selector (shared by replace and prepend)
        apply_frame = ttk.Frame(ops_frame)
        apply_frame.pack(fill='x')
        ttk.Label(apply_frame, text="Apply to:").pack(side='left', padx=(0, 5))
        self.apply_to_var = tk.StringVar(value="Selected")
        apply_combo = ttk.Combobox(apply_frame, textvariable=self.apply_to_var,
                                   values=["Selected", "All Loaded", "Filtered"],
                                   state='readonly', width=14)
        apply_combo.pack(side='left')

        # === Action buttons ===
        action_frame = ttk.Frame(self)
        action_frame.pack(fill='x', pady=(0, 5))

        ttk.Button(action_frame, text="Select All", command=self._select_all).pack(side='left', padx=(0, 5))
        ttk.Button(action_frame, text="Deselect All", command=self._deselect_all).pack(side='left', padx=(0, 5))
        ttk.Button(action_frame, text="Delete Selected", command=self._delete_selected).pack(side='left', padx=(0, 10))
        self.selection_var = tk.StringVar(value="")
        ttk.Label(action_frame, textvariable=self.selection_var).pack(side='right')

        # === Main content: PanedWindow with list + preview ===
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True)

        # Left pane: Treeview
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        # Treeview with scrollbar
        tree_container = ttk.Frame(left_frame)
        tree_container.pack(fill='both', expand=True)

        self.tree = ttk.Treeview(tree_container, columns=('name', 'folder', 'type', 'caption'),
                                 show='headings', selectmode='extended')
        self.tree.heading('name', text='Name', command=lambda: self._sort_column('name'))
        self.tree.heading('folder', text='Folder', command=lambda: self._sort_column('folder'))
        self.tree.heading('type', text='Type', command=lambda: self._sort_column('type'))
        self.tree.heading('caption', text='Caption', command=lambda: self._sort_column('caption'))

        self.tree.column('name', width=160, minwidth=100)
        self.tree.column('folder', width=100, minwidth=60)
        self.tree.column('type', width=70, minwidth=50)
        self.tree.column('caption', width=250, minwidth=100)

        vsb = ttk.Scrollbar(tree_container, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        self.tree.bind('<Delete>', lambda e: self._delete_selected())

        # Right pane: Preview + Caption Editor
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)

        # Image preview
        preview_label_frame = ttk.LabelFrame(right_frame, text="Preview", padding=5)
        preview_label_frame.pack(fill='both', expand=True, pady=(0, 5))

        self.preview_canvas = tk.Canvas(preview_label_frame, bg='#2b2b2b', highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True)
        self.preview_canvas.bind('<Configure>', self._on_preview_resize)
        self._pending_preview_path = None

        # Caption editor
        caption_frame = ttk.LabelFrame(right_frame, text="Caption", padding=5)
        caption_frame.pack(fill='x', pady=(0, 5))

        self.caption_source_var = tk.StringVar(value="")
        ttk.Label(caption_frame, textvariable=self.caption_source_var,
                  font=('TkDefaultFont', 8)).pack(anchor='w')

        self.caption_text = tk.Text(caption_frame, height=6, wrap='word')
        self.caption_text.pack(fill='x', expand=True, pady=(2, 5))

        btn_row = ttk.Frame(caption_frame)
        btn_row.pack(fill='x')
        self.save_btn = ttk.Button(btn_row, text="Save Caption", command=self._save_caption, state='disabled')
        self.save_btn.pack(side='left', padx=(0, 5))
        self.revert_btn = ttk.Button(btn_row, text="Revert", command=self._revert_caption, state='disabled')
        self.revert_btn.pack(side='left')
        self.multi_label = ttk.Label(btn_row, text="")
        self.multi_label.pack(side='right')

        # Keyboard bindings
        self.tree.bind('<Control-a>', lambda e: self._select_all())

        # Sort state
        self._sort_reverse = {}

        # Right-click context menu
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Copy Caption", command=self._copy_caption)
        self.context_menu.add_command(label="Open File Location", command=self._open_file_location)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Delete", command=self._delete_selected)
        self.tree.bind('<Button-3>', self._show_context_menu)

    # ── Directory Loading ──────────────────────────────────────────

    def _browse_folder(self):
        from tkinter import filedialog
        path = filedialog.askdirectory()
        if path:
            self.folder_var.set(path)

    def _load_folder(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Please select a folder first.")
            return
        folder_path = Path(folder)
        if not folder_path.is_dir():
            messagebox.showerror("Invalid Folder", f"'{folder}' is not a valid directory.")
            return

        self.root_folder = folder_path
        self.is_loading = True
        self.cancel_loading = False
        self.load_btn.config(state='disabled')
        self.stop_load_btn.config(state='normal')

        # Clear existing data
        self.tree.delete(*self.tree.get_children())
        for iid in self.detached_ids:
            try:
                self.tree.delete(iid)
            except Exception:
                pass
        self.entries.clear()
        self.all_item_ids.clear()
        self.detached_ids.clear()
        self._clear_preview()
        self.status_var.set("Scanning...")

        thread = threading.Thread(target=self._scan_directory, args=(folder_path,), daemon=True)
        thread.start()

    def _stop_loading(self):
        self.cancel_loading = True

    def _scan_directory(self, folder: Path):
        """Scan directory for images and their captions. Runs in background thread."""
        batch = []
        count = 0

        for root, dirs, files in os.walk(folder):
            if self.cancel_loading:
                break
            root_path = Path(root)
            for fname in files:
                if self.cancel_loading:
                    break
                fpath = root_path / fname
                if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                caption_path, caption_type = find_caption_file(fpath)
                caption_text = read_caption(caption_path)

                try:
                    rel = fpath.parent.relative_to(folder)
                    rel_str = str(rel) if str(rel) != '.' else ''
                except ValueError:
                    rel_str = ''

                entry = ImageEntry(
                    image_path=fpath,
                    caption_path=caption_path,
                    caption_text=caption_text,
                    caption_type=caption_type,
                    relative_path=rel_str,
                )
                batch.append(entry)
                count += 1

                if len(batch) >= 500:
                    self._insert_batch(batch.copy(), count)
                    batch.clear()

        if batch and not self.cancel_loading:
            self._insert_batch(batch.copy(), count)

        self.after(0, self._loading_finished, count)

    def _insert_batch(self, batch: List[ImageEntry], total_so_far: int):
        """Schedule batch insertion on the main thread."""
        self.after(0, self._do_insert_batch, batch, total_so_far)

    def _do_insert_batch(self, batch: List[ImageEntry], total_so_far: int):
        for entry in batch:
            caption_snippet = entry.caption_text[:100].replace('\n', ' ') if entry.caption_text else '(no caption)'
            type_label = {'autocaption': 'auto', 'original': 'txt', 'none': '-'}[entry.caption_type]
            iid = self.tree.insert('', 'end', values=(
                entry.image_path.name,
                entry.relative_path,
                type_label,
                caption_snippet,
            ))
            entry.item_id = iid
            self.entries[iid] = entry
            self.all_item_ids.append(iid)
        self.status_var.set(f"Loading... {total_so_far:,} images found")

    def _loading_finished(self, total: int):
        self.is_loading = False
        self.load_btn.config(state='normal')
        self.stop_load_btn.config(state='disabled')
        visible = len(self.all_item_ids) - len(self.detached_ids)
        self.status_var.set(f"Showing {visible:,} of {len(self.all_item_ids):,} images")
        self.save_settings()

    # ── Selection ──────────────────────────────────────────────────

    def _select_all(self):
        visible = [iid for iid in self.all_item_ids if iid not in self.detached_ids]
        if visible:
            self.tree.selection_set(visible)
        self._update_selection_count()
        return 'break'  # prevent default Ctrl+A behavior

    def _deselect_all(self):
        self.tree.selection_remove(*self.tree.selection())
        self._update_selection_count()

    def _update_selection_count(self):
        count = len(self.tree.selection())
        if count:
            self.selection_var.set(f"{count:,} selected")
        else:
            self.selection_var.set("")

    def _on_select(self, event=None):
        self._update_selection_count()
        selection = self.tree.selection()
        if len(selection) == 1:
            self._show_single_preview(selection[0])
        elif len(selection) > 1:
            self._show_multi_selection(len(selection))
        else:
            self._clear_preview()

    # ── Image Preview ──────────────────────────────────────────────

    def _show_single_preview(self, item_id: str):
        entry = self.entries.get(item_id)
        if not entry:
            return

        self.multi_label.config(text="")
        self.save_btn.config(state='normal')
        self.revert_btn.config(state='normal')
        self.caption_text.config(state='normal')

        # Update caption editor
        self.caption_text.delete('1.0', 'end')
        self.caption_text.insert('1.0', entry.caption_text)

        if entry.caption_type == 'autocaption':
            self.caption_source_var.set(f"Source: .autocaption.txt")
        elif entry.caption_type == 'original':
            self.caption_source_var.set(f"Source: .txt")
        else:
            self.caption_source_var.set("No caption file (will create .autocaption.txt on save)")

        # Load image preview
        self._pending_preview_path = entry.image_path
        self._load_preview_image(entry.image_path)

    def _load_preview_image(self, image_path: Path):
        if Image is None:
            self.preview_canvas.delete('all')
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                text="PIL/Pillow not installed", fill='white'
            )
            return

        try:
            img = Image.open(image_path)
            img.load()  # force load to catch errors early
        except Exception as e:
            self.preview_canvas.delete('all')
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                text=f"Cannot load image:\n{e}", fill='white', justify='center'
            )
            return

        self._fit_image_to_canvas(img)

    def _fit_image_to_canvas(self, img):
        """Resize image to fit canvas and display it."""
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            # Canvas not yet rendered, defer
            self.after(50, self._fit_image_to_canvas, img)
            return

        # Calculate fit size
        img_w, img_h = img.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * ratio))
        new_h = max(1, int(img_h * ratio))

        try:
            resized = img.resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            resized = img.resize((new_w, new_h))

        self.current_preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.current_preview_image, anchor='center'
        )

    def _on_preview_resize(self, event=None):
        """Re-render preview when canvas is resized."""
        if self._pending_preview_path and self._pending_preview_path.exists():
            # Debounce resize events
            if hasattr(self, '_resize_after_id'):
                self.after_cancel(self._resize_after_id)
            self._resize_after_id = self.after(150, self._load_preview_image, self._pending_preview_path)

    def _show_multi_selection(self, count: int):
        self.multi_label.config(text=f"{count:,} items selected")
        self.caption_text.config(state='normal')
        self.caption_text.delete('1.0', 'end')
        self.caption_text.insert('1.0', f"({count:,} images selected - use bulk operations)")
        self.caption_text.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.revert_btn.config(state='disabled')
        self.caption_source_var.set("")
        self.preview_canvas.delete('all')
        self.current_preview_image = None
        self._pending_preview_path = None

    def _clear_preview(self):
        self.preview_canvas.delete('all')
        self.current_preview_image = None
        self._pending_preview_path = None
        self.caption_text.config(state='normal')
        self.caption_text.delete('1.0', 'end')
        self.caption_text.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.revert_btn.config(state='disabled')
        self.caption_source_var.set("")
        self.multi_label.config(text="")

    # ── Caption Save / Revert ──────────────────────────────────────

    def _save_caption(self):
        selection = self.tree.selection()
        if len(selection) != 1:
            return
        entry = self.entries.get(selection[0])
        if not entry:
            return

        new_text = self.caption_text.get('1.0', 'end').strip()

        # Determine save path
        if entry.caption_path:
            save_path = entry.caption_path
        else:
            save_path = entry.image_path.with_suffix('.autocaption.txt')

        try:
            save_path.write_text(new_text, encoding='utf-8')
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save caption:\n{e}")
            return

        # Update internal state
        entry.caption_text = new_text
        entry.caption_path = save_path
        if entry.caption_type == 'none':
            entry.caption_type = 'autocaption'
            self.caption_source_var.set("Source: .autocaption.txt")

        # Update treeview
        snippet = new_text[:100].replace('\n', ' ') if new_text else '(no caption)'
        type_label = {'autocaption': 'auto', 'original': 'txt', 'none': '-'}[entry.caption_type]
        self.tree.item(selection[0], values=(
            entry.image_path.name, entry.relative_path, type_label, snippet
        ))

    def _revert_caption(self):
        selection = self.tree.selection()
        if len(selection) != 1:
            return
        entry = self.entries.get(selection[0])
        if not entry:
            return

        # Re-read from disk
        text = read_caption(entry.caption_path)
        entry.caption_text = text
        self.caption_text.config(state='normal')
        self.caption_text.delete('1.0', 'end')
        self.caption_text.insert('1.0', text)

    # ── Search & Filter ────────────────────────────────────────────

    def _apply_filter(self):
        query = self.search_var.get().strip()
        if not query:
            self._clear_filter()
            return

        required, excluded = parse_search_query(query)
        if not required and not excluded:
            self._clear_filter()
            return

        # Reattach everything first
        for iid in list(self.detached_ids):
            try:
                self.tree.reattach(iid, '', 'end')
            except Exception:
                pass
        self.detached_ids.clear()

        # Detach non-matching
        for iid in self.all_item_ids:
            entry = self.entries.get(iid)
            if entry:
                caption_lower = entry.caption_text.lower()
                if not caption_matches_query(caption_lower, required, excluded):
                    self.tree.detach(iid)
                    self.detached_ids.add(iid)

        visible = len(self.all_item_ids) - len(self.detached_ids)
        self.status_var.set(f"Showing {visible:,} of {len(self.all_item_ids):,} images (filtered)")
        self._deselect_all()

    def _clear_filter(self):
        self.search_var.set('')
        for iid in list(self.detached_ids):
            try:
                self.tree.reattach(iid, '', 'end')
            except Exception:
                pass
        self.detached_ids.clear()
        self.status_var.set(f"Showing {len(self.all_item_ids):,} of {len(self.all_item_ids):,} images")

    # ── Bulk Operations ────────────────────────────────────────────

    def _get_target_ids(self) -> List[str]:
        """Get item IDs based on the 'Apply to' selection."""
        mode = self.apply_to_var.get()
        if mode == "Selected":
            return list(self.tree.selection())
        elif mode == "Filtered":
            return [iid for iid in self.all_item_ids if iid not in self.detached_ids]
        else:  # All Loaded
            return list(self.all_item_ids)

    def _do_replace(self):
        find_text = self.find_var.get()
        replace_text = self.replace_var.get()
        if not find_text:
            messagebox.showwarning("Empty Find", "Please enter text to find.")
            return

        target_ids = self._get_target_ids()
        if not target_ids:
            messagebox.showinfo("No Targets", "No images match the current 'Apply to' selection.")
            return

        # Count how many will be affected
        affected = [iid for iid in target_ids
                    if iid in self.entries and find_text in self.entries[iid].caption_text]
        if not affected:
            messagebox.showinfo("No Matches", f"'{find_text}' was not found in any of the {len(target_ids):,} targeted captions.")
            return

        if not messagebox.askyesno("Confirm Replace",
                                   f"Replace '{find_text}' with '{replace_text}' in {len(affected):,} caption(s)?\n\n"
                                   f"This will write changes to disk immediately."):
            return

        errors = 0
        for iid in affected:
            entry = self.entries[iid]
            new_text = entry.caption_text.replace(find_text, replace_text)
            save_path = entry.caption_path or entry.image_path.with_suffix('.autocaption.txt')
            try:
                save_path.write_text(new_text, encoding='utf-8')
                entry.caption_text = new_text
                entry.caption_path = save_path
                if entry.caption_type == 'none':
                    entry.caption_type = 'autocaption'
                snippet = new_text[:100].replace('\n', ' ') if new_text else '(no caption)'
                type_label = {'autocaption': 'auto', 'original': 'txt', 'none': '-'}[entry.caption_type]
                self.tree.item(iid, values=(entry.image_path.name, entry.relative_path, type_label, snippet))
            except Exception:
                errors += 1

        # Refresh preview if single selection
        sel = self.tree.selection()
        if len(sel) == 1:
            self._show_single_preview(sel[0])

        msg = f"Replaced in {len(affected) - errors:,} caption(s)."
        if errors:
            msg += f"\n{errors} file(s) could not be written."
        messagebox.showinfo("Replace Complete", msg)

    def _do_prepend(self):
        prepend_text = self.prepend_var.get()
        if not prepend_text:
            messagebox.showwarning("Empty Prepend", "Please enter text to prepend.")
            return

        target_ids = self._get_target_ids()
        if not target_ids:
            messagebox.showinfo("No Targets", "No images match the current 'Apply to' selection.")
            return

        # Filter to entries that actually exist
        valid = [iid for iid in target_ids if iid in self.entries]
        if not valid:
            return

        if not messagebox.askyesno("Confirm Prepend",
                                   f"Prepend '{prepend_text}' to {len(valid):,} caption(s)?\n\n"
                                   f"This will write changes to disk immediately."):
            return

        errors = 0
        for iid in valid:
            entry = self.entries[iid]
            new_text = prepend_text + entry.caption_text
            save_path = entry.caption_path or entry.image_path.with_suffix('.autocaption.txt')
            try:
                save_path.write_text(new_text, encoding='utf-8')
                entry.caption_text = new_text
                entry.caption_path = save_path
                if entry.caption_type == 'none':
                    entry.caption_type = 'autocaption'
                snippet = new_text[:100].replace('\n', ' ') if new_text else '(no caption)'
                type_label = {'autocaption': 'auto', 'original': 'txt', 'none': '-'}[entry.caption_type]
                self.tree.item(iid, values=(entry.image_path.name, entry.relative_path, type_label, snippet))
            except Exception:
                errors += 1

        sel = self.tree.selection()
        if len(sel) == 1:
            self._show_single_preview(sel[0])

        msg = f"Prepended to {len(valid) - errors:,} caption(s)."
        if errors:
            msg += f"\n{errors} file(s) could not be written."
        messagebox.showinfo("Prepend Complete", msg)

    # ── Delete ─────────────────────────────────────────────────────

    def _delete_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Nothing Selected", "Please select images to delete.")
            return

        # Count files that will be deleted
        file_count = 0
        for iid in selection:
            entry = self.entries.get(iid)
            if entry:
                file_count += 1  # image
                # Count associated caption files
                img = entry.image_path
                for suffix in ['.txt', '.autocaption.txt', '.quality.txt']:
                    if img.with_suffix(suffix).exists():
                        file_count += 1

        if not messagebox.askyesno("Confirm Delete",
                                   f"Permanently delete {len(selection):,} image(s) and their caption files?\n"
                                   f"({file_count:,} total files will be deleted)\n\n"
                                   f"This cannot be undone!",
                                   icon='warning'):
            return

        errors = 0
        for iid in selection:
            entry = self.entries.get(iid)
            if not entry:
                continue
            img = entry.image_path
            # Delete associated files
            for suffix in ['.txt', '.autocaption.txt', '.quality.txt']:
                cap_file = img.with_suffix(suffix)
                if cap_file.exists():
                    try:
                        cap_file.unlink()
                    except Exception:
                        errors += 1
            # Delete image
            try:
                img.unlink()
            except Exception:
                errors += 1

            # Remove from treeview and data
            try:
                self.tree.delete(iid)
            except Exception:
                pass
            self.entries.pop(iid, None)
            if iid in self.all_item_ids:
                self.all_item_ids.remove(iid)
            self.detached_ids.discard(iid)

        self._clear_preview()
        visible = len(self.all_item_ids) - len(self.detached_ids)
        self.status_var.set(f"Showing {visible:,} of {len(self.all_item_ids):,} images")
        self._update_selection_count()

        if errors:
            messagebox.showwarning("Delete Warning", f"Deleted, but {errors} file(s) could not be removed.")

    # ── Sorting ────────────────────────────────────────────────────

    def _sort_column(self, col: str):
        reverse = self._sort_reverse.get(col, False)
        col_index = {'name': 0, 'folder': 1, 'type': 2, 'caption': 3}[col]

        # Get all visible items with their values
        items = []
        for iid in self.tree.get_children(''):
            vals = self.tree.item(iid, 'values')
            items.append((vals[col_index], iid))

        items.sort(key=lambda x: x[0].lower(), reverse=reverse)

        for idx, (_, iid) in enumerate(items):
            self.tree.move(iid, '', idx)

        self._sort_reverse[col] = not reverse

    # ── Context Menu ───────────────────────────────────────────────

    def _show_context_menu(self, event):
        iid = self.tree.identify_row(event.y)
        if iid:
            if iid not in self.tree.selection():
                self.tree.selection_set(iid)
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def _copy_caption(self):
        selection = self.tree.selection()
        if not selection:
            return
        entry = self.entries.get(selection[0])
        if entry:
            self.clipboard_clear()
            self.clipboard_append(entry.caption_text)

    def _open_file_location(self):
        selection = self.tree.selection()
        if not selection:
            return
        entry = self.entries.get(selection[0])
        if not entry:
            return

        import subprocess
        import sys
        folder = str(entry.image_path.parent)
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', '/select,', str(entry.image_path)])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', str(entry.image_path)])
        else:
            subprocess.Popen(['xdg-open', folder])

    # ── Settings ───────────────────────────────────────────────────

    def save_settings(self):
        settings = {
            'folder': self.folder_var.get(),
        }
        self.settings_manager.set_tab_settings(self.tab_name, settings)

    def load_settings(self):
        settings = self.settings_manager.get_tab_settings(self.tab_name)
        if settings:
            self.folder_var.set(settings.get('folder', ''))

    def get_settings(self):
        return {'folder': self.folder_var.get()}

    def set_settings(self, settings):
        if 'folder' in settings:
            self.folder_var.set(settings['folder'])
