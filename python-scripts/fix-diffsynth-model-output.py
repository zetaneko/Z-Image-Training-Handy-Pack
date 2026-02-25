#!/usr/bin/env python3
"""
Converts DiffSynth-Studio fine-tuned model checkpoints to standard format, or vice versa.

Forward (default): DiffSynth training output -> standard format for ComfyUI
  Fuses unfused attention keys (to_q/to_k/to_v -> qkv) and merges with original base model.

Reverse (--reverse): Standard/fixed checkpoint -> DiffSynth training format
  Strips ComfyUI 'model.diffusion_model.' prefix if present, unfuses attention keys
  (qkv -> to_q/to_k/to_v), and restores all_x_embedder/all_final_layer key names.

Usage:
  python fix-diffsynth-model-output.py --original <base_model> --input <finetuned> --output <output>
  python fix-diffsynth-model-output.py --reverse --input <fixed> --output <restored>

Examples:
  python fix-diffsynth-model-output.py \\
    --original /path/to/zImageBase_base.safetensors \\
    --input /path/to/step-2000.safetensors \\
    --output /path/to/step-2000-fixed.safetensors

  python fix-diffsynth-model-output.py --reverse \\
    --input /path/to/step-2000-fixed.safetensors \\
    --output /path/to/step-2000-restored.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def print_status(message: str):
    """Print status message with flush for GUI compatibility."""
    print(message, flush=True)


def convert_state_dict(ft_sd, original_sd):
    """Fuse unfused attention keys and rename special prefixed keys."""
    # Fuse unfused attention keys
    keys = list(ft_sd.keys())
    for key in keys:
        if 'to_q.weight' in key:
            base = key.replace('to_q.weight', '')
            q = ft_sd.pop(base + 'to_q.weight')
            k = ft_sd.pop(base + 'to_k.weight')
            v = ft_sd.pop(base + 'to_v.weight')
            ft_sd[base + 'qkv.weight'] = torch.cat([q, k, v], dim=0)
            if base + 'to_out.0.weight' in ft_sd:
                ft_sd[base + 'out.weight'] = ft_sd.pop(base + 'to_out.0.weight')
            if base + 'norm_q.weight' in ft_sd:
                ft_sd[base + 'q_norm.weight'] = ft_sd.pop(base + 'norm_q.weight')
            if base + 'norm_k.weight' in ft_sd:
                ft_sd[base + 'k_norm.weight'] = ft_sd.pop(base + 'norm_k.weight')

    # Rename special prefixed keys (e.g., for embedder and final layer)
    if 'all_x_embedder.2-1.weight' in ft_sd:
        ft_sd['x_embedder.weight'] = ft_sd.pop('all_x_embedder.2-1.weight')
        ft_sd['x_embedder.bias'] = ft_sd.pop('all_x_embedder.2-1.bias')
    if 'all_final_layer.2-1.linear.weight' in ft_sd:
        ft_sd['final_layer.linear.weight'] = ft_sd.pop('all_final_layer.2-1.linear.weight')
        ft_sd['final_layer.linear.bias'] = ft_sd.pop('all_final_layer.2-1.linear.bias')
        ft_sd['final_layer.adaLN_modulation.1.weight'] = ft_sd.pop('all_final_layer.2-1.adaLN_modulation.1.weight')
        ft_sd['final_layer.adaLN_modulation.1.bias'] = ft_sd.pop('all_final_layer.2-1.adaLN_modulation.1.bias')

    # Merge: Use fine-tuned weights where available, fall back to original for missing
    merged_sd = original_sd.copy()
    merged_sd.update(ft_sd)
    return merged_sd


def reverse_convert_state_dict(sd):
    """Unfuse fused attention keys back to DiffSynth training format.

    Steps:
    1. Strip 'model.diffusion_model.' prefix if present (ComfyUI convention).
    2. Split qkv.weight into equal thirds -> to_q, to_k, to_v. Safe for Z-Image
       because all attention layers use equal q/k/v dimensions (self-attention).
       Also reverses out/norm_q/norm_k key renames.
    3. Rename x_embedder.* -> all_x_embedder.2-1.* and final_layer.* ->
       all_final_layer.2-1.* to match the layer-groups training checkpoint format.
    """
    # Step 1: strip ComfyUI model.diffusion_model. prefix
    prefix = 'model.diffusion_model.'
    if any(k.startswith(prefix) for k in sd):
        print_status("  Detected ComfyUI 'model.diffusion_model.' prefix — stripping")
        sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

    # Step 2: unfuse attention keys
    result = dict(sd)
    keys = list(result.keys())
    for key in keys:
        if 'qkv.weight' not in key:
            continue
        base = key.replace('qkv.weight', '')
        tensor = result.pop(key)
        total = tensor.shape[0]
        chunk = total // 3
        result[base + 'to_q.weight'] = tensor[:chunk]
        result[base + 'to_k.weight'] = tensor[chunk:2 * chunk]
        result[base + 'to_v.weight'] = tensor[2 * chunk:]
        if base + 'out.weight' in result:
            result[base + 'to_out.0.weight'] = result.pop(base + 'out.weight')
        if base + 'q_norm.weight' in result:
            result[base + 'norm_q.weight'] = result.pop(base + 'q_norm.weight')
        if base + 'k_norm.weight' in result:
            result[base + 'norm_k.weight'] = result.pop(base + 'k_norm.weight')

    # Step 3: reverse special key renames (restore all_* prefixes for layer-groups format)
    if 'x_embedder.weight' in result:
        result['all_x_embedder.2-1.weight'] = result.pop('x_embedder.weight')
        if 'x_embedder.bias' in result:
            result['all_x_embedder.2-1.bias'] = result.pop('x_embedder.bias')
    if 'final_layer.linear.weight' in result:
        result['all_final_layer.2-1.linear.weight'] = result.pop('final_layer.linear.weight')
        if 'final_layer.linear.bias' in result:
            result['all_final_layer.2-1.linear.bias'] = result.pop('final_layer.linear.bias')
        if 'final_layer.adaLN_modulation.1.weight' in result:
            result['all_final_layer.2-1.adaLN_modulation.1.weight'] = result.pop('final_layer.adaLN_modulation.1.weight')
        if 'final_layer.adaLN_modulation.1.bias' in result:
            result['all_final_layer.2-1.adaLN_modulation.1.bias'] = result.pop('final_layer.adaLN_modulation.1.bias')

    return result


def validate_checkpoint(path, mode=None, reference_path=None):
    """Inspect a checkpoint's key structure, dtypes, and optionally compare against a reference.

    mode: 'forward' (expect fused qkv), 'reverse' (expect unfused to_q/k/v), or None (report only).
    """
    sd = load_file(str(path))

    qkv_keys  = [k for k in sd if 'qkv.weight' in k]
    to_q_keys = [k for k in sd if 'to_q.weight' in k]

    dtype_counts = {}
    for t in sd.values():
        dt = str(t.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

    print_status(f"\n=== Checkpoint: {path.name} ===")
    print_status(f"  Total tensors : {len(sd)}")
    print_status(f"  Fused qkv     : {len(qkv_keys)}")
    print_status(f"  Unfused to_q  : {len(to_q_keys)}")
    print_status(f"  Dtypes        : {dtype_counts}")

    if mode == 'forward':
        if qkv_keys:
            print_status(f"  OK: {len(qkv_keys)} fused attention layers")
        else:
            print_status("  WARNING: No fused qkv.weight keys found — conversion may have failed")
        if to_q_keys:
            print_status(f"  WARNING: {len(to_q_keys)} unfused to_q.weight keys still present")

    elif mode == 'reverse':
        if to_q_keys:
            print_status(f"  OK: {len(to_q_keys)} unfused attention layers")
        else:
            print_status("  WARNING: No unfused to_q.weight keys found — conversion may have failed")
        if qkv_keys:
            print_status(f"  WARNING: {len(qkv_keys)} fused qkv.weight keys still present")

        # Verify q/k/v shapes are consistent at every attention base
        bad = []
        for key in sd:
            if 'to_q.weight' not in key:
                continue
            base = key.replace('to_q.weight', '')
            q = sd.get(base + 'to_q.weight')
            k = sd.get(base + 'to_k.weight')
            v = sd.get(base + 'to_v.weight')
            if q is None or k is None or v is None:
                bad.append(f"{base}: missing to_k or to_v")
            elif q.shape != k.shape or q.shape != v.shape:
                bad.append(f"{base}: to_q={q.shape} to_k={k.shape} to_v={v.shape}")
        if bad:
            print_status(f"  WARNING: {len(bad)} attention shape inconsistencies:")
            for b in bad[:5]:
                print_status(f"    {b}")
            if len(bad) > 5:
                print_status(f"    ... ({len(bad) - 5} more)")
        elif to_q_keys:
            print_status(f"  OK: All {len(to_q_keys)} q/k/v shapes are consistent")

    if reference_path is None:
        return

    ref = load_file(str(reference_path))
    ref_keys = set(ref.keys())
    out_keys = set(sd.keys())
    only_ref = sorted(ref_keys - out_keys)
    only_out = sorted(out_keys - ref_keys)

    print_status(f"\n=== Comparison vs {reference_path.name} ===")

    if not only_ref and not only_out:
        print_status(f"  OK: Key sets match ({len(ref_keys)} keys)")
    else:
        if only_ref:
            preview = only_ref[:5]
            tail = f" ... (+{len(only_ref) - 5} more)" if len(only_ref) > 5 else ""
            print_status(f"  Missing from output ({len(only_ref)}): {preview}{tail}")
        if only_out:
            preview = only_out[:5]
            tail = f" ... (+{len(only_out) - 5} more)" if len(only_out) > 5 else ""
            print_status(f"  Extra in output ({len(only_out)}): {preview}{tail}")

    shape_bad = [(k, ref[k].shape, sd[k].shape)
                 for k in ref_keys & out_keys if ref[k].shape != sd[k].shape]
    if shape_bad:
        print_status(f"  Shape mismatches ({len(shape_bad)}):")
        for k, rs, os_ in shape_bad[:5]:
            print_status(f"    {k}: ref={rs} out={os_}")
        if len(shape_bad) > 5:
            print_status(f"    ... ({len(shape_bad) - 5} more)")
    else:
        print_status(f"  OK: All shared-key shapes match")

    dtype_pairs = {}
    for k in ref_keys & out_keys:
        if ref[k].dtype != sd[k].dtype:
            pair = (str(ref[k].dtype), str(sd[k].dtype))
            dtype_pairs[pair] = dtype_pairs.get(pair, 0) + 1
    if dtype_pairs:
        print_status(f"  Dtype mismatches:")
        for (rd, od), count in dtype_pairs.items():
            print_status(f"    {count} keys: ref={rd} -> out={od}")
    else:
        print_status(f"  OK: All shared-key dtypes match")


def main():
    parser = argparse.ArgumentParser(
        description='Convert DiffSynth-Studio checkpoints to/from standard format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forward: DiffSynth training output -> ComfyUI-compatible
  python fix-diffsynth-model-output.py \\
    --original /path/to/zImageBase_base.safetensors \\
    --input /path/to/step-2000.safetensors \\
    --output /path/to/step-2000-fixed.safetensors

  # Reverse: fixed checkpoint -> DiffSynth training format
  python fix-diffsynth-model-output.py --reverse \\
    --input /path/to/step-2000-fixed.safetensors \\
    --output /path/to/step-2000-restored.safetensors

  # Validate after reverse conversion (compare against HF base model)
  python fix-diffsynth-model-output.py --reverse --validate \\
    --reference /path/to/zImageBase_base.safetensors \\
    --input /path/to/step-2000-fixed.safetensors \\
    --output /path/to/step-2000-restored.safetensors

  # Inspect an existing checkpoint without converting
  python fix-diffsynth-model-output.py --validate --reverse \\
    --input /path/to/step-2000-restored.safetensors
        """
    )
    parser.add_argument('--reverse', '-r', action='store_true',
                        help='Reverse mode: convert fixed checkpoint back to DiffSynth format')
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Validate checkpoint structure and dtypes. '
                             'With --output: validates after conversion. '
                             'Without --output: validates --input directly (no conversion).')
    parser.add_argument('--original', '-o', type=Path, default=None,
                        help='Path to original base model (required for forward mode)')
    parser.add_argument('--input', '-i', required=True, type=Path,
                        help='Path to input checkpoint')
    parser.add_argument('--output', '-O', type=Path, default=None,
                        help='Path for output checkpoint (omit with --validate to inspect only)')
    parser.add_argument('--reference', type=Path, default=None,
                        help='Reference checkpoint to compare against during validation '
                             '(e.g., the HF base model or a known-good training output)')

    args = parser.parse_args()

    # Validate-only mode: just inspect --input, no conversion
    if args.validate and args.output is None:
        if not args.input.exists():
            print(f"Error: Input checkpoint not found: {args.input}")
            return 1
        if args.reference and not args.reference.exists():
            print(f"Error: Reference checkpoint not found: {args.reference}")
            return 1
        mode = 'reverse' if args.reverse else 'forward'
        validate_checkpoint(args.input, mode=mode, reference_path=args.reference)
        return 0

    # Conversion mode (with optional validate afterwards)
    if args.output is None:
        print("Error: --output is required for conversion (or use --validate without --output to inspect only)")
        return 1
    if not args.reverse and args.original is None:
        print("Error: --original is required for forward mode (omit it only with --reverse)")
        return 1
    if not args.reverse and not args.original.exists():
        print(f"Error: Original model not found: {args.original}")
        return 1
    if not args.input.exists():
        print(f"Error: Input checkpoint not found: {args.input}")
        return 1
    if args.reference and not args.reference.exists():
        print(f"Error: Reference checkpoint not found: {args.reference}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.reverse:
        print_status(f"Loading fixed checkpoint: {args.input}")
        sd = load_file(str(args.input))
        print_status(f"  Loaded {len(sd)} tensors")

        print_status("Reversing conversion (unfusing attention keys)...")
        converted_sd = reverse_convert_state_dict(sd)
        print_status(f"  Output: {len(converted_sd)} tensors")

        print_status(f"Saving restored checkpoint: {args.output}")
        save_file(converted_sd, str(args.output))
        print_status("Done. Restored checkpoint can be loaded by DiffSynth for training.")

        if args.validate:
            validate_checkpoint(args.output, mode='reverse', reference_path=args.reference)
    else:
        print_status(f"Loading original model: {args.original}")
        original_sd = load_file(str(args.original))
        print_status(f"  Loaded {len(original_sd)} tensors from original model")

        print_status(f"Loading fine-tuned checkpoint: {args.input}")
        ft_sd = load_file(str(args.input))
        print_status(f"  Loaded {len(ft_sd)} tensors from fine-tuned checkpoint")

        print_status("Converting state dict...")
        converted_sd = convert_state_dict(ft_sd, original_sd)
        print_status(f"  Merged to {len(converted_sd)} tensors")

        print_status(f"Saving converted checkpoint: {args.output}")
        save_file(converted_sd, str(args.output))
        print_status("Converted checkpoint saved. Load this in ComfyUI.")

        if args.validate:
            validate_checkpoint(args.output, mode='forward', reference_path=args.reference)

    return 0


if __name__ == '__main__':
    exit(main())
