#!/usr/bin/env python3
"""
Converts DiffSynth-Studio fine-tuned model checkpoints to standard format.
Fuses unfused attention keys and merges with original base model weights.
This fixes compatibility issues with ComfyUI and other inference software.

Usage:
  python fix-diffsynth-model-output.py --original <base_model> --input <finetuned> --output <output>

Example:
  python fix-diffsynth-model-output.py \
    --original /path/to/zImageBase_base.safetensors \
    --input /path/to/step-2000.safetensors \
    --output /path/to/step-2000-fixed.safetensors
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


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


def main():
    parser = argparse.ArgumentParser(
        description='Convert DiffSynth-Studio fine-tuned checkpoints to standard format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python fix-diffsynth-model-output.py \\
    --original /path/to/zImageBase_base.safetensors \\
    --input /path/to/step-2000.safetensors \\
    --output /path/to/step-2000-fixed.safetensors
        """
    )
    parser.add_argument('--original', '-o', required=True, type=Path,
                        help='Path to original base model (e.g., zImageBase_base.safetensors)')
    parser.add_argument('--input', '-i', required=True, type=Path,
                        help='Path to fine-tuned checkpoint to convert')
    parser.add_argument('--output', '-O', required=True, type=Path,
                        help='Path for converted output checkpoint')

    args = parser.parse_args()

    # Validate inputs
    if not args.original.exists():
        print(f"Error: Original model not found: {args.original}")
        return 1
    if not args.input.exists():
        print(f"Error: Input checkpoint not found: {args.input}")
        return 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading original model: {args.original}")
    original_sd = load_file(str(args.original))

    print(f"Loading fine-tuned checkpoint: {args.input}")
    ft_sd = load_file(str(args.input))

    print("Converting state dict...")
    converted_sd = convert_state_dict(ft_sd, original_sd)

    print(f"Saving converted checkpoint: {args.output}")
    save_file(converted_sd, str(args.output))

    print("Converted checkpoint saved. Load this in ComfyUI.")
    return 0


if __name__ == '__main__':
    exit(main())
