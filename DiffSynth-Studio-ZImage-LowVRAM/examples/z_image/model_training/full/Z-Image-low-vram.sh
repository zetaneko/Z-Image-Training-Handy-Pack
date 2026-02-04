#!/bin/bash
# Low VRAM single GPU training for Z-Image
# Optimized for ~12-16GB VRAM cards (e.g., RTX 3080/3090/4080)
#
# Key optimizations:
#   - Single GPU accelerate config (no DeepSpeed)
#   - Gradient checkpointing with CPU offload
#   - FP8 precision for text encoder and VAE (non-trainable)
#   - Smaller max_pixels (512x512 = 262144)
#   - Gradient accumulation to simulate larger batch
#
# Adjust max_pixels and gradient_accumulation_steps based on your VRAM:
#   - 8GB:  max_pixels=131072 (362x362), accumulation=8
#   - 12GB: max_pixels=262144 (512x512), accumulation=4
#   - 16GB: max_pixels=524288 (724x724), accumulation=2
#   - 24GB: max_pixels=786432 (886x886), accumulation=1

# Get the directory where this script is located, then navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Change to project root so relative paths work
cd "$PROJECT_ROOT"

accelerate launch \
  --config_file examples/z_image/model_training/full/accelerate_config_single_gpu.yaml \
  examples/z_image/model_training/train.py \
  --dataset_base_path "/media/misty/M2 Drive/Trainer/AI Toolkit/al-toolkit-linux/ai-toolkit/datasets/pregnant" \
  --dataset_metadata_path "/media/misty/M2 Drive/Trainer/AI Toolkit/al-toolkit-linux/ai-toolkit/datasets/pregnant/metadata.csv" \
  --max_pixels 262144 \
  --dataset_repeat 25 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --save_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image_full_low_vram" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --fp8_models "text_encoder,vae" \
  --gradient_accumulation_steps 4 \
  --dataset_num_workers 2
