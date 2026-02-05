#!/bin/bash
# Low VRAM single GPU LoRA training for Z-Image
# Suitable for 12-16GB VRAM cards
#
# LoRA training only trains small adapter weights, not the full model,
# making it much more memory efficient than full fine-tuning.
#
# Adjust max_pixels based on your VRAM:
#   - 12GB: max_pixels=262144 (512x512)
#   - 16GB: max_pixels=524288 (724x724)
#   - 24GB: max_pixels=786432 (886x886)

# Get the directory where this script is located, then navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Change to project root so relative paths work
cd "$PROJECT_ROOT"

# To use a custom model storage location, add:
#   --model_base_path "/path/to/your/models"
# or set the environment variable before running:
#   export DIFFSYNTH_MODEL_BASE_PATH="/path/to/your/models"

accelerate launch \
  --config_file examples/z_image/model_training/full/accelerate_config_single_gpu.yaml \
  examples/z_image/model_training/train.py \
  --dataset_base_path "/media/misty/M2 Drive/Trainer/AI Toolkit/al-toolkit-linux/ai-toolkit/datasets/pregnant" \
  --dataset_metadata_path "/media/misty/M2 Drive/Trainer/AI Toolkit/al-toolkit-linux/ai-toolkit/datasets/pregnant/metadata.csv" \
  --max_pixels 262144 \
  --dataset_repeat 25 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --save_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image_lora_low_vram" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --fp8_models "text_encoder,vae" \
  --gradient_accumulation_steps 4 \
  --dataset_num_workers 2
