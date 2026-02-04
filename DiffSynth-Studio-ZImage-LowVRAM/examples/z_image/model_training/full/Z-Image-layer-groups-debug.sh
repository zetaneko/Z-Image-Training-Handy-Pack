#!/bin/bash
# Layer Group Training for Z-Image on Low VRAM (12-16GB)
#
# This script uses layer group offloading to enable full fine-tuning on
# consumer GPUs with limited VRAM.
#
# How it works:
#   - All models are loaded to CPU first
#   - The DIT's 30 transformer layers are split into groups (default 6 groups of 5 layers)
#   - Text encoder and VAE are moved to GPU only when needed, then back to CPU
#   - BATCH OPTIMIZATION: Multiple images are processed through each layer group
#     before swapping, dramatically reducing GPU<->CPU transfers
#   - AdamW optimizer states are kept on CPU RAM, loaded per-group during updates
#
# Memory usage:
#   - GPU: ~8-12GB (one layer group + activations for batch)
#   - CPU RAM: ~32GB+ (all models + optimizer states + boundary activations)
#
# Swap efficiency (with images_per_group_batch=4, num_layer_groups=6):
#   - Without batching: 4 images × 6 groups × 2 passes = 48 swaps
#   - With batching: 6 groups × 2 passes = 12 swaps (4x improvement!)
#
# Recommended settings based on VRAM:
#   - 12GB: num_layer_groups=10, images_per_group_batch=2, max_pixels=131072 (362x362)
#   - 16GB: num_layer_groups=6,  images_per_group_batch=4, max_pixels=262144 (512x512)
#   - 24GB: num_layer_groups=4,  images_per_group_batch=8, max_pixels=524288 (724x724)

# Get the directory where this script is located, then navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Change to project root so relative paths work
cd "$PROJECT_ROOT"

# Print GPU info
echo "=============================================="
echo "Z-Image Layer Group Training"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=============================================="

# ==========================================
# Configuration - adjust these for your setup
# ==========================================

# Dataset settings
DATASET_PATH="/media/misty/M2 Drive/Trainer/AI Toolkit/al-toolkit-linux/ai-toolkit/datasets/pregnant"
DATASET_METADATA="${DATASET_PATH}/metadata.csv"
DATASET_REPEAT=25
MAX_PIXELS=1048576

# Model settings
MODEL_PATHS="Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors"

# Layer group settings (tune based on VRAM)
NUM_LAYER_GROUPS=6          # More groups = less VRAM, more swaps
IMAGES_PER_GROUP_BATCH=2    # Images processed per group before swapping (KEY OPTIMIZATION)

# Training settings
LEARNING_RATE=1e-4
NUM_EPOCHS=1
SAVE_STEPS=1
GRADIENT_ACCUMULATION=1     # With batch processing, often don't need additional accumulation

# AdamW optimizer settings
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPS=1e-8
WEIGHT_DECAY=0.01

# LR scheduler settings
WARMUP_STEPS=100
LR_SCHEDULER="cosine"       # cosine, linear, or constant
MIN_LR_RATIO=0.1            # Final LR = LEARNING_RATE * MIN_LR_RATIO

# Output settings
OUTPUT_PATH="./models/train/Z-Image_layer_groups"

# Reproducibility
SEED=42

# ==========================================
# Run training
# ==========================================

python examples/z_image/model_training/train_layer_groups.py \
  --dataset_base_path "$DATASET_PATH" \
  --dataset_metadata_path "$DATASET_METADATA" \
  --dataset_repeat $DATASET_REPEAT \
  --max_pixels $MAX_PIXELS \
  --model_id_with_origin_paths "$MODEL_PATHS" \
  --trainable_models "dit" \
  --num_layer_groups $NUM_LAYER_GROUPS \
  --images_per_group_batch $IMAGES_PER_GROUP_BATCH \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --save_steps $SAVE_STEPS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --adam_eps $ADAM_EPS \
  --weight_decay $WEIGHT_DECAY \
  --warmup_steps $WARMUP_STEPS \
  --lr_scheduler $LR_SCHEDULER \
  --min_lr_ratio $MIN_LR_RATIO \
  --output_path "$OUTPUT_PATH" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --seed $SEED \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload

# To resume from a checkpoint, add:
#   --resume_from_checkpoint "$OUTPUT_PATH/training_state_step_XXX.pt"

# To enable verbose logging (useful for debugging), add:
#   --verbose

# To enable detailed profiling (GPU/CPU memory, timing breakdown), add:
#   --profile --profile_report_interval 5
#
# This will show where time is spent:
#   - text_encoder_load/offload: Time loading/offloading text encoder
#   - vae_load/offload: Time loading/offloading VAE
#   - layer_group_N_load/forward/offload: Time per layer group
#   - optimizer_step: Time updating weights
#
# If GPU utilization is low, consider:
#   1. Reducing num_layer_groups (bigger groups = more GPU work per swap)
#   2. Increasing images_per_group_batch (more images per swap)
#   3. The bottleneck may be CPU->GPU data transfer
