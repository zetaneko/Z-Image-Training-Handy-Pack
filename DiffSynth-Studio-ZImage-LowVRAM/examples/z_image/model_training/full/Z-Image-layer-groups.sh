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

# Model storage location (optional)
# By default, models are downloaded to ./models in the project directory.
# Uncomment and set MODEL_BASE_PATH to use a custom location:
# - Useful for storing models on a different drive
# - Allows sharing models between projects
# - Can point to existing Hugging Face cache (e.g., ~/.cache/huggingface/hub)
# Examples:
#   MODEL_BASE_PATH="/data/ai-models"
#   MODEL_BASE_PATH="$HOME/.cache/huggingface/hub"
#   MODEL_BASE_PATH="/mnt/storage/models"

# Dataset settings
# Option A: Folder with images + metadata.csv (traditional)
DATASET_PATH="/path"
DATASET_METADATA="${DATASET_PATH}/metadata.csv"
# Option B: Zitpack archives (packed dataset)
# Set ZITPACK_DIR to a directory containing .zitpack files to use instead of DATASET_PATH.
# Create zitpacks with: python3 python-scripts/pack_dataset.py --input <folder> --output <dir>
# ZITPACK_DIR="/path/to/local/zitpacks"
# Per-dataset repeat multipliers (optional, only used with ZITPACK_DIR).
# Files matching the prefix get sampled that many times more per epoch.
# Example: anime_chunk_001.zitpack gets 3x, portrait_chunk_001.zitpack gets 2x.
# ZITPACK_REPEATS="anime:3,portrait:2"

# Google Drive sync (optional) — downloads missing .zitpack files into ZITPACK_DIR
# before training starts, so remote workers need zero manual file setup.
#
# Option 1: rclone (recommended for personal Google accounts)
#   One-time setup: install rclone (https://rclone.org/install/), then run 'rclone config'
#   to add a Google Drive remote. Copy ~/.config/rclone/rclone.conf to remote workers.
#   Example remote path: "gdrive:MyDatasets/zitpacks" or "myremote:path/to/zitpacks"
# RCLONE_REMOTE="gdrive:MyDatasets/zitpacks"
#
# Option 2: Service account (for GCP / team setups)
#   Create a service account in Google Cloud Console, download the JSON key, and share
#   the Drive folder with the service account's email (shown in the JSON as "client_email").
# GDRIVE_FOLDER_ID="1AbCdEfGhIjKlMnOpQrStUvWxYz"
# GDRIVE_CREDENTIALS="/path/to/service-account.json"
DATASET_REPEAT=25
MAX_PIXELS=1048576

# Model settings
MODEL_PATHS="Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image:text_encoder/*.safetensors,Tongyi-MAI/Z-Image:vae/diffusion_pytorch_model.safetensors"
# MODEL_BASE_PATH=""  # Uncomment and set to override default ./models location (e.g., "/data/models" or "$HOME/.cache/huggingface")

# Multi-GPU settings
# NUM_GPUS defaults to the number of CUDA GPUs on this machine.
# Set explicitly to use fewer GPUs, e.g. NUM_GPUS=1 for single-GPU mode.
# Each GPU gets its own data shard; gradients are averaged across GPUs after each batch.
# RAM note: each GPU rank loads a full copy of the models to CPU (~28-42 GB per rank).
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

# Layer group settings (tune based on VRAM)
NUM_LAYER_GROUPS=6          # More groups = less VRAM, more swaps
IMAGES_PER_GROUP_BATCH=128    # Images processed per group before swapping (KEY OPTIMIZATION)

# Training settings
# NOTE: These are the ORIGINAL working hyperparameters from 9cfd1fb
# For experimentation with higher LR, see commented alternatives below
LEARNING_RATE=1e-4
NUM_EPOCHS=1
SAVE_STEPS=10
GRADIENT_ACCUMULATION=1     # With batch processing, often don't need additional accumulation

# AdamW optimizer settings
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPS=1e-8
WEIGHT_DECAY=0.01
OPTIMIZER_STATE_DTYPE="float32"

# LR scheduler settings
WARMUP_STEPS=10
LR_SCHEDULER="cosine"       # cosine, linear, or constant
MIN_LR_RATIO=0.1            # Final LR = LEARNING_RATE * MIN_LR_RATIO (for cosine/linear)

# Alternative higher LR settings (use with caution):
# LEARNING_RATE=5e-4    # Higher LR for batch 64
# ADAM_BETA2=0.99       # Faster adaptation
# WEIGHT_DECAY=0.005    # Less regularization
# WARMUP_STEPS=300      # More warmup for stability
# LR_SCHEDULER="cosine" # Better convergence

# Output settings
OUTPUT_PATH="./models/train/Z-Image_layer_groups_testrbigger"

# Reproducibility
SEED=42

# ==========================================
# Run training
# ==========================================

# Continue from checkpoint (if exists) - set to true to resume training
CONTINUE_TRAINING=false

torchrun --nproc_per_node=$NUM_GPUS examples/z_image/model_training/train_layer_groups.py \
  $([ -n "$ZITPACK_DIR" ] && echo "--zitpacks \"$ZITPACK_DIR\"" || echo "--dataset_base_path \"$DATASET_PATH\" --dataset_metadata_path \"$DATASET_METADATA\"") \
  $([ -n "$ZITPACK_REPEATS" ] && echo "--zitpack_repeats \"$ZITPACK_REPEATS\"") \
  $([ -n "$RCLONE_REMOTE" ] && echo "--rclone_remote \"$RCLONE_REMOTE\"") \
  $([ -n "$GDRIVE_FOLDER_ID" ] && echo "--gdrive_folder_id \"$GDRIVE_FOLDER_ID\"") \
  $([ -n "$GDRIVE_CREDENTIALS" ] && echo "--gdrive_credentials \"$GDRIVE_CREDENTIALS\"") \
  --dataset_repeat $DATASET_REPEAT \
  --max_pixels $MAX_PIXELS \
  --model_id_with_origin_paths "$MODEL_PATHS" \
  $([ -n "$MODEL_BASE_PATH" ] && echo "--model_base_path \"$MODEL_BASE_PATH\"") \
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
  --use_gradient_checkpointing_offload \
  --optimizer-state-dtype "$OPTIMIZER_STATE_DTYPE" \
  $([ "$CONTINUE_TRAINING" = true ] && echo "--continue_training")

# ==========================================
# Resuming Training
# ==========================================
#
# Option 1 (Recommended): Set CONTINUE_TRAINING=true above
#   - Automatically resumes from latest checkpoint in OUTPUT_PATH
#   - Safe to use even if no checkpoint exists (starts fresh)
#   - Optimizer state files are overwritten each save to save disk space
#   - Model checkpoints (model_step_N.safetensors) are kept for each save_steps
#
# Option 2: Explicit checkpoint path
#   --resume_from_checkpoint "$OUTPUT_PATH"
#   (Can also point to a specific training_state_latest.pt file)
#
# Checkpoint files saved:
#   - training_state_latest.pt (metadata, overwritten each save)
#   - optimizer_group_N.safetensors (optimizer states, overwritten each save)
#   - optimizer_persistent.safetensors (persistent params, overwritten each save)
#   - model_step_N.safetensors (model weights, kept for each save_steps)

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
