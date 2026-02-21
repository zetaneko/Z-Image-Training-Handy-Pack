#!/bin/bash
# Z-Image Full Fine-Tuning (multi-GPU via accelerate)
# Tested on 8*A100
#
# To use a custom model storage location, add:
#   --model_base_path "/path/to/your/models"
# or set the environment variable:
#   export DIFFSYNTH_MODEL_BASE_PATH="/path/to/your/models"

# ==========================================
# Configuration - adjust these for your setup
# ==========================================

# Dataset settings
# Option A: Folder with images + metadata.csv (traditional)
DATASET_PATH="data/example_image_dataset"
DATASET_METADATA="${DATASET_PATH}/metadata.csv"
# Option B: Zitpack archives (packed dataset)
# Set ZITPACK_DIR to a directory containing .zitpack files to use instead of DATASET_PATH.
# Create zitpacks with: python3 python-scripts/pack_dataset.py --input <folder> --output <dir>
# ZITPACK_DIR="/path/to/zitpacks"
# Per-dataset repeat multipliers (optional, only used with ZITPACK_DIR).
# Files matching the prefix get sampled that many times more per epoch.
# Example: anime_chunk_001.zitpack gets 3x, portrait_chunk_001.zitpack gets 2x.
# ZITPACK_REPEATS="anime:3,portrait:2"
DATASET_REPEAT=400
MAX_PIXELS=1048576

# Model settings
MODEL_PATHS="Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors"
# MODEL_BASE_PATH=""  # Uncomment and set to override default ./models location

# Training settings
LEARNING_RATE=1e-5
NUM_EPOCHS=2
NUM_WORKERS=8

# Output settings
OUTPUT_PATH="./models/train/Z-Image_full"

# ==========================================
# Run training
# ==========================================

accelerate launch --config_file examples/z_image/model_training/full/accelerate_config.yaml examples/z_image/model_training/train.py \
  $([ -n "$ZITPACK_DIR" ] && echo "--zitpacks \"$ZITPACK_DIR\"" || echo "--dataset_base_path \"$DATASET_PATH\" --dataset_metadata_path \"$DATASET_METADATA\"") \
  $([ -n "$ZITPACK_REPEATS" ] && echo "--zitpack_repeats \"$ZITPACK_REPEATS\"") \
  --max_pixels $MAX_PIXELS \
  --dataset_repeat $DATASET_REPEAT \
  --model_id_with_origin_paths "$MODEL_PATHS" \
  $([ -n "$MODEL_BASE_PATH" ] && echo "--model_base_path \"$MODEL_BASE_PATH\"") \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers $NUM_WORKERS
