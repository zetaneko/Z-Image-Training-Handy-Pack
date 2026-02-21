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
DATASET_REPEAT=400
MAX_PIXELS=1048576

# Model settings
MODEL_PATHS="Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors"
# MODEL_BASE_PATH=""  # Uncomment and set to override default ./models location

# Training settings
LEARNING_RATE=1e-5
NUM_EPOCHS=2
NUM_WORKERS=8
SAVE_STEPS=100   # Save a checkpoint every N steps (set to 0 or remove to save per epoch only)

# Output settings
OUTPUT_PATH="./models/train/Z-Image_full"

# ==========================================
# Run training
# ==========================================

accelerate launch --config_file examples/z_image/model_training/full/accelerate_config.yaml examples/z_image/model_training/train.py \
  $([ -n "$ZITPACK_DIR" ] && echo "--zitpacks \"$ZITPACK_DIR\"" || echo "--dataset_base_path \"$DATASET_PATH\" --dataset_metadata_path \"$DATASET_METADATA\"") \
  $([ -n "$ZITPACK_REPEATS" ] && echo "--zitpack_repeats \"$ZITPACK_REPEATS\"") \
  $([ -n "$RCLONE_REMOTE" ] && echo "--rclone_remote \"$RCLONE_REMOTE\"") \
  $([ -n "$GDRIVE_FOLDER_ID" ] && echo "--gdrive_folder_id \"$GDRIVE_FOLDER_ID\"") \
  $([ -n "$GDRIVE_CREDENTIALS" ] && echo "--gdrive_credentials \"$GDRIVE_CREDENTIALS\"") \
  --max_pixels $MAX_PIXELS \
  --dataset_repeat $DATASET_REPEAT \
  --model_id_with_origin_paths "$MODEL_PATHS" \
  $([ -n "$MODEL_BASE_PATH" ] && echo "--model_base_path \"$MODEL_BASE_PATH\"") \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --save_steps $SAVE_STEPS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers $NUM_WORKERS
