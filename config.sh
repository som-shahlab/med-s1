#!/bin/bash

# Create directories if they don't exist
CACHE_DIR="/share/pi/nigam/users/calebwin/hf_cache"
DATA_DIR="/share/pi/nigam/users/calebwin/hf_cache/med-s1k"
mkdir -p "$CACHE_DIR"
mkdir -p "$DATA_DIR"

# Set all HuggingFace related cache environment variables
export HF_DATASETS_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_CACHE_DIR="$CACHE_DIR"

# Set med-s1k output directory
export MED_S1K_OUTPUT="$DATA_DIR"

# HF and Gemini credentials

# Debug: Print environment variables
echo "Cache directory: $CACHE_DIR"
echo "Data directory: $DATA_DIR"
echo "HF Token: ${HUGGING_FACE_HUB_TOKEN:0:10}..."
echo "Gemini Key: ${GEMINI_API_KEY:0:10}..."