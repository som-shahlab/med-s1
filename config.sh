#!/bin/bash

# Set directory paths

# # michael
# MED_S1_DIR="/share/pi/nigam/mwornow/meds1/med-s1"
# CACHE_DIR="/share/pi/nigam/users/calebwin/hf_cache"
# DATA_DIR="/share/pi/nigam/users/calebwin/hf_cache/med-s1k"
# RESULTS_JSON="/share/pi/nigam/mwornow/meds1/med-s1/results_michael.json"

# caleb
MED_S1_DIR="/share/pi/nigam/users/calebwin/med-s1"
CACHE_DIR="/share/pi/nigam/users/calebwin/hf_cache"
DATA_DIR="/share/pi/nigam/users/calebwin/hf_cache/med-s1k"
RESULTS_JSON="/share/pi/nigam/users/calebwin/med-s1/results.json"

# Export paths for scripts
export MED_S1_DIR
export CACHE_DIR
export DATA_DIR
export RESULTS_JSON
mkdir -p "$CACHE_DIR"
mkdir -p "$DATA_DIR"

# Set all HuggingFace related cache environment variables
export HF_DATASETS_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_CACHE_DIR="$CACHE_DIR"

# Set med-s1k output directory
export MED_S1K_OUTPUT="$DATA_DIR"

# HF, Gemini, and Wandb credentials

# Debug: Print environment variables
echo "Wandb Key: ${WANDB_API_KEY:0:10}..."
echo "Cache directory: $CACHE_DIR"
echo "Data directory: $DATA_DIR"
echo "HF Token: ${HUGGING_FACE_HUB_TOKEN:0:10}..."
echo "Gemini Key: ${GEMINI_API_KEY:0:10}..."

