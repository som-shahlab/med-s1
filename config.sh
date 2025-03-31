#!/bin/bash

# Set paths
if [ "$(whoami)" == "calebwin" ]; then
    export MED_S1_DIR="/share/pi/nigam/users/calebwin/med-s1"
    export CACHE_DIR="/share/pi/nigam/users/calebwin/hf_cache"
    export DATA_DIR="/share/pi/nigam/users/calebwin/hf_cache/med-s1k"
    export RESULTS_JSON="/share/pi/nigam/users/calebwin/med-s1/results.json"
    export CONDA_PATH="/share/pi/nigam/users/calebwin/nfs_conda.sh"
elif [ "$(whoami)" == "mwornow" ]; then
    # # michael
    export MED_S1_DIR="/share/pi/nigam/mwornow/med-s1"
    export CACHE_DIR="/share/pi/nigam/mwornow/hf_cache"
    export DATA_DIR="/share/pi/nigam/mwornow/hf_cache/med-s1k"
    export RESULTS_JSON="/share/pi/nigam/mwornow/med-s1/results.json"
    export CONDA_PATH="/share/pi/nigam/mwornow/conda.sh"
else
    echo "Unknown user: $(whoami)"
    exit 1
fi

# Export paths for scripts
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

