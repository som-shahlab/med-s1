#!/bin/bash

# Source conda and environment
source /share/pi/nigam/users/calebwin/nfs_conda.sh
conda activate med-s1

# Source config for environment variables
source config.sh

# Clear GPU cache
nvidia-smi --gpu-reset

# Set memory growth
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training with minimal settings for testing
python train/sft.py \
    --per_device_train_batch_size=1 \
    --max_steps=1 \
    --num_train_epochs=1 \
    --output_dir="ckpts/med-s1-test" \
    --eval_strategy="no" \
    --gradient_checkpointing=True \
    --save_steps=999999 \
    --logging_steps=999999