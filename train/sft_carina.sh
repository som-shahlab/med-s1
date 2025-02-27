#!/bin/bash
#SBATCH --job-name=med-s1-train
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-train-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-train-%j.err
#SBATCH --partition=gpu-long
#SBATCH --constraint="GPU_SKU:A100_PCIE"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --account=nigam

# Check if experiment name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

experiment_name=$1

# Source configuration first to get environment variables
echo "Sourcing config.sh..."
source "/share/pi/nigam/users/calebwin/med-s1/config.sh" || { echo "Failed to source config.sh"; exit 1; }

# Get experiment config from results.json
config=$(jq -r ".experiments[\"$experiment_name\"].config" "$RESULTS_JSON")

if [ "$config" = "null" ]; then
    echo "Error: Experiment '$experiment_name' not found in $RESULTS_JSON"
    exit 1
fi

# Get model key from config
model_key=$(jq -r ".model_key" <<< "$config")
model=$(jq -r ".models[\"$model_key\"].hf_path" < "${MED_S1_DIR}/config.json")

# Create logs directory
echo "Creating logs directory..."
mkdir -p "${MED_S1_DIR}/logs"

# Get training params
learning_rate=$(jq -r ".training_params.learning_rate" <<< "$config")
batch_size=$(jq -r ".training_params.batch_size" <<< "$config")
num_epochs=$(jq -r ".training_params.num_epochs" <<< "$config")
weight_decay=$(jq -r ".training_params.weight_decay // \"1e-4\"" <<< "$config")  # Default to 1e-4 if not set

# Set strategy
strategy="fsdp"
uid="$(date +%Y%m%d_%H%M%S)"
echo "Starting job..."

# Setup environment
echo "Setting up conda environment..."
source /share/pi/nigam/users/calebwin/nfs_conda.sh || { echo "Failed to source nfs_conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }

# Set environment variables
echo "Setting environment variables..."

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NVLS_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=8

# CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable wandb
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Calculate gradient accumulation steps based on batch size and GPU count
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$((batch_size/gpu_count))

echo "Number of GPUs: $gpu_count"
echo "Gradient accumulation steps: $grad_acc"
echo "Memory per GPU: 80GB"

# Get dataset path from curation results
dataset_path=$(jq -r ".experiments[\"$experiment_name\"].results.curation.dataset_path" "$RESULTS_JSON")

if [ "$dataset_path" = "null" ]; then
    echo "Error: Dataset path not found in $RESULTS_JSON. Has curation been run for this experiment?"
    exit 1
fi

# Set up data directory
if [ -d "/local-scratch" ]; then
    echo "Using local scratch directory..."
    LOCAL_DATA_DIR="/local-scratch/${SLURM_JOB_ID}"
    mkdir -p $LOCAL_DATA_DIR || { echo "Failed to create local scratch directory"; exit 1; }
    
    # Copy data to local scratch
    echo "Copying data to local scratch..."
    echo "  Source: ${dataset_path}"
    echo "  Destination: $LOCAL_DATA_DIR"
    cp -r "${dataset_path}" $LOCAL_DATA_DIR/ || { echo "Failed to copy data"; exit 1; }
else
    echo "Local scratch not available, using dataset directory directly..."
    LOCAL_DATA_DIR=$(dirname "$dataset_path")
fi

# Launch training
echo "Starting training..."

# Clean experiment name same way as curation (remove all non-alphanumeric except hyphens)
safe_experiment_name=$(echo "$experiment_name" | sed 's/[^a-zA-Z0-9-]//g')
checkpoint_dir="${CACHE_DIR}/ckpts/${safe_experiment_name}"

# Set master address for distributed training
master_addr=$(hostname)
master_port=29500
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port

echo "Outputting to: ${checkpoint_dir}"
echo "Train file path: ${LOCAL_DATA_DIR}/med_s1k_formatted"

if [ "$strategy" = "fsdp" ]; then
        # --block_size=32768 \
    torchrun \
        --nproc_per_node=$gpu_count \
        "${MED_S1_DIR}/train/sft.py" \
        --block_size=4096 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=$grad_acc \
        --num_train_epochs=${num_epochs} \
        --train_file_path="${LOCAL_DATA_DIR}/med_s1k_formatted" \
        --model_name="${model}" \
        --warmup_ratio=0.05 \
        --report_to="none" \
        --bf16=False \
        --fp16=False \
        --eval_strategy="no" \
        --logging_steps=100 \
        --save_strategy="no" \
        --lr_scheduler_type="cosine" \
        --learning_rate=${learning_rate} \
        --weight_decay=${weight_decay} \
        --adam_beta1=0.9 \
        --adam_beta2=0.95 \
        --output_dir="${checkpoint_dir}" \
        --push_to_hub=false \
        --save_only_model=True \
        --save_safetensors=True \
        --ddp_find_unused_parameters=False \
        --ddp_timeout=3600 \
        --fsdp="full_shard auto_wrap" \
        --fsdp_config="${MED_S1_DIR}/train/fsdp_config_llama_cpu.json"
else
    # Non-FSDP training path
    torchrun \
        --nproc_per_node=$gpu_count \
        "${MED_S1_DIR}/train/sft.py" \
        --block_size=4096 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=$grad_acc \
        --num_train_epochs=${num_epochs} \
        --train_file_path="${LOCAL_DATA_DIR}/med_s1k_formatted" \
        --model_name="${model}" \
        --warmup_ratio=0.05 \
        --report_to="none" \
        --bf16=True \
        --eval_strategy="no" \
        --logging_steps=100 \
        --save_strategy="epoch" \
        --lr_scheduler_type="cosine" \
        --learning_rate=${learning_rate} \
        --weight_decay=${weight_decay} \
        --adam_beta1=0.9 \
        --adam_beta2=0.95 \
        --output_dir="${checkpoint_dir}" \
        --push_to_hub=false \
        --save_only_model=True \
        --ddp_find_unused_parameters=False \
        --ddp_timeout=3600
fi

# Cleanup local scratch if we created it
if [ -d "/local-scratch" ]; then
    echo "Cleaning up local scratch..."
    rm -rf $LOCAL_DATA_DIR
fi

# Update results.json with model path and timestamp
model_path="${checkpoint_dir}"
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Use jq to update results.json
jq --arg path "$model_path" --arg time "$timestamp" \
  ".experiments[\"$experiment_name\"].results.training = {
    \"model_path\": \$path,
    \"timestamp\": \$time,
    \"metrics\": null
  }" \
  "$RESULTS_JSON" > "${RESULTS_JSON}.tmp" && mv "${RESULTS_JSON}.tmp" "$RESULTS_JSON"

echo "Training complete!"