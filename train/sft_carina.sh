#!/bin/bash
#SBATCH --job-name=med-s1-train
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-train-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-train-%j.err
#SBATCH --partition=nigam-h100
#SBATCH --constraint="GPU_SKU:H100_PCIE"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --account=nigam

# Script for running distributed training on 4 H100 GPUs using DeepSpeed
# Usage: sbatch sft_carina.sh [--debug] <experiment_name>
# The experiment configuration is read from results.json
#
# Training configuration matches HuatuoGPT paper:
# - Total batch size: 128
#   * Per-GPU batch size: 2
#   * Number of GPUs: 4
#   * Gradient accumulation steps: 16
#   * 2 * 4 * 16 = 128 total batch size
# - Learning rate: 5e-6
# - Number of epochs: 3
# - Sequence length: 8192

set -e  # Exit on any error

# Parse arguments
debug=false
experiment_name=""

# Print all arguments for debugging
echo "Arguments received: $@"

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            debug=true
            echo "Debug mode enabled"
            shift
            ;;
        *)
            if [ -z "$experiment_name" ]; then
                experiment_name="$1"
                echo "Experiment name set to: $experiment_name"
            else
                echo "Error: Unexpected argument: $1"
                echo "Usage: $0 [--debug] <experiment_name>"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$experiment_name" ]; then
    echo "Error: No experiment name provided"
    echo "Usage: $0 [--debug] <experiment_name>"
    exit 1
fi

# Source configuration
echo "Sourcing config.sh..."
source "config.sh" || { echo "Failed to source config.sh"; exit 1; }

# Get experiment config from results.json
echo "Reading experiment configuration..."
config=$(jq -r ".experiments[\"$experiment_name\"].config" "$RESULTS_JSON")

if [ "$config" = "null" ]; then
    echo "Error: Experiment '$experiment_name' not found in $RESULTS_JSON"
    exit 1
fi

# Get model info
model_key=$(jq -r ".model_key" <<< "$config")
model=$(jq -r ".models[\"$model_key\"].hf_path" < "${MED_S1_DIR}/config.json")

# Create logs directory
echo "Creating logs directory..."
mkdir -p "${MED_S1_DIR}/logs"

# Get training params with defaults
echo "Extracting training parameters..."

# Required parameters (no defaults)
learning_rate=$(jq -r ".training_params.learning_rate" <<< "$config")
batch_size=$(jq -r ".training_params.batch_size" <<< "$config")
num_epochs=$(jq -r ".training_params.num_epochs" <<< "$config")
grad_acc=$(jq -r ".training_params.gradient_accumulation_steps" <<< "$config")

# Validate required parameters
if [ "$learning_rate" = "null" ] || [ "$batch_size" = "null" ] || [ "$num_epochs" = "null" ] || [ "$grad_acc" = "null" ]; then
    echo "Error: Missing required training parameters in results.json"
    echo "Required parameters:"
    echo "  learning_rate: $learning_rate"
    echo "  batch_size: $batch_size"
    echo "  num_epochs: $num_epochs"
    echo "  gradient_accumulation_steps: $grad_acc"
    exit 1
fi

# Verify batch size configuration matches paper
total_batch_size=$((batch_size * 4 * grad_acc))
if [ "$total_batch_size" -ne 128 ]; then
    echo "Warning: Total batch size ($total_batch_size) does not match paper (128)"
    echo "Check batch_size ($batch_size) and gradient_accumulation_steps ($grad_acc)"
fi

# Optional parameters with defaults
weight_decay=$(jq -r ".training_params.weight_decay // \"0.1\"" <<< "$config")
warmup_ratio=$(jq -r ".training_params.warmup_ratio // \"0.05\"" <<< "$config")

# Get optimizer params
adam_beta1=$(jq -r ".training_params.optimizer.adam_beta1 // \"0.9\"" <<< "$config")
adam_beta2=$(jq -r ".training_params.optimizer.adam_beta2 // \"0.95\"" <<< "$config")
adam_epsilon=$(jq -r ".training_params.optimizer.adam_epsilon // \"1e-8\"" <<< "$config")

# Print training configuration
echo -e "\nTraining Configuration:"
echo "Debug mode: $debug"
echo "Model:"
echo "  Key: $model_key"
echo "  Path: $model"
echo "Training Parameters:"
echo "  Learning Rate: $learning_rate"
echo "  Batch Size per GPU: $batch_size"
echo "  Number of GPUs: 4"
echo "  Gradient Accumulation Steps: $grad_acc"
echo "  Total Batch Size: $total_batch_size"
echo "  Number of Epochs: $num_epochs"
echo "  Weight Decay: $weight_decay"
echo "  Warmup Ratio: $warmup_ratio"
echo "  Sequence Length: 8192"
echo "Optimizer Parameters:"
echo "  Adam Beta1: $adam_beta1"
echo "  Adam Beta2: $adam_beta2"
echo "  Adam Epsilon: $adam_epsilon"

# Setup environment
echo "Setting up conda environment..."
source /share/pi/nigam/users/calebwin/nfs_conda.sh || { echo "Failed to source nfs_conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }

# Get network interface for NCCL
export NCCL_SOCKET_IFNAME=$(ip route show default | awk '/default/ {print $5}')
echo "Using network interface: $NCCL_SOCKET_IFNAME"

# Set NCCL environment variables for optimal multi-GPU performance
echo "Configuring NCCL..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_FAMILY=AF_INET  # Force IPv4
export NCCL_IB_HCA=mlx5_0:1
export NCCL_BUFFSIZE=2097152
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=5
export NCCL_SHM_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# Additional NCCL optimizations
export NCCL_IB_TC=106
export NCCL_IB_SL=3
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# DeepSpeed specific settings
echo "Configuring DeepSpeed..."
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_MIN_NCHANNELS=4
export DS_ACCELERATOR=cuda
export DS_ACCELERATOR_BACKEND=nccl

# Print NCCL configuration
echo "NCCL Configuration:"
echo "  Network interface: $NCCL_SOCKET_IFNAME"
echo "  Socket family: $NCCL_SOCKET_FAMILY"
echo "  IB HCA: $NCCL_IB_HCA"
echo "  Debug level: $NCCL_DEBUG"

# Get dataset path from results.json
echo "Getting dataset path..."
dataset_path=$(jq -r ".experiments[\"$experiment_name\"].results.curation.dataset_path" "$RESULTS_JSON")

if [ "$dataset_path" = "null" ]; then
    echo "Error: Dataset path not found in $RESULTS_JSON. Has curation been run?"
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

# Set output directory and clean it
safe_experiment_name=$(echo "$experiment_name" | sed 's/[^a-zA-Z0-9-]//g')
checkpoint_dir="${CACHE_DIR}/ckpts/${safe_experiment_name}"

echo "Setting up checkpoint directory: $checkpoint_dir"
if [ -d "$checkpoint_dir" ]; then
    echo "Cleaning up existing checkpoint directory"
    rm -rf "$checkpoint_dir"
fi
mkdir -p "$checkpoint_dir"

# Debug mode overrides
if [ "$debug" = true ]; then
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>>>>> USING DEBUG MODE <<<<<<"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    model="meta-llama/Llama-3.2-1B"
    num_epochs=2
    export CUDA_VISIBLE_DEVICES=0
fi

# Launch training
echo "Starting training..."
gpu_count=$(nvidia-smi -L | wc -l)

# Check wandb API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY not set in config.sh"
    exit 1
fi

# Check accelerate config exists
accelerate_config="${MED_S1_DIR}/train/accelerate_config.yaml"
if [ ! -f "$accelerate_config" ]; then
    echo "Error: Accelerate config not found at $accelerate_config"
    exit 1
fi

# Get a random port in a safer range (avoid common ports)
MASTER_PORT=$(shuf -i 40000-45000 -n 1)

# Change to med-s1 directory and add to Python path
cd "${MED_S1_DIR}"
export PYTHONPATH="${MED_S1_DIR}:${PYTHONPATH}"

# Build accelerate command
# Note: Only add --debug flag if debug mode is enabled
debug_flag=""
if [ "$debug" = true ]; then
    debug_flag="--debug"
fi

# Launch training command with Accelerate
echo "Launching training with accelerate..."
cmd="accelerate launch \
    --config_file \"${accelerate_config}\" \
    --main_process_port $MASTER_PORT \
    train/sft.py \
    --experiment_name=\"${experiment_name}\" \
    --results_json=\"${RESULTS_JSON}\" \
    --model_name=\"${model}\" \
    --train_file_path=\"${LOCAL_DATA_DIR}/med_s1k_formatted\" \
    --output_dir=\"${checkpoint_dir}\" \
    --block_size=8192 \
    --per_device_train_batch_size=${batch_size} \
    --gradient_accumulation_steps=${grad_acc} \
    --learning_rate=${learning_rate} \
    --weight_decay=${weight_decay} \
    --warmup_ratio=${warmup_ratio} \
    --adam_beta1=${adam_beta1} \
    --adam_beta2=${adam_beta2} \
    --adam_epsilon=${adam_epsilon} \
    --num_train_epochs=${num_epochs} \
    $debug_flag"

# Print launch configuration
echo -e "\nAccelerate Launch Configuration:"
echo "Command: $cmd"

# Execute command
echo "Running command: $cmd"
eval "$cmd"

# Cleanup local scratch
if [ -d "/local-scratch" ]; then
    echo "Cleaning up local scratch..."
    rm -rf $LOCAL_DATA_DIR
fi

# Update results.json with model path and timestamp
model_path="${checkpoint_dir}"
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Function to update results.json with retries
update_results_json() {
    local max_retries=5
    local retry_delay=1
    local attempt=1
    
    while [ $attempt -le $max_retries ]; do
        echo "Attempting to update results.json (attempt $attempt/$max_retries)..."
        
        # Create a unique temporary file
        local tmp_file="${RESULTS_JSON}.tmp.$$"
        
        # Read the latest version and update it
        if jq --arg path "$model_path" --arg time "$timestamp" \
            ".experiments[\"$experiment_name\"].results.training = {
                \"model_path\": \$path,
                \"timestamp\": \$time,
                \"metrics\": null
            }" \
            "$RESULTS_JSON" > "$tmp_file"; then
            
            # Atomic move of the temp file
            if mv "$tmp_file" "$RESULTS_JSON"; then
                echo "Successfully updated results.json"
                return 0
            fi
        fi
        
        # Clean up temp file if it exists
        [ -f "$tmp_file" ] && rm "$tmp_file"
        
        echo "Failed to update results.json, retrying in ${retry_delay}s..."
        sleep $retry_delay
        attempt=$((attempt + 1))
    done
    
    echo "Failed to update results.json after $max_retries attempts"
    return 1
}

# Try to update results.json
if ! update_results_json; then
    echo "Error: Failed to update results.json"
    exit 1
fi

echo "Training complete!"