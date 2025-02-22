#!/bin/bash
#SBATCH --job-name=med-s1-train
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-train-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-train-%j.err
#SBATCH --partition=nigam-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --account=nigam

# Create logs directory
echo "Creating logs directory..."
mkdir -p /share/pi/nigam/users/calebwin/med-s1/logs

# Set training parameters
lr=1e-5
epochs=5
batch_size=8  # Reduced from 16 to be more conservative
weight_decay=1e-4
path_to_train_dataset="/share/pi/nigam/data/med_s1k/s1_replication/med_s1k_formatted"
dataset_name=$(basename "${path_to_train_dataset}")

uid="$(date +%Y%m%d_%H%M%S)"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --path_to_train_dataset) path_to_train_dataset="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

echo "Starting job..."

# Source configuration
echo "Sourcing config.sh..."
source $(pwd)/config.sh || { echo "Failed to source config.sh"; exit 1; }

# Setup environment
echo "Setting up conda environment..."
# source /share/pi/nigam/users/calebwin/nfs_conda.sh || { echo "Failed to source nfs_conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
# conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }

# Set environment variables
echo "Setting environment variables..."
# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NVLS_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Disable wandb
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Calculate gradient accumulation steps
gpu_count=$(nvidia-smi -L | wc -l) # use max number of GPUs on node
grad_acc=$((batch_size/gpu_count))

echo "Number of GPUs: $gpu_count"
echo "Gradient accumulation steps: $grad_acc"
echo "Memory per GPU: 80GB"

# Create local scratch directory for data
echo "Creating local scratch directory..."
LOCAL_DATA_DIR="/local-scratch/${SLURM_JOB_ID}"
mkdir -p $LOCAL_DATA_DIR || { echo "Failed to create local scratch directory"; exit 1; }

# Copy data to local scratch
echo "Copying data to local scratch..."
echo "  Source: ${path_to_train_dataset}"
echo "  Destination: $LOCAL_DATA_DIR"
cp -rv "${path_to_train_dataset}" $LOCAL_DATA_DIR/ || { echo "Failed to copy data"; exit 1; }

# Launch training
echo "Starting training..."
run_name="med_s1__${dataset_name}_bs${batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}_${uid}"
# Set master address for distributed training
master_addr=$(hostname)
master_port=29500
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
echo "Outputting to: ${CACHE_DIR}/ckpts/${run_name}"
echo "Train file path: ${LOCAL_DATA_DIR}/med_s1k_formatted"

torchrun \
    --nproc_per_node=$gpu_count \
    $(pwd)/train/sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=${epochs} \
    --train_file_path="${LOCAL_DATA_DIR}/med_s1k_formatted" \
    --model_name="meta-llama/Llama-3.1-8B-Instruct" \
    --warmup_ratio=0.05 \
    --report_to="none" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="$(pwd)/train/fsdp_config_llama_cpu.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="epoch" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="${CACHE_DIR}/ckpts/${run_name}" \
    --push_to_hub=false \
    --save_only_model=True \
    --ddp_find_unused_parameters=False \
    --ddp_timeout=3600

# Copy results back and cleanup
echo "Copying results back to shared storage..."
cp -r $LOCAL_DATA_DIR/ckpts/* "${CACHE_DIR}/ckpts/"

echo "Cleaning up local scratch..."
rm -rf $LOCAL_DATA_DIR

echo "Training complete!"