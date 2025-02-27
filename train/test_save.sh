#!/bin/bash
#SBATCH --job-name=test-save
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/test-save-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/test-save-%j.err
#SBATCH --partition=gpu
#SBATCH --constraint="GPU_SKU:A100_PCIE"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --account=nigam

# Source configuration first to get environment variables
echo "Sourcing config.sh..."
source "/share/pi/nigam/users/calebwin/med-s1/config.sh" || { echo "Failed to source config.sh"; exit 1; }

# Setup environment
echo "Setting up conda environment..."
source /share/pi/nigam/users/calebwin/nfs_conda.sh || { echo "Failed to source nfs_conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }

# Get model path from config
model_key="llama3.1:8b"  # Using same model key as in config.json
model=$(jq -r ".models[\"$model_key\"].hf_path" < "${MED_S1_DIR}/config.json")

echo "Using model: $model"

# Set checkpoint directory
checkpoint_dir="${CACHE_DIR}/ckpts/test-save"

# Create checkpoint directory and add dummy shard files to simulate pre-existing state
mkdir -p "$checkpoint_dir"
for i in {1..4}; do
    touch "${checkpoint_dir}/model-0000${i}-of-00004.safetensors"
done

# Create a tiny test dataset
TEST_DATA_DIR="/share/pi/nigam/users/calebwin/med-s1/test_data"
mkdir -p "$TEST_DATA_DIR"

# Create minimal test dataset
python3 << END
from datasets import Dataset, DatasetDict
import json

# Create tiny dataset with just 2 examples
data = {
    'train': Dataset.from_dict({
        'text': [
            'user: What is diabetes?\nassistant: Diabetes is a chronic condition affecting blood sugar levels.\n\n',
            'user: What are symptoms of a cold?\nassistant: Common cold symptoms include runny nose, cough, and sore throat.\n\n'
        ]
    }),
    'test': Dataset.from_dict({
        'text': [
            'user: What is hypertension?\nassistant: Hypertension is high blood pressure.\n\n'
        ]
    })
}

# Save as DatasetDict
dataset = DatasetDict(data)
dataset.save_to_disk('${TEST_DATA_DIR}/tiny_test')
END

# Debug mode settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

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

# Set master address for distributed training
master_addr=$(hostname)
master_port=29500
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port

# Launch training with minimal epochs and batch size
cmd="torchrun \
    --nproc_per_node=4 \
    \"${MED_S1_DIR}/train/sft.py\" \
    --block_size=4096 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=1 \
    --train_file_path=\"${TEST_DATA_DIR}/tiny_test\" \
    --model_name=\"${model}\" \
    --warmup_ratio=0.05 \
    --report_to=\"none\" \
    --eval_strategy=\"no\" \
    --lr_scheduler_type=\"cosine\" \
    --learning_rate=1e-5 \
    --weight_decay=1e-4 \
    --output_dir=\"${checkpoint_dir}\" \
    --push_to_hub=false \
    --save_only_model=True \
    --save_safetensors=True \
    --ddp_find_unused_parameters=False \
    --ddp_timeout=3600 \
    --fsdp=\"full_shard auto_wrap\" \
    --fsdp_config=\"${MED_S1_DIR}/train/fsdp_config_llama_cpu.json\" \
    --save_strategy=no \
    --logging_steps=1"

# Execute command
echo "Running command: $cmd"
eval "$cmd"

# Cleanup test data
rm -rf "$TEST_DATA_DIR"