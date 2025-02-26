#!/bin/bash
#SBATCH --job-name=med-s1-eval
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-eval-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-eval-%j.err
#SBATCH --partition=nigam-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=nigam

# Check if experiment name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

experiment_name=$1

# Get experiment config from results.json
config=$(jq -r ".experiments[\"$experiment_name\"].config" med-s1/results.json)

if [ "$config" = "null" ]; then
    echo "Error: Experiment '$experiment_name' not found in results.json"
    exit 1
fi

# Get model info based on experiment
if [ "$experiment_name" = "base" ]; then
    # For base experiment, get model path from config.json
    model_key=$(jq -r ".experiments[\"$experiment_name\"].config.model_key" med-s1/results.json)
    model_path=$(jq -r ".models[\"$model_key\"].hf_path" med-s1/config.json)
    echo "Using base model: $model_path"
else
    # For other experiments, get model path from training results
    model_path=$(jq -r ".experiments[\"$experiment_name\"].results.training.model_path" med-s1/results.json)
    if [ "$model_path" = "null" ]; then
        echo "Error: Model path not found in results.json. Has training been completed for this experiment?"
        exit 1
    fi
fi

# Create logs directory
echo "Creating logs directory..."
mkdir -p /share/pi/nigam/users/calebwin/med-s1/logs

# Source configuration
echo "Sourcing config.sh..."
source $(pwd)/config.sh || { echo "Failed to source config.sh"; exit 1; }

# Setup environment
echo "Setting up conda environment..."
source /share/pi/nigam/users/calebwin/nfs_conda.sh || { echo "Failed to source nfs_conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }

# Clean experiment name for filenames
safe_experiment_name=$(echo "$experiment_name" | sed 's/[^a-zA-Z0-9-]//g')

# Create output directory
output_dir="${CACHE_DIR}/eval/${safe_experiment_name}"
mkdir -p "$output_dir"

# Set CUDA environment
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Run evaluation using vllm
echo "Starting evaluation..."
echo "Model path: ${model_path}"
echo "Output directory: ${output_dir}"

python $(pwd)/eval/eval.py \
    --experiment_name "${experiment_name}" \
    --model_path "${model_path}" \
    --path_to_eval_json "$(pwd)/eval/data/eval_data.json" \
    --path_to_output_dir "${output_dir}" \
    --use_vllm true \
    --batch_size 1024

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo "Evaluation complete!"