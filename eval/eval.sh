#!/bin/bash
#SBATCH --job-name=med-s1-eval
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-eval-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-eval-%j.err
#SBATCH --partition=nigam-h100
#SBATCH --constraint="GPU_SKU:H100_PCIE"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00  # Will be overridden if debug mode
#SBATCH --account=nigam

# Parse arguments and set time limit
debug=false
test_time_scaling=false
experiment_name=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            debug=true
            shift
            ;;
        --test-time-scaling)
            test_time_scaling=true
            shift
            ;;
        *)
            if [ -z "$experiment_name" ]; then
                experiment_name="$1"
            else
                echo "Usage: $0 [--debug] [--test-time-scaling] <experiment_name>"
                echo "Note: test_time_scaling will be automatically enabled if specified in experiment config"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$experiment_name" ]; then
    echo "Usage: $0 [--debug] [--test-time-scaling] <experiment_name>"
    echo "Note: test_time_scaling will be automatically enabled if specified in experiment config"
    exit 1
fi

# Source configuration first to get environment variables
echo "Sourcing config.sh..."
source "/share/pi/nigam/users/calebwin/med-s1/config.sh" || { echo "Failed to source config.sh"; exit 1; }

# Get experiment config from results.json
config=$(jq -r ".experiments[\"$experiment_name\"].config" "$RESULTS_JSON")

if [ "$config" = "null" ]; then
    echo "Error: Experiment '$experiment_name' not found in $RESULTS_JSON"
    exit 1
fi

# Check if test_time_scaling is enabled in config
config_test_time_scaling=$(jq -r ".experiments[\"$experiment_name\"].config.test_time_scaling" "$RESULTS_JSON")
if [ "$config_test_time_scaling" = "true" ]; then
    echo "Enabling test time scaling from config"
    test_time_scaling=true
fi

# Get model info based on experiment
# First check if there's a trained model path
model_path=$(jq -r ".experiments[\"$experiment_name\"].results.training.model_path" "$RESULTS_JSON")

if [ "$model_path" != "null" ] && [ -d "$model_path" ]; then
    echo "Using fine-tuned model: $model_path"
else
    # Fallback to pre-trained model path from config.json
    model_key=$(jq -r ".experiments[\"$experiment_name\"].config.model_key" "$RESULTS_JSON")
    if [ "$model_key" = "null" ]; then
        echo "Error: Neither training model_path nor model_key found in $RESULTS_JSON"
        exit 1
    fi
    model_path=$(jq -r ".models[\"$model_key\"].hf_path" "${MED_S1_DIR}/config.json")
    echo "Using pre-trained model: $model_path"
fi

# Only verify directory exists for fine-tuned models
if [[ "$model_path" == /* ]] && [ ! -d "$model_path" ]; then
    echo "Error: Fine-tuned model directory not found: $model_path"
    exit 1
fi

# Create logs directory
echo "Creating logs directory..."
mkdir -p "${MED_S1_DIR}/logs"

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

# Build eval.py command
cmd="python ${MED_S1_DIR}/eval/eval.py \
    --experiment_name ${experiment_name} \
    --model_path ${model_path} \
    --path_to_eval_json ${MED_S1_DIR}/eval/data/eval_data.json \
    --path_to_output_dir ${output_dir}"

# Add debug flags if in debug mode
if [ "$debug" = true ]; then
    cmd="$cmd --debug --debug_samples 1"
fi

# Add test time scaling flag if enabled
if [ "$test_time_scaling" = true ]; then
    cmd="$cmd --test_time_scaling"
fi

# Run evaluation
eval "$cmd"

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo "Evaluation complete!"