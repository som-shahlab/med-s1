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
#SBATCH --time=04:00:00
#SBATCH --account=nigam


if [ "$(whoami)" == "calebwin" ]; then
    export MED_S1_DIR="/share/pi/nigam/users/calebwin/med-s1"
elif [ "$(whoami)" == "mwornow" ]; then
    # # michael
    export MED_S1_DIR="/share/pi/nigam/mwornow/med-s1"
fi

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
source "${MED_S1_DIR}/config.sh" || { echo "Failed to source config.sh"; exit 1; }

# Get experiment config and parameters using Python script
echo "Resolving experiment configuration..."
resolved_config=$(python "${MED_S1_DIR}/eval/resolve_eval_config.py" \
    "$experiment_name" \
    --results-json "$RESULTS_JSON")

if [ $? -ne 0 ]; then
    echo "Error: Failed to resolve configuration"
    exit 1
fi

# Extract values from resolved config
config=$(echo "$resolved_config" | jq -r '.config')
model_key=$(echo "$resolved_config" | jq -r '.model_key')
model_path=$(echo "$resolved_config" | jq -r '.model_path')

# Check if test_time_scaling is enabled in config
config_test_time_scaling=$(echo "$resolved_config" | jq -r '.test_time_scaling')
if [ "$config_test_time_scaling" = "true" ]; then
    echo "Enabling test time scaling from config"
    test_time_scaling=true
fi

# Create logs directory
echo "Creating logs directory..."
mkdir -p "${MED_S1_DIR}/logs"

# Setup environment
echo "Setting up conda environment..."
source "${CONDA_PATH}" || { echo "Failed to source conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
if [ "$(whoami)" == "calebwin" ]; then
    conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }
elif [ "$(whoami)" == "mwornow" ]; then
    conda activate /local-scratch/nigam/users/mwornow/envs/meds1 || { echo "Failed to activate med-s1 environment"; exit 1; }
else
    echo "Unknown user: $(whoami)"
    exit 1
fi

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
echo "Output directory: ${output_dir}"

# Build eval.py command
# cmd="python ${MED_S1_DIR}/eval/eval.py \
#     --experiment_name ${experiment_name} \
#     --path_to_eval_json ${MED_S1_DIR}/eval/data/eval_data.json \
#     --path_to_output_dir ${output_dir}"
cmd="python ${MED_S1_DIR}/eval/eval.py \
    --experiment_name ${experiment_name} \
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