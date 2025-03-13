#!/bin/bash

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

# Create logs directory if it doesn't exist
mkdir -p "${MED_S1_DIR}/logs"

# Export experiment name for Python script
export EXPERIMENT_NAME="$experiment_name"

# Use the new Python script to determine whether to use GPU or CPU
echo "Determining whether to use GPU or CPU..."
device=$(python "${MED_S1_DIR}/select_curation_device.py" --experiment "$experiment_name")

if [ "$device" = "cpu" ]; then
    echo "Using CPU for processing..."
    sbatch "${MED_S1_DIR}/data/curate_med_s1k_cpu.sh" "$experiment_name"
else
    echo "Using GPU for processing..."
    sbatch "${MED_S1_DIR}/data/curate_med_s1k_gpu.sh" "$experiment_name"
fi