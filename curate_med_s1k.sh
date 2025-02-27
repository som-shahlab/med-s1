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

# Check if base dataset exists in hf_cache
if [ -d "$DATA_DIR/plumbing_test_001_20250219_145607" ] && [ -f "$DATA_DIR/plumbing_test_001_20250219_145607/med_s1k_filtered.parquet" ]; then
    echo "Base dataset already exists at $DATA_DIR/plumbing_test_001_20250219_145607"
    echo "Using CPU for processing since we only need to read and process existing data..."
    sbatch "${MED_S1_DIR}/data/curate_med_s1k_cpu.sh" "$experiment_name"
else
    echo "Base dataset needs to be created"
    echo "Using GPU for processing since we need to run model inference..."
    sbatch "${MED_S1_DIR}/data/curate_med_s1k_gpu.sh" "$experiment_name"
fi