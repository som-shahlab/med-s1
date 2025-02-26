#!/bin/bash

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

# Create logs directory if it doesn't exist
mkdir -p /share/pi/nigam/users/calebwin/med-s1/logs

# Check if base dataset exists in hf_cache
dataset_dir="/share/pi/nigam/users/calebwin/hf_cache/med-s1k"
if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/med_s1k_filtered.parquet" ]; then
    echo "Base dataset already exists at $dataset_dir"
    echo "Using CPU for processing since we only need to read and process existing data..."
    sbatch med-s1/data/curate_med_s1k_cpu.sh "$experiment_name"
else
    echo "Base dataset needs to be created"
    echo "Using GPU for processing since we need to run model inference..."
    sbatch med-s1/data/curate_med_s1k_gpu.sh "$experiment_name"
fi