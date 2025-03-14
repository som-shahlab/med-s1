#!/bin/bash
# Script to run evaluation for all experiments in results.json
# Usage: ./eval_all.sh [--debug] [--test-time-scaling]

# Parse arguments
debug=false
test_time_scaling=false

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
            echo "Unknown argument: $1"
            echo "Usage: $0 [--debug] [--test-time-scaling]"
            exit 1
            ;;
    esac
done

# Source configuration to get environment variables
echo "Sourcing config.sh..."
source "/share/pi/nigam/users/calebwin/med-s1/config.sh" || { echo "Failed to source config.sh"; exit 1; }

# Get all experiment names from results.json
echo "Extracting experiment names from results.json..."
experiment_names=$(jq -r '.experiments | keys[]' "$RESULTS_JSON")

# Count experiments
experiment_count=$(echo "$experiment_names" | wc -l)
echo "Found $experiment_count experiments to evaluate"

# Build debug and test-time-scaling flags
debug_flag=""
if [ "$debug" = true ]; then
    debug_flag="--debug"
    echo "Debug mode enabled for all evaluations"
fi

tts_flag=""
if [ "$test_time_scaling" = true ]; then
    tts_flag="--test-time-scaling"
    echo "Test time scaling enabled for all evaluations"
fi

# Create a log directory for this batch run
timestamp=$(date +"%Y%m%d_%H%M%S")
batch_log_dir="logs/eval_batch_${timestamp}"
mkdir -p "$batch_log_dir"
echo "Batch log directory: $batch_log_dir"

# Create a file to track job IDs
job_tracking_file="${batch_log_dir}/job_ids.txt"
touch "$job_tracking_file"

# Function to check if an experiment already has evaluation results
has_eval_results() {
    local exp_name=$1
    local has_results=$(jq -r ".experiments[\"$exp_name\"].results.eval" "$RESULTS_JSON")
    if [ "$has_results" != "null" ] && [ "$has_results" != "" ]; then
        return 0  # True, has results
    else
        return 1  # False, no results
    fi
}

# Function to check if an experiment has training results (needed for evaluation)
has_training_results() {
    local exp_name=$1
    local has_results=$(jq -r ".experiments[\"$exp_name\"].results.training" "$RESULTS_JSON")
    if [ "$has_results" != "null" ] && [ "$has_results" != "" ]; then
        return 0  # True, has training results
    else
        # Check if it's a base model that doesn't need training
        local model_key=$(jq -r ".experiments[\"$exp_name\"].config.model_key" "$RESULTS_JSON")
        if [ "$model_key" != "null" ] && [ "$model_key" != "" ]; then
            return 0  # True, has model_key
        else
            return 1  # False, no training results or model_key
        fi
    fi
}

# Submit evaluation jobs for each experiment
echo "Submitting evaluation jobs..."
for experiment_name in $experiment_names; do
    # Skip experiments that already have evaluation results unless in debug mode
    if [ "$debug" != true ] && has_eval_results "$experiment_name"; then
        echo "Skipping $experiment_name (already has evaluation results)"
        continue
    fi
    
    # Skip experiments without training results (except base models)
    if ! has_training_results "$experiment_name"; then
        echo "Skipping $experiment_name (no training results or model_key)"
        continue
    fi
    
    echo "Submitting evaluation job for $experiment_name..."
    
    # Submit the job
    job_id=$(sbatch $debug_flag $tts_flag "eval/eval.sh" "$experiment_name" | awk '{print $NF}')
    
    # Check if job submission was successful
    if [[ $job_id =~ ^[0-9]+$ ]]; then
        echo "  Job submitted with ID: $job_id"
        echo "$experiment_name: $job_id" >> "$job_tracking_file"
    else
        echo "  Failed to submit job for $experiment_name"
    fi
    
    # Add a small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All evaluation jobs submitted. Job IDs are recorded in $job_tracking_file"
echo "To check status of all jobs: cat $job_tracking_file | cut -d':' -f2 | xargs squeue -j"

# -------------------------------------------------------------------------
# Individual evaluation commands for copy-paste convenience
# -------------------------------------------------------------------------
# Uncomment and run individually as needed:

# sbatch eval/eval.sh huatuo-25k
# sbatch eval/eval.sh huatuo-1k-random
# sbatch eval/eval.sh huatuo-5k-random
# sbatch eval/eval.sh huatuo-100-random
# sbatch eval/eval.sh huatuo-1k-embedding-similarity-question
# sbatch eval/eval.sh huatuo-1k-novelty-answer
# sbatch eval/eval.sh huatuo
# sbatch eval/eval.sh base

# sbatch eval/eval.sh huatuo-1k-embedding-diversity-question-cluster-10-outlier-5
# sbatch eval/eval.sh huatuo-1k-embedding-diversity-cot-cluster-10-outlier-5
# sbatch eval/eval.sh huatuo-1k-difficulty-substring

# With test time scaling:
# sbatch --test-time-scaling eval/eval.sh huatuo-25k
# sbatch --test-time-scaling eval/eval.sh huatuo-1k-random
# sbatch --test-time-scaling eval/eval.sh huatuo-5k-random
# sbatch --test-time-scaling eval/eval.sh huatuo-100-random
# sbatch --test-time-scaling eval/eval.sh huatuo-1k-embedding-similarity-question
# sbatch --test-time-scaling eval/eval.sh huatuo-1k-embedding-diversity-question-cluster-10-outlier-5
# sbatch --test-time-scaling eval/eval.sh huatuo-1k-embedding-diversity-cot-cluster-10-outlier-5
# sbatch --test-time-scaling eval/eval.sh huatuo-1k-difficulty-substring
# sbatch --test-time-scaling eval/eval.sh huatuo-1k-novelty-answer
# sbatch --test-time-scaling eval/eval.sh huatuo
# sbatch --test-time-scaling eval/eval.sh base

# Debug mode (faster, uses fewer samples):
# sbatch --debug eval/eval.sh huatuo-25k