
#!/bin/bash

# Check if experiment name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

experiment_name=$1

# Submit training job
echo "Submitting training job for $experiment_name..."
train_job=$(sbatch train/sft_carina.sh "$experiment_name" | awk '{print $4}')
echo "Training job submitted with ID: $train_job"

# Submit eval job with dependency
echo "Submitting eval job (will run after training)..."
eval_job=$(sbatch --dependency=afterok:$train_job eval/eval.sh "$experiment_name" | awk '{print $4}')
echo "Eval job submitted with ID: $eval_job"

echo "Jobs submitted! Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/med-s1-train-$train_job.out"
echo "  tail -f logs/med-s1-eval-$eval_job.out"