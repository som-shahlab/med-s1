#!/bin/bash
#SBATCH --job-name=med-s1k-curation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --time=06:00:00
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/curation_%j.log
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/curation_%j.err

# Debug mode
set -x

# Create logs directory
LOGS_DIR="/share/pi/nigam/users/calebwin/med-s1/logs"
mkdir -p $LOGS_DIR
chmod 755 $LOGS_DIR

# Print detailed environment info
echo "=================== Job Start ==================="
date
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Current dir: $(pwd)"
nvidia-smi
echo "=============================================="

# Load conda environment
echo "Loading conda environment..."
cd /share/pi/nigam/users/calebwin/
source /share/pi/nigam/users/calebwin/nfs_conda.sh
conda activate med-s1
which python
python --version
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Go to project directory and run
echo "Changing to project directory..."
cd /share/pi/nigam/users/calebwin/med-s1

echo "Loading configuration..."
source config.sh

echo "Starting curation script..."
# Enable Python unbuffered output
export PYTHONUNBUFFERED=1
# Run the script
python data/curate_med_s1k.py

echo "Job finished at: $(date)"

# To monitor this job:
# 1. View job status:    squeue -j $SLURM_JOB_ID
# 2. Watch logs:         tail -f $LOGS_DIR/curation_$SLURM_JOB_ID.log
# 3. GPU usage:          srun --jobid $SLURM_JOB_ID --pty nvidia-smi
# 4. Cancel if needed:   scancel $SLURM_JOB_ID