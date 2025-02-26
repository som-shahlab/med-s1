#!/bin/bash
#SBATCH --job-name=med-s1-curate
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-curate-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-curate-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=28
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --account=nigam

# Source configuration
echo "Sourcing config.sh..."
source $(pwd)/config.sh || { echo "Failed to source config.sh"; exit 1; }

# Setup environment
echo "Setting up conda environment..."
source /share/pi/nigam/users/calebwin/nfs_conda.sh || { echo "Failed to source nfs_conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }

# Run curation script
python med-s1/data/curate_med_s1k.py --experiment "$1"