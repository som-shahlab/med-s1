#!/bin/bash
#SBATCH --job-name=med-s1-curate
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-curate-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/med-s1-curate-%j.err
#SBATCH --partition=nigam-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --time=06:00:00
#SBATCH --account=nigam

# Set paths
if [ "$(whoami)" == "calebwin" ]; then
    MED_S1_DIR="/share/pi/nigam/users/calebwin/med-s1"
    CONDA_PATH="/share/pi/nigam/users/calebwin/nfs_conda.sh"
elif [ "$(whoami)" == "mwornow" ]; then
    MED_S1_DIR="/share/pi/nigam/mwornow/med-s1"
    CONDA_PATH="/share/pi/nigam/mwornow/conda.sh"
else
    echo "Unknown user: $(whoami)"
    exit 1
fi

# Source configuration first to get environment variables
echo "Sourcing config.sh..."
source "${MED_S1_DIR}/config.sh" || { echo "Failed to source config.sh"; exit 1; }

# Setup environment
echo "Setting up conda environment..."
source "${CONDA_PATH}" || { echo "Failed to source conda.sh"; exit 1; }
echo "Activating med-s1 environment..."
conda activate med-s1 || { echo "Failed to activate med-s1 environment"; exit 1; }

# Run curation script
python "${MED_S1_DIR}/data/curate_med_s1k_new.py" --experiment "$1"