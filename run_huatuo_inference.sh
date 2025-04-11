#!/bin/bash
#SBATCH --job-name=huatuo-inference
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/huatuo-inference-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/huatuo-inference-%j.err
#SBATCH --partition=nigam-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=nigam

if [ "$(whoami)" == "calebwin" ]; then
    export MED_S1_DIR="/share/pi/nigam/users/calebwin/med-s1"
elif [ "$(whoami)" == "mwornow" ]; then
    export MED_S1_DIR="/share/pi/nigam/mwornow/med-s1"
fi

# Source configuration to get environment variables
echo "Sourcing config.sh..."
source "${MED_S1_DIR}/config.sh" || { echo "Failed to source config.sh"; exit 1; }

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

# Set CUDA environment
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Install sglang if not already installed
pip install -q sglang || { echo "Failed to install sglang"; exit 1; }

# Run inference
python ${MED_S1_DIR}/run_huatuo_inference.py

echo "Inference complete!"