#!/bin/bash
#SBATCH --job-name=test-llm
#SBATCH --output=/share/pi/nigam/users/calebwin/med-s1/logs/test-llm-%j.out
#SBATCH --error=/share/pi/nigam/users/calebwin/med-s1/logs/test-llm-%j.err
#SBATCH --partition=gpu-long
#SBATCH --constraint="GPU_SKU:A100_PCIE"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=nigam

# Source configuration
source "/share/pi/nigam/users/calebwin/med-s1/config.sh"

# Setup environment
source /share/pi/nigam/users/calebwin/nfs_conda.sh
conda activate med-s1

# Run test
python /share/pi/nigam/users/calebwin/med-s1/test_llm.py