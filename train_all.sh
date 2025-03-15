#!/bin/bash

# This script runs training for all experiments in results.json
# Usage: bash train_all.sh

# Base and huatuo don't need training (pre-trained models)

# Run training for each experiment
sbatch train/sft_carina.sh medqa-1k-random
sbatch train/sft_carina.sh medqa-1k-embedding-diversity-question-cluster-10-outlier-5
sbatch train/sft_carina.sh medqa-1k-embedding-difficulty-diversity-question-cluster-10-outlier-5
sbatch train/sft_carina.sh medqa-1k-random-step-extract
sbatch train/sft_carina.sh medqa-1k-random-no-cot
sbatch train/sft_carina.sh medqa-1k-random-1-sentence-extract
sbatch train/sft_carina.sh medqa-5k-random-no-cot
sbatch train/sft_carina.sh medqa-10k-random-no-cot
sbatch train/sft_carina.sh medqa-25k