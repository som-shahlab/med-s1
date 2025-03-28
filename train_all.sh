#!/bin/bash

# This script runs training for all experiments in results.json
# Usage: bash train_all.sh

# Base and huatuo don't need training (pre-trained models)

# Run training for each experiment
sbatch train/sft_carina.sh medqa-1k-random
sbatch train/sft_carina.sh medqa-1k-embedding-diversity-question-cluster-10-outlier-5
sbatch train/sft_carina.sh medqa-1k-embedding-difficulty-diversity-question-cluster-10-outlier-5
sbatch train/sft_carina.sh medqa-1k-random-step-extract


# Not converging:
sbatch train/sft_carina.sh medqa-1k-random-no-cot
sbatch train/sft_carina.sh medqa-1k-random-1-sentence-extract
sbatch train/sft_carina.sh medqa-5k-random-no-cot
sbatch train/sft_carina.sh medqa-10k-random-no-cot
sbatch train/sft_carina.sh medqa-25k

# Ablations
sbatch train/sft_carina.sh curate_med_s1k.sh medqa-1k-random-step-extract
sbatch train/sft_carina.sh curate_med_s1k.sh medqa-1k-random-evidence-extract
sbatch train/sft_carina.sh curate_med_s1k.sh medqa-1k-random-markdown-extract
sbatch train/sft_carina.sh curate_med_s1k.sh medqa-1k-random-list-extract
sbatch train/sft_carina.sh curate_med_s1k.sh medqa-1k-random-note-extract

# Run these commands after job 89838 completes
sbatch train/sft_carina.sh medqa-1k-random-step-extract
sbatch train/sft_carina.sh medqa-1k-random-evidence-extract
sbatch train/sft_carina.sh medqa-1k-random-markdown-extract
sbatch train/sft_carina.sh medqa-1k-random-list-extract
sbatch train/sft_carina.sh medqa-1k-random-note-extract
