#!/bin/bash

# This script runs training for all experiments in results.json
# Usage: bash train_all.sh

# Base and huatuo don't need training (pre-trained models)

# # Data selection
# sbatch train/sft_carina.sh medqa-1k-embedding-diversity-question-cluster-10-outlier-5
# sbatch train/sft_carina.sh medqa-1k-embedding-difficulty-diversity-question-cluster-10-outlier-5

# # Not converging:
# sbatch train/sft_carina.sh medqa-1k-random-no-cot
# sbatch train/sft_carina.sh medqa-1k-random-1-sentence-extract
# sbatch train/sft_carina.sh medqa-5k-random-no-cot
# sbatch train/sft_carina.sh medqa-10k-random-no-cot
# sbatch train/sft_carina.sh medqa-25k

# Hyperparameter tuning
sbatch train/sft_carina.sh medqa-25k-no-cot
sbatch train/sft_carina.sh medqa-25k
sbatch train/sft_carina.sh medqa-25k-step-extract

# Case reports

# focus on fine-tuning these:
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-qwen
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-qwen-tuned
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-cot-extract-qwen
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-cot-extract-qwen-tuned
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-nejmcr-extract-qwen-tuned
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-nejmcr-extract-qwen-tuned-m1
sbatch --dependency=afterany:914057 train/sft_carina.sh medqa-nejmcr-1k-random-nejmcr-extract-qwen-tuned

sbatch train/sft_carina.sh medqa-nejmcr-1k-random-step-extract-qwen

sbatch train/sft_carina.sh medqa-nejmcr-1k-random
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-step-extract
sbatch train/sft_carina.sh medqa-nejmcr-1k-random-cot-extract

# Base reasoning LLM
sbatch train/sft_carina.sh medqa-1k-nemotron
sbatch train/sft_carina.sh medqa-1k-step-extract-nemotron
sbatch train/sft_carina.sh medqa-1k-qwen
sbatch train/sft_carina.sh medqa-1k-step-extract-qwen

# Extractions
sbatch train/sft_carina.sh medqa-1k-random
sbatch train/sft_carina.sh medqa-1k-random-step-extract
sbatch train/sft_carina.sh medqa-1k-random-evidence-extract
sbatch train/sft_carina.sh medqa-1k-random-markdown-extract
sbatch train/sft_carina.sh medqa-1k-random-list-extract
sbatch train/sft_carina.sh medqa-1k-random-note-extract
sbatch train/sft_carina.sh medqa-1k-random-qa-extract
sbatch train/sft_carina.sh medqa-1k-random-socratic-extract
sbatch train/sft_carina.sh medqa-1k-random-decision-tree-extract

# Perturbation experiments
# Collapse consecutive
sbatch train/sft_carina.sh medqa-1k-random-collapse-33
sbatch train/sft_carina.sh medqa-1k-random-collapse-66
sbatch train/sft_carina.sh medqa-1k-random-collapse-100

# Skip steps
sbatch train/sft_carina.sh medqa-1k-random-skip-33
sbatch train/sft_carina.sh medqa-1k-random-skip-66
sbatch train/sft_carina.sh medqa-1k-random-skip-100

# Shuffle steps
sbatch train/sft_carina.sh medqa-1k-random-shuffle-33
sbatch train/sft_carina.sh medqa-1k-random-shuffle-66
sbatch train/sft_carina.sh medqa-1k-random-shuffle-100

# Add irrelevant steps
sbatch train/sft_carina.sh medqa-1k-random-irrelevant-33
sbatch train/sft_carina.sh medqa-1k-random-irrelevant-66
sbatch train/sft_carina.sh medqa-1k-random-irrelevant-100

# Wrong answer
sbatch train/sft_carina.sh medqa-1k-random-wrong-answer-33
sbatch train/sft_carina.sh medqa-1k-random-wrong-answer-66
sbatch train/sft_carina.sh medqa-1k-random-wrong-answer-100

# Restoration experiments
# Collapse consecutive
sbatch train/sft_carina.sh medqa-1k-random-collapse-33-restore
sbatch train/sft_carina.sh medqa-1k-random-collapse-66-restore
sbatch train/sft_carina.sh medqa-1k-random-collapse-100-restore

# Skip steps
sbatch train/sft_carina.sh medqa-1k-random-skip-33-restore
sbatch train/sft_carina.sh medqa-1k-random-skip-66-restore
sbatch train/sft_carina.sh medqa-1k-random-skip-100-restore

# Shuffle steps
sbatch train/sft_carina.sh medqa-1k-random-shuffle-33-restore
sbatch train/sft_carina.sh medqa-1k-random-shuffle-66-restore
sbatch train/sft_carina.sh medqa-1k-random-shuffle-100-restore

# Add irrelevant steps
sbatch train/sft_carina.sh medqa-1k-random-irrelevant-33-restore
sbatch train/sft_carina.sh medqa-1k-random-irrelevant-66-restore
sbatch train/sft_carina.sh medqa-1k-random-irrelevant-100-restore