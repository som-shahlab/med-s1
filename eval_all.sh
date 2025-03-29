#!/bin/bash

# This script runs evaluation for all experiments in results.json
# Usage: bash eval_all.sh

# # Baselines an data selection
# sbatch eval/eval.sh base
# sbatch eval/eval.sh huatuo
# sbatch eval/eval.sh medqa-1k-random
# sbatch eval/eval.sh medqa-1k-embedding-diversity-question-cluster-10-outlier-5
# sbatch eval/eval.sh medqa-1k-embedding-difficulty-diversity-question-cluster-10-outlier-5
# sbatch eval/eval.sh medqa-1k-random-step-extract
# sbatch eval/eval.sh medqa-1k-random-no-cot
# sbatch eval/eval.sh medqa-1k-random-1-sentence-extract
# sbatch eval/eval.sh medqa-5k-random-no-cot
# sbatch eval/eval.sh medqa-10k-random-no-cot
# sbatch eval/eval.sh medqa-25k

# Extractions
sbatch eval/eval.sh medqa-1k-random
sbatch eval/eval.sh medqa-1k-random-step-extract
sbatch eval/eval.sh medqa-1k-random-evidence-extract
sbatch eval/eval.sh medqa-1k-random-markdown-extract
sbatch eval/eval.sh medqa-1k-random-list-extract
sbatch eval/eval.sh medqa-1k-random-note-extract
sbatch eval/eval.sh medqa-1k-random-qa-extract
sbatch eval/eval.sh medqa-1k-random-socratic-extract
sbatch eval/eval.sh medqa-1k-random-decision-tree-extract

# Perturbation experiments
# Collapse consecutive
sbatch eval/eval.sh medqa-1k-random-collapse-33
sbatch eval/eval.sh medqa-1k-random-collapse-66
sbatch eval/eval.sh medqa-1k-random-collapse-100

# Skip steps
sbatch eval/eval.sh medqa-1k-random-skip-33
sbatch eval/eval.sh medqa-1k-random-skip-66
sbatch eval/eval.sh medqa-1k-random-skip-100

# Shuffle steps
sbatch eval/eval.sh medqa-1k-random-shuffle-33
sbatch eval/eval.sh medqa-1k-random-shuffle-66
sbatch eval/eval.sh medqa-1k-random-shuffle-100

# Add irrelevant steps
sbatch eval/eval.sh medqa-1k-random-irrelevant-33
sbatch eval/eval.sh medqa-1k-random-irrelevant-66
sbatch eval/eval.sh medqa-1k-random-irrelevant-100

# Wrong answer
sbatch eval/eval.sh medqa-1k-random-wrong-answer-33
sbatch eval/eval.sh medqa-1k-random-wrong-answer-66
sbatch eval/eval.sh medqa-1k-random-wrong-answer-100

# Restoration experiments
# Collapse consecutive
sbatch eval/eval.sh medqa-1k-random-collapse-33-restore
sbatch eval/eval.sh medqa-1k-random-collapse-66-restore
sbatch eval/eval.sh medqa-1k-random-collapse-100-restore

# Skip steps
sbatch eval/eval.sh medqa-1k-random-skip-33-restore
sbatch eval/eval.sh medqa-1k-random-skip-66-restore
sbatch eval/eval.sh medqa-1k-random-skip-100-restore

# Shuffle steps
sbatch eval/eval.sh medqa-1k-random-shuffle-33-restore
sbatch eval/eval.sh medqa-1k-random-shuffle-66-restore
sbatch eval/eval.sh medqa-1k-random-shuffle-100-restore

# Add irrelevant steps
sbatch eval/eval.sh medqa-1k-random-irrelevant-33-restore
sbatch eval/eval.sh medqa-1k-random-irrelevant-66-restore
sbatch eval/eval.sh medqa-1k-random-irrelevant-100-restore