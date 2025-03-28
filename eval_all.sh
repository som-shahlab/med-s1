#!/bin/bash

# This script runs evaluation for all experiments in results.json
# Usage: bash eval_all.sh

# Run evaluation for each experiment
sbatch eval/eval.sh base
sbatch eval/eval.sh huatuo
sbatch eval/eval.sh medqa-1k-random
sbatch eval/eval.sh medqa-1k-embedding-diversity-question-cluster-10-outlier-5
sbatch eval/eval.sh medqa-1k-embedding-difficulty-diversity-question-cluster-10-outlier-5
sbatch eval/eval.sh medqa-1k-random-step-extract
sbatch eval/eval.sh medqa-1k-random-no-cot
sbatch eval/eval.sh medqa-1k-random-1-sentence-extract
sbatch eval/eval.sh medqa-5k-random-no-cot
sbatch eval/eval.sh medqa-10k-random-no-cot
sbatch eval/eval.sh medqa-25k

sbatch eval/eval.sh medqa-1k-random-step-extract
sbatch eval/eval.sh medqa-1k-random-evidence-extract
sbatch eval/eval.sh medqa-1k-random-markdown-extract
sbatch eval/eval.sh medqa-1k-random-list-extract
sbatch eval/eval.sh medqa-1k-random-note-extract