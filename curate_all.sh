#!/bin/bash

# This script runs curation for all experiments in results.json
# Usage: bash curate_all.sh

# Base and huatuo don't need curation (pre-trained models)

# # Data selection
# bash curate_med_s1k.sh medqa-1k-random
# bash curate_med_s1k.sh medqa-1k-embedding-diversity-question-cluster-10-outlier-5
# bash curate_med_s1k.sh medqa-1k-embedding-difficulty-diversity-question-cluster-10-outlier-5 # (*)
# bash curate_med_s1k.sh medqa-25k
# bash curate_med_s1k.sh medqa-1k-random-step-extract # (*)
# bash curate_med_s1k.sh medqa-1k-random-no-cot # (*)
# bash curate_med_s1k.sh medqa-1k-random-1-sentence-extract # (*)
# bash curate_med_s1k.sh medqa-5k-random-no-cot
# bash curate_med_s1k.sh medqa-10k-random-no-cot

# Case reports
bash curate_med_s1k.sh medqa-nejmcr-1k-random
bash curate_med_s1k.sh medqa-nejmcr-1k-random-step-extract
bash curate_med_s1k.sh medqa-nejmcr-1k-random-cot-extract

# Base reasoning LLM
bash curate_med_s1k.sh medqa-1k-nemotron
bash curate_med_s1k.sh medqa-1k-step-extract-nemotron
bash curate_med_s1k.sh medqa-1k-qwen
bash curate_med_s1k.sh medqa-1k-step-extract-qwen

# Extractions
bash curate_med_s1k.sh medqa-1k-random
bash curate_med_s1k.sh medqa-1k-random-step-extract
bash curate_med_s1k.sh medqa-1k-random-evidence-extract
bash curate_med_s1k.sh medqa-1k-random-markdown-extract
bash curate_med_s1k.sh medqa-1k-random-list-extract
bash curate_med_s1k.sh medqa-1k-random-note-extract
bash curate_med_s1k.sh medqa-1k-random-qa-extract
bash curate_med_s1k.sh medqa-1k-random-socratic-extract
bash curate_med_s1k.sh medqa-1k-random-decision-tree-extract

# Perturbation experiments
# Collapse consecutive
bash curate_med_s1k.sh medqa-1k-random-collapse-33
bash curate_med_s1k.sh medqa-1k-random-collapse-66
bash curate_med_s1k.sh medqa-1k-random-collapse-100

# Skip steps
bash curate_med_s1k.sh medqa-1k-random-skip-33
bash curate_med_s1k.sh medqa-1k-random-skip-66
bash curate_med_s1k.sh medqa-1k-random-skip-100

# Shuffle steps
bash curate_med_s1k.sh medqa-1k-random-shuffle-33
bash curate_med_s1k.sh medqa-1k-random-shuffle-66
bash curate_med_s1k.sh medqa-1k-random-shuffle-100

# Add irrelevant steps
bash curate_med_s1k.sh medqa-1k-random-irrelevant-33
bash curate_med_s1k.sh medqa-1k-random-irrelevant-66
bash curate_med_s1k.sh medqa-1k-random-irrelevant-100

# Wrong answer
bash curate_med_s1k.sh medqa-1k-random-wrong-answer-33
bash curate_med_s1k.sh medqa-1k-random-wrong-answer-66
bash curate_med_s1k.sh medqa-1k-random-wrong-answer-100

# Restoration experiments
# Collapse consecutive
bash curate_med_s1k.sh medqa-1k-random-collapse-33-restore
bash curate_med_s1k.sh medqa-1k-random-collapse-66-restore
bash curate_med_s1k.sh medqa-1k-random-collapse-100-restore

# Skip steps
bash curate_med_s1k.sh medqa-1k-random-skip-33-restore
bash curate_med_s1k.sh medqa-1k-random-skip-66-restore
bash curate_med_s1k.sh medqa-1k-random-skip-100-restore

# Shuffle steps
bash curate_med_s1k.sh medqa-1k-random-shuffle-33-restore
bash curate_med_s1k.sh medqa-1k-random-shuffle-66-restore
bash curate_med_s1k.sh medqa-1k-random-shuffle-100-restore

# Add irrelevant steps
bash curate_med_s1k.sh medqa-1k-random-irrelevant-33-restore
bash curate_med_s1k.sh medqa-1k-random-irrelevant-66-restore
bash curate_med_s1k.sh medqa-1k-random-irrelevant-100-restore