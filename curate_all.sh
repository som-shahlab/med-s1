#!/bin/bash

# This script runs curation for all experiments in results.json
# Usage: bash curate_all.sh

# Base and huatuo don't need curation (pre-trained models)

# Run curation for each experiment
bash curate_med_s1k.sh medqa-1k-random
bash curate_med_s1k.sh medqa-1k-embedding-diversity-question-cluster-10-outlier-5
bash curate_med_s1k.sh medqa-1k-embedding-difficulty-diversity-question-cluster-10-outlier-5 # (*)
bash curate_med_s1k.sh medqa-25k
bash curate_med_s1k.sh medqa-1k-random-step-extract # (*)
bash curate_med_s1k.sh medqa-1k-random-no-cot # (*)
bash curate_med_s1k.sh medqa-1k-random-1-sentence-extract # (*)
bash curate_med_s1k.sh medqa-5k-random-no-cot
bash curate_med_s1k.sh medqa-10k-random-no-cot