#!/usr/bin/env python3
"""
Script to determine whether to use GPU or CPU for curation based on:
1. The curation method specified in the experiment config
2. The existence of pre-computed datasets
"""

import os
import json
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def load_experiment_config(experiment_name: str, results_json_path: str) -> dict:
    """Load experiment configuration from results.json"""
    with open(results_json_path, "r") as f:
        results = json.load(f)
    
    if experiment_name not in results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in {results_json_path}")
    
    return results["experiments"][experiment_name]["config"]

def select_device(experiment_name: str, results_json_path: str, data_dir: str) -> str:
    """
    Determine whether to use GPU or CPU for curation based on method and file existence.
    
    Returns:
        str: "gpu" or "cpu"
    """
    # Load experiment config
    config = load_experiment_config(experiment_name, results_json_path)
    curation_method = config["curation"]["method"]
    
    # Define paths to check
    filtered_dataset_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    embeddings_dir_path = os.path.join(data_dir, "embeddings-25k")
    
    # Log the method and paths we're checking
    logging.info(f"Curation method: {curation_method}")
    logging.info(f"Checking for filtered dataset: {filtered_dataset_path}")
    logging.info(f"Checking for embeddings directory: {embeddings_dir_path}")
    
    # Simple methods that always use CPU
    if curation_method in ["all", "random", "difficulty-substring"]:
        logging.info(f"Method '{curation_method}' always uses CPU")
        return "cpu"
    
    # S1 method - check for filtered dataset
    elif curation_method == "s1":
        if os.path.exists(filtered_dataset_path):
            logging.info(f"Method '{curation_method}' using CPU (filtered dataset exists)")
            return "cpu"
        else:
            logging.info(f"Method '{curation_method}' using GPU (filtered dataset does not exist)")
            return "gpu"
    
    # Novelty-answer method - check for both filtered dataset and embeddings
    elif curation_method == "novelty-answer":
        if os.path.exists(filtered_dataset_path) and os.path.exists(embeddings_dir_path):
            logging.info(f"Method '{curation_method}' using CPU (both filtered dataset and embeddings exist)")
            return "cpu"
        else:
            logging.info(f"Method '{curation_method}' using GPU (filtered dataset or embeddings do not exist)")
            return "gpu"
    
    # Embedding methods - check for embeddings directory
    elif curation_method in ["embedding-similarity", "embedding-diversity"]:
        if os.path.exists(embeddings_dir_path):
            logging.info(f"Method '{curation_method}' using CPU (embeddings exist)")
            return "cpu"
        else:
            logging.info(f"Method '{curation_method}' using GPU (embeddings do not exist)")
            return "gpu"
    
    # Default to GPU for unknown methods
    else:
        logging.warning(f"Unknown curation method '{curation_method}', defaulting to GPU")
        return "gpu"

def main():
    parser = argparse.ArgumentParser(description="Select curation device (GPU or CPU)")
    parser.add_argument("--experiment", required=True, help="Name of experiment from results.json")
    args = parser.parse_args()
    
    # Get paths from environment variables
    results_json_path = os.environ.get('RESULTS_JSON')
    data_dir = os.environ.get('DATA_DIR')
    
    if not results_json_path:
        raise ValueError("RESULTS_JSON environment variable not set")
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    
    # Select device
    device = select_device(args.experiment, results_json_path, data_dir)
    
    # Print result for shell script to capture
    print(device)

if __name__ == "__main__":
    main()