#!/usr/bin/env python3
"""
Analyze token counts in curated datasets for each experiment in results.json.
This script calculates min, max, average, and total token counts for reasoning+answer
portions of each sample in the datasets.
"""

import os
import json
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def load_results_json(path):
    """Load the results.json file."""
    with open(path, 'r') as f:
        return json.load(f)

def get_tokenizer(model_key):
    """Get the appropriate tokenizer based on model key."""
    # Map model keys to HuggingFace model paths
    model_map = {
        "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
        "huatuo:8b": "FreedomIntelligence/HuatuoGPT-o1-8B"
    }
    
    if model_key in model_map:
        model_path = model_map[model_key]
        logging.info(f"Loading tokenizer for {model_key} from {model_path}")
        return AutoTokenizer.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

def get_curated_parquet_path(dataset_path):
    """
    Convert the formatted dataset path to the curated.parquet path.
    
    Args:
        dataset_path: Path to the formatted dataset directory
        
    Returns:
        Path to the curated.parquet file
    """
    # Extract the experiment directory from the formatted dataset path
    # Example: /share/pi/nigam/users/calebwin/hf_cache/med-s1k/medqa-1k-random_20250315_174440/med_s1k_formatted
    # Becomes: /share/pi/nigam/users/calebwin/hf_cache/med-s1k/medqa-1k-random_20250315_174440/med_s1k_curated.parquet
    
    if not dataset_path:
        return None
        
    # Replace the formatted directory with curated.parquet
    return dataset_path.replace("med_s1k_formatted", "med_s1k_curated.parquet")

def analyze_dataset_tokens(dataset_path, tokenizer, experiment_name, extract_method=None):
    """
    Analyze token counts in a dataset.
    
    Args:
        dataset_path: Path to the formatted dataset directory
        tokenizer: Tokenizer to use for counting tokens
        experiment_name: Name of the experiment
        extract_method: Extraction method used (if any)
        
    Returns:
        Dictionary with token statistics
    """
    # Get the path to the curated.parquet file
    curated_path = get_curated_parquet_path(dataset_path)
    
    if not curated_path:
        logging.warning(f"Could not determine curated.parquet path for {experiment_name}")
        return None
        
    logging.info(f"Analyzing dataset at {curated_path}")
    
    # Check if curated.parquet exists
    if not os.path.exists(curated_path):
        logging.warning(f"Curated dataset not found at {curated_path}")
        return None
    
    try:
        # Load the curated dataset as a pandas DataFrame
        df = pd.read_parquet(curated_path)
        
        # Determine what columns to analyze
        token_counts = []
        
        # Print columns for debugging
        logging.info(f"Dataset columns: {df.columns}")
        
        # Check for different possible column combinations
        # The dataset uses 'Response' instead of 'Answer'
        if 'Complex_CoT' in df.columns and 'Response' in df.columns:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {experiment_name}"):
                # For standard format with CoT
                if extract_method == "none":
                    # No CoT, just answer
                    text = row['Response']
                elif extract_method == "step":
                    # Step extraction (might be in a different column)
                    text = row.get('Extracted_Steps', row['Complex_CoT']) + " " + row['Response']
                elif extract_method == "1-sentence":
                    # 1-sentence extraction
                    text = row.get('Extracted_Sentence', row['Complex_CoT']) + " " + row['Response']
                else:
                    # Full CoT
                    text = row['Complex_CoT'] + " " + row['Response']
                
                token_count = len(tokenizer.encode(text))
                token_counts.append(token_count)
        else:
            logging.warning(f"Required columns not found in dataset. Need 'Complex_CoT' and 'Response'.")
        
        # Calculate statistics
        if token_counts:
            stats = {
                "experiment": experiment_name,
                "dataset_size": len(token_counts),
                "min_tokens": min(token_counts),
                "max_tokens": max(token_counts),
                "avg_tokens": sum(token_counts) / len(token_counts),
                "total_tokens": sum(token_counts)
            }
            return stats
        else:
            logging.warning(f"No token counts calculated for {experiment_name}")
            return None
            
    except Exception as e:
        logging.error(f"Error analyzing dataset {curated_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze token counts in curated datasets")
    parser.add_argument("--results-json", default="/share/pi/nigam/users/calebwin/med-s1/results.json", 
                        help="Path to results.json file")
    args = parser.parse_args()
    
    # Load results.json
    results = load_results_json(args.results_json)
    
    # Store results for all experiments
    all_stats = []
    
    # Process each experiment
    for exp_name, exp_data in results["experiments"].items():
        logging.info(f"Processing experiment: {exp_name}")
        
        # Skip experiments without curation results
        if "results" not in exp_data or "curation" not in exp_data["results"] or not exp_data["results"]["curation"]:
            logging.info(f"Skipping {exp_name} - no curation results")
            continue
        
        # Get model key and dataset path
        model_key = exp_data["config"]["model_key"]
        dataset_path = exp_data["results"]["curation"]["dataset_path"]
        
        # Get extraction method if specified
        extract_method = exp_data["config"].get("curation", {}).get("extract", None)
        
        # Get tokenizer
        try:
            tokenizer = get_tokenizer(model_key)
            
            # Analyze dataset
            stats = analyze_dataset_tokens(dataset_path, tokenizer, exp_name, extract_method)
            
            if stats:
                all_stats.append(stats)
                
        except Exception as e:
            logging.error(f"Error processing {exp_name}: {str(e)}")
    
    # Print results in a formatted table
    print("\nToken Count Analysis Results:")
    print("=" * 100)
    print(f"{'Experiment':<40} {'Dataset Size':<15} {'Min':<8} {'Max':<8} {'Avg':<12} {'Total':<15}")
    print("-" * 100)
    
    for stats in all_stats:
        print(f"{stats['experiment']:<40} {stats['dataset_size']:<15} "
              f"{stats['min_tokens']:<8} {stats['max_tokens']:<8} "
              f"{stats['avg_tokens']:<12.2f} {stats['total_tokens']:<15,}")
    
    print("=" * 100)
    print("\nRecommendations for hyperparameter settings:")
    print("-" * 100)
    
    for stats in all_stats:
        total_tokens = stats['total_tokens']
        dataset_size = stats['dataset_size']
        avg_tokens = stats['avg_tokens']
        
        # Calculate recommended hyperparameters based on dataset size and token counts
        # These are heuristics based on papers like LIMA, LIMO, etc.
        if dataset_size <= 1000:
            # For smaller datasets (1k examples)
            lr = 2e-6
            epochs = max(3, min(15, int(1_000_000 / total_tokens)))
            warmup = 0.15
        elif dataset_size <= 5000:
            # For medium datasets (1k-5k examples)
            lr = 3e-6
            epochs = max(2, min(8, int(2_000_000 / total_tokens)))
            warmup = 0.1
        else:
            # For larger datasets (>5k examples)
            lr = 4e-6
            epochs = max(1, min(5, int(3_000_000 / total_tokens)))
            warmup = 0.075
        
        print(f"{stats['experiment']} (n={dataset_size}, {total_tokens:,} total tokens):")
        print(f"  - Learning rate: {lr}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Warmup ratio: {warmup}")
        print(f"  - Batch size: 2 (with gradient accumulation steps of 16)")
        print("-" * 100)

if __name__ == "__main__":
    main()