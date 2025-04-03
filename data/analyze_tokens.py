#!/usr/bin/env python3
"""
Analyze token counts in datasets.
This script calculates min, max, average, median, and total token counts for specified columns
in datasets, whether they are from disk, parquet files, or HuggingFace.
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import load_from_disk, load_dataset
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

def load_dataset_from_path(path, format_type=None):
    """
    Load a dataset from various formats.
    
    Args:
        path: Path to the dataset
        format_type: One of 'disk', 'parquet', 'huggingface', or None (auto-detect)
        
    Returns:
        Dataset object or DataFrame
    """
    if format_type is None:
        # Auto-detect format
        if path.endswith('.parquet'):
            format_type = 'parquet'
        elif '/' in path and not os.path.exists(path):
            format_type = 'huggingface'
        else:
            format_type = 'disk'
    
    logging.info(f"Loading dataset from {path} as {format_type}")
    
    if format_type == 'disk':
        return load_from_disk(path)
    elif format_type == 'parquet':
        return pd.read_parquet(path)
    elif format_type == 'huggingface':
        return load_dataset(path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")

def analyze_dataset_tokens(dataset, tokenizer, columns, combine_with_space=True):
    """
    Analyze token counts in a dataset.
    
    Args:
        dataset: Dataset object or DataFrame
        tokenizer: Tokenizer to use for counting tokens
        columns: List of column names to analyze
        combine_with_space: Whether to combine columns with spaces between them
        
    Returns:
        Dictionary with token statistics
    """
    token_counts = []
    
    # Convert dataset to DataFrame if needed
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)
    
    # Verify columns exist
    missing_cols = [col for col in columns if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")
    
    logging.info(f"Analyzing columns: {columns}")
    
    # Process each row
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Tokenizing"):
        # Combine specified columns
        if combine_with_space:
            text = " ".join(str(row[col]) for col in columns if pd.notna(row[col]))
        else:
            text = "".join(str(row[col]) for col in columns if pd.notna(row[col]))
        
        token_count = len(tokenizer.encode(text))
        token_counts.append(token_count)
    
    # Calculate statistics
    token_counts = np.array(token_counts)
    under_8192 = np.sum(token_counts < 8192)
    
    stats = {
        "dataset_size": len(token_counts),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "avg_tokens": float(np.mean(token_counts)),
        "med_tokens": float(np.median(token_counts)),
        "total_tokens": int(np.sum(token_counts)),
        "under_8192_count": int(under_8192),
        "under_8192_percent": float(under_8192 * 100 / len(token_counts)),
        "analyzed_columns": columns
    }
    
    return stats

def get_training_recommendations(stats):
    """Get training hyperparameter recommendations based on dataset statistics."""
    total_tokens = stats['total_tokens']
    dataset_size = stats['dataset_size']
    
    # Calculate recommended hyperparameters
    if dataset_size <= 1000:
        lr = 2e-6
        epochs = max(3, min(15, int(1_000_000 / total_tokens)))
        warmup = 0.15
    elif dataset_size <= 5000:
        lr = 3e-6
        epochs = max(2, min(8, int(2_000_000 / total_tokens)))
        warmup = 0.1
    else:
        lr = 4e-6
        epochs = max(1, min(5, int(3_000_000 / total_tokens)))
        warmup = 0.075
    
    return {
        "learning_rate": lr,
        "epochs": epochs,
        "warmup_ratio": warmup,
        "batch_size": 2,
        "gradient_accumulation_steps": 16
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze token counts in datasets")
    parser.add_argument("--path", required=True, help="Path to dataset")
    parser.add_argument("--format", choices=['disk', 'parquet', 'huggingface'], 
                        help="Dataset format (auto-detected if not specified)")
    parser.add_argument("--model", default="llama3.1:8b", 
                        help="Model key for tokenizer (default: llama3.1:8b)")
    parser.add_argument("--columns", nargs='+', required=True,
                        help="Columns to analyze")
    parser.add_argument("--no-space", action='store_true',
                        help="Don't add spaces between columns when combining")
    args = parser.parse_args()
    
    try:
        # Load dataset
        dataset = load_dataset_from_path(args.path, args.format)
        
        # Get tokenizer
        tokenizer = get_tokenizer(args.model)
        
        # Analyze dataset
        stats = analyze_dataset_tokens(
            dataset, 
            tokenizer, 
            args.columns,
            not args.no_space
        )
        
        # Print results
        print("\nToken Count Analysis Results:")
        print("=" * 80)
        print(f"Dataset path: {args.path}")
        print(f"Analyzed columns: {stats['analyzed_columns']}")
        print(f"Dataset size: {stats['dataset_size']:,}")
        print(f"Min tokens: {stats['min_tokens']:,}")
        print(f"Max tokens: {stats['max_tokens']:,}")
        print(f"Avg tokens: {stats['avg_tokens']:.2f}")
        print(f"Med tokens: {stats['med_tokens']:.2f}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Samples under 8192 tokens: {stats['under_8192_count']:,} ({stats['under_8192_percent']:.1f}%)")
        
        # Print training recommendations
        print("\nRecommended Training Hyperparameters:")
        print("=" * 80)
        recs = get_training_recommendations(stats)
        for param, value in recs.items():
            print(f"{param}: {value}")
            
    except Exception as e:
        logging.error(f"Error analyzing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()