"""Script to generate validation dataset from eval data."""

import json
import random
import os
import argparse
from typing import Dict, List

def load_eval_data(eval_data_path: str) -> List[Dict]:
    """Load evaluation data from JSON file."""
    with open(eval_data_path, 'r') as f:
        return json.load(f)

def filter_validation_samples(eval_data: List[Dict], samples_per_source: int = 50, seed: int = 42) -> List[Dict]:
    """Filter validation samples from eval data.
    
    Args:
        eval_data: List of evaluation samples
        samples_per_source: Number of samples to select per source
        seed: Random seed for reproducibility
        
    Returns:
        List of selected validation samples
    """
    # Set random seed
    random.seed(seed)
    
    # Group data by source
    data_by_source = {}
    for item in eval_data:
        source = item.get('source')
        if source not in ['MedDS', 'MedDS_NOTA']:  # Skip MedDS sources
            if source not in data_by_source:
                data_by_source[source] = []
            data_by_source[source].append(item)
    
    # Select random samples from each source
    validation_data = []
    for source, items in data_by_source.items():
        selected = random.sample(items, min(samples_per_source, len(items)))
        validation_data.extend(selected)
        print(f"Selected {len(selected)} samples from {source}")
    
    return validation_data

def save_validation_data(validation_data: List[Dict], output_path: str):
    """Save validation data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(validation_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Generate validation dataset from eval data')
    parser.add_argument('--eval_data_path', type=str, required=True,
                      help='Path to eval_data.json')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save val_data.json')
    parser.add_argument('--samples_per_source', type=int, default=50,
                      help='Number of samples to select per source')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    args = parser.parse_args()

    # Load eval data
    print(f"Loading eval data from {args.eval_data_path}")
    eval_data = load_eval_data(args.eval_data_path)
    print(f"Loaded {len(eval_data)} total samples")

    # Filter validation samples
    validation_data = filter_validation_samples(
        eval_data,
        samples_per_source=args.samples_per_source,
        seed=args.seed
    )
    print(f"\nSelected {len(validation_data)} total validation samples")

    # Save validation data
    print(f"Saving validation data to {args.output_path}")
    save_validation_data(validation_data, args.output_path)
    print("Done!")

if __name__ == "__main__":
    main()