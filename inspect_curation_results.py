#!/usr/bin/env python3
"""
This script inspects the curated datasets for each experiment in results.json.
It prints out the question, response, complex CoT, and original CoT if it exists,
as well as the formatted text from the HuggingFace dataset.
"""

import os
import json
import pandas as pd
from datasets import load_from_disk
import argparse
from typing import Dict, Any, Optional

def load_results_json(results_json_path: str) -> Dict[str, Any]:
    """Load the results.json file."""
    with open(results_json_path, 'r') as f:
        return json.load(f)

def print_separator(experiment_name: str):
    """Print a separator with the experiment name."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)

def inspect_parquet_dataset(dataset_path: str):
    """Inspect the first example in the parquet dataset."""
    if not os.path.exists(dataset_path):
        print(f"Parquet dataset not found at: {dataset_path}")
        return
    
    try:
        df = pd.read_parquet(dataset_path)
        
        # Filter to only selected examples
        selected_df = df[df['selected_for_training']]
        
        if len(selected_df) == 0:
            print("No selected examples found in the dataset.")
            return
        
        # Get the first example
        first_example = selected_df.iloc[0]
        
        print("\nPARQUET DATASET FIRST EXAMPLE:")
        
        # Print Question
        print("\nQUESTION:")
        print(first_example.get('Question', 'N/A'))
        
        # Print Response
        print("\nRESPONSE:")
        print(first_example.get('Response', 'N/A'))
        
        # Print Complex_CoT
        print("\nCOMPLEX_COT:")
        print(first_example.get('Complex_CoT', 'N/A'))
        
        # Check if Complex_CoT_orig exists
        if 'Complex_CoT_orig' in first_example and pd.notna(first_example['Complex_CoT_orig']):
            print("\nCOMPLEX_COT_ORIG:")
            print(first_example['Complex_CoT_orig'])
            
            # If both exist and are different, show a note about transformation
            if pd.notna(first_example.get('Complex_CoT', '')) and first_example['Complex_CoT'] != first_example['Complex_CoT_orig']:
                print("\nNOTE: CoT transformation was applied to this example.")
    
    except Exception as e:
        print(f"Error inspecting parquet dataset: {e}")

def inspect_hf_dataset(formatted_path: str):
    """Inspect the first example in the HuggingFace formatted dataset."""
    if not os.path.exists(formatted_path):
        print(f"HuggingFace dataset not found at: {formatted_path}")
        return
    
    try:
        # Load the dataset
        train_path = os.path.join(formatted_path, 'train')
        if not os.path.exists(train_path):
            print(f"Train dataset not found at: {train_path}")
            return
            
        dataset = load_from_disk(train_path)
        
        if len(dataset) == 0:
            print("No examples found in the HuggingFace dataset.")
            return
        
        # Get the first example
        first_example = dataset[0]
        
        print("\nHUGGINGFACE FORMATTED DATASET FIRST EXAMPLE:")
        print(f"Text: {first_example['text']}")
        
    except Exception as e:
        print(f"Error inspecting HuggingFace dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description='Inspect curated datasets for experiments in results.json')
    parser.add_argument('--results_json', type=str, default='/share/pi/nigam/users/calebwin/med-s1/results.json',
                        help='Path to results.json file')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Specific experiment to inspect (optional)')
    args = parser.parse_args()
    
    # Load results.json
    results = load_results_json(args.results_json)
    
    # Filter experiments if specified
    experiments = {}
    if args.experiment:
        if args.experiment in results['experiments']:
            experiments[args.experiment] = results['experiments'][args.experiment]
        else:
            print(f"Experiment '{args.experiment}' not found in results.json")
            return
    else:
        experiments = results['experiments']
    
    # Iterate through each experiment
    for experiment_name, experiment_data in experiments.items():
        print_separator(experiment_name)
        
        # Skip experiments without curation results
        if 'results' not in experiment_data or 'curation' not in experiment_data['results'] or experiment_data['results']['curation'] is None:
            print(f"No curation results found for experiment: {experiment_name}")
            continue
        
        # Get dataset paths
        curated_path = experiment_data['results']['curation'].get('dataset_path')
        if not curated_path:
            print(f"No dataset path found for experiment: {experiment_name}")
            continue
            
        # Get the parent directory
        parent_dir = os.path.dirname(curated_path)
        
        # Construct paths
        parquet_path = os.path.join(parent_dir, 'med_s1k_curated.parquet')
        formatted_path = os.path.join(parent_dir, 'med_s1k_formatted')
        
        # Print experiment config details
        if 'config' in experiment_data and 'curation' in experiment_data['config'] and experiment_data['config']['curation']:
            curation_config = experiment_data['config']['curation']
            print("\nCURATION CONFIG:")
            for key, value in curation_config.items():
                print(f"{key}: {value}")
        
        # Inspect datasets
        inspect_parquet_dataset(parquet_path)
        inspect_hf_dataset(formatted_path)

if __name__ == "__main__":
    main()