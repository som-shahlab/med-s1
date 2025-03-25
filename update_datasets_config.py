#!/usr/bin/env python3
"""
Script to update config.json and results.json to specify train_datasets and eval_datasets.
This script:
1. Updates config.json to add train_datasets and eval_datasets
2. Updates all experiments in results.json to add datasets: {curate: <train_dataset_name>, eval: <eval_dataset_name>}
"""

import os
import json
import argparse

def update_config_json(config_path):
    """Update config.json to add train_datasets and eval_datasets"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add train_datasets if not present
    if 'train_datasets' not in config:
        config['train_datasets'] = {
            "huatuo-sft": "FreedomIntelligence/medical-o1-reasoning-SFT",
        }
    
    # Add eval_datasets if not present
    if 'eval_datasets' not in config:
        config['eval_datasets'] = {
            "huatuo-eval": "${MED_S1_DIR}/eval/data/eval_data.json",
        }
    
    # Write updated config back to file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated {config_path} with train_datasets and eval_datasets")
    return config

def update_results_json(results_path, config):
    """Update all experiments in results.json to add datasets field"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get dataset mappings from config
    train_datasets = config.get('train_datasets', {})
    eval_datasets = config.get('eval_datasets', {})
    
    # Default dataset names
    default_train = next(iter(train_datasets.keys())) if train_datasets else None
    default_eval = next(iter(eval_datasets.keys())) if eval_datasets else None
    
    # Update each experiment
    for exp_name, exp_data in results.get('experiments', {}).items():
        if 'config' not in exp_data:
            exp_data['config'] = {}
        
        # Skip if datasets already defined
        if 'datasets' in exp_data['config']:
            print(f"Experiment {exp_name} already has datasets defined, skipping")
            continue
        
        # Determine appropriate datasets based on experiment type
        if exp_name == 'base' or exp_name == 'huatuo':
            # Base models don't have training data
            exp_data['config']['datasets'] = {
                'curate': None,
                'eval': default_eval
            }
        else:
            # Regular experiments use the default datasets
            exp_data['config']['datasets'] = {
                'curate': default_train,
                'eval': default_eval
            }
        
        print(f"Updated experiment {exp_name} with datasets configuration")
    
    # Write updated results back to file
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Updated {results_path} with datasets for all experiments")

def main():
    parser = argparse.ArgumentParser(description='Update config.json and results.json with dataset specifications')
    parser.add_argument('--config', default='/share/pi/nigam/users/calebwin/med-s1/config.json', help='Path to config.json')
    parser.add_argument('--results', default='/share/pi/nigam/users/calebwin/med-s1/results.json', help='Path to results.json')
    args = parser.parse_args()
    
    # Update config.json first
    updated_config = update_config_json(args.config)
    
    # Then update results.json
    update_results_json(args.results, updated_config)
    
    print("Updates completed successfully!")

if __name__ == "__main__":
    main()