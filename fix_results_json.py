#!/usr/bin/env python3
"""
Script to convert legacy results.json to the new format with separate files.
This script:
1. Reads the current results.json
2. For each experiment, extracts large data (curation stats, eval summary)
3. Creates separate files for this data in the results directory
4. Updates results.json with references to these files
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

def get_results_storage_path(med_s1_dir, experiment_name, stage):
    """Get path for storing detailed results in separate files"""
    results_dir = os.path.join(med_s1_dir, "results")
    safe_name = ''.join(c for c in experiment_name if c.isalnum() or c == '-')
    return os.path.join(results_dir, f"{safe_name}_{stage}.json")

def save_results_to_file(data, file_path):
    """Save results data to a separate JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Created file: {file_path}")

def fix_curation_results(experiment_name, experiment_data, med_s1_dir):
    """Extract curation stats to a separate file"""
    if 'results' not in experiment_data or 'curation' not in experiment_data['results']:
        return experiment_data
    
    curation_data = experiment_data['results']['curation']
    
    # Skip if already converted
    if 'stats_file' in curation_data:
        print(f"Experiment {experiment_name} curation data already converted")
        return experiment_data
    
    # Skip if no stats
    if 'stats' not in curation_data:
        print(f"Experiment {experiment_name} has no curation stats to extract")
        return experiment_data
    
    # Extract stats to separate file
    stats = curation_data.pop('stats')
    stats_file_path = get_results_storage_path(med_s1_dir, experiment_name, 'curation')
    save_results_to_file(stats, stats_file_path)
    
    # Update reference in results.json
    curation_data['stats_file'] = os.path.abspath(stats_file_path)
    
    return experiment_data

def fix_eval_results(experiment_name, experiment_data, med_s1_dir):
    """Extract eval summary to a separate file"""
    if 'results' not in experiment_data or 'eval' not in experiment_data['results']:
        return experiment_data
    
    eval_data = experiment_data['results']['eval']
    
    # Skip if already converted
    if 'summary_file' in eval_data:
        print(f"Experiment {experiment_name} eval data already converted")
        return experiment_data
    
    # Skip if no summary
    if 'summary' not in eval_data:
        print(f"Experiment {experiment_name} has no eval summary to extract")
        return experiment_data
    
    # Extract data to store in separate file
    summary_data = {}
    
    # Extract summary
    summary_data['summary'] = eval_data.pop('summary')
    
    # Extract test time scaling data if present
    if 'test_time_scaling' in eval_data:
        summary_data['test_time_scaling'] = eval_data['test_time_scaling']
    
    if 'reasoning_tokens' in eval_data:
        summary_data['reasoning_tokens'] = eval_data.pop('reasoning_tokens')
    
    if 'test_time_scaling_plot' in eval_data:
        summary_data['test_time_scaling_plot'] = eval_data['test_time_scaling_plot']
    
    # Save to file
    summary_file_path = get_results_storage_path(med_s1_dir, experiment_name, 'eval')
    save_results_to_file(summary_data, summary_file_path)
    
    # Update reference in results.json
    eval_data['summary_file'] = os.path.abspath(summary_file_path)
    
    return experiment_data

def main():
    parser = argparse.ArgumentParser(description="Convert legacy results.json to new format with separate files")
    parser.add_argument("--results_json", help="Path to results.json (default: $MED_S1_DIR/results.json)")
    parser.add_argument("--med_s1_dir", help="Path to med-s1 directory (default: $MED_S1_DIR)")
    parser.add_argument("--backup", action="store_true", help="Create backup of original results.json")
    args = parser.parse_args()
    
    # Get MED_S1_DIR
    med_s1_dir = args.med_s1_dir or os.environ.get('MED_S1_DIR')
    if not med_s1_dir:
        med_s1_dir = '/share/pi/nigam/users/calebwin/med-s1'
        print(f"MED_S1_DIR not provided, using default: {med_s1_dir}")
    
    # Get results.json path
    results_json_path = args.results_json
    if not results_json_path:
        results_json_path = os.path.join(med_s1_dir, 'results.json')
        print(f"results_json path not provided, using default: {results_json_path}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(med_s1_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Read results.json
    try:
        with open(results_json_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading results.json: {e}")
        sys.exit(1)
    
    # Create backup if requested
    if args.backup:
        backup_path = f"{results_json_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Created backup: {backup_path}")
    
    # Process each experiment
    for experiment_name, experiment_data in results.get('experiments', {}).items():
        print(f"\nProcessing experiment: {experiment_name}")
        
        # Fix curation results
        experiment_data = fix_curation_results(experiment_name, experiment_data, med_s1_dir)
        
        # Fix eval results
        experiment_data = fix_eval_results(experiment_name, experiment_data, med_s1_dir)
    
    # Write updated results.json
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nUpdated {results_json_path} with references to separate files")
    print(f"Detailed results are now stored in: {results_dir}")

if __name__ == "__main__":
    main()