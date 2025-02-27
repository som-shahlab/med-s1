import os
import json
from datetime import datetime

def clean_experiment_name(name: str) -> str:
    """Clean experiment name for use in paths"""
    return ''.join(c for c in name if c.isalnum() or c == '-')

def get_experiment_dir(output_dir: str, experiment_name: str) -> str:
    """Get experiment-specific directory"""
    safe_name = clean_experiment_name(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{safe_name}_{timestamp}")

def get_formatted_dataset_path(output_dir: str, experiment_name: str) -> str:
    """Get path for formatted dataset"""
    exp_dir = get_experiment_dir(output_dir, experiment_name)
    return os.path.join(exp_dir, "med_s1k_formatted")

def get_intermediate_path(output_dir: str, experiment_name: str, stage: str, timestamp: str) -> str:
    """Get path for intermediate outputs"""
    safe_name = clean_experiment_name(experiment_name)
    return os.path.join(output_dir, f"med_s1k_{safe_name}_{stage}_{timestamp}.parquet")

def get_final_paths(output_dir: str, experiment_name: str) -> dict:
    """Get paths for final outputs"""
    exp_dir = get_experiment_dir(output_dir, experiment_name)
    return {
        'filtered': os.path.join(exp_dir, "med_s1k_filtered.parquet"),
        'curated': os.path.join(exp_dir, "med_s1k_curated.parquet"),
        'formatted': os.path.join(exp_dir, "med_s1k_formatted")
    }

def update_results_json(results_json_path: str, experiment_name: str, stage: str, paths: dict, timestamp: str, stats: dict = None) -> None:
    """Update results.json with paths and stats"""
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # Update experiment results
    if stage not in results["experiments"][experiment_name]["results"]:
        results["experiments"][experiment_name]["results"][stage] = {}
    
    results["experiments"][experiment_name]["results"][stage].update({
        "dataset_path": paths['formatted'],
        "timestamp": timestamp
    })
    
    if stats:
        results["experiments"][experiment_name]["results"][stage]["stats"] = stats
    
    # Write back safely
    with open(results_json_path + '.tmp', 'w') as f:
        json.dump(results, f, indent=2)
    os.rename(results_json_path + '.tmp', results_json_path)