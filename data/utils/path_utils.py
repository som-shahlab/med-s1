import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

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

def get_results_storage_path(experiment_name: str, stage: str) -> str:
    """Get path for storing detailed results in separate files"""
    med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
    results_dir = os.path.join(med_s1_dir, "results")
    safe_name = clean_experiment_name(experiment_name)
    return os.path.join(results_dir, f"{safe_name}_{stage}.json")

def save_results_to_file(data: Dict[str, Any], file_path: str) -> None:
    """Save results data to a separate JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def update_results_json(results_json_path: str, experiment_name: str, stage: str, paths: dict, timestamp: str, stats: dict = None) -> None:
    """Update results.json with paths and stats"""
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # Update experiment results
    if "results" not in results["experiments"][experiment_name]:
        results["experiments"][experiment_name]["results"] = {}
    if stage not in results["experiments"][experiment_name]["results"]:
        results["experiments"][experiment_name]["results"][stage] = {}
    
    # Create a separate file for detailed results
    results_file_path = get_results_storage_path(experiment_name, stage)
    
    if stage == "curation":
        # Keep basic info in results.json
        results["experiments"][experiment_name]["results"][stage] = {
            "dataset_path": paths['formatted'],
            "timestamp": timestamp
        }
        
        # Save detailed stats to separate file
        if stats:
            save_results_to_file(stats, results_file_path)
            # Add reference to the file
            results["experiments"][experiment_name]["results"][stage]["stats_file"] = os.path.abspath(results_file_path)
    
    elif stage == "training":
        # Training data is already minimal, keep it in results.json
        results["experiments"][experiment_name]["results"][stage] = {
            "model_path": os.path.abspath(paths.get('checkpoint', paths.get('model_path', ''))),
            "timestamp": timestamp,
            "metrics": None
        }
    
    elif stage == "eval":
        # Keep basic info in results.json
        results["experiments"][experiment_name]["results"][stage] = {
            "outputs_path": os.path.abspath(paths.get('results', paths.get('outputs_path', ''))),
            "metrics_path": os.path.abspath(paths.get('metrics_path', paths.get('metrics', ''))),
            "timestamp": timestamp
        }
        
        # Save detailed summary to separate file if it exists
        if stats and "summary" in stats:
            save_results_to_file({"summary": stats["summary"]}, results_file_path)
            # Add reference to the file
            results["experiments"][experiment_name]["results"][stage]["summary_file"] = os.path.abspath(results_file_path)
    
    # Write back safely
    with open(results_json_path + '.tmp', 'w') as f:
        json.dump(results, f, indent=2)
    os.rename(results_json_path + '.tmp', results_json_path)