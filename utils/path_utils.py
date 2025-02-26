import os
import re
from typing import Optional
from datetime import datetime

def clean_experiment_name(name: str) -> str:
    """Clean experiment name for use in filenames by removing all non-alphanumeric characters except hyphens"""
    return re.sub(r'[^a-zA-Z0-9-]', '', name)

def get_experiment_dir(output_dir: str, experiment_name: str) -> str:
    """Get experiment directory using cleaned experiment name"""
    safe_name = clean_experiment_name(experiment_name)
    return os.path.join(output_dir, safe_name)

def get_formatted_dataset_path(output_dir: str, experiment_name: str) -> str:
    """Get path to formatted dataset using cleaned experiment name"""
    return os.path.join(get_experiment_dir(output_dir, experiment_name), "med_s1k_formatted")

def get_intermediate_path(output_dir: str, experiment_name: str, stage: str, timestamp: str) -> str:
    """Get path for intermediate files during curation"""
    safe_name = clean_experiment_name(experiment_name)
    return os.path.join(output_dir, f"{safe_name}_post_{stage}_{timestamp}.parquet")

def get_final_paths(output_dir: str, experiment_name: str) -> dict:
    """Get paths for final curation outputs"""
    experiment_dir = get_experiment_dir(output_dir, experiment_name)
    return {
        'filtered': os.path.join(experiment_dir, "med_s1k_filtered.parquet"),
        'curated': os.path.join(experiment_dir, "med_s1k_curated.parquet"),
        'formatted': os.path.join(experiment_dir, "med_s1k_formatted")
    }

def get_training_paths(cache_dir: str, experiment_name: str, batch_size: int, learning_rate: float, num_epochs: int) -> dict:
    """Get paths for training outputs"""
    safe_name = clean_experiment_name(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{safe_name}_bs{batch_size}_lr{learning_rate}_epoch{num_epochs}_{timestamp}"
    ckpt_dir = os.path.join(cache_dir, "ckpts", run_name)
    return {
        'checkpoint': ckpt_dir,
        'local_data': "/local-scratch/${SLURM_JOB_ID}/med_s1k_formatted",
        'run_name': run_name
    }

def get_eval_paths(output_dir: str, experiment_name: str) -> dict:
    """Get paths for evaluation outputs"""
    experiment_dir = get_experiment_dir(output_dir, experiment_name)
    return {
        'results': os.path.join(experiment_dir, "eval_results.json"),
        'predictions': os.path.join(experiment_dir, "eval_predictions.jsonl")
    }

def update_results_json(results_json_path: str, experiment_name: str, stage: str, paths: dict, timestamp: Optional[str] = None, stats: Optional[dict] = None) -> None:
    """Update results.json with paths, timestamp, and optional stats"""
    import json
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    with open(results_json_path, "r") as f:
        results = json.load(f)
    
    # Update paths and timestamp
    if stage == "curation":
        results["experiments"][experiment_name]["results"]["curation"] = {
            "dataset_path": os.path.abspath(paths['formatted']),
            "timestamp": timestamp,
            "stats": stats if stats else {}
        }
    elif stage == "training":
        results["experiments"][experiment_name]["results"]["training"] = {
            "model_path": os.path.abspath(paths['checkpoint']),
            "timestamp": timestamp
        }
    elif stage == "eval":
        results["experiments"][experiment_name]["results"]["eval"] = {
            "results_path": os.path.abspath(paths['results']),
            "timestamp": timestamp
        }
    
    # Write back to file
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)