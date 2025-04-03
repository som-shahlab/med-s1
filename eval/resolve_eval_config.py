#!/usr/bin/env python3
import json
import sys
import argparse
from typing import Dict, Any, Optional
import os

def resolve_reference(ref: Any, experiments: Dict[str, Any], field: str) -> Any:
    """Resolve a reference to its actual value."""
    # If it's a string reference, resolve it
    if isinstance(ref, str) and ref.startswith("same as "):
        referenced_exp = ref[8:]  # Remove "same as " prefix
        if referenced_exp not in experiments:
            raise ValueError(f"Referenced experiment {referenced_exp} not found")
        
        value = experiments[referenced_exp]["config"].get(field)
        if isinstance(value, str) and value.startswith("same as "):
            # Handle nested references
            return resolve_reference(value, experiments, field)
        # If we get a dictionary back, make a copy to avoid modifying the original
        elif isinstance(value, dict):
            return value.copy()
        return value
    # If it's a dictionary, resolve any references in its values
    elif isinstance(ref, dict):
        resolved = {}
        for k, v in ref.items():
            resolved[k] = resolve_reference(v, experiments, f"{field}.{k}")
        return resolved
    # Otherwise return as-is
    return ref

def resolve_eval_config(experiment_name: str, results_json_path: str) -> Dict[str, Any]:
    """Resolve experiment configuration for evaluation."""
    # Read results.json
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    if experiment_name not in results["experiments"]:
        raise ValueError(f"Experiment '{experiment_name}' not found in {results_json_path}")
    
    # Get experiment config
    experiment = results["experiments"][experiment_name]
    config = experiment["config"]
    
    # Resolve references in config
    for key in ["training_params", "datasets"]:
        if isinstance(config.get(key), str):
            config[key] = resolve_reference(
                config[key],
                results["experiments"],
                key
            )
        elif isinstance(config.get(key), dict):
            config[key] = resolve_reference(
                config[key],
                results["experiments"],
                key
            )
    
    # Resolve datasets reference if needed
    if isinstance(config.get("datasets"), str):
        config["datasets"] = resolve_reference(
            config["datasets"], 
            results["experiments"], 
            "datasets"
        )
    
    # Get model info
    model_key = config["model_key"]
    
    # Get training results or use pre-trained model path
    training_results = experiment.get("results", {}).get("training")
    model_path = None
    
    # If training results exist, use the model path from there
    if training_results and training_results.get("model_path"):
        model_path = training_results.get("model_path")
    # Otherwise, check if it's a pre-trained model
    elif model_key:
        # Load config.json to get the model path
        med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
        with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
            global_config = json.load(f)
        if model_key in global_config.get("models", {}):
            model_path = global_config["models"][model_key]["hf_path"]
    
    if not model_path:
        raise ValueError("Could not determine model path. Either provide training results or use a valid pre-trained model_key.")
    
    # Return resolved configuration
    return {
        "config": config,
        "model_key": model_key,
        "model_path": model_path,
        "test_time_scaling": config.get("test_time_scaling", False)
    }

def main():
    parser = argparse.ArgumentParser(description="Resolve evaluation configuration")
    parser.add_argument("experiment_name", help="Name of the experiment")
    parser.add_argument("--results-json", required=True, help="Path to results.json")
    args = parser.parse_args()
    
    try:
        result = resolve_eval_config(args.experiment_name, args.results_json)
        # Output as JSON for easy parsing in bash
        print(json.dumps(result))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()