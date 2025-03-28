#!/usr/bin/env python3
import json
import sys
import argparse
from typing import Dict, Any, Optional

def resolve_reference(ref: str, experiments: Dict[str, Any], field: str) -> Any:
    """Resolve a reference to its actual value."""
    if not isinstance(ref, str) or not ref.startswith("same as "):
        return ref
        
    referenced_exp = ref[8:]  # Remove "same as " prefix
    if referenced_exp not in experiments:
        raise ValueError(f"Referenced experiment {referenced_exp} not found")
        
    value = experiments[referenced_exp]["config"].get(field)
    if isinstance(value, str) and value.startswith("same as "):
        # Handle nested references
        return resolve_reference(value, experiments, field)
    return value

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
    
    # Resolve training_params reference if needed
    if isinstance(config.get("training_params"), str):
        config["training_params"] = resolve_reference(
            config["training_params"], 
            results["experiments"], 
            "training_params"
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
    
    # Get training results
    training_results = experiment.get("results", {}).get("training")
    if not training_results:
        raise ValueError("No training results found. Has training been completed?")
    
    model_path = training_results.get("model_path")
    if not model_path:
        raise ValueError("No model path found in training results")
    
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