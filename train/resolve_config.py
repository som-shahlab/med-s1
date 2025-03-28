#!/usr/bin/env python3
import json
import sys
from typing import Dict, Any, Optional
import argparse

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

def resolve_config(experiment_name: str, results_json_path: str, num_gpus: int, scale_down_mem_usage: int = 1) -> Dict[str, Any]:
    """Resolve experiment configuration and compute training parameters."""
    # Read results.json
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    if experiment_name not in results["experiments"]:
        raise ValueError(f"Experiment '{experiment_name}' not found in {results_json_path}")
    
    # Get experiment config
    config = results["experiments"][experiment_name]["config"]
    
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
    
    # Extract and compute training parameters
    training_params = config.get("training_params", {})
    if not training_params:
        raise ValueError("No training parameters found")
        
    # Compute scaled parameters
    base_batch_size = training_params["batch_size"]
    batch_size = base_batch_size // scale_down_mem_usage
    base_grad_acc = training_params["gradient_accumulation_steps"]
    grad_acc = base_grad_acc * scale_down_mem_usage * 4 // num_gpus
    
    # Verify total batch size matches paper (128)
    total_batch_size = batch_size * num_gpus * grad_acc
    if total_batch_size != 128:
        print(f"Warning: Total batch size ({total_batch_size}) does not match paper (128)", 
              file=sys.stderr)
    
    # Return resolved config and computed parameters
    return {
        "config": config,
        "training_params": {
            "learning_rate": training_params["learning_rate"],
            "batch_size": batch_size,
            "base_batch_size": base_batch_size,
            "num_epochs": training_params["num_epochs"],
            "weight_decay": training_params.get("weight_decay", 0.1),
            "warmup_ratio": training_params.get("warmup_ratio", 0.05),
            "gradient_accumulation_steps": grad_acc,
            "base_gradient_accumulation_steps": base_grad_acc,
            "optimizer": {
                "adam_beta1": training_params.get("optimizer", {}).get("adam_beta1", 0.9),
                "adam_beta2": training_params.get("optimizer", {}).get("adam_beta2", 0.95),
                "adam_epsilon": training_params.get("optimizer", {}).get("adam_epsilon", 1e-8),
                "no_decay_params": training_params.get("optimizer", {}).get(
                    "no_decay_params", ["bias", "LayerNorm.weight"]
                )
            }
        },
        "total_batch_size": total_batch_size
    }

def main():
    parser = argparse.ArgumentParser(description="Resolve experiment configuration")
    parser.add_argument("experiment_name", help="Name of the experiment")
    parser.add_argument("--results-json", required=True, help="Path to results.json")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--scale-down-mem", type=int, default=1, 
                       help="Factor to scale down memory usage")
    args = parser.parse_args()
    
    try:
        result = resolve_config(
            args.experiment_name,
            args.results_json,
            args.num_gpus,
            args.scale_down_mem
        )
        # Output as JSON for easy parsing in bash
        print(json.dumps(result))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()