#!/usr/bin/env python3
import json
import copy
import sys
import os
from typing import Dict, Any, Optional, Set, Tuple
import hashlib
from pathlib import Path

def validate_json_file(path: str) -> None:
    """Validate that the file exists and is valid JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    
    try:
        with open(path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading {path}: {str(e)}")

def hash_dict(d: Dict) -> str:
    """Create a stable hash of a dictionary."""
    # Sort the dictionary to ensure stable hashing
    serialized = json.dumps(d, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]

def find_first_occurrence(experiments: Dict[str, Any], target_hash: str, field: str) -> Optional[str]:
    """Find the first experiment name that has the same hash for the given field."""
    for exp_name, exp_data in experiments.items():
        if "config" in exp_data and field in exp_data["config"]:
            current = exp_data["config"][field]
            if current is not None and hash_dict(current) == target_hash:
                return exp_name
    return None

def clean_up_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up results by replacing duplicate training_params and datasets with references."""
    # Create a deep copy to avoid modifying the input
    results = copy.deepcopy(results)
    
    # Track processed hashes to avoid circular references
    processed_training_params: Set[str] = set()
    processed_datasets: Set[str] = set()
    
    # Track statistics for reporting
    stats = {
        "training_params_replaced": 0,
        "datasets_replaced": 0
    }
    
    # Process each experiment
    for exp_name, exp_data in results["experiments"].items():
        if "config" not in exp_data:
            continue
            
        # Process training_params
        if "training_params" in exp_data["config"] and exp_data["config"]["training_params"] is not None:
            tp_hash = hash_dict(exp_data["config"]["training_params"])
            if tp_hash in processed_training_params:
                first_occurrence = find_first_occurrence(results["experiments"], tp_hash, "training_params")
                if first_occurrence and first_occurrence != exp_name:
                    exp_data["config"]["training_params"] = f"same as {first_occurrence}"
                    stats["training_params_replaced"] += 1
            else:
                processed_training_params.add(tp_hash)
        
        # Process datasets
        if "datasets" in exp_data["config"] and exp_data["config"]["datasets"] is not None:
            ds_hash = hash_dict(exp_data["config"]["datasets"])
            if ds_hash in processed_datasets:
                first_occurrence = find_first_occurrence(results["experiments"], ds_hash, "datasets")
                if first_occurrence and first_occurrence != exp_name:
                    exp_data["config"]["datasets"] = f"same as {first_occurrence}"
                    stats["datasets_replaced"] += 1
            else:
                processed_datasets.add(ds_hash)
    
    # Print statistics
    print(f"\nReplacements made:")
    print(f"  Training params: {stats['training_params_replaced']}")
    print(f"  Datasets: {stats['datasets_replaced']}")
    
    return results

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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up results.json by replacing duplicate configs with references")
    parser.add_argument("--results-json", default="results.json",
                       help="Path to results.json file (default: results.json)")
    parser.add_argument("--backup", action="store_true",
                       help="Create a backup of the original file before modifying")
    args = parser.parse_args()
    
    try:
        # Validate input file
        validate_json_file(args.results_json)
        
        # Create backup if requested
        if args.backup:
            backup_path = f"{args.results_json}.bak"
            print(f"Creating backup at {backup_path}...")
            import shutil
            shutil.copy2(args.results_json, backup_path)
        
        # Read the results.json file
        print(f"Reading {args.results_json}...")
        with open(args.results_json, "r") as f:
            results = json.load(f)
        
        if "experiments" not in results:
            raise ValueError(f"{args.results_json} does not contain an 'experiments' section")
        
        # Clean up the results
        print("Processing experiments...")
        cleaned_results = clean_up_results(results)
        
        # Write the cleaned results back to the file
        print(f"Writing cleaned results back to {args.results_json}...")
        with open(args.results_json, "w") as f:
            json.dump(cleaned_results, f, indent=2)
        
        print("\nSuccess! File has been optimized.")
        if args.backup:
            print(f"Original file backed up to: {backup_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()