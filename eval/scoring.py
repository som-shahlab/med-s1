import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scorer import get_results

def save_and_score_results(
    args,
    results: List[Dict],
    model_path: str,
    task_name: str
) -> Tuple[str, str, Dict, Optional[List[str]]]:
    """Save evaluation results and compute metrics.
    
    Args:
        args: Command line arguments
        results: List of results with multiple runs per sample
        model_path: Path to the model used
        task_name: Name of the evaluation task
    
    Returns:
        Tuple of:
        - Path to output directory
        - Path to metrics file
        - Metrics dictionary
        - List of approaches (if any)
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.path_to_output_dir, f"{task_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(output_dir, "eval_data.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    # Compute metrics
    metrics = get_results(results_file)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "eval_data_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Print metrics summary
    print("\nMetrics summary:")
    for source, source_metrics in metrics.items():
        print(f"\n{source}:")
        print(f"  Accuracy: {source_metrics['accuracy']*100:.1f}%")
        print(f"  95% CI: [{source_metrics['accuracy_ci'][0]*100:.1f}%, {source_metrics['accuracy_ci'][1]*100:.1f}%]")
        print(f"  Total samples: {source_metrics['total_samples']}")
        print(f"  Runs per sample: {source_metrics['runs_per_sample']}")
    
    return output_dir, metrics_file, metrics, None

def update_results_json(
    args,
    path_to_output: str,
    metrics_file: str,
    metrics: Dict,
    approaches: Optional[List[str]],
    results: List[Dict]
) -> None:
    """Update results.json with evaluation results.
    
    Args:
        args: Command line arguments
        path_to_output: Path to output directory
        metrics_file: Path to metrics file
        metrics: Metrics dictionary
        approaches: List of approaches (if any)
        results: List of results with multiple runs
    """
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
    
    # Load existing results
    with open(results_json, "r") as f:
        all_results = json.load(f)
    
    # Get experiment
    if args.experiment_name not in all_results["experiments"]:
        all_results["experiments"][args.experiment_name] = {}
    experiment = all_results["experiments"][args.experiment_name]
    
    # Initialize results if needed
    if "results" not in experiment:
        experiment["results"] = {}
    if "eval" not in experiment["results"]:
        experiment["results"]["eval"] = {}
    
    # Update eval results - store only file paths, not metrics
    eval_results = experiment["results"]["eval"]
    eval_results.update({
        "timestamp": datetime.now().isoformat(),
        "output_dir": path_to_output,
        "metrics_file": metrics_file
    })
    
    # Add approaches if present
    if approaches:
        eval_results["approaches"] = approaches
    
    # Save updated results
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated {results_json} with evaluation results")