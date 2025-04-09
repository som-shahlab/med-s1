import os
import glob
import pandas as pd
import re
import json
import datasets
import wandb
import numpy as np
from datasets import Dataset
from typing import Dict, Tuple, Any, Optional, List, Literal, Union, Literal
from transformers import AutoTokenizer
import numpy as np
from scorer import match_choice

def load_results_json() -> dict:
    """Load the results.json file."""
    with open("/share/pi/nigam/users/calebwin/med-s1/results.json", 'r') as f:
        return json.load(f)

def get_latest_dataset_path(base_path: str, pattern: str) -> str:
    """Get the most recent dataset path matching the pattern."""
    matching_dirs = glob.glob(os.path.join(base_path, pattern))
    if not matching_dirs:
        return None
    return sorted(matching_dirs, key=lambda x: re.search(r'_(\d+_\d+)', x).group(1), reverse=True)[0]

def get_experiment_paths(experiment_name: str) -> Tuple[str, str]:
    """Get paths to raw and formatted datasets for an experiment."""
    results = load_results_json()
    if experiment_name not in results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in results.json")
    
    exp_data = results["experiments"][experiment_name]
    
    # Get raw dataset path
    raw_base_dir = "/share/pi/nigam/users/calebwin/hf_cache/med-s1k/"
    # Ensure exact experiment name match by adding underscore
    raw_pattern = f"{experiment_name}_*"
    raw_dir = get_latest_dataset_path(raw_base_dir, raw_pattern)
    raw_path = os.path.join(raw_dir, "med_s1k_curated.parquet") if raw_dir else None
    
    # Get formatted dataset path from results.json
    formatted_path = None
    if "results" in exp_data and "curation" in exp_data["results"]:
        # First try to get path directly from results
        formatted_path = exp_data["results"]["curation"]["dataset_path"]
        
        # If path doesn't exist, try to find latest version
        if not os.path.exists(formatted_path):
            base_formatted_path = formatted_path
            formatted_dir = os.path.dirname(base_formatted_path)
            formatted_pattern = f"{os.path.basename(formatted_dir)}_*"
            latest_path = get_latest_dataset_path(os.path.dirname(formatted_dir), formatted_pattern)
            if latest_path:
                formatted_path = os.path.join(latest_path, "med_s1k_formatted")
    
    return raw_path, formatted_path

def get_train_example(experiment_name: str, index: int, print_format: bool = False) -> Dict[str, Any]:
    """Get both raw and formatted training examples at specified index.
    
    Args:
        experiment_name: Name of the experiment (e.g., "medqa-nejmcr-1k-random").
                       Will only match exact experiment name followed by timestamp.
        index: Index of the example to retrieve from the dataset.
        print_format: If True, prints the example in a formatted way with sections:
                     QUESTION, REASONING, ANSWER
    
    Returns:
        Dictionary containing:
        - raw: Raw example from parquet file with fields:
            - Question: The question text
            - Complex_CoT: The reasoning trace
            - Response: The final answer
            - Complex_CoT_orig: Original reasoning (if available)
            - Complex_CoT_extracted: Extracted reasoning (if available)
        - formatted: Formatted example from HuggingFace dataset (if available)
    
    Example:
        >>> get_train_example("medqa-nejmcr-1k-random", 0, print_format=True)
        QUESTION:
        [question text]
        
        REASONING:
        [reasoning text]
        
        ANSWER:
        [answer text]
        
        >>> result = get_train_example("medqa-nejmcr-1k-random", 0)
        >>> print(result["raw"]["Question"])
    """
    raw_path, formatted_path = get_experiment_paths(experiment_name)
    
    # Get raw example
    raw_example = None
    if raw_path and os.path.exists(raw_path):
        df = pd.read_parquet(raw_path)
        if index < len(df):
            raw_example = dict(df.iloc[index])
    
    # Get formatted example from HuggingFace dataset
    formatted_example = None
    if formatted_path and os.path.exists(formatted_path):
        try:
            # Try loading as a complete dataset first
            dataset = Dataset.load_from_disk(formatted_path)
            if index < len(dataset):
                formatted_example = dataset[index]
        except Exception:
            # If that fails, try loading the train split
            train_path = os.path.join(formatted_path, "train")
            if os.path.exists(train_path):
                dataset = Dataset.load_from_disk(train_path)
                if index < len(dataset):
                    formatted_example = dataset[index]
    
    result = {
        "raw": raw_example,
        "formatted": formatted_example
    }
    
    if print_format and result["raw"]:
        print("QUESTION:")
        print(result["raw"]["Question"])
        print("\nREASONING:")
        print(result["raw"]["Complex_CoT"])
        print("\nANSWER:")
        print(result["raw"]["Response"])
    
    return None if print_format else result

def score_example(example: Dict) -> bool:
    """Score a single example as correct or wrong."""
    output = example['output']
    ans, _ = match_choice(output, example['options'])
    return ans[0].lower() == example['answer_idx'].lower()

def get_eval_example(
    experiment_names: Union[str, List[str]],
    index: int,
    print_format: bool = False,
    filter_score: Union[Literal["any"], List[Literal["correct", "wrong"]]] = "any"
) -> Dict[str, Any]:
    """Get evaluation example and results at specified index for one or more experiments.
    
    Args:
        experiment_names: Single experiment name or list of experiment names to compare.
        index: Index of the example to retrieve after filtering.
        print_format: If True, prints examples in a formatted way:
                     - Shows question and expected answer once
                     - Shows response and score for each experiment
        filter_score: How to filter examples:
                     - "any": No filtering
                     - List of "correct"/"wrong": Must match pattern for each experiment
                       e.g., ["correct", "wrong"] finds examples where first experiment
                       is correct and second is wrong
    
    Returns:
        Dictionary containing:
        - examples: List of (experiment_name, example) tuples where example contains:
            - question: The question text
            - options: Answer choices
            - answer_idx: Correct answer index
            - answer: Correct answer text
            - output: Model's response
        - metrics: Dictionary mapping experiment names to their metrics
    
    Examples:
        # Get single example
        >>> get_eval_example("medqa-nejmcr-1k-random", 0, print_format=True)
        
        # Compare responses from two experiments
        >>> get_eval_example(
        ...     ["exp1", "exp2"],
        ...     0,
        ...     print_format=True
        ... )
        
        # Find examples where first is correct but second is wrong
        >>> get_eval_example(
        ...     ["exp1", "exp2"],
        ...     0,
        ...     filter_score=["correct", "wrong"]
        ... )
    """
    results = load_results_json()
    
    # Convert single experiment to list for uniform handling
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]
        if filter_score != "any":
            filter_score = [filter_score]
    
    # Validate filter_score
    if filter_score != "any" and len(filter_score) != len(experiment_names):
        raise ValueError("If filter_score is a list, it must match length of experiment_names")
    
    all_examples = []
    for i, experiment_name in enumerate(experiment_names):
        if experiment_name not in results["experiments"]:
            raise ValueError(f"Experiment {experiment_name} not found in results.json")
        
        exp_data = results["experiments"][experiment_name]
        
        # Get eval outputs
        if "results" in exp_data and "eval" in exp_data["results"]:
            outputs_path = exp_data["results"]["eval"]["outputs_path"]
            if outputs_path and os.path.exists(outputs_path):
                with open(outputs_path, 'r') as f:
                    eval_data = json.load(f)
                if index < len(eval_data):
                    # Get all examples that match filter criteria
                    filtered_data = []
                    for j, ex in enumerate(eval_data):
                        if filter_score == "any":
                            filtered_data.append((j, ex))
                        else:
                            is_correct = score_example(ex)
                            if (filter_score[i] == "correct" and is_correct) or \
                            (filter_score[i] == "wrong" and not is_correct):
                                filtered_data.append((j, ex))
                    
                    if filtered_data and index < len(filtered_data):
                        _, example = filtered_data[index]
                        all_examples.append((experiment_name, example))
    
    if not all_examples:
        return {"examples": None, "metrics": None}

    # Get metrics for all experiments
    metrics = {}
    for experiment_name, _ in all_examples:
        metrics_path = os.path.join(os.path.dirname(results["experiments"][experiment_name]["results"]["eval"]["outputs_path"]), "eval_data_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics[experiment_name] = json.load(f)
    
    result = {
        "examples": all_examples,
        "metrics": metrics
    }
    
    if print_format and result["examples"]:
        # Print shared information once
        first_example = result["examples"][0][1]
        print("QUESTION:")
        print(first_example["question"])
        print("\nEXPECTED ANSWER:")
        print(f"Index: {first_example['answer_idx']}")
        print(f"Answer: {first_example['answer']}")
        
        # Print responses and scores for each experiment
        for experiment_name, example in result["examples"]:
            print(f"\nRESPONSE ({experiment_name}):")
            print(example["output"])
            
            # Add scoring output
            is_correct = score_example(example)
            print(f"\nSCORE ({experiment_name}):")
            print("✅ Correct" if is_correct else "❌ Wrong")
    
    return None if print_format else result

def get_token_stats(
    experiment_name: str,
    component: Literal["questions", "reasoning_answers", "full", "reasoning"] = "full",
    print_format: bool = False
) -> Dict[str, Any]:
    """Get token count statistics for different components of the dataset.
    
    This function analyzes token counts for questions, reasoning+answers, or full examples
    from a dataset to help inform hyperparameter choices and training decisions.
    
    Args:
        experiment_name: Name of the experiment to analyze
        component: Which component to analyze:
            - "questions": Only the questions
            - "reasoning_answers": Reasoning and answers combined
            - "full": Questions + reasoning + answers (default)
            - "reasoning": Only the reasoning
        print_format: If True, prints a summary of statistics
        
    Returns:
        Dictionary containing:
        - dataset_size: Number of examples analyzed
        - min_tokens: Minimum token count
        - max_tokens: Maximum token count
        - avg_tokens: Average token count
        - med_tokens: Median token count
        - total_tokens: Total tokens across all examples
        - under_8192_count: Number of examples under 8192 tokens
        - under_8192_percent: Percentage of examples under 8192 tokens
    """
    # Get dataset path
    raw_path, _ = get_experiment_paths(experiment_name)
    if not raw_path or not os.path.exists(raw_path):
        raise ValueError(f"Could not find raw dataset for experiment {experiment_name}")
    
    # Load dataset
    df = pd.read_parquet(raw_path)
    
    # Load model config
    with open("/share/pi/nigam/users/calebwin/med-s1/config.json", 'r') as f:
        config = json.load(f)
    
    # Get model key from experiment config
    results = load_results_json()
    model_key = results["experiments"][experiment_name]["config"]["model_key"]
    
    # Get HF path from config
    if model_key not in config["models"]:
        raise ValueError(f"Unknown model key: {model_key}")
    
    hf_path = config["models"][model_key]["hf_path"]
    
    # Get tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    # Define which columns to analyze based on component
    if component == "questions":
        columns = ["Question"]
    elif component == "reasoning_answers":
        columns = ["Complex_CoT", "Response"]
    elif component == "reasoning":
        columns = ["Complex_CoT"]
    else:  # full
        columns = ["Question", "Complex_CoT", "Response"]
    
    # Verify columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")
    
    # Process each row
    token_counts = []
    for _, row in df.iterrows():
        # Combine specified columns with space between
        text = " ".join(str(row[col]) for col in columns if pd.notna(row[col]))
        token_count = len(tokenizer.encode(text))
        token_counts.append(token_count)
    
    # Calculate statistics
    token_counts = np.array(token_counts)
    under_8192 = np.sum(token_counts < 8192)
    
    stats = {
        "dataset_size": len(token_counts),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "avg_tokens": float(np.mean(token_counts)),
        "med_tokens": float(np.median(token_counts)),
        "total_tokens": int(np.sum(token_counts)),
        "under_8192_count": int(under_8192),
        "under_8192_percent": float(under_8192 * 100 / len(token_counts)),
        "analyzed_columns": columns
    }
    
    if print_format:
        print(f"\nToken Count Analysis for {experiment_name} ({component}):")
        print("=" * 80)
        print(f"Analyzed columns: {stats['analyzed_columns']}")
        print(f"Dataset size: {stats['dataset_size']:,}")
        print(f"Min tokens: {stats['min_tokens']:,}")
        print(f"Max tokens: {stats['max_tokens']:,}")
        print(f"Avg tokens: {stats['avg_tokens']:.2f}")
        print(f"Med tokens: {stats['med_tokens']:.2f}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Samples under 8192 tokens: {stats['under_8192_count']:,} ({stats['under_8192_percent']:.1f}%)")
        
        return None
    
    return stats

def get_training_metrics(experiment_name: str, print_format: bool = False) -> Dict[str, Any]:
    """Get training metrics from WANDB logs for a given experiment.
    
    This function loads the WANDB logs for the latest run of an experiment and extracts
    key metrics at important points during training (start, inflection, end, and evenly
    spaced points).
    
    Args:
        experiment_name: Name of the experiment to analyze
        print_format: If True, prints a summary of metrics in a readable format
        
    Returns:
        Dictionary containing metrics at key points:
        - start: Metrics at start of training
        - inflection: Metrics at inflection point (where improvement slows)
        - end: Metrics at end of training
        - step-1, step-2, step-3: Metrics at evenly spaced points between start and end
        - run_info: Contains total_steps and total_epochs
        
        Each metrics point contains:
        - train_loss: Training loss
        - val_loss: Validation loss
        - train_accuracy: Training accuracy
        - val_accuracy: Validation accuracy
        - learning_rate: Learning rate
        - step: Training step number
        - epoch: Training epoch number
    """
    # Initialize WANDB
    api = wandb.Api()
    
    # Find the latest run for this experiment
    runs = api.runs("ehr-fm/med-s1", {"config.experiment_name": experiment_name})
    if not runs:
        raise ValueError(f"No WANDB runs found for experiment {experiment_name}")
    
    # Get the latest run
    run = runs[0]
    
    # Get the history of metrics
    history = pd.DataFrame(run.history())
    
    # Define the metrics we want to track
    metric_cols = {
        'train_loss': 'train_loss',
        'val_loss': 'val_loss',
        'train_accuracy': 'train_accuracy',
        'val_accuracy': 'val_accuracy',
        'learning_rate': 'learning_rate',
        'epoch': 'epoch',
        '_step': 'step'
    }
    
    # Helper function to get metrics at a specific step
    def get_metrics_at_step(step_idx):
        metrics = {'step': step_idx}  # Always use the actual step index
        
        # Helper to find closest non-NaN value
        def get_closest_value(col_name):
            if col_name not in history.columns:
                return None
            values = history[col_name]
            non_nan_indices = values.dropna().index
            if len(non_nan_indices) == 0:
                return None
            closest_idx = non_nan_indices[np.abs(non_nan_indices - step_idx).argmin()]
            return values[closest_idx]
        
        # Get closest value for each metric
        metrics['epoch'] = get_closest_value(metric_cols['epoch'])
        metrics['learning_rate'] = get_closest_value(metric_cols['learning_rate'])
        metrics['train_loss'] = get_closest_value(metric_cols['train_loss'])
        metrics['train_accuracy'] = get_closest_value(metric_cols['train_accuracy'])
        metrics['val_loss'] = get_closest_value(metric_cols['val_loss'])
        metrics['val_accuracy'] = get_closest_value(metric_cols['val_accuracy'])
        
        return metrics
    
    # Helper function to format metrics for printing
    def format_metrics(metrics, point_name):
        if metrics is None:
            return ""
        
        output = f"{point_name} (Step {metrics['step']}, Epoch {metrics['epoch']}):\n"
        if metrics.get('train_loss') is not None:
            output += f"  Train Loss: {metrics['train_loss']:.3f}\n"
        if metrics.get('val_loss') is not None:
            output += f"  Val Loss: {metrics['val_loss']:.3f}\n"
        if metrics.get('train_accuracy') is not None:
            output += f"  Train Accuracy: {metrics['train_accuracy']:.3f}\n"
        if metrics.get('val_accuracy') is not None:
            output += f"  Val Accuracy: {metrics['val_accuracy']:.3f}\n"
        if metrics.get('learning_rate') is not None:
            output += f"  Learning Rate: {metrics['learning_rate']:.2e}\n"
        return output
    
    # Helper function to format hyperparameters
    def format_hyperparameters(config):
        output = "Training Configuration:\n"
        output += f"  Learning Rate: {config.get('learning_rate', 'N/A'):.2e}\n"
        output += f"  Batch Size: {config.get('per_device_train_batch_size', 1) * config.get('gradient_accumulation_steps', 1)}\n"
        output += f"  Epochs: {config.get('num_train_epochs', 'N/A')}\n"
        output += f"  Warmup Ratio: {config.get('warmup_ratio', 'N/A'):.2f}\n"
        output += f"  Weight Decay: {config.get('weight_decay', 'N/A'):.2f}\n"
        output += f"  Block Size: {config.get('block_size', 'N/A'):,}\n"
        output += f"  Adam: β₁={config.get('adam_beta1', 'N/A'):.3f}, β₂={config.get('adam_beta2', 'N/A'):.3f}, ε={config.get('adam_epsilon', 'N/A'):.1e}\n"
        return output
    
    # Get total number of steps
    total_steps = len(history)
    if total_steps == 0:
        return None
    
    # Get metrics at key points
    start_metrics = get_metrics_at_step(0)  # Always start at step 0
    end_metrics = get_metrics_at_step(total_steps - 1)  # Use actual last step
    
    # Find inflection point using validation loss
    inflection_metrics = None
    val_loss_column = metric_cols['val_loss']
    if val_loss_column and val_loss_column in history.columns:
        val_losses = history[val_loss_column].dropna()
        if len(val_losses) > 2:
            # Simple inflection detection - point where improvement rate changes most
            diffs = np.diff(val_losses.values)
            inflection_idx = val_losses.index[np.argmax(np.abs(np.diff(diffs))) + 1]
            inflection_metrics = get_metrics_at_step(inflection_idx)
    
    # Get middle 3 parts evenly spaced across all steps
    if total_steps >= 5:
        # Get indices at approximately 25%, 50%, and 75% of total steps
        middle_indices = [
            total_steps // 4,  # 25%
            total_steps // 2,  # 50%
            (3 * total_steps) // 4  # 75%
        ]
        part_metrics = {
            f'part-{i+1}': get_metrics_at_step(idx)
            for i, idx in enumerate(middle_indices)
        }
    else:
        # If we have fewer than 5 points, just use what we have for middle points
        part_metrics = {}
        if total_steps > 2:
            for i in range(min(3, total_steps-2)):
                part_metrics[f'part-{i+1}'] = get_metrics_at_step(i+1)
    
    # Get hyperparameters from run config
    hyperparameters = {
        'learning_rate': run.config.get('learning_rate'),
        'batch_size': run.config.get('per_device_train_batch_size', 1) * run.config.get('gradient_accumulation_steps', 1),
        'num_epochs': run.config.get('num_train_epochs'),
        'warmup_ratio': run.config.get('warmup_ratio'),
        'weight_decay': run.config.get('weight_decay'),
        'block_size': run.config.get('block_size'),
        'adam_beta1': run.config.get('adam_beta1'),
        'adam_beta2': run.config.get('adam_beta2'),
        'adam_epsilon': run.config.get('adam_epsilon')
    }
    
    result = {
        'start': start_metrics,
        'inflection': inflection_metrics,
        'end': end_metrics,
        **part_metrics,  # Add step-1, step-2, step-3
        'run_info': {
            'total_steps': total_steps,
            'total_epochs': end_metrics['epoch'] if end_metrics and end_metrics.get('epoch') is not None else None,
            'hyperparameters': hyperparameters
        }
    }
    
    if print_format:
        print(f"Training Metrics Summary for {experiment_name}")
        print("-" * (35 + len(experiment_name)))
        print("\nHyperparameters:")
        print("-" * 15)
        print(format_hyperparameters(run.config))
        print("\nTraining Progress:")
        print("-" * 17)
        print("\n" + format_metrics(start_metrics, "Start"))
        if inflection_metrics:
            print("\n" + format_metrics(inflection_metrics, "Inflection Point"))
        for part_name, metrics in part_metrics.items():
            print("\n" + format_metrics(metrics, part_name.replace('-', ' ').title()))
        print("\n" + format_metrics(end_metrics, "End"))
    
    return result if not print_format else None