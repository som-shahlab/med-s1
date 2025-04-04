import os
import glob
import pandas as pd
import re
import json
import datasets
from datasets import Dataset
from typing import Dict, Tuple, Any, Optional, List, Literal, Union
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
    
    # Get formatted dataset path from curation results
    formatted_path = None
    if "results" in exp_data and "curation" in exp_data["results"]:
        base_formatted_path = exp_data["results"]["curation"]["dataset_path"]
        formatted_dir = os.path.dirname(base_formatted_path)
        # Ensure exact experiment name match for formatted path
        formatted_pattern = f"{os.path.basename(formatted_dir)}_*"
        formatted_path = get_latest_dataset_path(os.path.dirname(formatted_dir), formatted_pattern)
    
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
    
    # Get formatted example
    formatted_example = None
    if formatted_path:
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