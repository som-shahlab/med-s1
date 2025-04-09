"""
Helper functions for analyzing different extraction methods and transformations.

This module provides functions to analyze and compare different methods of extracting
clinical reasoning steps from medical case discussions.
"""

import os
import json
import glob
import logging
import asyncio
import pandas as pd
import re
from typing import List, Optional, Dict, Any, Union, Tuple
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get MED_S1_DIR from environment
MED_S1_DIR = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')

def load_results_json() -> dict:
    """Load the results.json file."""
    with open(os.path.join(MED_S1_DIR, "results.json"), 'r') as f:
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
    
    # Get raw dataset path from results.json
    raw_path = None
    if "results" in exp_data and "curation" in exp_data["results"]:
        # Get path to curated dataset
        raw_path = exp_data["results"]["curation"].get("dataset_path")
        if raw_path:
            # Convert formatted path to curated path
            raw_path = os.path.join(os.path.dirname(raw_path), "med_s1k_curated.parquet")
    
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
    """Get both raw and formatted training examples at specified index."""
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

# Import clinical formatting functions
from curation_methods.clinical_formatting import (
    transform_to_cot,
    transform_to_nejmcr_steps,
    transform_to_gemini,
    transform_to_nejmcr_transform,
    transform_to_gemini_nejmcr,
    transform_to_nejmcr_qa,
    transform_to_nejmcr_reason
)

async def transform_async(index: int, transforms: List[Union[str, Tuple[str, str]]], experiment_name: str = "medqa-nejmcr-1k-random") -> Dict[str, Any]:
    """Analyze different extraction methods and transformations on a specific example.
    
    Args:
        index: Index in the dataset to analyze
        transforms: List of transformations to apply. Each can be either:
            - "cot" - Apply Chain of Thought extraction
            - "nejmcr" - Apply NEJM Case Report extraction
            - "nejm-qa" or "nejmcr-qa" - Extract Q&A from NEJM Case Report
            - "nejmcr-reason" - Extract reasoning from NEJM Case Report
            - A string with a custom prompt template - will transform the CoT field
            - Tuple[str, str] where:
                - First element is the field to transform:
                    - "cot" - Transform the chain of thought reasoning
                    - "question" - Transform the question text
                    - "answer" - Transform the final answer
                    - "question-answer" - Transform both question and answer, returns in format:
                        Question: <question>
                        Answer: <answer>
                - Second element is the custom prompt template that will be formatted with:
                    - {cot}: The chain of thought reasoning
                    - {question}: The question text
                    - {answer}: The final answer
            - Any other string will be used as a custom prompt for transforming the CoT (for backwards compatibility)
        experiment_name: Name of experiment to get example from (default: "medqa-nejmcr-1k-random")
            
    Returns:
        Dictionary containing:
        - original: Original example from dataset with fields:
            - Question: The question text
            - Complex_CoT: The reasoning trace
            - Response: The final answer
        - transformed: Final transformed version after all transformations
        - intermediate: List of intermediate results after each transform
    """
    logger.info(f"Analyzing example {index} from {experiment_name}")
    logger.info(f"Applying transforms: {transforms}")

    # Validate transforms
    if not transforms:
        raise ValueError("Must provide at least one transformation")
    
    # Load config to get model key
    try:
        with open(os.path.join(MED_S1_DIR, "config.json"), 'r') as f:
            config = json.load(f)
        model_key = config.get("model_choices", {}).get("curation", "gemini-2.0-flash")
        logger.info(f"Using model: {model_key}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise
    
    # Get example from dataset
    example = get_train_example(experiment_name, index)
    if not example or not example["raw"]:
        raise ValueError(f"Could not find example at index {index}")
    
    # Track original and intermediate results
    # Initialize result with original content
    result = {
        "original": {
            "question": example["raw"]["Question"],
            "reasoning": example["raw"]["Complex_CoT"],
            "answer": example["raw"]["Response"]
        },
        "transformed": {
            "question": example["raw"]["Question"].replace("\nWhat is the diagnosis of the patient?", ""),
            "reasoning": example["raw"]["Complex_CoT"],
            "answer": example["raw"]["Response"]
        },
        "intermediate": []
    }
    
    # Apply each transformation in sequence
    for transform in transforms:
        try:
            # Validate and normalize transform
            if isinstance(transform, tuple):
                field, _ = transform
                if field not in ["cot", "question", "answer", "question-answer"]:
                    raise ValueError(f"Invalid field to transform: {field}")
                transform_type = field  # Use field as transform type
            else:
                transform_type = transform.lower() if isinstance(transform, str) else str(transform)
                # Default to cot for string transforms
                field = "cot"
                valid_transforms = ["cot", "nejmcr", "gemini", "nejmcr-transform", "gemini-nejmcr", "nejmcr-qa", "nejmcr-reason", "nejm-qa", "nejm-reason"]
                # Normalize transform names
                if transform_type == "nejm-qa":
                    transform_type = "nejmcr-qa"
                elif transform_type == "nejm-reason":
                    transform_type = "nejmcr-reason"
                if transform_type not in valid_transforms:
                    # For non-standard transforms, treat as custom CoT prompt
                    if not isinstance(transform, str) or not transform.strip():
                        raise ValueError(f"Invalid transform: {transform}")
            
            logger.info(f"Applying transform: {transform[:100]}...")
            
            # Store current state before transform
            intermediate = {
                "transform": transform,
                "question": result["transformed"]["question"],
                "reasoning": result["transformed"]["reasoning"],
                "answer": result["transformed"]["answer"]
            }

            if transform_type == "cot":
                # For CoT, generate new reasoning from Q&A
                transformed = await transform_to_cot(
                    result["transformed"]["question"],
                    result["transformed"]["answer"],
                    model_key
                )
                result["transformed"]["reasoning"] = transformed
            elif transform_type == "nejmcr":
                # For NEJMCR, transform reasoning using answer
                transformed = await transform_to_nejmcr_steps(
                    result["transformed"]["reasoning"],
                    model_key,
                    result["transformed"]["answer"]
                )
                result["transformed"]["reasoning"] = transformed
            elif transform_type == "gemini":
                # For Gemini, generate new reasoning from Q&A
                transformed = await transform_to_gemini(
                    result["transformed"]["question"],
                    model_key,
                    result["transformed"]["answer"]
                )
                result["transformed"]["reasoning"] = transformed
            elif transform_type == "nejmcr-transform":
                # For NEJMCR transform, transform reasoning using answer
                transformed = await transform_to_nejmcr_transform(
                    result["transformed"]["reasoning"],
                    model_key,
                    result["transformed"]["answer"]
                )
                result["transformed"]["reasoning"] = transformed
            elif transform_type == "gemini-nejmcr":
                # For Gemini-NEJMCR, enhance reasoning using all components
                transformed = await transform_to_gemini_nejmcr(
                    result["transformed"]["question"],
                    model_key,
                    result["transformed"]["answer"],
                    result["transformed"]["reasoning"]
                )
                result["transformed"]["reasoning"] = transformed
            elif transform_type == "nejmcr-qa":
                # For NEJMCR Q&A, transform Q&A
                transformed = await transform_to_nejmcr_qa(
                    result["transformed"]["question"],
                    result["transformed"]["reasoning"],
                    result["transformed"]["answer"],
                    model_key
                )
                # Parse Q&A result
                lines = transformed.strip().split('\n')
                for line in lines:
                    if line.startswith("Question:"):
                        result["transformed"]["question"] = line[9:].strip()
                    elif line.startswith("Answer:"):
                        result["transformed"]["answer"] = line[7:].strip()
            elif transform_type == "nejmcr-reason":
                # For NEJMCR reason, generate new reasoning based on Q&A
                transformed = await transform_to_nejmcr_reason(
                    result["transformed"]["reasoning"],  # cot
                    result["transformed"]["question"],   # question
                    result["transformed"]["answer"],     # answer
                    model_key
                )
                result["transformed"]["reasoning"] = transformed
            else:
                # Handle custom prompt transformation
                from utils.openai_utils import get_model_response
                
                # Format prompt with current transformed values
                current_values = {
                    "cot": result["transformed"]["reasoning"],
                    "question": result["transformed"]["question"],
                    "answer": result["transformed"]["answer"]
                }
                
                if isinstance(transform, tuple) and len(transform) == 2:
                    field, prompt_template = transform
                    # Format prompt with all values
                    prompt = prompt_template.format(
                        cot=current_values["cot"],
                        question=current_values["question"],
                        answer=current_values["answer"]
                    )
                    # Transform specified field
                    transformed = await get_model_response(prompt, model=model_key, max_tokens=8192)
                    
                    # Handle special case for question-answer
                    if field == "question-answer":
                        # Parse transformed result into question and answer
                        lines = transformed.strip().split('\n')
                        question = ""
                        answer = ""
                        for line in lines:
                            if line.startswith("Question:"):
                                question = line[9:].strip()
                            elif line.startswith("Answer:"):
                                answer = line[7:].strip()
                        if question and answer:
                            result["transformed"]["question"] = question
                            result["transformed"]["answer"] = answer
                        else:
                            raise ValueError("Transformed result did not contain valid Question/Answer format")
                    elif field == "cot":
                        result["transformed"]["reasoning"] = transformed
                    elif field == "question":
                        result["transformed"]["question"] = transformed
                    elif field == "answer":
                        result["transformed"]["answer"] = transformed
                    else:
                        raise ValueError(f"Invalid field to transform: {field}")
                else:
                    # Legacy support - transform CoT only
                    prompt = transform.format(
                        cot=current_values["cot"],
                        question=current_values["question"],
                        answer=current_values["answer"]
                    )
                    transformed = await get_model_response(prompt, model=model_key, max_tokens=8192)
                    result["transformed"]["reasoning"] = transformed
            
            if not transformed:
                raise ValueError(f"Transform produced no result: {transform}")
            
            # Store intermediate state after this transform
            result["intermediate"].append(intermediate)
            logger.info(f"Transform complete. Updated {field}")
            
        except Exception as e:
            logger.error(f"Error applying transform {transform}: {e}")
            raise
    
    # Print results
    print("\nORIGINAL:")
    print("=" * 80)
    print(f"Question: {result['original']['question']}")
    print(f"Reasoning: {result['original']['reasoning']}")
    print(f"Answer: {result['original']['answer']}")
    
    print("\nTRANSFORMED:")
    print("=" * 80)
    print(f"Question: {result['transformed']['question']}")
    print(f"Reasoning: {result['transformed']['reasoning']}")
    print(f"Answer: {result['transformed']['answer']}")
    
    logger.info("Analysis complete")
    return result

def transform(index: int, transforms: List[str], experiment_name: str = "medqa-nejmcr-1k-random") -> Dict[str, Any]:
    """Non-async wrapper for transform_async. See transform_async for full documentation."""
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    return asyncio.run(transform_async(index, transforms, experiment_name))

def main():
    """Command line interface for testing different extraction methods."""
    import argparse
    parser = argparse.ArgumentParser(description='Test different extraction methods on specific examples')
    parser.add_argument('index', type=int, help='Index in the dataset to analyze')
    parser.add_argument('--method', type=str, choices=['gemini', 'nejmcr', 'nejmcr-transform', 'gemini-nejmcr'],
                      required=True, help='Extraction method to test')
    parser.add_argument('--experiment', type=str, default="medqa-nejmcr-1k-random",
                      help='Name of experiment to get example from')
    
    args = parser.parse_args()
    
    # Run the transformation
    result = transform(args.index, [args.method], args.experiment)
    # Print results in a clear format
    print("\nQUESTION:")
    print("=" * 80)
    print(result["original"]["question"])
    
    print("\nORIGINAL REASONING:")
    print("=" * 80)
    print(result["original"]["reasoning"])
    
    print("\nCORRECT ANSWER:")
    print("=" * 80)
    print(result["original"]["answer"])
    
    print(f"\n{args.method.upper()} TRANSFORMED:")
    print("=" * 80)
    print(f"Question: {result['transformed']['question']}")
    print(f"Reasoning: {result['transformed']['reasoning']}")
    print(f"Answer: {result['transformed']['answer']}")
    print(result["transformed"])

if __name__ == "__main__":
    main()