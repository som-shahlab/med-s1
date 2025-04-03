import os
import json
import re
import time
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from collections import Counter
from typing import Dict, List, Sequence, Optional
import logging
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.openai_utils import verify_answer
from utils.model_utils import get_base_model_answers
from utils.specialty_utils import load_specialties, batch_classify_specialties
from utils.workers import run_pipeline
from utils.formatting import format_for_training
from utils.path_utils import (
    get_experiment_dir,
    get_formatted_dataset_path,
    get_intermediate_path,
    get_final_paths,
    update_results_json,
    clean_experiment_name
)
from datetime import datetime
import random
import asyncio
import argparse

# Import curation methods
from curation_methods.base import full_dataset, random_sample_dataset, quality_filter
from curation_methods.s1 import process_s1_dataset, check_s1_prerequisites
from curation_methods.advanced import (
    novelty_answer_curation,
    difficulty_substring_curation,
    embedding_similarity_curation,
    embedding_diversity_curation,
    check_novelty_answer_prerequisites,
    check_embedding_prerequisites
)
from curation_methods.ngram_difficulty import ngram_difficulty_curation
from curation_methods.step_extraction import apply_step_extraction

def setup_logging():
    """Configure logging with consistent format"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Disable HTTP request logging
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)

def load_config() -> Dict:
    """Load configuration from config.json"""
    assert os.getenv("MED_S1_DIR") is not None, "MED_S1_DIR environment variable not set"
    with open(os.path.join(os.getenv("MED_S1_DIR"), "config.json"), "r") as f:
        return json.load(f)
    
def resolve_config_reference(config: Dict, key: str, results: Dict, visited: set = None) -> Dict:
    """Recursively resolve 'same as' references in config"""
    if visited is None:
        visited = set()
        
    # Base case: not a reference
    if not isinstance(config.get(key), str) or not config[key].startswith("same as "):
        return config.get(key, {})
        
    # Get referenced experiment
    ref_exp = config[key].replace("same as ", "")
    
    # Check for circular references
    if ref_exp in visited:
        raise ValueError(f"Circular reference detected: {ref_exp}")
    visited.add(ref_exp)
    
    # Get referenced config
    if ref_exp not in results["experiments"]:
        raise ValueError(f"Referenced experiment {ref_exp} not found")
    ref_config = results["experiments"][ref_exp]["config"]
    
    # Recursively resolve if the referenced config also has references
    if isinstance(ref_config.get(key), str) and ref_config[key].startswith("same as "):
        return resolve_config_reference(ref_config, key, results, visited)
        
    return ref_config.get(key, {})

def load_experiment_config(experiment_name: str) -> Dict:
    """Load experiment configuration from results.json and resolve references"""
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
        
    with open(results_json, "r") as f:
        results = json.load(f)

    # Handle both old and new format
    experiments = results.get("experiments", results)
    if experiment_name not in experiments:
        raise ValueError(f"Experiment {experiment_name} not found in {results_json}")
    
    # Get raw config
    exp_data = experiments[experiment_name]
    if not isinstance(exp_data, dict) or "config" not in exp_data:
        raise ValueError(f"Invalid experiment data format for {experiment_name}")
        
    config = exp_data["config"]
    if config is None:
        raise ValueError(f"Configuration for experiment {experiment_name} is None")
    
    # Resolve references for each top-level key
    resolved_config = {}
    for key in ["curation", "training_params", "datasets"]:
        resolved_config[key] = resolve_config_reference(config, key, {"experiments": experiments})
    
    # Copy other keys as-is
    for key in config:
        if key not in resolved_config:
            resolved_config[key] = config[key]
    
    return resolved_config

def get_output_dir() -> str:
    """Get the output directory from environment"""
    output_dir = os.environ.get('MED_S1K_OUTPUT')
    if not output_dir:
        raise ValueError("MED_S1K_OUTPUT environment variable not set")
    return output_dir

def load_base_dataset(experiment_config: Dict, config: Dict) -> pd.DataFrame:
    """Load the base dataset and initialize metadata columns"""
    # Get dataset name from experiment config
    dataset_name = experiment_config.get("datasets", {}).get("curate")
    
    if not dataset_name:
        logging.warning("No dataset specified in experiment config, using default")
        dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    else:
        # Get dataset config from config.json
        if dataset_name not in config.get("train_datasets", {}):
            raise ValueError(f"Dataset {dataset_name} not found in config.json")
        
        dataset_config = config["train_datasets"][dataset_name]
        
        # Load dataset based on config
        if "hf_path" in dataset_config:
            # Load from Hugging Face
            hf_path = dataset_config["hf_path"]
            hf_config = dataset_config.get("hf_config", None)
            hf_split = dataset_config.get("hf_split", "train")
            
            logging.info(f"Loading dataset {dataset_name} from {hf_path}")
            dataset = load_dataset(hf_path, hf_config, split=hf_split)
        elif "file_path" in dataset_config:
            # Load from local file
            file_path = dataset_config["file_path"]
            # Replace environment variables
            file_path = file_path.replace("${MED_S1_DIR}", os.environ.get("MED_S1_DIR", ""))
            
            logging.info(f"Loading dataset {dataset_name} from {file_path}")
            # Determine file type and load accordingly
            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data)
            elif file_path.endswith(".parquet"):
                dataset = Dataset.from_parquet(file_path)
            elif os.path.isdir(file_path):
                logging.info(f"Loading dataset from directory {file_path}")
                try:
                    dataset = load_from_disk(file_path)
                    logging.info(f"Successfully loaded dataset with {len(dataset)} examples")
                except Exception as e:
                    logging.error(f"Failed to load dataset from {file_path}: {e}")
                    raise
            else:
                raise ValueError(f"Unsupported file format for {file_path}")
        else:
            raise ValueError(f"Dataset {dataset_name} has no hf_path or file_path")
    
    # Convert to DataFrame and initialize metadata columns
    df = pd.DataFrame(dataset)
    logging.info(f"Converted dataset to DataFrame with {len(df)} rows and columns: {list(df.columns)}")
    
    # Do some dataset-specific cleaning
    if dataset_name == "s1-gemini-raw":
        # Align column names with HuatuoGPT SFT dataset
        df = df.rename(columns={
            "thinking_trajectories": "Complex_CoT",
            "question": "Question",
            "solution": "Response"
        })
        # For some reason, `thinking_trajectories` is a list of strings (of length 1). Unwrap it.
        df["Complex_CoT"] = df["Complex_CoT"].apply(lambda x: x[0] if isinstance(x, list) else x)
    elif dataset_name == "nejmcr":
        # Process NEJM case reports dataset
        logging.info(f"Processing NEJM case reports dataset with {len(df)} samples")
        
        # Format the columns
        df["Question"] = df["question"] + "\nWhat is the diagnosis of the patient?"
        df["Complex_CoT"] = df["thinking"]
        
        # Choose diagnosis based on priority order
        diagnosis_fields = [
            'diagnosis_final',
            'diagnosis_clinical_and_final',
            'diagnosis_pathological',
            'diagnosis_anatomical',
            'diagnosis_diagnosis_and_management',
            'diagnosis_diagnosis',
            'diagnosis_clinical',
            'diagnosis_laboratory',
            'diagnosis_psychiatric'
        ]
        
        def get_diagnosis(row):
            for field in diagnosis_fields:
                if field in row and pd.notna(row[field]):
                    return row[field]
            return "No diagnosis available"
            
        df["Response"] = df.apply(get_diagnosis, axis=1)
        
        # Calculate total tokens for each sample
        model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def get_total_tokens(row):
            text = f"{row['Question']} {row['Complex_CoT']} {row['Response']}"
            return len(tokenizer(text).input_ids)
        
        df['token_length'] = df.apply(get_total_tokens, axis=1)
        
        # Filter to samples under 8192 tokens
        n_total = len(df)
        df = df[df['token_length'] <= 8192].copy()
        n_filtered = len(df)
        
        logging.info(f"Filtered NEJM dataset from {n_total} to {n_filtered} samples (token length ≤ 8192)")
        
        # Keep only needed columns
        df = df[["Question", "Complex_CoT", "Response"]]
    
    # Sort by Question to ensure deterministic ordering across runs
    # This is critical for reproducibility when random sampling
    logging.info("Sorting dataset by Question for deterministic ordering")
    df = df.sort_values('Question', ascending=True).reset_index(drop=True)
    
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    df['selected_for_training'] = False
    return df

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    # Set seeds for other libraries that might be used
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Name of experiment from results.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Setup logging and load configs
    setup_logging()
    config = load_config()
    experiment_config = load_experiment_config(args.experiment)
    
    # Set all random seeds before any data loading or processing
    set_random_seeds(args.seed)
    logging.info(f"Set random seed to {args.seed}")
    
    # Get tokenizer for length checks
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Clean experiment name for filenames
    safe_experiment_name = clean_experiment_name(args.experiment)
    
    # Get curation parameters
    curation_method = experiment_config["curation"]["method"]
    n_samples = experiment_config["curation"]["n_samples"]
    specialty_weights = experiment_config["curation"].get("specialty_weights")
    
    # Get data directory
    data_dir = os.environ.get('DATA_DIR')
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    
    # Get output directory
    output_dir = get_output_dir()
    
    # Get dataset name
    dataset_name = experiment_config.get("datasets", {}).get("curate")
    
    # Check if base dataset exists
    filtered_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    
    # Process based on curation method
    if curation_method == "all":
        # For "all" method, load dataset and use full dataset
        if dataset_name != "nejmcr" and os.path.exists(filtered_path):
            df = pd.read_parquet(filtered_path)
            logging.info(f"Loaded existing dataset from {filtered_path}")
        else:
            df = load_base_dataset(experiment_config, config)
        df = full_dataset(df, experiment_config)
    
    elif curation_method == "random":
        # For "random" method, load dataset and randomly sample
        if dataset_name != "nejmcr" and os.path.exists(filtered_path):
            df = pd.read_parquet(filtered_path)
            logging.info(f"Loaded existing dataset from {filtered_path}")
        else:
            df = load_base_dataset(experiment_config, config)
        df = random_sample_dataset(df, n_samples, seed=args.seed)
    
    elif curation_method == "s1":
        # For "s1" method, process with S1 pipeline
        if os.path.exists(filtered_path):
            df = pd.read_parquet(filtered_path)
            logging.info(f"Loaded existing dataset from {filtered_path}")
            df = await process_s1_dataset(
                df, config, tokenizer, n_samples,
                specialty_weights, safe_experiment_name,
                output_dir, None,  # No need to run pipeline again
                seed=args.seed
            )
        else:
            df = load_base_dataset()
            df = await process_s1_dataset(
                df, config, tokenizer, n_samples,
                specialty_weights, safe_experiment_name,
                output_dir, run_pipeline,
                seed=args.seed
            )

            # Save processed dataset for future use
            os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
            df.to_parquet(filtered_path)
    
    elif curation_method == "difficulty-substring":
        # For "difficulty-substring" method, always use CPU and load directly
        df = difficulty_substring_curation(experiment_config, n_samples, seed=args.seed)
        
    elif curation_method == "difficulty-n-gram":
        # For "difficulty-n-gram" method, use n-gram based difficulty scoring
        df = ngram_difficulty_curation(experiment_config, n_samples, seed=args.seed)
    
    elif curation_method == "novelty-answer":
        # For "novelty-answer" method, need filtered dataset and embeddings
        if os.path.exists(filtered_path):
            df = pd.read_parquet(filtered_path)
            logging.info(f"Loaded existing dataset from {filtered_path}")
        else:
            # Create and save filtered dataset first
            df = load_base_dataset(experiment_config, config)
            df = await process_s1_dataset(
                df, config, tokenizer, n_samples,
                specialty_weights, safe_experiment_name,
                output_dir, run_pipeline,
                seed=args.seed
            )
            os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
            df.to_parquet(filtered_path)
        
        # Check if base_model_response column exists
        if 'base_model_response' not in df.columns:
            logging.error("base_model_response column not found in dataset. This is required for novelty-answer curation.")
            raise ValueError("base_model_response column not found in dataset. This is required for novelty-answer curation.")
        
        # Apply novelty-answer curation
        df = novelty_answer_curation(df, experiment_config, n_samples, seed=args.seed)
    
    elif curation_method in ["embedding-similarity", "embedding-diversity"]:
        # For embedding methods, need dataset with embeddings
        if os.path.exists(filtered_path):
            df = pd.read_parquet(filtered_path)
            logging.info(f"Loaded existing dataset from {filtered_path}")
        else:
            df = load_base_dataset(experiment_config, config)
        
        # Get column from config
        curation_params = experiment_config.get("curation", {})
        column = curation_params.get("column", "Complex_CoT")
        
        # Parse column from experiment name if not in config
        if column == "Complex_CoT" and "-question" in curation_method.lower():
            column = "Question"
        elif column == "Complex_CoT" and "-cot" in curation_method.lower():
            column = "Complex_CoT"
        
        logging.info(f"Using column '{column}' for embedding-based curation")
        
        # Apply appropriate embedding method
        if curation_method == "embedding-similarity":
            df = embedding_similarity_curation(df, experiment_config, n_samples, seed=args.seed)
        else:  # embedding-diversity
            df = embedding_diversity_curation(df, experiment_config, n_samples, seed=args.seed)
    else:
        raise ValueError(f"Unknown curation method: {curation_method}")
    
    # Get paths for final outputs
    paths = get_final_paths(output_dir, args.experiment)
    
    # Create experiment directory
    os.makedirs(os.path.dirname(paths['filtered']), exist_ok=True)
    
    # Apply extraction if configured
    if experiment_config.get("curation", {}).get("extract"):
        logging.info(f"Applying {experiment_config['curation']['extract']} extraction to selected examples")
        df = await apply_step_extraction(df, experiment_config)
    
    # Save filtered dataset with all examples and their filtering status
    df.to_parquet(paths['filtered'])
    logging.info(f"Saved filtered dataset to {paths['filtered']}")
    
    # Save curated dataset (selected examples only)
    df[df['selected_for_training']].to_parquet(paths['curated'])
    logging.info(f"Saved curated dataset to {paths['curated']}")
    
    # Save formatted dataset for training (with validation split)
    train_dataset, validation_dataset = format_for_training(
        df[df['selected_for_training']], config, experiment_config, args.experiment,
        validation_split=0.1, seed=args.seed
    )
    
    # Create paths for train and validation datasets
    train_path = os.path.join(paths['formatted'], 'train')
    val_path = os.path.join(paths['formatted'], 'validation')
    
    # Save datasets
    os.makedirs(train_path, exist_ok=True)
    train_dataset.save_to_disk(train_path)
    logging.info(f"Saved training dataset to {train_path}")
    
    if validation_dataset:
        os.makedirs(val_path, exist_ok=True)
        validation_dataset.save_to_disk(val_path)
        logging.info(f"Saved validation dataset to {val_path}")
    
    # Update results.json with paths and stats
    stats = {
        "total_examples": len(df),
        "selected_examples": len(df[df['selected_for_training']]),
        "filtered_examples": len(df[df['filter_status'] == 'removed']),
        "filter_reasons": df[df['filter_status'] == 'removed']['filter_reason'].value_counts().to_dict(),
    }
    
    # Add specialty distribution if the column exists
    if 'specialty' in df.columns:
        stats["specialty_distribution"] = df[df['selected_for_training']]['specialty'].value_counts().to_dict()
    else:
        logging.warning("'specialty' column not found in dataframe, skipping specialty distribution stats")
    
    # Get results.json path from environment
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
    
    # Update results.json with paths and stats
    update_results_json(
        results_json_path=results_json,
        experiment_name=args.experiment,
        stage="curation",
        paths=paths,
        timestamp=datetime.now().isoformat(),
        stats=stats
    )
    
    logging.info("Curation complete!")

if __name__ == "__main__":
    asyncio.run(main())