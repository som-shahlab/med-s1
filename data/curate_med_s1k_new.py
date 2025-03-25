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
    with open("/share/pi/nigam/users/calebwin/med-s1/config.json", "r") as f:
        return json.load(f)

def load_experiment_config(experiment_name: str) -> Dict:
    """Load experiment configuration from results.json"""
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
        
    with open(results_json, "r") as f:
        results = json.load(f)
    if experiment_name not in results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in {results_json}")
    return results["experiments"][experiment_name]["config"]

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
                dataset = load_from_disk(file_path)
            else:
                raise ValueError(f"Unsupported file format for {file_path}")
        else:
            raise ValueError(f"Dataset {dataset_name} has no hf_path or file_path")
    
    # Convert to DataFrame and initialize metadata columns
    df = pd.DataFrame(dataset)
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
    
    # Check if base dataset exists
    filtered_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    
    # Process based on curation method
    if curation_method == "all":
        # For "all" method, load dataset and use full dataset
        if os.path.exists(filtered_path):
            df = pd.read_parquet(filtered_path)
            logging.info(f"Loaded existing dataset from {filtered_path}")
        else:
            df = load_base_dataset(experiment_config, config)
        df = full_dataset(df, experiment_config)
    
    elif curation_method == "random":
        # For "random" method, load dataset and randomly sample
        if os.path.exists(filtered_path):
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
    extract_method = experiment_config.get("curation", {}).get("extract", "")
    if extract_method != "":
        logging.info(f"Applying {extract_method} extraction to selected examples")
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