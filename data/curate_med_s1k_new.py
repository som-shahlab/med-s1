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

def load_base_dataset() -> pd.DataFrame:
    """Load the base dataset and initialize metadata columns"""
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    df = pd.DataFrame(dataset)
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    df['selected_for_training'] = False
    return df

def quality_filter(df: pd.DataFrame, config: Dict, tokenizer) -> pd.DataFrame:
    """Filter out empty/null values and exact 1024 token responses"""
    logging.info(f"Starting quality filter with {len(df)} examples...")
    
    # Check for null values
    quality_mask = df[['Question', 'Complex_CoT', 'Response']].isna().any(axis=1)
    df.loc[quality_mask, 'filter_status'] = 'removed'
    df.loc[quality_mask, 'filter_stage'] = 'quality'
    df.loc[quality_mask, 'filter_reason'] = df[quality_mask].apply(
        lambda x: "missing_" + ",".join([
            col.lower() for col, value in zip(['Question', 'Complex_CoT', 'Response'],
                                           [x['Question'], x['Complex_CoT'], x['Response']])
            if pd.isna(value)
        ]),
        axis=1
    )
    
    # Check for exact 1024 token responses
    df['token_length'] = df['Complex_CoT'].fillna('').apply(lambda x: len(tokenizer(x).input_ids))
    token_mask = (df['filter_status'] == 'kept') & (df['token_length'] == 1024)
    df.loc[token_mask, 'filter_status'] = 'removed'
    df.loc[token_mask, 'filter_stage'] = 'quality'
    df.loc[token_mask, 'filter_reason'] = 'exact_1024_tokens'
    
    # Add timestamp
    df['quality_filter_timestamp'] = datetime.now().isoformat()
    
    # Log quality filter results
    quality_filtered = df[df['filter_stage'] == 'quality']
    logging.info("=== Quality Filter Results ===")
    logging.info(f"Total examples: {len(df)}")
    logging.info(f"Kept: {len(df[df['filter_status'] == 'kept'])}")
    logging.info(f"Removed: {len(quality_filtered)}")
    logging.info("\nRemoval reasons:")
    for reason, count in quality_filtered['filter_reason'].value_counts().items():
        logging.info(f"- {reason}: {count}")
    
    return df

def random_sample_dataset(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Randomly sample n examples from dataset"""
    if n_samples >= len(df):
        df['selected_for_training'] = True
        return df
    
    # Random sample
    sampled_indices = random.sample(range(len(df)), n_samples)
    df['selected_for_training'] = df.index.isin(sampled_indices)
    
    # Mark unselected examples (preserve all rows)
    df.loc[~df['selected_for_training'], 'filter_status'] = 'removed'
    df.loc[~df['selected_for_training'], 'filter_stage'] = 'random'
    df.loc[~df['selected_for_training'], 'filter_reason'] = 'not_selected_in_sampling'
    
    return df

def diversity_sample(df: pd.DataFrame, target_size: int, tokenizer, specialty_weights: Optional[dict] = None) -> pd.DataFrame:
    """Do difficulty-weighted diversity sampling across specialties"""
    # Only sample from examples that passed quality and difficulty filtering
    available_df = df[df['filter_status'] == 'kept'].copy()
    
    # Calculate CoT lengths
    available_df['cot_length'] = available_df['Complex_CoT'].fillna('').apply(lambda x: len(tokenizer(x).input_ids))
    
    # Initialize selected set S
    S = set()
    
    # First add all examples with long CoT
    long_examples = available_df[available_df['cot_length'] >= 1000]
    for idx in long_examples.index:
        S.add(idx)
    
    # Initialize domain pools with optional weights
    all_domains = list(available_df['specialty'].unique())
    benchmark_domains = []
    if specialty_weights:
        for domain in all_domains:
            weight = specialty_weights.get(domain, 1.0)
            benchmark_domains.extend([domain] * int(weight * 10))
    else:
        benchmark_domains = all_domains.copy()
    
    # Sample until we reach target size
    while len(S) < target_size and (len(all_domains) > 0 or len(benchmark_domains) > 0):
        # First phase: uniform sampling across all domains (70%)
        if len(S) < min(int(target_size * 0.7), target_size):
            if len(all_domains) == 0:
                break
            d = random.choice(all_domains)
            domain_pool = all_domains
        # Second phase: weighted sampling for remaining 30%
        else:
            if len(benchmark_domains) == 0:
                break
            d = random.choice(benchmark_domains)
            domain_pool = benchmark_domains
        
        # Get questions in domain d (excluding already selected)
        Qd = available_df[(available_df['specialty'] == d)]
        Qd = Qd[~Qd.index.isin(S)]
        
        if len(Qd) == 0:
            if d in domain_pool:
                domain_pool.remove(d)
            continue
        
        # Rank by thinking length
        lengths = Qd['cot_length'].values
        ranks = len(lengths) - 1 - np.argsort(np.argsort(lengths))
        weights = np.power(2.0, -ranks)
        weights = weights / weights.sum()
        
        # Sample one question
        selected_idx = np.random.choice(Qd.index, p=weights)
        S.add(selected_idx)
    
    # Mark selected examples in original dataframe (preserve all rows)
    df['selected_for_training'] = df.index.isin(S)
    
    # Mark unselected examples that were kept after filtering
    diversity_mask = ~df['selected_for_training'] & (df['filter_status'] == 'kept')
    df.loc[diversity_mask, 'filter_status'] = 'removed'
    df.loc[diversity_mask, 'filter_stage'] = 'diversity'
    df.loc[diversity_mask, 'filter_reason'] = 'not_selected_in_sampling'
    
    return df

async def process_s1_dataset(df: pd.DataFrame, config: Dict, tokenizer, n_samples: int, specialty_weights: Optional[dict] = None, experiment_name: str = None) -> pd.DataFrame:
    """Process dataset using s1 method (quality filter -> difficulty filter -> diversity sample)"""
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run quality filter
    df = quality_filter(df, config, tokenizer)
    df.to_parquet(get_intermediate_path(output_dir, experiment_name, "quality", timestamp))
    
    # Run answer verification and specialty labeling pipeline
    df = await run_pipeline(df, config, batch_size=4)
    df.to_parquet(get_intermediate_path(output_dir, experiment_name, "difficulty", timestamp))
    df.to_parquet(get_intermediate_path(output_dir, experiment_name, "specialty", timestamp))
    
    # Run diversity sampling on kept examples
    df = diversity_sample(df, n_samples, tokenizer, specialty_weights)
    df.to_parquet(get_intermediate_path(output_dir, experiment_name, "diversity", timestamp))
    
    return df

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Name of experiment from results.json")
    args = parser.parse_args()
    
    # Setup logging and load configs
    setup_logging()
    config = load_config()
    experiment_config = load_experiment_config(args.experiment)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Get tokenizer for length checks
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Clean experiment name for filenames
    safe_experiment_name = clean_experiment_name(args.experiment)
    
    # Get curation parameters
    curation_method = experiment_config["curation"]["method"]
    n_samples = experiment_config["curation"]["n_samples"]
    specialty_weights = experiment_config["curation"].get("specialty_weights")
    
    # Check if base dataset exists
    data_dir = os.environ.get('DATA_DIR')
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    filtered_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    
    if os.path.exists(filtered_path):
        # Load existing dataset with all metadata
        df = pd.read_parquet(filtered_path)
        logging.info(f"Loaded existing dataset from {filtered_path}")
        
        # Apply appropriate sampling method
        if curation_method == "all":
            df['selected_for_training'] = True
        elif curation_method == "random":
            df = random_sample_dataset(df, n_samples)
        else:  # s1
            df = diversity_sample(df, n_samples, tokenizer, specialty_weights)
    else:
        # Load base dataset
        df = load_base_dataset()
        
        # Apply appropriate processing method
        if curation_method == "all":
            df['selected_for_training'] = True
        elif curation_method == "random":
            df = random_sample_dataset(df, n_samples)
        else:  # s1
            df = await process_s1_dataset(df, config, tokenizer, n_samples, specialty_weights, safe_experiment_name)
            df.to_parquet(filtered_path)  # Save processed dataset for future use
    
    # Get output directory and paths
    output_dir = get_output_dir()
    paths = get_final_paths(output_dir, args.experiment)
    
    # Create experiment directory
    os.makedirs(os.path.dirname(paths['filtered']), exist_ok=True)
    
    # Save filtered dataset with all examples and their filtering status
    df.to_parquet(paths['filtered'])
    logging.info(f"Saved filtered dataset to {paths['filtered']}")
    
    # Save curated dataset (selected examples only)
    df[df['selected_for_training']].to_parquet(paths['curated'])
    logging.info(f"Saved curated dataset to {paths['curated']}")
    
    # Save formatted dataset for training
    dataset = format_for_training(df[df['selected_for_training']], config, args.experiment)
    dataset.save_to_disk(paths['formatted'])
    logging.info(f"Saved formatted dataset to {paths['formatted']}")
    # Update results.json with paths and stats
    stats = {
        "total_examples": len(df),
        "selected_examples": len(df[df['selected_for_training']]),
        "filtered_examples": len(df[df['filter_status'] == 'removed']),
        "filter_reasons": df[df['filter_status'] == 'removed']['filter_reason'].value_counts().to_dict(),
        "specialty_distribution": df[df['selected_for_training']]['specialty'].value_counts().to_dict()
    }
    
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