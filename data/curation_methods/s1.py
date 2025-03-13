"""
S1 curation method for the med-s1k dataset.
"""

import pandas as pd
import numpy as np
import logging
import random
import os
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

def diversity_sample(df: pd.DataFrame, target_size: int, tokenizer, specialty_weights: Optional[dict] = None) -> pd.DataFrame:
    """
    Do difficulty-weighted diversity sampling across specialties.
    
    Args:
        df: Input dataframe with examples that passed quality and difficulty filtering
        target_size: Number of examples to sample
        tokenizer: Tokenizer for length checks
        specialty_weights: Optional weights for specialties
        
    Returns:
        DataFrame with diversity-sampled examples selected for training
    """
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
    
    logging.info(f"Diversity sampling selected {len(S)} examples")
    return df

async def process_s1_dataset(
    df: pd.DataFrame, 
    config: Dict, 
    tokenizer, 
    n_samples: int, 
    specialty_weights: Optional[dict] = None, 
    experiment_name: str = None,
    output_dir: str = None,
    run_pipeline: Callable = None
) -> pd.DataFrame:
    """
    Process dataset using s1 method (quality filter -> difficulty filter -> diversity sample).
    
    This method includes GPU-intensive operations for:
    1. Answer verification (checking if base model can answer correctly)
    2. Specialty classification
    
    Args:
        df: Input dataframe with all examples
        config: Curation configuration
        tokenizer: Tokenizer for length checks
        n_samples: Number of examples to sample
        specialty_weights: Optional weights for specialties
        experiment_name: Name of experiment
        output_dir: Output directory for intermediate files
        run_pipeline: Function to run the pipeline for answer verification and specialty labeling
        
    Returns:
        DataFrame with s1-processed examples selected for training
    """
    from curation_methods.base import quality_filter
    from utils.path_utils import get_intermediate_path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run quality filter
    df = quality_filter(df, config, tokenizer)
    if output_dir and experiment_name:
        df.to_parquet(get_intermediate_path(output_dir, experiment_name, "quality", timestamp))
    
    # Run answer verification and specialty labeling pipeline
    # This is the GPU-intensive part that:
    # 1. Gets base model answers to check if questions are too easy
    # 2. Classifies specialties for each question
    if run_pipeline:
        logging.info("Running GPU-intensive pipeline for answer verification and specialty classification...")
        df = await run_pipeline(df, config, batch_size=4)
        
        # Save intermediate results
        if output_dir and experiment_name:
            df.to_parquet(get_intermediate_path(output_dir, experiment_name, "difficulty", timestamp))
            df.to_parquet(get_intermediate_path(output_dir, experiment_name, "specialty", timestamp))
    else:
        logging.warning("No run_pipeline function provided, skipping difficulty filtering and specialty classification")
    
    # Run diversity sampling on kept examples
    df = diversity_sample(df, n_samples, tokenizer, specialty_weights)
    if output_dir and experiment_name:
        df.to_parquet(get_intermediate_path(output_dir, experiment_name, "diversity", timestamp))
    
    return df

def check_s1_prerequisites(data_dir: str) -> bool:
    """
    Check if the prerequisites for S1 curation exist.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        bool: True if prerequisites exist, False otherwise
    """
    filtered_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    return os.path.exists(filtered_path)