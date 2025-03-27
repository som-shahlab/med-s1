"""
Base curation methods for the med-s1k dataset.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional

def full_dataset(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Select all examples from the dataset.
    
    Args:
        df: Input dataframe with all examples
        config: Curation configuration
        
    Returns:
        DataFrame with all examples selected for training
    """
    logging.info(f"Using full dataset with {len(df)} examples")
    df['selected_for_training'] = True
    return df

def random_sample_dataset(df: pd.DataFrame, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Randomly sample n examples from dataset with deterministic ordering.
    
    Note: This function assumes the input DataFrame has a deterministic ordering
    (e.g., sorted by a stable key like 'Question'). This is critical for
    reproducibility as the random sampling uses DataFrame indices.
    
    Args:
        df: Input dataframe with all examples (must have deterministic ordering)
        n_samples: Number of examples to sample
        seed: Random seed for reproducibility (uses global numpy seed if None)
        
    Returns:
        DataFrame with randomly sampled examples selected for training
    """
    if n_samples >= len(df):
        logging.info(f"Requested samples ({n_samples}) >= dataset size ({len(df)}), using full dataset")
        df['selected_for_training'] = True
        return df
    
    # Use local random state if seed provided
    if seed is not None:
        rng = np.random.RandomState(seed)
        logging.info(f"Using provided random seed: {seed}")
    else:
        rng = np.random.RandomState()
        logging.info("Using global numpy random state")
    
    # Random sample with deterministic ordering
    all_indices = np.arange(len(df))
    sampled_indices = rng.choice(all_indices, size=n_samples, replace=False)
    sampled_indices.sort()  # Sort for deterministic ordering
    
    df['selected_for_training'] = df.index.isin(sampled_indices)
    
    # Mark unselected examples (preserve all rows)
    df.loc[~df['selected_for_training'], 'filter_status'] = 'removed'
    df.loc[~df['selected_for_training'], 'filter_stage'] = 'random'
    df.loc[~df['selected_for_training'], 'filter_reason'] = 'not_selected_in_sampling'
    
    # Log sampling details
    logging.info(f"Randomly sampled {n_samples} examples from {len(df)} total examples")
    logging.info(f"First 5 sampled indices: {sampled_indices[:5]}")
    return df

def quality_filter(df: pd.DataFrame, config: Dict, tokenizer) -> pd.DataFrame:
    """
    Filter out empty/null values and exact 1024 token responses.
    
    Args:
        df: Input dataframe with all examples
        config: Curation configuration
        tokenizer: Tokenizer for length checks
        
    Returns:
        DataFrame with quality-filtered examples
    """
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