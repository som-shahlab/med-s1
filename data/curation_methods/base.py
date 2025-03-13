"""
Base curation methods for the med-s1k dataset.
"""

import pandas as pd
import random
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

def random_sample_dataset(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Randomly sample n examples from dataset.
    
    Args:
        df: Input dataframe with all examples
        n_samples: Number of examples to sample
        
    Returns:
        DataFrame with randomly sampled examples selected for training
    """
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
    
    logging.info(f"Randomly sampled {n_samples} examples from {len(df)} total examples")
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