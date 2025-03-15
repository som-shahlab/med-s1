"""
N-gram difficulty-based curation method.

This module implements a curation method that ranks examples based on the presence
of specific n-grams that are indicative of complex reasoning in medical texts.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import re
from collections import Counter

# Define the important n-grams based on TF-IDF and frequency analysis
# These n-grams are indicative of complex reasoning in medical texts
IMPORTANT_NGRAMS = {
    # Top unigrams (reasoning indicators)
    'like': 5.0,
    'let': 5.0,
    'think': 4.5,
    'really': 4.0,
    'right': 4.0,
    'symptoms': 4.0,
    'pain': 4.0,
    'blood': 4.0,
    'usually': 3.5,
    'pretty': 3.5,
    'heart': 3.5,
    'makes': 3.5,
    'cells': 3.5,
    'cell': 3.5,
    'alright': 3.5,
    'okay': 3.5,
    'hmm': 3.5,
    'sense': 3.5,
    
    # Top bigrams (reasoning patterns)
    'let s': 8.0,
    'think about': 8.0,
    's think': 8.0,
    'we re': 7.0,
    'i m': 7.0,
    'doesn t': 7.0,
    's not': 7.0,
    
    # Top trigrams (complex reasoning patterns)
    'let s think': 10.0,
    's think about': 10.0,
    'alright let s': 10.0,
    'okay let s': 10.0,
    'we ve got': 9.0,
    'we have a': 9.0,
    'it s like': 9.0,
    'let s see': 9.0,
    'think about this': 9.0,
    'think about what': 9.0
}

def preprocess_text(text: str) -> str:
    """Preprocess text for n-gram matching."""
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces to ensure proper word boundaries
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_ngrams(text: str, ngrams: Dict[str, float]) -> Dict[str, int]:
    """Count occurrences of specified n-grams in text."""
    text = preprocess_text(text)
    counts = {}
    for ngram in ngrams:
        counts[ngram] = text.count(ngram)
    return counts

def calculate_ngram_score(text: str, ngrams: Dict[str, float]) -> float:
    """Calculate a score based on the presence of important n-grams."""
    if not text or pd.isna(text):
        return 0.0
    
    # Count occurrences of each n-gram
    counts = count_ngrams(text, ngrams)
    
    # Calculate weighted score
    score = sum(counts[ngram] * weight for ngram, weight in ngrams.items())
    
    # Normalize by text length to avoid bias towards longer texts
    # Add 1 to avoid division by zero
    normalized_score = score / (len(text.split()) + 1)
    
    return normalized_score

def ngram_difficulty_curation(experiment_config: Dict, n_samples: int) -> pd.DataFrame:
    """
    Curate dataset based on n-gram difficulty.
    
    Args:
        experiment_config: Configuration for the experiment
        n_samples: Number of samples to select
        
    Returns:
        DataFrame with selected examples
    """
    logging.info("Starting n-gram difficulty-based curation")
    
    # Load the dataset
    dataset_path = os.environ.get('DATA_DIR')
    if not dataset_path:
        raise ValueError("DATA_DIR environment variable not set")
    
    # Load the filtered dataset
    filtered_path = os.path.join(dataset_path, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    if not os.path.exists(filtered_path):
        raise FileNotFoundError(f"Filtered dataset not found at {filtered_path}")
    
    df = pd.read_parquet(filtered_path)
    logging.info(f"Loaded dataset with {len(df)} examples")
    
    # Initialize selection column
    df['selected_for_training'] = False
    
    # Calculate n-gram difficulty score for each example
    logging.info("Calculating n-gram difficulty scores...")
    df['ngram_score'] = df['Complex_CoT'].apply(
        lambda x: calculate_ngram_score(x, IMPORTANT_NGRAMS)
    )
    
    # Sort by score in descending order
    df_sorted = df.sort_values('ngram_score', ascending=False)
    
    # Select top n_samples
    selected_indices = df_sorted.index[:n_samples]
    df.loc[selected_indices, 'selected_for_training'] = True
    
    # Log statistics
    selected_df = df[df['selected_for_training']]
    logging.info(f"Selected {len(selected_df)} examples based on n-gram difficulty")
    logging.info(f"Average n-gram score of selected examples: {selected_df['ngram_score'].mean():.4f}")
    logging.info(f"Min n-gram score of selected examples: {selected_df['ngram_score'].min():.4f}")
    logging.info(f"Max n-gram score of selected examples: {selected_df['ngram_score'].max():.4f}")
    
    # Add filter status for non-selected examples
    df.loc[~df['selected_for_training'], 'filter_status'] = 'removed'
    df.loc[~df['selected_for_training'], 'filter_stage'] = 'ngram_difficulty'
    df.loc[~df['selected_for_training'], 'filter_reason'] = 'low_ngram_score'
    
    return df