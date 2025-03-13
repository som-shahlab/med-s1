"""
Advanced curation methods for the med-s1k dataset.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import random
from typing import Dict, Optional, List, Tuple
from sklearn.cluster import KMeans
from utils.embedding_utils import get_or_generate_embeddings, cosine_similarity

def novelty_answer_curation(df: pd.DataFrame, config: Dict, n_samples: int) -> pd.DataFrame:
    """
    Novelty-answer curation method.
    
    Selects examples with the lowest cosine similarity between base_model_response and Response.
    
    Args:
        df: Input dataframe with all examples
        config: Curation configuration
        n_samples: Number of examples to sample
        
    Returns:
        DataFrame with novelty-answer curated examples selected for training
    """
    logging.info(f"Starting novelty-answer curation with {len(df)} examples...")
    
    # Check if filtered dataset exists
    data_dir = os.environ.get('DATA_DIR')
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    
    filtered_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    
    # Use filtered dataset if it exists (it should have base_model_response)
    if os.path.exists(filtered_path) and 'base_model_response' not in df.columns:
        logging.info(f"Loading filtered dataset from {filtered_path}")
        df = pd.read_parquet(filtered_path)
    
    # Reset filter status and reason for all examples
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    df['selected_for_training'] = False
    
    # Check if base_model_response column exists
    if 'base_model_response' not in df.columns:
        raise ValueError("base_model_response column not found in dataframe. This is required for novelty-answer curation.")
    
    # Get or generate embeddings
    columns = ["Response", "base_model_response"]
    logging.info(f"Getting embeddings for columns: {columns}")
    
    embeddings = get_or_generate_embeddings(
        df,
        data_dir,
        columns=columns
    )
    
    # Calculate cosine similarity between Response and base_model_response
    response_embeddings = embeddings["Response"]
    base_model_embeddings = embeddings["base_model_response"]
    
    # Only consider examples with both embeddings
    valid_mask = (~df['Response'].isna()) & (~df['base_model_response'].isna())
    valid_indices = df[valid_mask].index
    
    if len(valid_indices) < n_samples:
        logging.warning(f"Only {len(valid_indices)} valid examples with both Response and base_model_response")
        n_samples = len(valid_indices)
    
    # Calculate similarities for valid examples
    similarities = []
    for i, idx in enumerate(valid_indices):
        response_idx = df.index.get_loc(idx)
        response_emb = response_embeddings[response_idx]
        base_model_emb = base_model_embeddings[response_idx]
        
        # Calculate cosine similarity
        similarity = np.dot(response_emb, base_model_emb) / (
            np.linalg.norm(response_emb) * np.linalg.norm(base_model_emb)
        )
        similarities.append((idx, similarity))
    
    # Sort by similarity (ascending)
    similarities.sort(key=lambda x: x[1])
    
    # Select top n_samples with lowest similarity
    selected_indices = [idx for idx, _ in similarities[:n_samples]]
    
    # Mark selected examples
    df['selected_for_training'] = df.index.isin(selected_indices)
    
    # Mark unselected examples
    unselected_mask = ~df['selected_for_training'] & (df['filter_status'] == 'kept')
    df.loc[unselected_mask, 'filter_status'] = 'removed'
    df.loc[unselected_mask, 'filter_stage'] = 'novelty-answer'
    df.loc[unselected_mask, 'filter_reason'] = 'not_selected_in_sampling'
    
    logging.info(f"Novelty-answer curation selected {len(selected_indices)} examples")
    return df

def difficulty_substring_curation(config: Dict, n_samples: int) -> pd.DataFrame:
    """
    Difficulty-substring curation method.
    
    Selects examples where Complex_CoT contains specific substrings indicating difficulty.
    This method always uses CPU and reads from the original dataset.
    
    Args:
        config: Curation configuration
        n_samples: Number of examples to sample
        
    Returns:
        DataFrame with difficulty-substring curated examples selected for training
    """
    # Load the base dataset directly
    from datasets import load_dataset
    
    logging.info("Loading base dataset directly for difficulty-substring curation...")
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    df = pd.DataFrame(dataset)
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    df['selected_for_training'] = False
    
    logging.info(f"Starting difficulty-substring curation with {len(df)} examples...")
    
    # Define difficulty substrings
    difficulty_substrings = ['confus', 'mislead', 'overlook', 'double-check', 'confirm']
    
    # Create filter for examples containing any of the substrings
    difficulty_mask = df['Complex_CoT'].fillna('').str.contains(
        '|'.join(difficulty_substrings),
        case=False
    )
    
    # Get indices of examples with difficulty substrings
    difficult_indices = df[difficulty_mask].index.tolist()
    logging.info(f"Found {len(difficult_indices)} examples containing difficulty substrings")
    
    # If we have more examples than needed, randomly sample
    if len(difficult_indices) > n_samples:
        selected_indices = random.sample(difficult_indices, n_samples)
    else:
        selected_indices = difficult_indices
        if len(selected_indices) < n_samples:
            logging.warning(
                f"Only found {len(selected_indices)} examples with difficulty substrings, "
                f"needed {n_samples}"
            )
    
    # Mark selected examples
    df['selected_for_training'] = df.index.isin(selected_indices)
    
    # Mark unselected examples
    unselected_mask = ~df['selected_for_training'] & (df['filter_status'] == 'kept')
    df.loc[unselected_mask, 'filter_status'] = 'removed'
    df.loc[unselected_mask, 'filter_stage'] = 'difficulty-substring'
    df.loc[unselected_mask, 'filter_reason'] = 'not_selected_in_sampling'
    
    logging.info(f"Difficulty-substring curation selected {len(selected_indices)} examples")
    return df

def embedding_similarity_curation(df: pd.DataFrame, config: Dict, n_samples: int) -> pd.DataFrame:
    """
    Embedding-similarity curation method.
    
    Implements the algorithm from embedding_similarity_algorithm.txt:
    1. Initialize an empty selection list L
    2. Iterate until L reaches the desired size n:
       - For each v in V (round-robin fashion):
         a. Find the highest-scoring d in D according to S[v, d]
         b. Add d to L
         c. Set S[v, d] to -infinity to prevent re-selection
         d. If L reaches n, exit early
    3. Return L, the final selected dataset
    
    Args:
        df: Input dataframe with all examples
        config: Curation configuration
        n_samples: Number of examples to sample
        
    Returns:
        DataFrame with embedding-similarity curated examples selected for training
    """
    logging.info(f"Starting embedding-similarity curation with {len(df)} examples...")
    
    # Check if filtered dataset exists
    data_dir = os.environ.get('DATA_DIR')
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    
    # Get med-s1 directory from environment
    med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
    
    # Reset filter status and reason for all examples
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    df['selected_for_training'] = False
    
    # Get parameters from config
    curation_params = config.get("curation", {})
    
    # Determine column to use for embeddings
    column = curation_params.get("column", "Question")  # Default to Question for embedding-similarity
    
    # Parse column from experiment name if not in config
    curation_method = curation_params.get("method", "")
    if column == "Question" and "-cot" in curation_method.lower():
        column = "Complex_CoT"
    
    logging.info(f"Using column '{column}' for embedding similarity")
    
    # Load representative queries from eval_data_samples.json
    eval_samples_path = os.path.join(med_s1_dir, 'eval', 'data', 'eval_data_samples.json')
    eval_embeddings_path = os.path.join(med_s1_dir, 'eval', 'data', 'eval_data_samples.npy')
    
    # Check if the samples and embeddings exist
    if not os.path.exists(eval_samples_path) or not os.path.exists(eval_embeddings_path):
        logging.error(f"Representative samples or embeddings not found. Please run find_samples.py first.")
        raise FileNotFoundError(f"Required files not found: {eval_samples_path} or {eval_embeddings_path}")
    
    # Load representative samples
    with open(eval_samples_path, 'r') as f:
        eval_samples = json.load(f)
    
    # Load representative embeddings
    eval_embeddings = np.load(eval_embeddings_path)
    
    logging.info(f"Loaded {len(eval_samples)} representative samples with embeddings")
    
    # Get or generate embeddings for the dataset
    embeddings = get_or_generate_embeddings(df, data_dir, columns=[column])
    column_embeddings = embeddings[column]
    
    # Calculate cosine similarity matrix between representative samples and dataset
    # Shape: (n_samples, n_dataset)
    similarity_matrix = cosine_similarity(eval_embeddings, column_embeddings)
    
    # Initialize selection list
    selected_indices = []
    
    # Create a mask to track which examples are still available
    available_mask = np.ones(len(df), dtype=bool)
    
    # Implement round-robin selection
    while len(selected_indices) < n_samples:
        for i in range(len(eval_samples)):
            if len(selected_indices) >= n_samples:
                break
                
            # Get similarities for this representative sample
            similarities = similarity_matrix[i].copy()
            
            # Set similarities of already selected examples to -infinity
            similarities[~available_mask] = -np.inf
            
            # Find the highest-scoring example
            if np.max(similarities) > -np.inf:
                best_idx = np.argmax(similarities)
                selected_indices.append(df.index[best_idx])
                available_mask[best_idx] = False
            else:
                # No more examples with positive similarity for this representative
                continue
    
    logging.info(f"Selected {len(selected_indices)} examples using embedding-similarity")
    
    # Mark selected examples
    df['selected_for_training'] = df.index.isin(selected_indices)
    
    # Mark unselected examples
    unselected_mask = ~df['selected_for_training'] & (df['filter_status'] == 'kept')
    df.loc[unselected_mask, 'filter_status'] = 'removed'
    df.loc[unselected_mask, 'filter_stage'] = 'embedding-similarity'
    df.loc[unselected_mask, 'filter_reason'] = 'not_selected_in_sampling'
    
    return df

def embedding_diversity_curation(df: pd.DataFrame, config: Dict, n_samples: int) -> pd.DataFrame:
    """
    Embedding-diversity curation method.
    
    Implements the algorithm from embedding_diversity_algorithm.txt:
    1. Cluster all points into k = CP% * n clusters
    2. Select OP% * n outliers (points furthest from their centroids)
    3. Evenly sample from remaining clusters to reach n samples
    
    Args:
        df: Input dataframe with all examples
        config: Curation configuration
        n_samples: Number of examples to sample
        
    Returns:
        DataFrame with embedding-diversity curated examples selected for training
    """
    logging.info(f"Starting embedding-diversity curation with {len(df)} examples...")
    
    # Check if filtered dataset exists
    data_dir = os.environ.get('DATA_DIR')
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set")
    
    filtered_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    
    # Use filtered dataset if it exists
    if os.path.exists(filtered_path):
        logging.info(f"Loading filtered dataset from {filtered_path}")
        df = pd.read_parquet(filtered_path)
    
    # Reset filter status and reason for all examples
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    df['selected_for_training'] = False
    
    # Get parameters from config
    curation_params = config.get("curation", {})
    
    # Determine column to use for embeddings
    column = curation_params.get("column", "Complex_CoT")
    
    # Parse column from experiment name if not in config
    curation_method = curation_params.get("method", "")
    if column == "Complex_CoT" and "-question" in curation_method.lower():
        column = "Question"
    elif column == "Complex_CoT" and "-cot" in curation_method.lower():
        column = "Complex_CoT"
    
    logging.info(f"Using column '{column}' for embedding diversity")
    
    # Get clustering parameters
    cluster_percentage = curation_params.get("cluster_percentage", 10)
    outlier_percentage = curation_params.get("outlier_percentage", 5)
    
    logging.info(f"Using cluster_percentage={cluster_percentage}%, outlier_percentage={outlier_percentage}%")
    
    # Get or generate embeddings
    embeddings = get_or_generate_embeddings(df, data_dir, columns=[column])
    column_embeddings = embeddings[column]
    
    # Calculate number of clusters and outliers
    num_clusters = max(1, int(cluster_percentage * n_samples / 100))
    num_outliers = max(0, int(outlier_percentage * n_samples / 100))
    
    logging.info(f"Using {num_clusters} clusters and selecting {num_outliers} outliers")
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(column_embeddings)
    
    # Calculate distances to centroids
    distances = []
    for i, (idx, row) in enumerate(df.iterrows()):
        centroid = kmeans.cluster_centers_[cluster_labels[i]]
        distance = np.linalg.norm(column_embeddings[i] - centroid)
        distances.append((idx, distance, cluster_labels[i]))
    
    # Sort by distance (descending)
    distances.sort(key=lambda x: x[1], reverse=True)
    
    # Select outliers (points furthest from centroids)
    outlier_indices = [idx for idx, _, _ in distances[:num_outliers]]
    
    # Create a set of selected indices
    selected_indices = set(outlier_indices)
    
    # Count points per cluster (excluding outliers)
    cluster_counts = {}
    for idx, _, cluster in distances:
        if idx not in selected_indices:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    
    # Calculate how many points to sample from each cluster
    remaining_samples = n_samples - num_outliers
    points_per_cluster = remaining_samples // num_clusters
    extra_points = remaining_samples % num_clusters
    
    # Distribute points evenly across clusters
    cluster_points = {i: points_per_cluster + (1 if i < extra_points else 0)
                     for i in range(num_clusters)}
    
    # Sample from each cluster
    for cluster in range(num_clusters):
        # Get indices for this cluster (excluding already selected outliers)
        cluster_indices = [idx for idx, _, c in distances if c == cluster and idx not in selected_indices]
        
        # Sample from this cluster
        to_sample = min(cluster_points[cluster], len(cluster_indices))
        if to_sample > 0:
            sampled = random.sample(cluster_indices, to_sample)
            selected_indices.update(sampled)
    
    # If we still need more samples, take from remaining points
    if len(selected_indices) < n_samples:
        remaining_indices = [idx for idx, _, _ in distances if idx not in selected_indices]
        additional_needed = n_samples - len(selected_indices)
        if additional_needed > 0 and remaining_indices:
            additional_samples = random.sample(remaining_indices, min(additional_needed, len(remaining_indices)))
            selected_indices.update(additional_samples)
    
    # Mark selected examples
    df['selected_for_training'] = df.index.isin(selected_indices)
    
    # Mark unselected examples
    unselected_mask = ~df['selected_for_training'] & (df['filter_status'] == 'kept')
    df.loc[unselected_mask, 'filter_status'] = 'removed'
    df.loc[unselected_mask, 'filter_stage'] = 'embedding-diversity'
    df.loc[unselected_mask, 'filter_reason'] = 'not_selected_in_sampling'
    
    logging.info(f"Embedding-diversity curation selected {len(selected_indices)} examples")
    return df

def check_novelty_answer_prerequisites(data_dir: str) -> bool:
    """
    Check if the prerequisites for novelty-answer curation exist.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        bool: True if prerequisites exist, False otherwise
    """
    filtered_path = os.path.join(data_dir, "plumbing_test_001_20250219_145607/med_s1k_filtered.parquet")
    embeddings_dir = os.path.join(data_dir, "embeddings-25k")
    return os.path.exists(filtered_path) and os.path.exists(embeddings_dir)

def check_embedding_prerequisites(data_dir: str, method: str = None) -> bool:
    """
    Check if the prerequisites for embedding-based curation exist.
    
    Args:
        data_dir: Path to the data directory
        method: The specific embedding method to check for (optional)
        
    Returns:
        bool: True if prerequisites exist, False otherwise
    """
    # Get med-s1 directory from environment
    med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
    
    # Check for basic embedding prerequisites
    embeddings_dir = os.path.join(data_dir, "embeddings-25k")
    basic_prereqs = os.path.exists(embeddings_dir)
    
    # If method is embedding-similarity, also check for eval samples
    if method == "embedding-similarity":
        eval_samples_path = os.path.join(med_s1_dir, 'eval', 'data', 'eval_data_samples.json')
        eval_embeddings_path = os.path.join(med_s1_dir, 'eval', 'data', 'eval_data_samples.npy')
        return basic_prereqs and os.path.exists(eval_samples_path) and os.path.exists(eval_embeddings_path)
    
    return basic_prereqs