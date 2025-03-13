"""
Utilities for generating and loading embeddings for the med-s1k dataset.
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Default model for embeddings
DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to get sentence embeddings.
    
    Args:
        model_output: Output from the model
        attention_mask: Attention mask from tokenizer
        
    Returns:
        Mean-pooled embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = None
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the model to use
        batch_size: Batch size for processing
        device: Device to use (None for auto-detection)
        
    Returns:
        Array of embeddings
    """
    if device is None:
        # Check if CUDA is available and if we're running on a GPU node
        if torch.cuda.is_available():
            device = "cuda"
            logging.info("CUDA is available, using GPU for embedding generation")
        else:
            device = "cpu"
            logging.warning("CUDA is not available, using CPU for embedding generation (this will be slow)")
    
    logging.info(f"Generating embeddings using {model_name} on {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Process in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Mean pooling
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        # Move to CPU and convert to numpy
        all_embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all batches
    return np.vstack(all_embeddings)

def generate_and_save_embeddings(
    df: pd.DataFrame,
    data_dir: str,
    columns: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = None
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for specified columns and save to disk.
    
    Args:
        df: DataFrame containing the texts
        data_dir: Directory to save embeddings
        columns: Columns to generate embeddings for
        model_name: Name of the model to use
        batch_size: Batch size for processing
        device: Device to use (None for auto-detection)
        
    Returns:
        Dictionary mapping column names to embedding arrays
    """
    embeddings_dir = os.path.join(data_dir, "embeddings-25k")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    embeddings = {}
    for column in columns:
        if column not in df.columns:
            logging.warning(f"Column '{column}' not found in dataframe, skipping embedding generation")
            continue
            
        logging.info(f"Generating embeddings for column: {column}")
        
        # Generate embeddings
        texts = df[column].fillna("").tolist()
        column_embeddings = generate_embeddings(texts, model_name, batch_size, device)
        
        # Save to disk
        output_path = os.path.join(embeddings_dir, f"{column}_embeddings.npy")
        np.save(output_path, column_embeddings)
        logging.info(f"Saved {column} embeddings to {output_path}")
        
        embeddings[column] = column_embeddings
    
    return embeddings

def load_embeddings(
    data_dir: str,
    columns: List[str]
) -> Dict[str, np.ndarray]:
    """
    Load pre-computed embeddings from disk.
    
    Args:
        data_dir: Directory containing embeddings
        columns: Columns to load embeddings for
        
    Returns:
        Dictionary mapping column names to embedding arrays
    """
    embeddings_dir = os.path.join(data_dir, "embeddings-25k")
    
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    embeddings = {}
    for column in columns:
        embedding_path = os.path.join(embeddings_dir, f"{column}_embeddings.npy")
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embeddings file not found: {embedding_path}")
        
        embeddings[column] = np.load(embedding_path)
        logging.info(f"Loaded {column} embeddings from {embedding_path}")
    
    return embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of vectors.
    
    Args:
        a: First set of vectors (n x d)
        b: Second set of vectors (m x d)
        
    Returns:
        Similarity matrix (n x m)
    """
    # Normalize if not already normalized
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    
    # Compute similarity
    return np.dot(a_norm, b_norm.T)

def get_or_generate_embeddings(
    df: pd.DataFrame,
    data_dir: str,
    columns: List[str] = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    force_device: str = None
) -> Dict[str, np.ndarray]:
    """
    Load embeddings if they exist, otherwise generate and save them.
    
    Args:
        df: DataFrame containing the texts
        data_dir: Directory to save/load embeddings
        columns: Columns to generate embeddings for (if None, will use all 4 columns if available)
        model_name: Name of the model to use
        batch_size: Batch size for processing
        force_device: Force using a specific device ("cpu" or "cuda")
        
    Returns:
        Dictionary mapping column names to embedding arrays
    """
    # Determine which columns to use
    default_columns = ["Complex_CoT", "Question", "Response"]
    
    if columns is None:
        # If no columns specified, use all 4 if base_model_response exists, otherwise use default 3
        if 'base_model_response' in df.columns:
            columns = default_columns + ["base_model_response"]
            logging.info("Using all 4 columns including base_model_response")
        else:
            columns = default_columns
            logging.info("Using default 3 columns (base_model_response not found)")
    
    embeddings_dir = os.path.join(data_dir, "embeddings-25k")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Check which columns already have embeddings
    existing_embeddings = {}
    missing_columns = []
    
    for column in columns:
        embedding_path = os.path.join(embeddings_dir, f"{column}_embeddings.npy")
        if os.path.exists(embedding_path):
            # Load existing embedding
            existing_embeddings[column] = np.load(embedding_path)
            logging.info(f"Loaded existing embeddings for {column}")
        else:
            # Mark column as missing
            missing_columns.append(column)
    
    # If all embeddings exist, return them
    if not missing_columns:
        logging.info("All required embeddings already exist")
        return existing_embeddings
    
    # Generate missing embeddings
    logging.info(f"Generating embeddings for missing columns: {missing_columns}")
    
    # Determine device to use
    if force_device:
        device = force_device
    elif torch.cuda.is_available():
        device = "cuda"
        logging.info("Using GPU for embedding generation")
    else:
        device = "cpu"
        logging.warning("CUDA not available, using CPU for embedding generation (this will be slow)")
    
    # Generate embeddings for missing columns
    for column in missing_columns:
        if column not in df.columns:
            logging.error(f"Column '{column}' not found in dataframe, skipping embedding generation")
            continue
            
        logging.info(f"Generating embeddings for column: {column}")
        texts = df[column].fillna("").tolist()
        column_embeddings = generate_embeddings(texts, model_name, batch_size, device)
        
        # Save to disk
        output_path = os.path.join(embeddings_dir, f"{column}_embeddings.npy")
        np.save(output_path, column_embeddings)
        logging.info(f"Saved {column} embeddings to {output_path}")
        
        existing_embeddings[column] = column_embeddings
    
    return existing_embeddings