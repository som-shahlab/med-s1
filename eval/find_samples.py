#!/usr/bin/env python3
"""
Script to randomly select samples from eval_data.json for embedding similarity curation.
This creates:
1. eval_data_samples.json - 100 randomly selected data points
2. eval_data_samples.npy - embeddings of the questions from these data points
"""

import os
import json
import random
import numpy as np
import logging
import argparse
import sys

# Add the med-s1/data directory to the path to import embedding_utils
sys.path.append(os.path.join(os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1'), 'data'))
from utils.embedding_utils import generate_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def load_eval_data(eval_data_path):
    """Load evaluation data from JSON file."""
    logging.info(f"Loading evaluation data from {eval_data_path}")
    with open(eval_data_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        # Flatten the dictionary into a list
        flattened_data = []
        for dataset_name, examples in data.items():
            for example in examples:
                example['source'] = dataset_name
                flattened_data.append(example)
        return flattened_data
    else:
        return data

def select_samples(data, num_samples=100, seed=42):
    """Randomly select samples from the data."""
    random.seed(seed)
    if len(data) <= num_samples:
        logging.warning(f"Data only contains {len(data)} examples, using all of them")
        return data
    
    return random.sample(data, num_samples)

def extract_questions(samples):
    """Extract questions from the samples for embedding generation."""
    return [sample.get('question', '') for sample in samples]

def main():
    parser = argparse.ArgumentParser(description='Select random samples from eval data for embedding similarity')
    parser.add_argument('--eval_data_path', type=str, default='data/eval_data.json',
                        help='Path to eval_data.json (relative to med-s1/eval)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save output files (relative to med-s1/eval)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to select')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Get the med-s1 directory from environment or use default
    med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
    
    # Construct absolute paths
    eval_dir = os.path.join(med_s1_dir, 'eval')
    eval_data_path = os.path.join(eval_dir, args.eval_data_path)
    output_dir = os.path.join(eval_dir, args.output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation data
    data = load_eval_data(eval_data_path)
    logging.info(f"Loaded {len(data)} examples from evaluation data")
    
    # Select random samples
    samples = select_samples(data, args.num_samples, args.seed)
    logging.info(f"Selected {len(samples)} random samples")
    
    # Save samples to JSON
    samples_path = os.path.join(output_dir, 'eval_data_samples.json')
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    logging.info(f"Saved samples to {samples_path}")
    
    # Extract questions for embedding generation
    questions = extract_questions(samples)
    logging.info(f"Extracted {len(questions)} questions for embedding generation")
    
    # Generate embeddings
    logging.info("Generating embeddings for questions")
    embeddings = generate_embeddings(questions)
    
    # Save embeddings to NPY file
    embeddings_path = os.path.join(output_dir, 'eval_data_samples.npy')
    np.save(embeddings_path, embeddings)
    logging.info(f"Saved embeddings to {embeddings_path}")
    
    logging.info("Done!")

if __name__ == "__main__":
    main()