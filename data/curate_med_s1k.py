import os
import json
import re
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from collections import Counter
from typing import Dict, List, Sequence
import logging
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.openai_utils import verify_answer, label_specialty
from utils.model_utils import get_base_model_answers
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO)

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def load_specialties() -> pd.DataFrame:
    """Load medical specialties from CSV"""
    config = load_config()
    return pd.read_csv(config["data_paths"]["specialties_list"])

def get_token_length(text: str, tokenizer) -> int:
    """Get number of tokens in text"""
    return len(tokenizer(text).input_ids)

def quality_filter(dataset, config: Dict, min_cot_length: int = 50) -> pd.DataFrame:
    """Filter out empty or too short reasoning traces"""
    logging.info("Applying quality filter...")
    
    # Convert to pandas for easier filtering
    df = pd.DataFrame(dataset)
    
    # Remove empty/null values
    df = df.dropna(subset=['Question', 'Complex_CoT', 'Response'])
    
    # Get token lengths of reasoning traces using configured model
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df['cot_length'] = df['Complex_CoT'].apply(lambda x: get_token_length(x, tokenizer))
    
    # Filter by minimum length
    df = df[df['cot_length'] >= min_cot_length]
    
    logging.info(f"After quality filtering: {len(df)} examples")
    return df

def batch_verify_answers(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Get base model answers and verify them in batches"""
    logging.info("Verifying base model answers...")
    
    questions = df['Question'].tolist()
    correct_answers = df['Response'].tolist()
    
    # Process in batches
    all_correct = []
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i + batch_size]
        batch_correct = correct_answers[i:i + batch_size]
        
        # Get base model answers
        batch_answers = get_base_model_answers(batch_questions)
        
        # Verify each answer
        for q, a, c in zip(batch_questions, batch_answers, batch_correct):
            is_correct = verify_answer(q, a, c) if a is not None else False
            all_correct.append(is_correct)
    
    df['base_correct'] = all_correct
    return df

def difficulty_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out questions the base model gets correct"""
    logging.info("Applying difficulty filter...")
    
    # Get base model performance
    df = batch_verify_answers(df)
    
    # Keep only questions base model got wrong
    df = df[~df['base_correct']]
    
    logging.info(f"After difficulty filtering: {len(df)} examples")
    return df

def batch_label_specialties(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Label questions with specialties in batches"""
    logging.info("Labeling specialties...")
    
    specialties_df = load_specialties()
    questions = df['Question'].tolist()
    
    # Process in batches
    all_specialties = []
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_questions = questions[i:i + batch_size]
        
        # Label each question
        for q in batch_questions:
            specialty = label_specialty(q, specialties_df)
            all_specialties.append(specialty)
    
    df['specialty'] = all_specialties
    
    # Remove any without labels
    df = df.dropna(subset=['specialty'])
    
    logging.info(f"After specialty labeling: {len(df)} examples")
    return df

def diversity_sample(df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """Do difficulty-weighted diversity sampling"""
    logging.info("Performing diversity sampling...")
    
    # Group by specialty
    specialty_groups = df.groupby('specialty')
    
    # Initialize selected samples
    selected = []
    
    while len(selected) < n_samples:
        # Randomly select a specialty with probability proportional to size
        specialty_weights = {s: len(g) for s, g in specialty_groups}
        total = sum(specialty_weights.values())
        specialty_weights = {s: w/total for s, w in specialty_weights.items()}
        
        specialty = np.random.choice(list(specialty_weights.keys()), 
                                   p=list(specialty_weights.values()))
        
        # Get questions for this specialty
        specialty_questions = specialty_groups.get_group(specialty)
        
        if len(specialty_questions) == 0:
            continue
            
        # Rank by reasoning trace length
        lengths = specialty_questions['cot_length'].values
        ranks = len(lengths) - 1 - np.argsort(np.argsort(lengths))
        weights = np.power(2.0, -ranks)
        weights = weights / weights.sum()
        
        # Sample one question
        selected_idx = np.random.choice(len(specialty_questions), p=weights)
        selected.append(specialty_questions.iloc[selected_idx])
        
        # Remove selected question from pool
        specialty_groups = df[~df.index.isin([selected[-1].name])].groupby('specialty')
    
    result = pd.DataFrame(selected)
    logging.info(f"Final dataset size: {len(result)}")
    return result

def preprocess(text):
    """Preprocess text same as tokenization.py"""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text  # Add return statement

def format_for_training(df: pd.DataFrame, config: Dict) -> Dataset:
    """Format data for training with sft.py"""
    logging.info("Formatting for training...")
    
    # Get model info from config
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Format each example
    formatted_data = []
    for _, row in df.iterrows():
        # Preprocess text
        question = preprocess(row['Question'])
        thinking = preprocess(row['Complex_CoT'])
        answer = preprocess(row['Response'])
        
        # Add "Answer:" prefix if needed
        answer = "Answer: " + answer if "Answer:" not in answer else answer
        
        # Format as chat with think/answer markers
        if "Llama" in model_name:
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": question},
                {
                    "role": "assistant", 
                    "content": f"<|start_header_id|>think<|end_header_id|>\n{thinking}\n" + 
                              f"<|start_header_id|>answer<|end_header_id|>\n{answer}"
                }
            ], tokenize=False)
        else:  # Qwen
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": question},
                {
                    "role": "assistant", 
                    "content": f"<|im_start|>think\n{thinking}\n" + 
                              f"<|im_start|>answer\n{answer}"
                }
            ], tokenize=False)
        formatted_data.append({"text": text})
    
    # Convert to HF dataset and create train/test split
    dataset = Dataset.from_list(formatted_data)
    
    # Split into train/test (90/10 split)
    split = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    
    return split

def main():
    # Load config
    config = load_config()
    curation_config = config["curation"]
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load dataset
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    logging.info(f"Loaded {len(dataset)} examples")
    
    # Take random sample if sample_size is provided
    sample_size = int(os.getenv('SAMPLE_SIZE', '0'))
    if sample_size > 0:
        indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)
        logging.info(f"Using random sample of {sample_size} examples")
    
    # Quality filtering
    df = quality_filter(dataset, config, min_cot_length=curation_config["min_cot_length"])
    
    # Difficulty filtering
    df = difficulty_filter(df)
    
    # Label specialties
    df = batch_label_specialties(df, batch_size=curation_config["batch_size"])
    
    # Diversity sampling
    df = diversity_sample(df, n_samples=curation_config["target_size"])
    
    # Format for training
    dataset = format_for_training(df, config)
    
    # Create versioned output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"{curation_config['version']}_{timestamp}"
    output_dir = config["data_paths"]["med_s1k_output"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw parquet for reference
    output_path = os.path.join(output_dir, f"med_s1k_{version}.parquet")
    df.to_parquet(output_path)
    logging.info(f"Saved raw dataset to {output_path}")
    
    # Save formatted dataset
    formatted_path = os.path.join(output_dir, f"med_s1k_{version}_formatted")
    dataset.save_to_disk(formatted_path)
    logging.info(f"Saved formatted dataset to {formatted_path}")
    
    # Save metadata
    metadata = {
        "version": version,
        "timestamp": timestamp,
        "original_size": len(dataset),
        "final_size": len(df),
        "config": curation_config,
        "specialty_distribution": df['specialty'].value_counts().to_dict()
    }
    
    metadata_path = os.path.join(output_dir, f"med_s1k_{version}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()