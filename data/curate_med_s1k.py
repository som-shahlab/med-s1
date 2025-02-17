import os
import json
import re
import time
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from collections import Counter
from typing import Dict, List, Sequence, Optional
import logging
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.openai_utils import verify_answer
from utils.model_utils import get_base_model_answers
from utils.specialty_utils import load_specialties, batch_classify_specialties
from datetime import datetime
import random
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def get_output_dir() -> str:
    """Get the output directory from environment"""
    output_dir = os.environ.get('MED_S1K_OUTPUT')
    if not output_dir:
        raise ValueError("MED_S1K_OUTPUT environment variable not set")
    return output_dir

def get_token_length(text: str, tokenizer) -> int:
    """Get number of tokens in text"""
    return len(tokenizer(text).input_ids)

def quality_filter(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Filter out empty/null values"""
    logging.info(f"Starting quality filter with {len(df)} examples...")
    
    # Add quality metadata columns
    df['has_question'] = ~df['Question'].isna()
    df['has_cot'] = ~df['Complex_CoT'].isna()
    df['has_response'] = ~df['Response'].isna()
    df['quality_score'] = df[['has_question', 'has_cot', 'has_response']].sum(axis=1)
    
    # Initialize filter tracking
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    
    # Mark quality filter status
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
    
    # Add timestamp
    df['quality_filter_timestamp'] = datetime.now().isoformat()
    
    # Save intermediate state
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(output_dir, f"med_s1k_{config['curation']['version']}_post_quality_{timestamp}.parquet"))
    
    # Log quality filter results
    quality_filtered = df[df['filter_stage'] == 'quality']
    logging.info("=== Quality Filter Results ===")
    logging.info(f"Total examples: {len(df)}")
    logging.info(f"Kept: {len(df[df['filter_status'] == 'kept'])}")
    logging.info(f"Removed: {len(quality_filtered)}")
    logging.info("\nRemoval reasons:")
    for reason, count in quality_filtered['filter_reason'].value_counts().items():
        logging.info(f"- {reason}: {count}")
    logging.info(f"\nQuality score distribution:\n{df['quality_score'].value_counts().sort_index()}")
    
    return df

async def verify_answers_batch(questions: List[str], answers: List[str], correct_answers: List[str]) -> List[tuple[bool, str]]:
    """Verify a batch of answers using async calls"""
    tasks = []
    for q, a, c in zip(questions, answers, correct_answers):
        if a is not None:
            tasks.append(verify_answer(q, a, c))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task for None answers
    
    results = await asyncio.gather(*tasks)
    return [r if isinstance(r, tuple) else (False, "Failed to process") for r in results]

def batch_verify_answers(df: pd.DataFrame, batch_size: int = 4) -> pd.DataFrame:
    """Get base model answers and verify them in batches with rate limiting"""
    config = load_config()  # Load config at start of function
    start_time = time.time()
    total_examples = len(df)
    logging.info(f"\n=== Starting Base Model Verification ===")
    logging.info(f"Processing {total_examples} examples with batch_size={batch_size}")
    
    # Only process examples that passed quality filter
    df_to_process = df[df['filter_status'] == 'kept'].copy()
    questions = df_to_process['Question'].tolist()
    correct_answers = df_to_process['Response'].tolist()
    
    # Process in batches
    all_correct = []
    all_model_answers = []
    total_tokens = 0
    pbar = tqdm(range(0, len(questions), batch_size), desc="Verifying answers")
    
    for i in pbar:
        batch_questions = questions[i:i + batch_size]
        batch_correct = correct_answers[i:i + batch_size]
        
        # Get base model answers
        batch_answers = get_base_model_answers(batch_questions)
        all_model_answers.extend(batch_answers)
        
        # Verify answers asynchronously
        results = asyncio.run(verify_answers_batch(
            batch_questions, batch_answers, batch_correct
        ))
        all_correct.extend(results)
        
        # Approximate token count
        total_tokens += len(batch_questions) * 500
        
        # Update progress bar with cost estimate
        estimated_cost = (total_tokens / 1000) * 0.01
        pbar.set_postfix({'Est. Cost': f'${estimated_cost:.2f}'})
        
        # Rate limiting: 15 RPM = 4 seconds per batch
        time.sleep(4)
    
    # Update dataframe with results
    df.loc[df['filter_status'] == 'kept', 'base_model_response'] = all_model_answers
    
    # Extract correctness and judgments from verification results
    correctness = []
    judgments = []
    for result in all_correct:
        is_correct, judgment = result
        correctness.append(is_correct)
        judgments.append(judgment)
    
    df.loc[df['filter_status'] == 'kept', 'base_model_correct'] = pd.Series(correctness, dtype=bool)
    df.loc[df['filter_status'] == 'kept', 'base_model_judgment'] = pd.Series(judgments)
    df['difficulty_filter_timestamp'] = datetime.now().isoformat()
    
    # Mark examples that base model got correct (1 = hard, 0 = easy)
    df.loc[df['filter_status'] == 'kept', 'difficulty_score'] = df.loc[df['filter_status'] == 'kept', 'base_model_correct'].map({True: 0, False: 1})
    difficulty_mask = (df['filter_status'] == 'kept') & (df['base_model_correct'] == True)
    df.loc[difficulty_mask, 'filter_status'] = 'removed'
    df.loc[difficulty_mask, 'filter_stage'] = 'difficulty'
    df.loc[difficulty_mask, 'filter_reason'] = 'base_model_correct'
    
    # Save intermediate state
    config = load_config()  # Load config here
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(output_dir, f"med_s1k_{config['curation']['version']}_post_difficulty_{timestamp}.parquet"))
    
    elapsed_time = time.time() - start_time
    difficulty_filtered = df[df['filter_stage'] == 'difficulty']
    
    logging.info("=== Difficulty Filter Results ===")
    logging.info(f"Processing time: {elapsed_time:.2f}s")
    logging.info(f"Estimated cost: ${(total_tokens / 1000) * 0.01:.2f}")
    
    logging.info("\nBase model performance:")
    logging.info(f"Total processed: {len(df[df['base_model_response'].notna()])}")
    logging.info(f"Correct: {len(df[df['base_model_correct'] == True])}")
    logging.info(f"Incorrect: {len(df[df['base_model_correct'] == False])}")
    logging.info(f"Failed to process: {len(df[df['base_model_response'].isna()])}")
    
    logging.info("\nFiltering results:")
    logging.info(f"Kept from previous stage: {len(df[df['filter_status'] == 'kept'])}")
    logging.info(f"Removed in this stage: {len(difficulty_filtered)}")
    for reason, count in difficulty_filtered['filter_reason'].value_counts().items():
        logging.info(f"- {reason}: {count}")
    
    return df

def batch_label_specialties(df: pd.DataFrame, batch_size: int = 4) -> pd.DataFrame:
    """Label questions with specialties in batches with rate limiting"""
    config = load_config()
    start_time = time.time()
    total_examples = len(df)
    logging.info(f"Starting specialty classification for {total_examples} examples")
    
    # Only process examples that passed previous filters
    df_to_process = df[df['filter_status'] == 'kept'].copy()
    questions = df_to_process['Question'].tolist()
    
    # Process in batches
    total_tokens = 0  # Initialize token counter
    all_specialties = asyncio.run(batch_classify_specialties(
        questions=questions,
        model=config["model_choices"]["specialty_labeler"],
        batch_size=batch_size
    ))
    
    # Update dataframe
    df.loc[df['filter_status'] == 'kept', 'specialty'] = all_specialties
    df['specialty_label_timestamp'] = datetime.now().isoformat()
    
    # Mark examples without labels
    specialty_mask = (df['filter_status'] == 'kept') & df['specialty'].isna()
    df.loc[specialty_mask, 'filter_status'] = 'removed'
    df.loc[specialty_mask, 'filter_stage'] = 'specialty'
    df.loc[specialty_mask, 'filter_reason'] = 'no_specialty_assigned'
    
    # Save intermediate state
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(output_dir, f"med_s1k_{config['curation']['version']}_post_specialty_{timestamp}.parquet"))
    
    elapsed_time = time.time() - start_time
    specialty_filtered = df[df['filter_stage'] == 'specialty']
    
    logging.info("=== Specialty Classification Results ===")
    logging.info(f"Processing time: {elapsed_time:.2f}s")
    
    logging.info("\nClassification results:")
    logging.info(f"Total examples processed: {len(df_to_process)}")
    logging.info(f"Successfully classified: {len(df[df['specialty'].notna() & (df['filter_status'] == 'kept')])}")
    logging.info(f"Failed to classify: {len(specialty_filtered)}")
    
    logging.info("\nFiltering results:")
    logging.info(f"Kept from previous stage: {len(df_to_process)}")
    logging.info(f"Removed in this stage: {len(specialty_filtered)}")
    for reason, count in specialty_filtered['filter_reason'].value_counts().items():
        logging.info(f"- {reason}: {count}")
    
    logging.info("\nSpecialty distribution for kept examples:")
    for specialty, count in df[df['filter_status'] == 'kept']['specialty'].value_counts().items():
        logging.info(f"- {specialty}: {count}")
    
    return df

def diversity_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Do difficulty-weighted diversity sampling across specialties"""
    config = load_config()  # Load config at start of function
    start_time = time.time()
    
    # Get available examples count
    available_examples = len(df[df['filter_status'] == 'kept'])
    target_size = min(1000, available_examples)
    
    logging.info(f"\n=== Starting Diversity Sampling ===")
    logging.info(f"Available examples: {available_examples}")
    logging.info(f"Target size: {target_size}")
    
    # Load tokenizer for length calculation
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Calculate CoT lengths using tokenizer
    df['cot_length'] = df['Complex_CoT'].fillna('').apply(lambda x: len(tokenizer(x).input_ids))
    
    # Initialize selected set S
    S = set()
    
    # First add all examples with long CoT (0.1%)
    long_examples = df[(df['cot_length'] >= 1000) & (df['filter_status'] == 'kept')]
    for idx in long_examples.index:
        S.add(idx)
    logging.info(f"Added {len(S)} examples with CoT length >= 1000 tokens")
    
    # TODO: For phase 2, consider weighting certain specialties higher based on:
    # 1. Core medical domains (Internal Medicine, Emergency Medicine)
    # 2. Specialties with complex diagnostic reasoning
    # 3. Specialties handling life-threatening conditions
    # 4. Specialties with strong interdisciplinary overlap
    # Similar to how math questions were weighted in the original algorithm
    
    # Initialize domain pools
    all_domains = list(df[df['filter_status'] == 'kept']['specialty'].unique())
    benchmark_domains = all_domains.copy()  # For second phase sampling
    logging.info(f"Initial domain pool size: {len(all_domains)}")
    
    # Create progress bar
    pbar = tqdm(total=target_size-len(S), desc="Sampling questions")
    
    # While |S| < target_size and we have domains to sample from
    while len(S) < target_size and (len(all_domains) > 0 or len(benchmark_domains) > 0):
        # First phase: uniform sampling across all domains (70%)
        if len(S) < min(700, target_size * 0.7):
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
        
        # Get questions Qd in domain d (excluding already selected)
        Qd = df[(df['specialty'] == d) & (df['filter_status'] == 'kept')]
        Qd = Qd[~Qd.index.isin(S)]
        
        if len(Qd) == 0:
            # Remove empty domain from appropriate pool
            if d in all_domains:
                all_domains.remove(d)
            if d in benchmark_domains:
                benchmark_domains.remove(d)
            
            # If either pool is empty but we still need examples,
            # continue with the other pool
            if len(S) < target_size:
                if len(all_domains) == 0 and len(benchmark_domains) > 0:
                    logging.info("First phase pool exhausted, continuing with second phase")
                elif len(benchmark_domains) == 0 and len(all_domains) > 0:
                    logging.info("Second phase pool exhausted, continuing with first phase")
            continue
        
        # Rank by thinking length
        lengths = Qd['cot_length'].values
        ranks = len(lengths) - 1 - np.argsort(np.argsort(lengths))
        weights = np.power(2.0, -ranks)
        weights = weights / weights.sum()
        
        # Sample one question
        selected_idx = np.random.choice(Qd.index, p=weights)
        
        # Add to S
        S.add(selected_idx)
        pbar.update(1)
        
        # If domain is empty after removing selected question
        remaining = df[(df['specialty'] == d) & (df['filter_status'] == 'kept')]
        remaining = remaining[~remaining.index.isin(S)]
        if len(remaining) == 0:
            if d in all_domains:
                all_domains.remove(d)
            if d in benchmark_domains:
                benchmark_domains.remove(d)
    
    pbar.close()
    
    # Mark selected examples
    df['selected_for_training'] = df.index.isin(S)
    diversity_mask = ~df['selected_for_training'] & (df['filter_status'] == 'kept')
    df.loc[diversity_mask, 'filter_status'] = 'removed'
    df.loc[diversity_mask, 'filter_stage'] = 'diversity'
    df.loc[diversity_mask, 'filter_reason'] = 'not_selected_in_sampling'
    df['diversity_sample_timestamp'] = datetime.now().isoformat()
    
    # Save intermediate state
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(output_dir, f"med_s1k_{config['curation']['version']}_post_diversity_{timestamp}.parquet"))
    
    elapsed_time = time.time() - start_time
    diversity_filtered = df[df['filter_stage'] == 'diversity']
    
    logging.info("=== Diversity Sampling Results ===")
    logging.info(f"Processing time: {elapsed_time:.2f}s")
    
    logging.info("\nSampling statistics:")
    logging.info(f"Available examples: {available_examples}")
    logging.info(f"Target size: {target_size}")
    logging.info(f"Long CoT examples (≥1000 tokens): {len(long_examples)}")
    
    # Calculate phase statistics
    phase1_target = min(700, int(target_size * 0.7))
    selected_indices = list(S)
    phase1_count = sum(1 for i, _ in enumerate(selected_indices) if i < phase1_target)
    
    logging.info("\nSampling phases:")
    logging.info(f"Phase 1 (uniform): {phase1_count} examples")
    logging.info(f"Phase 2 (weighted): {len(selected_indices) - phase1_count} examples")
    
    logging.info("\nFiltering results:")
    logging.info(f"Kept from previous stage: {len(df[df['filter_status'] == 'kept'])}")
    logging.info(f"Selected for training: {len(df[df['selected_for_training']])}")
    logging.info(f"Removed in this stage: {len(diversity_filtered)}")
    
    logging.info("\nFinal specialty distribution:")
    specialty_dist = df[df['selected_for_training']]['specialty'].value_counts()
    for specialty, count in specialty_dist.items():
        logging.info(f"- {specialty}: {count}")
    
    # Verify no duplicates in selected examples
    selected_indices = df[df['selected_for_training']].index
    if len(selected_indices) != len(set(selected_indices)):
        duplicates = selected_indices[selected_indices.duplicated()]
        logging.error("WARNING: Duplicates found in selected examples!")
        logging.error(f"Duplicate indices: {duplicates.tolist()}")
        raise ValueError("Duplicate examples found in final dataset")
    
    return df

def preprocess(text):
    """Preprocess text same as tokenization.py"""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def format_for_training(df: pd.DataFrame, config: Dict) -> Dataset:
    """Format data for training with sft.py"""
    logging.info("Formatting for training...")
    
    # Get model info from config
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Only format selected examples
    df_selected = df[df['selected_for_training']].copy()
    
    # Format each example
    formatted_data = []
    for idx, row in df_selected.iterrows():
        logging.info(f"\nProcessing example {idx}:")
        
        # Preprocess text
        question = preprocess(row['Question'])
        thinking = preprocess(row['Complex_CoT'])
        answer = preprocess(row['Response'])
        
        logging.info(f"Question length: {len(question)}")
        logging.info(f"Thinking length: {len(thinking)}")
        logging.info(f"Answer length: {len(answer)}")
        
        # Add "Answer:" prefix if needed
        answer = "Answer: " + answer if "Answer:" not in answer else answer
        
        # Log first 100 chars of each
        logging.info(f"\nQuestion preview: {question[:100]}")
        logging.info(f"Thinking preview: {thinking[:100]}")
        logging.info(f"Answer preview: {answer[:100]}")
        
        # Format as chat with think/answer markers
        if "Llama" in model_name:
            assistant_content = f"<|start_header_id|>think<|end_header_id|>\n{thinking}\n" + \
                              f"<|start_header_id|>answer<|end_header_id|>\n{answer}"
            
            logging.info(f"\nAssistant content preview:")
            logging.info(assistant_content[:200])
            
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content}
            ], tokenize=False)
            
            logging.info(f"\nFormatted text preview:")
            logging.info(text[:200])
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
    
    # Load medical dataset
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    logging.info(f"Loaded {len(dataset)} examples")
    
    # Verify it's the medical dataset
    first_example = dataset[0]
    if "Question" not in first_example or "Complex_CoT" not in first_example:
        raise ValueError("Dataset does not have expected medical question format")
    
    # Take initial sample if configured
    if curation_config["initial_sample"] > 0:
        sample_size = min(curation_config["initial_sample"], len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)
        logging.info(f"Using initial sample of {sample_size} examples")
        if sample_size < 1000:
            logging.warning(f"Initial sample size ({sample_size}) is less than target size (1000)")
            logging.warning(f"Final dataset will contain at most {sample_size} examples")
    
    # Convert to pandas and add metadata columns
    df = pd.DataFrame(dataset)
    df['curation_version'] = curation_config['version']
    df['curation_start_timestamp'] = datetime.now().isoformat()
    
    # Quality filtering
    df = quality_filter(df, config)
    
    # Difficulty filtering with configured batch size
    df = batch_verify_answers(df, batch_size=curation_config["batch_size"])
    
    # Label specialties with configured batch size
    df = batch_label_specialties(df, batch_size=curation_config["batch_size"])
    
    # Diversity sampling
    df = diversity_sample(df)
    
    # Format for training
    dataset = format_for_training(df, config)
    
    # Create versioned output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"{curation_config['version']}_{timestamp}"
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    # Create version-specific directory
    version_dir = os.path.join(output_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Save filtered dataset with all examples and their filtering status
    filtered_path = os.path.join(version_dir, "med_s1k_filtered.parquet")
    df.to_parquet(filtered_path)
    logging.info(f"Saved filtered dataset to {filtered_path}")
    
    # Save curated dataset (selected examples only)
    curated_path = os.path.join(version_dir, "med_s1k_curated.parquet")
    df[df['selected_for_training']].to_parquet(curated_path)
    logging.info(f"Saved curated dataset to {curated_path}")
    
    # Save formatted dataset for training (this is what train/sft.py will use)
    formatted_path = os.path.join(version_dir, "med_s1k_formatted")
    dataset.save_to_disk(formatted_path)
    logging.info(f"Saved formatted dataset to {formatted_path}")
    
    # Save metadata
    metadata = {
        "version": version,
        "timestamp": timestamp,
        "dataset_info": {
            "original_size": len(dataset),
            "final_size": len(df[df['selected_for_training']]),
            "config": curation_config
        },
        "outputs": {
            "filtered": {
                "path": "med_s1k_filtered.parquet",
                "description": "Full dataset with all examples and their filtering status"
            },
            "curated": {
                "path": "med_s1k_curated.parquet",
                "description": "Selected examples only (training data)"
            },
            "formatted": {
                "path": "med_s1k_formatted",
                "description": "HuggingFace dataset formatted for train/sft.py"
            }
        },
        "filtering_stats": {
            "overall": {
                "total": len(df),
                "kept": len(df[df['filter_status'] == 'kept']),
                "removed": len(df[df['filter_status'] == 'removed']),
                "selected": len(df[df['selected_for_training']])
            },
            "by_stage": {
                stage: {
                    "count": int(stage_df['filter_status'].value_counts()['removed']),
                    "reasons": {k: int(v) for k, v in stage_df['filter_reason'].value_counts().to_dict().items()}
                }
                for stage, stage_df in df[df['filter_status'] == 'removed'].groupby('filter_stage')
            },
            "quality_score_distribution": {k: int(v) for k, v in df['quality_score'].value_counts().to_dict().items()},
            "specialty_distribution": {k: int(v) for k, v in df[df['selected_for_training']]['specialty'].value_counts().to_dict().items()},
            "cot_length_stats": {
                "mean": float(df[df['selected_for_training']]['cot_length'].mean()),
                "median": float(df[df['selected_for_training']]['cot_length'].median()),
                "min": int(df[df['selected_for_training']]['cot_length'].min()),
                "max": int(df[df['selected_for_training']]['cot_length'].max())
            }
        }
    }
    
    metadata_path = os.path.join(version_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()