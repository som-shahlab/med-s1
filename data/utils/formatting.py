import pandas as pd
import logging
import os
import re
from typing import Dict, Tuple, Optional
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer
import json

def preprocess_text(text: str) -> str:
    """Preprocess text by cleaning and standardizing format"""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def format_chat_template(question: str, thinking: str, answer: str, model_name: str, tokenizer, huatuo_format: bool = False) -> str:
    """Format example using appropriate chat template based on model"""
    # Add Answer: prefix if needed
    if not huatuo_format and "Answer:" not in answer:
        answer = "Answer: " + answer
        
    # Format assistant content based on format flag
    if huatuo_format:
        # HuatuoGPT format with ## markers
        assistant_content = f"## Thinking\n\n{thinking}\n\n## Final Response\n\n{answer}"
    else:
        # Default format with model-specific markers
        if "Llama" in model_name:
            assistant_content = f"<|start_header_id|>think<|end_header_id|>\n{thinking}\n" + \
                              f"<|start_header_id|>answer<|end_header_id|>\n{answer}"
        else:  # Qwen
            assistant_content = f"<|im_start|>think\n{thinking}\n" + \
                              f"<|im_start|>answer\n{answer}"
    
    # Apply chat template consistently
    return tokenizer.apply_chat_template([
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content}
    ], tokenize=False)

def format_for_training(df: pd.DataFrame, config: Dict, experiment_config: Dict, experiment_name: str, 
                       validation_split: float = 0.1) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Format data for training with sft.py.
    
    Args:
        df: DataFrame with selected examples
        config: Global configuration
        experiment_config: Experiment-specific configuration
        experiment_name: Name of experiment
        validation_split: Fraction of data to use for validation (0.0 to disable)
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    logging.info("Formatting for training...")
    
    # Check formatting mode from config
    huatuo_format = experiment_config["curation"].get("huatuo_format", False)
    logging.info(f"Formatting mode: {'HuatuoGPT-style' if huatuo_format else 'default'}")
    if huatuo_format:
        logging.info("Using '## Thinking' and '## Final Response' markers")
    else:
        logging.info("Using model-specific markers with Answer: prefix")
    
    # Get model info from config
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Only format selected examples
    df_selected = df[df['selected_for_training']].copy()
    
    # Format each example
    formatted_data = []
    for idx, row in df_selected.iterrows():
        # Preprocess text
        question = preprocess_text(row['Question'])
        thinking = preprocess_text(row['Complex_CoT']) if pd.notna(row['Complex_CoT']) else ""
        answer = preprocess_text(row['Response'])
        
        # Format using chat template with huatuo_format flag
        text = format_chat_template(
            question=question,
            thinking=thinking,
            answer=answer,
            model_name=model_name,
            tokenizer=tokenizer,
            huatuo_format=huatuo_format
        )
        formatted_data.append({"text": text})
    
    # Convert to HF dataset
    dataset = Dataset.from_dict({'text': [d['text'] for d in formatted_data]})
    
    # Create train/validation split if requested
    if validation_split > 0.0:
        split_dataset = dataset.train_test_split(test_size=validation_split, shuffle=True, seed=42)
        train_dataset = split_dataset["train"]
        validation_dataset = split_dataset["test"]
        logging.info(f"Split dataset into {len(train_dataset)} train and {len(validation_dataset)} validation examples")
        return train_dataset, validation_dataset
    else:
        logging.info(f"No validation split requested, using all {len(dataset)} examples for training")
        return dataset, None