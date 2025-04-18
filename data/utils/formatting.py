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

def format_chat_template(question: str, thinking: str, answer: str, model_name: str, tokenizer, format: str = "huatuo", extract: str = None) -> str:
    """Format example using appropriate chat template based on model"""
    # Add Answer: prefix if needed
    if format != "huatuo" and "Answer:" not in answer:
        answer = "Answer: " + answer
    
    # Check if we should skip the thinking part (no-cot mode)
    if extract == "none":
        # Skip the thinking part entirely
        if format == "huatuo":
            assistant_content = f"## Final Response\n\n{answer}"
        elif format == "nemotron":
            assistant_content = f"{answer}"
        else:
            if "Llama" in model_name:
                assistant_content = f"<|start_header_id|>answer<|end_header_id|>\n{answer}"
            else:  # Qwen - match the same format as with thinking
                assistant_content = f"<|im_start|>answer\n{answer}"
    else:
        # Format assistant content based on format flag
        if format == "huatuo":
            # HuatuoGPT format with ## markers
            assistant_content = f"## Thinking\n\n{thinking}\n\n## Final Response\n\n{answer}"
        elif format == "nemotron":
            # Nemotron format with <think> tags
            assistant_content = f"<think>{thinking}</think>{answer}"
        elif format == "qwen":  # Qwen
            # Match m1's simpler format without extra <|im_end|> tags
            assistant_content = f"<|im_start|>think\n{thinking}\n<|im_start|>answer\n{answer}"
        else:
            # Keep consistent format for other models
            assistant_content = f"<|start_header_id|>think<|end_header_id|>\n{thinking}\n" + \
                              f"<|start_header_id|>answer<|end_header_id|>\n{answer}"
    
    # Apply chat template with model-specific handling
    if format == "nemotron":
        # Get Nemotron system prompt from config
        system_prompt = "detailed thinking on"
        return tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ], tokenize=False)
    elif format == "qwen":
        # Simplified chat template without system message, matching m1's implementation
        return tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ], tokenize=False)
    else:
        # Default chat template for other models
        return tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ], tokenize=False)

def format_for_training(df: pd.DataFrame, config: Dict, experiment_config: Dict, experiment_name: str, 
                       validation_split: float = 0.1, seed: int = 42) -> Tuple[Dataset, Optional[Dataset]]:
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
    format = experiment_config["curation"].get("format", "huatuo")
    extract = experiment_config["curation"].get("extract", None)
    
    logging.info(f"Formatting mode: {format}")
    if extract == "none":
        logging.info("Skipping CoT/Thinking section (no-cot mode)")
    elif extract == "step":
        logging.info("Using step-by-step extracted CoT")
    elif extract == "1-sentence":
        logging.info("Using 1-sentence extracted CoT")
    if format == "huatuo":
        logging.info("Using '## Thinking' and '## Final Response' markers")
    elif format == "nemotron":
        logging.info("Using '<think>' tags")
    elif format == "qwen":
        logging.info("Using '<|im_start|>think' and '<|im_start|>answer' markers")
    else:
        logging.info("Using model-specific markers with Answer: prefix")

    # Get model info from config
    # Get model key from experiment config, falling back to base model
    model_key = experiment_config.get("model_key")
    if not model_key:
        model_key = config["model_choices"]["base"]
        print(f"Model key not found in experiment config, using base model: {model_key}")
    model_name = config["models"][model_key]["hf_path"]
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
        
        # Format using chat template with format flag and extract parameter
        text = format_chat_template(
            question=question,
            thinking=thinking,
            answer=answer,
            model_name=model_name,
            tokenizer=tokenizer,
            format=format,
            extract=extract
        )
        formatted_data.append({"text": text})
    
    # Convert to HF dataset
    dataset = Dataset.from_dict({'text': [d['text'] for d in formatted_data]})
    
    # Create train/validation split if requested
    if validation_split > 0.0:
        split_dataset = dataset.train_test_split(test_size=validation_split, shuffle=True, seed=seed)
        train_dataset = split_dataset["train"]
        validation_dataset = split_dataset["test"]
        logging.info(f"Split dataset into {len(train_dataset)} train and {len(validation_dataset)} validation examples")
        return train_dataset, validation_dataset
    else:
        logging.info(f"No validation split requested, using all {len(dataset)} examples for training")
        return dataset, None