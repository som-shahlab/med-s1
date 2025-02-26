import pandas as pd
import logging
import os
import re
from typing import Dict
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer

def preprocess_text(text: str) -> str:
    """Preprocess text by cleaning and standardizing format"""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def format_chat_template(question: str, thinking: str, answer: str, model_name: str, tokenizer) -> str:
    """Format example using appropriate chat template based on model"""
    # Add "Answer:" prefix if needed
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    
    # Format as chat with think/answer markers
    if "Llama" in model_name:
        assistant_content = f"<|start_header_id|>think<|end_header_id|>\n{thinking}\n" + \
                          f"<|start_header_id|>answer<|end_header_id|>\n{answer}"
        
        return tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ], tokenize=False)
    else:  # Qwen
        return tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {
                "role": "assistant", 
                "content": f"<|im_start|>think\n{thinking}\n" + 
                          f"<|im_start|>answer\n{answer}"
            }
        ], tokenize=False)

def format_for_training(df: pd.DataFrame, config: Dict, experiment_name: str) -> Dataset:
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
        # Preprocess text
        question = preprocess_text(row['Question'])
        thinking = preprocess_text(row['Complex_CoT']) if pd.notna(row['Complex_CoT']) else ""
        answer = preprocess_text(row['Response'])
        
        # Format using chat template
        text = format_chat_template(
            question=question,
            thinking=thinking,
            answer=answer,
            model_name=model_name,
            tokenizer=tokenizer
        )
        formatted_data.append({"text": text})
    
    # Convert to HF dataset and create train/test split
    dataset = Dataset.from_dict({'text': [d['text'] for d in formatted_data]})
    split = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    
    # Save dataset
    output_dir = os.environ.get('MED_S1K_OUTPUT')
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    formatted_path = os.path.join(experiment_dir, "med_s1k_formatted")
    split.save_to_disk(formatted_path)
    
    # Update results.json with dataset path
    with open("med-s1/results.json", "r") as f:
        results = json.load(f)
    results["experiments"][experiment_name]["results"]["curation"] = {
        "dataset_path": formatted_path,
        "timestamp": datetime.now().isoformat()
    }
    with open("med-s1/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Split dataset into {len(split['train'])} train and {len(split['test'])} test examples")
    logging.info(f"Saved formatted dataset to {formatted_path}")
    
    return split