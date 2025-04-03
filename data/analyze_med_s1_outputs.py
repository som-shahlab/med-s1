import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict

def match_choice(text, options):
    """Simplified version of match_choice from scorer.py"""
    # For strict prompt
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first, ans_last], 1

    # non strict
    match_options = 'ABCDEFGHIJKLMN'[:len(options)]
    matches = list(re.finditer(r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["+match_options+r"])(\W|[\u4e00-\u9fff]|$)", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first, ans_last], 1

    return None, 0

def analyze_output(data: List[Dict]) -> Dict:
    """Analyze the output data for various patterns and metrics"""
    print(f"\nAnalyzing {len(data)} examples...")
    
    stats = {
        "total_examples": len(data),
        "has_think_token": 0,
        "has_answer_token": 0,
        "multiple_think_tokens": 0,
        "contains_answer_is": 0,
        "correct_with_think": 0,
        "correct_without_think": 0,
        "total_with_think": 0,
        "total_without_think": 0
    }
    
    for i, item in enumerate(data):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing example {i}/{len(data)}...")
            
        output = item.get("output", "")
        
        # Count special tokens
        think_tokens = output.count("<|start_header_id|>think<|end_header_id|>")
        has_answer_token = "<|start_header_id|>answer" in output
        
        if think_tokens > 0:
            stats["has_think_token"] += 1
            stats["total_with_think"] += 1
        else:
            stats["total_without_think"] += 1
            
        if think_tokens > 1:
            stats["multiple_think_tokens"] += 1
            
        if has_answer_token:
            stats["has_answer_token"] += 1
            
        if "answer is" in output.lower():
            stats["contains_answer_is"] += 1
            
        # Check correctness
        ans, _ = match_choice(output, item["options"])
        if ans and ans[0].lower() == item["answer_idx"].lower():
            if think_tokens > 0:
                stats["correct_with_think"] += 1
            else:
                stats["correct_without_think"] += 1

    return stats

def analyze_training_data(dataset_path: str):
    """Analyze training data for token patterns"""
    print(f"\nAnalyzing training data: {dataset_path}")
    
    # Load dataset and tokenizer
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    import json
    
    # Get model path from config
    with open("/share/pi/nigam/users/calebwin/med-s1/config.json", "r") as f:
        config = json.load(f)
    model_path = config["models"]["llama3.1:8b"]["hf_path"]
    
    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Successfully loaded {len(dataset['train'])} training examples")
    
    # Analyze token patterns
    stats = {
        "total_examples": len(dataset['train']),
        "has_think_token": 0,
        "has_answer_token": 0,
        "multiple_think_tokens": 0,
        "truncated_examples": 0,
        "avg_tokens_between_think_answer": 0,
        "max_tokens_between_think_answer": 0,
        "token_lengths": [],
        "examples_over_8192": 0,
        "examples_over_2048": 0,
        "examples_over_1024": 0
    }
    
    token_distances = []
    for example in dataset['train']:
        text = example['text']
        
        # Count special tokens
        think_tokens = text.count("<|start_header_id|>think<|end_header_id|>")
        has_answer_token = "<|start_header_id|>answer" in text
        
        if think_tokens > 0:
            stats["has_think_token"] += 1
        if think_tokens > 1:
            stats["multiple_think_tokens"] += 1
        if has_answer_token:
            stats["has_answer_token"] += 1
            
        # Check for truncation
        if text.endswith("<|start_header_id|>think") or text.endswith("Wait"):
            stats["truncated_examples"] += 1
            
        # Analyze token lengths
        tokens = tokenizer.encode(text)
        token_length = len(tokens)
        stats["token_lengths"].append(token_length)
        
        if token_length > 8192:
            stats["examples_over_8192"] += 1
        if token_length > 2048:
            stats["examples_over_2048"] += 1
        if token_length > 1024:
            stats["examples_over_1024"] += 1
            
        # Measure distance between think and answer tokens
        if think_tokens > 0 and has_answer_token:
            think_idx = text.find("<|start_header_id|>think<|end_header_id|>")
            answer_idx = text.find("<|start_header_id|>answer")
            if think_idx >= 0 and answer_idx >= 0:
                # Get text between markers and count tokens
                thinking_text = text[think_idx + len("<|start_header_id|>think<|end_header_id|>"): answer_idx]
                distance = len(tokenizer.encode(thinking_text))
                token_distances.append(distance)
    
    # Calculate stats
    if token_distances:
        stats["avg_tokens_between_think_answer"] = sum(token_distances) / len(token_distances)
        stats["max_tokens_between_think_answer"] = max(token_distances)
    
    # Print results
    print("\n" + "="*50)
    print("Training Data Analysis")
    print("="*50)
    
    print("\nBasic Stats:")
    print(f"Total examples: {stats['total_examples']}")
    
    print("\nSpecial Token Usage:")
    print(f"- Think token present:     {stats['has_think_token']:5d} ({stats['has_think_token']/stats['total_examples']*100:6.2f}%)")
    print(f"- Answer token present:    {stats['has_answer_token']:5d} ({stats['has_answer_token']/stats['total_examples']*100:6.2f}%)")
    print(f"- Multiple think tokens:   {stats['multiple_think_tokens']:5d} ({stats['multiple_think_tokens']/stats['total_examples']*100:6.2f}%)")
    print(f"- Truncated examples:      {stats['truncated_examples']:5d} ({stats['truncated_examples']/stats['total_examples']*100:6.2f}%)")
    
    print("\nToken Length Distribution:")
    print(f"- Over 8192 tokens:        {stats['examples_over_8192']:5d} ({stats['examples_over_8192']/stats['total_examples']*100:6.2f}%)")
    print(f"- Over 2048 tokens:        {stats['examples_over_2048']:5d} ({stats['examples_over_2048']/stats['total_examples']*100:6.2f}%)")
    print(f"- Over 1024 tokens:        {stats['examples_over_1024']:5d} ({stats['examples_over_1024']/stats['total_examples']*100:6.2f}%)")
    print(f"- Average length:          {sum(stats['token_lengths'])/len(stats['token_lengths']):6.1f} tokens")
    print(f"- Maximum length:          {max(stats['token_lengths']):6.1f} tokens")
    
    print("\nThink-Answer Token Distance:")
    print(f"- Average distance:        {stats['avg_tokens_between_think_answer']:6.1f} tokens")
    print(f"- Maximum distance:        {stats['max_tokens_between_think_answer']:6.1f} tokens")
    
    # Print first few examples to check template
    print("\nFirst Example Template Check:")
    print("-" * 50)
    print(dataset['train'][0]['text'])
    print("-" * 50)
    
    print("\n" + "="*50)
    
    return stats

def main():
    """Main function to analyze training data"""
    # Analyze training data
    training_path = "/share/pi/nigam/users/calebwin/hf_cache/med-s1k/plumbing_test_001_20250219_145607/med_s1k_formatted"
    analyze_training_data(training_path)

if __name__ == "__main__":
    main()