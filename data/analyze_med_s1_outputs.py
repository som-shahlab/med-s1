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

def main(json_path: str):
    """Main function to analyze a JSON file"""
    print(f"\nAnalyzing file: {json_path}")
    
    # Load data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} examples")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
        
    # Analyze
    stats = analyze_output(data)
    
    # Calculate percentages
    think_accuracy = (stats["correct_with_think"] / stats["total_with_think"] * 100 
                     if stats["total_with_think"] > 0 else 0)
    no_think_accuracy = (stats["correct_without_think"] / stats["total_without_think"] * 100 
                        if stats["total_without_think"] > 0 else 0)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Total examples analyzed: {stats['total_examples']}")
    print(f"\nSpecial Token Statistics:")
    print(f"- Examples with think token: {stats['has_think_token']} ({stats['has_think_token']/stats['total_examples']*100:.2f}%)")
    print(f"- Examples with answer token: {stats['has_answer_token']} ({stats['has_answer_token']/stats['total_examples']*100:.2f}%)")
    print(f"- Examples with multiple think tokens: {stats['multiple_think_tokens']} ({stats['multiple_think_tokens']/stats['total_examples']*100:.2f}%)")
    print(f"- Examples containing 'answer is': {stats['contains_answer_is']} ({stats['contains_answer_is']/stats['total_examples']*100:.2f}%)")
    
    print(f"\nAccuracy Analysis:")
    print(f"- Accuracy with think token: {think_accuracy:.2f}% ({stats['correct_with_think']}/{stats['total_with_think']})")
    print(f"- Accuracy without think token: {no_think_accuracy:.2f}% ({stats['correct_without_think']}/{stats['total_without_think']})")

if __name__ == "__main__":
    # Example usage
    main("/share/pi/nigam/users/calebwin/hf_cache/eval/med-s1-1k-tuned/med-s1-1k-tunedeval_data_strict-prompt.json")
    # main("/share/pi/nigam/users/calebwin/hf_cache/eval/med-s1-1k-tuned/med-s1-1k-tunedeval_data_strict-prompt_debug.json")