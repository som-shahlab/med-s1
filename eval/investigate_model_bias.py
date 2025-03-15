#!/usr/bin/env python
"""
Script to investigate potential biases in the evaluation pipeline between
HuggingFace models and local fine-tuned models.

This script:
1. Loads models from both HuggingFace and local checkpoints
2. Compares their tokenizer configurations
3. Checks for chat templates
4. Tests tokenization of the same input
5. Examines special token handling
6. Prints out the chat format for each model

Usage:
    python investigate_model_bias.py --results_json path/to/results.json
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from jinja2 import Template
import pandas as pd
from tabulate import tabulate
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import re
import difflib


def parse_args():
    parser = argparse.ArgumentParser(description="Investigate model bias in evaluation")
    parser.add_argument(
        "--results_json",
        type=str,
        default="/share/pi/nigam/users/calebwin/med-s1/results.json",
        help="Path to results.json containing experiment configurations"
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default="/share/pi/nigam/users/calebwin/med-s1/config.json",
        help="Path to config.json containing model configurations"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="model_comparison.csv",
        help="Path to save the comparison results as CSV"
    )
    parser.add_argument(
        "--sample_prompt",
        type=str,
        default="Please answer the following multiple-choice question: What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Rome",
        help="Sample prompt to test tokenization"
    )
    parser.add_argument(
        "--show_chat_format",
        action="store_true",
        help="Show the chat format for each model"
    )
    parser.add_argument(
        "--compare_models",
        nargs='+',
        help="Compare specific models by name (e.g., 'huatuo-100-random base')"
    )
    return parser.parse_args()


def load_results_json(path: str) -> Dict:
    """Load the results.json file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_config_json(path: str) -> Dict:
    """Load the config.json file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_model_paths(results: Dict, config: Dict) -> List[Dict[str, Any]]:
    """
    Extract model paths from results.json and config.json.
    
    Returns a list of dictionaries with model information:
    {
        'name': experiment name,
        'type': 'hf' or 'local',
        'path': path to model,
        'model_key': original model key if applicable
    }
    """
    model_paths = []
    
    for exp_name, exp_data in results.get('experiments', {}).items():
        # Skip experiments without config
        if 'config' not in exp_data:
            continue
            
        # Check if it has a local trained model
        if (exp_data.get('results', {}).get('training', {}) and
            exp_data['results']['training'].get('model_path')):
            
            model_path = exp_data['results']['training']['model_path']
            model_key = exp_data['config'].get('model_key')
            
            model_paths.append({
                'name': exp_name,
                'type': 'local',
                'path': model_path,
                'model_key': model_key
            })
        
        # Also add the original HF model if model_key exists
        if exp_data['config'].get('model_key'):
            model_key = exp_data['config']['model_key']
            if model_key in config.get('models', {}):
                hf_path = config['models'][model_key]['hf_path']
                
                # Only add if not already in the list
                if not any(m['path'] == hf_path and m['type'] == 'hf' for m in model_paths):
                    model_paths.append({
                        'name': f"{model_key} (HF)",
                        'type': 'hf',
                        'path': hf_path,
                        'model_key': model_key
                    })
    
    return model_paths


def load_tokenizer(model_info: Dict) -> Optional[AutoTokenizer]:
    """Load tokenizer for a model, handling potential errors."""
    try:
        # First try standard loading
        tokenizer = AutoTokenizer.from_pretrained(
            model_info['path'],
            trust_remote_code=True,
            padding_side="left"
        )
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer for {model_info['name']}: {e}")
        
        # For local models, try to load the tokenizer from the base model
        if model_info['type'] == 'local' and model_info.get('model_key'):
            try:
                # Check if config.json exists and if model_type is missing
                config_path = os.path.join(model_info['path'], 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    if 'model_type' not in config_data:
                        print(f"  Note: model_type is missing from config.json in {model_info['path']}")
                
                # Try to load from the original HF model
                print(f"  Attempting to load tokenizer from base model: {model_info['model_key']}")
                
                # Load config.json to get the model path from model_key
                med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
                with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
                    config = json.load(f)
                
                if model_info['model_key'] in config.get("models", {}):
                    hf_path = config["models"][model_info['model_key']]["hf_path"]
                    tokenizer = AutoTokenizer.from_pretrained(
                        hf_path,
                        trust_remote_code=True,
                        padding_side="left"
                    )
                    print(f"  Successfully loaded tokenizer from base model: {hf_path}")
                    return tokenizer
            except Exception as e2:
                print(f"  Failed to load tokenizer from base model: {e2}")
        
        return None


def analyze_tokenizer(tokenizer, model_info: Dict, sample_prompt: str) -> Dict[str, Any]:
    """Analyze tokenizer properties and behavior."""
    if tokenizer is None:
        return {
            'name': model_info['name'],
            'type': model_info['type'],
            'error': 'Failed to load tokenizer'
        }
    
    # Basic tokenizer properties
    result = {
        'name': model_info['name'],
        'type': model_info['type'],
        'vocab_size': tokenizer.vocab_size,
        'model_max_length': tokenizer.model_max_length,
        'has_chat_template': hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None,
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': tokenizer.bos_token,
        'special_tokens': str(tokenizer.special_tokens_map),
    }
    
    # Test tokenization
    encoded = tokenizer.encode(sample_prompt)
    result['encoded_length'] = len(encoded)
    
    # Test chat template if available
    if result['has_chat_template']:
        try:
            template = Template(tokenizer.chat_template)
            chat_prompt = template.render(
                messages=[{"role": "user", "content": sample_prompt}],
                bos_token=tokenizer.bos_token,
                add_generation_prompt=True
            )
            chat_encoded = tokenizer.encode(chat_prompt)
            result['chat_template_works'] = True
            result['chat_encoded_length'] = len(chat_encoded)
            result['chat_template_adds_tokens'] = len(chat_encoded) - len(encoded)
            result['chat_formatted_prompt'] = chat_prompt
        except Exception as e:
            result['chat_template_works'] = False
            result['chat_template_error'] = str(e)
    else:
        # For models without chat template, use a simple format
        result['chat_formatted_prompt'] = sample_prompt
    
    # Store raw prompt for comparison
    result['raw_prompt'] = sample_prompt
    
    return result


def compare_tokenizers(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert tokenizer analysis results to a DataFrame for comparison."""
    return pd.DataFrame(results)


def print_chat_format(tokenizer_results: List[Dict[str, Any]]):
    """Print the chat format for each model in a clear, readable way."""
    print("\n" + "="*80)
    print("CHAT FORMAT COMPARISON")
    print("="*80)
    
    for result in tokenizer_results:
        print(f"\n\n{'-'*40}")
        print(f"Model: {result['name']} ({result['type']})")
        print(f"{'-'*40}")
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            continue
            
        print(f"Has chat template: {result.get('has_chat_template', False)}")
        
        if result.get('chat_template_works', False):
            print("\nChat formatted prompt:")
            print("-" * 30)
            # Format the chat prompt for better readability
            chat_prompt = result.get('chat_formatted_prompt', '')
            # Replace special tokens with highlighted versions
            chat_prompt = re.sub(r'<[^>]+>', lambda m: f"\033[1;33m{m.group(0)}\033[0m", chat_prompt)
            print(chat_prompt)
            print("-" * 30)
            print(f"Encoded length: {result.get('encoded_length', 0)} → {result.get('chat_encoded_length', 0)} (+{result.get('chat_template_adds_tokens', 0)} tokens)")
        else:
            if 'chat_template_error' in result:
                print(f"\nChat template error: {result['chat_template_error']}")
            print("\nRaw prompt (no chat template):")
            print("-" * 30)
            print(result.get('raw_prompt', ''))
            print("-" * 30)
            print(f"Encoded length: {result.get('encoded_length', 0)}")


def compare_specific_models(tokenizer_results: List[Dict[str, Any]], model_names: List[str]):
    """Compare specific models side by side."""
    # Filter results to only include the specified models
    filtered_results = [r for r in tokenizer_results if any(name.lower() in r['name'].lower() for name in model_names)]
    
    if len(filtered_results) < 2:
        print(f"\nNot enough models found for comparison. Found {len(filtered_results)} of {len(model_names)} requested.")
        return
    
    print("\n" + "="*80)
    print(f"SIDE-BY-SIDE COMPARISON OF {len(filtered_results)} MODELS")
    print("="*80)
    
    # Compare chat formats
    print("\nCHAT FORMAT COMPARISON:")
    
    # Get prompts for comparison
    prompts = []
    for result in filtered_results:
        prompt = result.get('chat_formatted_prompt', result.get('raw_prompt', ''))
        prompts.append((result['name'], prompt))
    
    # Print side by side with differences highlighted
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            name1, prompt1 = prompts[i]
            name2, prompt2 = prompts[j]
            
            print(f"\n{'-'*40}")
            print(f"Comparing {name1} vs {name2}")
            print(f"{'-'*40}")
            
            # Use difflib to highlight differences
            d = difflib.Differ()
            diff = list(d.compare(prompt1.splitlines(), prompt2.splitlines()))
            
            print("\nDifferences:")
            for line in diff:
                if line.startswith('+ '):
                    print(f"\033[32m{line}\033[0m")  # Green for additions
                elif line.startswith('- '):
                    print(f"\033[31m{line}\033[0m")  # Red for deletions
                elif line.startswith('? '):
                    continue  # Skip the markers
                else:
                    print(line)
    
    # Compare tokenization
    print("\nTOKENIZATION COMPARISON:")
    
    # Create a table for comparison
    comparison_data = []
    for result in filtered_results:
        row = {
            'Model': result['name'],
            'Type': result['type'],
            'Has Chat Template': result.get('has_chat_template', False),
            'Raw Encoded Length': result.get('encoded_length', 0),
            'Chat Encoded Length': result.get('chat_encoded_length', 0) if 'chat_encoded_length' in result else 'N/A',
            'Added Tokens': result.get('chat_template_adds_tokens', 0) if 'chat_template_adds_tokens' in result else 'N/A',
            'Special Tokens': str(result.get('special_tokens', ''))[:50] + '...' if len(str(result.get('special_tokens', ''))) > 50 else str(result.get('special_tokens', ''))
        }
        comparison_data.append(row)
    
    # Print as table
    print(tabulate(comparison_data, headers='keys', tablefmt='grid'))
    
    # Analyze potential biases
    print("\nPOTENTIAL BIASES:")
    
    # Check if there are significant differences in tokenization
    encoded_lengths = [r.get('encoded_length', 0) for r in filtered_results]
    chat_encoded_lengths = [r.get('chat_encoded_length', 0) if 'chat_encoded_length' in r else r.get('encoded_length', 0) for r in filtered_results]
    
    max_diff = max(chat_encoded_lengths) - min(chat_encoded_lengths)
    if max_diff > 10:
        print(f"- Large difference in encoded lengths: {max_diff} tokens")
        print("  This could affect model performance as some models receive more context than others.")
    
    # Check for chat template differences
    has_template = [r.get('has_chat_template', False) for r in filtered_results]
    if any(has_template) and not all(has_template):
        print("- Some models have chat templates while others don't")
        print("  This could affect how models interpret the input and generate responses.")
    
    # Check for special token differences
    special_tokens_diff = False
    for i in range(len(filtered_results)):
        for j in range(i+1, len(filtered_results)):
            tokens1 = filtered_results[i].get('special_tokens', '')
            tokens2 = filtered_results[j].get('special_tokens', '')
            if tokens1 != tokens2:
                special_tokens_diff = True
                break
    
    if special_tokens_diff:
        print("- Different special token configurations")
        print("  This could affect tokenization and model behavior.")


def main():
    args = parse_args()
    
    # Load configuration files
    results = load_results_json(args.results_json)
    config = load_config_json(args.config_json)
    
    # Get model paths
    model_paths = get_model_paths(results, config)
    print(f"Found {len(model_paths)} models to analyze")
    
    # Analyze tokenizers
    tokenizer_results = []
    for model_info in model_paths:
        print(f"Analyzing {model_info['name']} ({model_info['type']})...")
        tokenizer = load_tokenizer(model_info)
        result = analyze_tokenizer(tokenizer, model_info, args.sample_prompt)
        tokenizer_results.append(result)
    
    # Compare results
    comparison_df = compare_tokenizers(tokenizer_results)
    
    # Save to CSV
    comparison_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # Print summary
    print("\nTokenizer Comparison Summary:")
    print(tabulate(comparison_df[['name', 'type', 'has_chat_template', 'encoded_length', 'chat_encoded_length', 'chat_template_adds_tokens']],
                  headers=['Model', 'Type', 'Has Chat Template', 'Raw Length', 'Chat Length', 'Added Tokens'],
                  tablefmt='grid'))
    
    # Analyze potential biases
    hf_models = comparison_df[comparison_df['type'] == 'hf']
    local_models = comparison_df[comparison_df['type'] == 'local']
    
    print("\nPotential Biases Analysis:")
    
    # Check chat template availability
    hf_chat_template = hf_models['has_chat_template'].mean() * 100
    local_chat_template = local_models['has_chat_template'].mean() * 100
    print(f"Chat template availability: HF models: {hf_chat_template:.1f}%, Local models: {local_chat_template:.1f}%")
    
    # Check tokenization differences
    if not hf_models.empty and not local_models.empty:
        hf_encoded_length = hf_models['encoded_length'].mean()
        local_encoded_length = local_models['encoded_length'].mean()
        print(f"Average encoded length: HF models: {hf_encoded_length:.1f}, Local models: {local_encoded_length:.1f}")
        
        # Check if chat template works
        if 'chat_template_works' in hf_models.columns and 'chat_template_works' in local_models.columns:
            hf_template_works = hf_models['chat_template_works'].mean() * 100
            local_template_works = local_models['chat_template_works'].mean() * 100
            print(f"Chat template works: HF models: {hf_template_works:.1f}%, Local models: {local_template_works:.1f}%")
    
    # Show chat format if requested
    if args.show_chat_format:
        print_chat_format(tokenizer_results)
    
    # Compare specific models if requested
    if args.compare_models:
        compare_specific_models(tokenizer_results, args.compare_models)
    
    # Recommendations
    print("\nRecommendations:")
    if local_chat_template < hf_chat_template:
        print("- Ensure chat templates are properly saved during training")
    
    print("- Add explicit tokenizer configuration saving in trainer.py")
    print("- Consider adding a warm-up phase before evaluation")
    print("- Standardize special token handling for all models")
    print("- When using chat templates, ensure all models have the same template format")
    print("- Consider disabling chat templates for all models to ensure fair comparison")


if __name__ == "__main__":
    main()