import argparse
import json
import asyncio
import uvloop
import os
import sys
from transformers import AutoTokenizer
import sglang as sgl
from jinja2 import Template

from dataset import load_eval_dataset
from utils import (
    get_query_prompt, print_indented, postprocess_output,
    load_experiment_config
)
from model import process_data_batch, process_gpqa_with_multiple_runs
from scoring import save_and_score_results, update_results_json

# Add med-s1 to path for imports
med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
sys.path.insert(0, med_s1_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment from results.json')
    parser.add_argument('--path_to_eval_json', type=str, default=None, required=False, help='Path to the evaluation data (optional, will use dataset from config if not provided)')
    parser.add_argument('--path_to_output_dir', type=str, default='./results', help='Path to the output directory')
    parser.add_argument('--max_new_tokens', type=int, default=2000, help='Maximum number of new tokens to generate')
    parser.add_argument('--max_tokens', type=int, default=-1, help='Maximum number of tokens to generate. If -1, no truncation is performed')
    parser.add_argument('--use_chat_template', type=bool, default=True, help='Use chat template')
    parser.add_argument('--strict_prompt', action="store_true", help='Use strict prompt')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with subset of data')
    parser.add_argument('--debug_samples', type=int, default=100, help='Number of samples to use in debug mode')
    parser.add_argument('--test_time_scaling', action='store_true', help='Run test time scaling evaluation')
    parser.add_argument('--gpqa_repeats', type=int, default=5, help='Number of times to run GPQA_Medical_test for averaging')
    return parser.parse_args()

def get_model_path_from_results(experiment_name: str) -> tuple:
    """Get the model path from results.json based on experiment name."""
    print(f"\nGetting model path for experiment: {experiment_name}")
    
    # Use load_experiment_config to handle reference resolution
    experiment_config = load_experiment_config(experiment_name)
    
    # Get experiment data from results.json for results/training info
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
    with open(results_json, "r") as f:
        results = json.load(f)
    experiment = results["experiments"][experiment_name]
    print(f"Found experiment config: {experiment}")
    
    # Get model_path from results.json
    model_path = None
    
    # Check if this is a pre-trained model (training results should be null)
    print("\nChecking model path resolution:")
    print(f"Resolved config: {experiment_config}")
    print(f"Training results is None? {experiment.get('results', {}).get('training') is None}")
    print(f"Has model_key? {experiment_config.get('model_key')}")
    
    if experiment.get("results", {}).get("training") is None and experiment_config.get("model_key"):
        model_key = experiment_config["model_key"]
        print(f"Pre-trained model detected, using model_key: {model_key}")
        
        # Load config.json to get the model path from model_key
        med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
        with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        if model_key in config.get("models", {}):
            model_path = config["models"][model_key]["hf_path"]
            print(f"Found pre-trained model path in config.json: {model_path}")
        else:
            raise ValueError(f"Model key {model_key} not found in config.json")
    
    # If not pre-trained, try to get model_path from training results
    elif experiment.get("results", {}).get("training", {}) and experiment["results"]["training"].get("model_path"):
        base_model_path = experiment["results"]["training"]["model_path"]
        best_model_path = os.path.join(base_model_path, "best_model")
        
        # Check if best_model directory exists, otherwise use the base path
        if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
            model_path = best_model_path
            print(f"Using best_model from training results: {model_path}")
        else:
            model_path = base_model_path
            print(f"Using base model path from training results: {model_path}")
    # If not pre-trained and no training results, try model_key as fallback
    elif experiment_config.get("model_key"):
        model_key = experiment_config["model_key"]
        print(f"No training results found, using model_key as fallback: {model_key}")
        
        # Load config.json to get the model path from model_key
        med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
        with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        if model_key in config.get("models", {}):
            model_path = config["models"][model_key]["hf_path"]
            print(f"Found model path in config.json: {model_path}")
        else:
            raise ValueError(f"Model key {model_key} not found in config.json")
    else:
        # Provide more helpful error message
        if not experiment.get("config"):
            raise ValueError(f"No config found for experiment {experiment_name}")
        elif not experiment.get("config", {}).get("model_key"):
            raise ValueError(f"No model_key found in config for experiment {experiment_name}")
        else:
            raise ValueError(f"Could not determine model path for experiment {experiment_name}. " +
                           "Either set training results to null for pre-trained models, " +
                           "or provide training results with model_path.")
    
    return model_path, experiment

def prepare_data(args, experiment):
    """Load and prepare data for evaluation."""
    print(f"\nLoading evaluation data with {args=}...")
    input_data = load_eval_dataset(args.experiment_name, args.path_to_eval_json)
    print(f"Loaded {len(input_data)} total examples")
    
    # Separate GPQA_Medical_test data for multiple runs
    gpqa_data = [item for item in input_data if item['source'] == 'GPQA_Medical_test']
    non_gpqa_data = [item for item in input_data if item['source'] != 'GPQA_Medical_test']
    
    if gpqa_data:
        print(f"Found {len(gpqa_data)} GPQA_Medical_test examples for multiple runs")
    
    # Get eval_sample from config if test_time_scaling
    eval_sample = None
    if args.test_time_scaling:
        config = experiment["config"]
        eval_sample = config.get("eval_sample", None)
        
    # Set random seed for reproducibility
    import random
    random.seed(42)
    
    # If eval_sample is set, randomly sample that many examples
    if eval_sample:
        print(f"\nRandomly sampling {eval_sample} examples (seed 42)...")
        input_data = random.sample(input_data, eval_sample)
        # Re-separate after sampling
        gpqa_data = [item for item in input_data if item['source'] == 'GPQA_Medical_test']
        non_gpqa_data = [item for item in input_data if item['source'] != 'GPQA_Medical_test']
    
    # Print dataset distribution
    datasets = set(item['source'] for item in input_data)
    print("\nDataset distribution:")
    for dataset in datasets:
        count = sum(1 for item in input_data if item['source'] == dataset)
        print(f"  {dataset}: {count} examples")
    
    if args.debug:
        print(f"\nRunning in debug mode with {args.debug_samples} samples")
        # Take samples from each dataset to maintain distribution
        debug_data = []
        samples_per_dataset = max(1, args.debug_samples // len(datasets))
        
        # Sort datasets for consistent ordering
        sorted_datasets = sorted(list(datasets))
        for dataset in sorted_datasets:
            dataset_items = [item for item in input_data if item['source'] == dataset]
            if len(dataset_items) > 0:
                # Shuffle with same seed for reproducibility
                random.shuffle(dataset_items)
                debug_data.extend(dataset_items[:samples_per_dataset])
        
        input_data = debug_data
        # Re-separate after debug sampling
        gpqa_data = [item for item in input_data if item['source'] == 'GPQA_Medical_test']
        non_gpqa_data = [item for item in input_data if item['source'] != 'GPQA_Medical_test']
        
        print(f"\nDebug dataset sizes:")
        for dataset in sorted_datasets:
            count = sum(1 for item in debug_data if item['source'] == dataset)
            print(f"  {dataset}: {count} samples")
    
    return input_data, gpqa_data, non_gpqa_data

async def run_test_time_scaling(args, engine, tokenizer, input_data, template, experiment):
    """Run test time scaling evaluation."""
    from test_time_scaling import evaluate_test_time_scaling
    
    print("\nRunning test time scaling evaluation...")
    # Use a larger batch size for test time scaling since we're doing multiple passes
    test_time_batch_size = 48  # Smaller batches since we do multiple passes
    print(f"\nUsing batch size of {test_time_batch_size} for test time scaling evaluation")
    
    # Get reasoning approaches from config if specified
    reasoning_approaches = None
    if "reasoning_approaches" in experiment["config"]:
        reasoning_approaches = experiment["config"]["reasoning_approaches"]
        print(f"\nUsing reasoning approaches from config: {reasoning_approaches}")
    
    # Set environment variable for test_time_scaling.py to access
    os.environ['EXPERIMENT_NAME'] = args.experiment_name
    
    # Run evaluation
    return await evaluate_test_time_scaling(
        engine=engine,
        tokenizer=tokenizer,
        input_data=input_data,
        template=template,
        temperature=args.temperature,
        debug=args.debug,
        debug_samples=args.debug_samples,
        batch_size=test_time_batch_size,
        reasoning_approaches=reasoning_approaches
    )

async def run_standard_evaluation(args, engine, tokenizer, non_gpqa_data, gpqa_data, template, query_prompt, experiment=None):
    """Run standard evaluation with optional GPQA multiple runs."""
    results = []
    
    # Process non-GPQA data normally
    if non_gpqa_data:
        print(f"\nProcessing {len(non_gpqa_data)} non-GPQA examples...")
        non_gpqa_results = await process_data_batch(
            engine=engine,
            tokenizer=tokenizer,
            input_data=non_gpqa_data,
            template=template,
            query_prompt=query_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            is_use_chat_template=args.use_chat_template,
            max_tokens=args.max_tokens,
            batch_size=256
        )
        results.extend(non_gpqa_results)
    
    # Process GPQA data with multiple runs
    if gpqa_data:
        gpqa_results = await process_gpqa_with_multiple_runs(
            engine=engine,
            tokenizer=tokenizer,
            gpqa_data=gpqa_data,
            template=template,
            query_prompt=query_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            is_use_chat_template=args.use_chat_template,
            max_tokens=args.max_tokens,
            batch_size=256,
            num_runs=args.gpqa_repeats
        )
        results.extend(gpqa_results)
    
    return results

async def main_async():
    args = parse_args()
    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Get model path from results.json
    model_path, experiment = get_model_path_from_results(args.experiment_name)

    # Initialize model
    print(f"\nInitializing model: {model_path}")
    print(f"Using tensor parallel size: {args.tensor_parallel_size}")
    
    # Set PyTorch memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    print("Tokenizer loaded")
    if "Llama" in model_path:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in model_path:
        tokenizer.pad_token = "<|fim_pad|>"
    
    # Load config to get model-specific settings
    med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
    with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    # Load model
    print("Loading model...")
    engine = sgl.Engine(model_path=model_path)
    print("Model loaded")
    
    # Get model key from experiment config
    model_key = experiment.get("config", {}).get("model_key")
    
    # Set up chat template
    if args.use_chat_template:
        print("\nSetting up chat template:", flush=True)
        print(f"Model key: {model_key}", flush=True)
        print(f"Model path: {model_path}", flush=True)
        
        # Always use tokenizer's apply_chat_template for consistency with training
        first_message = True  # Flag to track first message
        def custom_template(messages):
            nonlocal first_message
            if model_key == "nemotron:8b":
                # Use Nemotron-specific template with system prompt
                model_config = config["models"][model_key]
                system_prompt = model_config["system_prompt"].format(thinking="on")
                print(f"Using Nemotron format with system prompt: {system_prompt}", flush=True)
                formatted_messages = [
                    {"role": "system", "content": system_prompt},
                    messages[0]  # User message
                ]
            elif model_key == "huatuo:8b":
                # Use default LLaMA format for huatuo since it's trained like LLaMA
                print(f"Using LLaMA format for huatuo", flush=True)
                formatted_messages = [
                    messages[0]  # User message
                ]
            else:
                # For other models, just use the user message
                print(f"Using default format for {model_key}", flush=True)
                formatted_messages = [messages[0]]
            
            # Apply chat template and print result for first example
            formatted_prompt = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
            if first_message:  # Print formatted prompt for first message
                print("\nFirst example after apply_chat_template:")
                print("=" * 80)
                print(formatted_prompt)
                print("=" * 80)
                first_message = False
            return formatted_prompt
        template = custom_template
    else:
        template = None

    # Prepare data
    input_data, gpqa_data, non_gpqa_data = prepare_data(args, experiment)
    
    # Get query prompt
    query_prompt = get_query_prompt(args, experiment)
    
    # Run evaluation
    if args.test_time_scaling:
        final_results = await run_test_time_scaling(args, engine, tokenizer, input_data, template, experiment)
    else:
        final_results = await run_standard_evaluation(args, engine, tokenizer, non_gpqa_data, gpqa_data, template, query_prompt, experiment)
    
    # Save and score results
    if args.path_to_eval_json:
        task_name: str = os.path.basename(args.path_to_eval_json).replace('.json', '')
    else:
        task_name = input_data[0]['source']
    path_to_output, metrics_file, metrics, approaches = save_and_score_results(args, final_results, model_path, task_name)
    
    # Update results.json
    update_results_json(args, path_to_output, metrics_file, metrics, approaches, final_results)
    
    print("\nEvaluation complete!")

def main():
    # Install uvloop as event loop policy
    uvloop.install()
    
    # Run the async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
