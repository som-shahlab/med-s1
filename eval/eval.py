import argparse
import asyncio
import uvloop
import sglang as sgl
import os
import json
import time
import numpy as np
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
from scorer import get_results, score, match_choice, calculate_confidence_interval
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment from results.json')
    parser.add_argument('--path_to_eval_json', type=str, required=True, help='Path to the evaluation data')
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

def print_indented(text: str):
    """Prints each line of the string with one tab indentation."""
    for line in text.split('\n'):
        print(f'\t{line}')

def postprocess_output(pred: str) -> str:
    """Postprocess the output of the model."""
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_samples_file(samples_path: str) -> List[Dict]:
    """Load the evaluation samples data from a JSON file."""
    try:
        with open(samples_path, 'r') as f:
            samples = json.load(f)
        return samples
    except FileNotFoundError:
        print(f"Warning: Samples file {samples_path} not found. No samples will be excluded.")
        return []

def load_file(input_fp: str, exclude_samples: bool = True) -> List[Dict]:
    """Load the evaluation data from a JSON file."""
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    
    # Exclude samples if requested
    if exclude_samples:
        # Get the directory of the input file
        input_dir = os.path.dirname(input_fp)
        samples_path = os.path.join(input_dir, 'eval_data_samples.json')
        samples = load_samples_file(samples_path)
        
        if samples:
            # Create a set of question strings for faster lookup
            sample_questions = {sample.get('question', '') for sample in samples}
            
            # Filter out samples
            original_count = len(input_data)
            input_data = [item for item in input_data if item.get('question', '') not in sample_questions]
            excluded_count = original_count - len(input_data)
            print(f"Excluded {excluded_count} samples from evaluation data")
    
    return input_data

async def call_model(
    engine: sgl.Engine, 
    prompts: List[str], 
    tokenizer: AutoTokenizer, 
    template: Template, 
    max_new_tokens: int = 50, 
    temperature: float = 0, 
    is_print_example: bool = False, 
    is_use_chat_template: bool = False, 
    max_tokens: int = -1
) -> Tuple[List[str], List[str]]:
    """Call the model to get the predicted output using sglang."""
    if is_print_example:
        print("Raw prompt:")
        print("```")
        print_indented(prompts[0])
        print("```")

    if is_use_chat_template:
        prompts = [template.render(messages=[{"role": "user", "content": p}],
                                   bos_token=tokenizer.bos_token,
                                   add_generation_prompt=True)
                   for p in prompts]
    
    if max_tokens > 0:
        new_prompts: List[str] = []
        for prompt in prompts:
            input_ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > max_tokens:
                input_ids = input_ids[:max_tokens]
                new_prompts.append(tokenizer.decode(input_ids))
            else:
                new_prompts.append(prompt[-max_tokens:])
        prompts = new_prompts

    if is_print_example:
        print("Tokenized prompt:")
        print("```")
        print_indented(prompts[0])
        print("```")

    # Generate completions using sglang
    tasks = [
        engine.async_generate(
            prompt=prompt,
            sampling_params={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": 0.9
            }
        )
        for prompt in prompts
    ]
    outputs = await asyncio.gather(*tasks)

    raw_preds = [output['text'] for output in outputs]
    preds = [postprocess_output(pred) for pred in raw_preds]
    
    if is_print_example:
        print("Postprocessed predicted output:")
        print("```")
        print_indented(preds[0])
        print("```")
        
    return preds, raw_preds

async def process_data_batch(
    engine: sgl.Engine,
    tokenizer: AutoTokenizer,
    input_data: List[Dict],
    template: Template,
    query_prompt: str,
    max_new_tokens: int,
    temperature: float,
    is_use_chat_template: bool,
    max_tokens: int,
    batch_size: int = 256
) -> List[Dict]:
    """Process a batch of data through the model."""
    results = []
    total_batches = (len(input_data) + batch_size - 1) // batch_size
    
    print(f"Processing {len(input_data)} examples in {total_batches} batches (size {batch_size})...")
    
    for i in tqdm(range(0, len(input_data), batch_size), desc="Processing batches"):
        batch = input_data[i:i + batch_size]
        current_batch = i // batch_size + 1
        print(f"\nBatch {current_batch}/{total_batches} ({len(batch)} examples)")
        
        # Format batch
        for item in batch:
            item['option_str'] = '\n'.join([f'{op}. {ans}' for op,ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)
        
        # Process batch
        preds, _ = await call_model(
            engine=engine,
            prompts=[item["input_str"] for item in batch],
            tokenizer=tokenizer,
            template=template,
            max_new_tokens=max_new_tokens,
            is_print_example=(i == 0),
            temperature=temperature,
            is_use_chat_template=is_use_chat_template,
            max_tokens=max_tokens
        )
        
        # Store results
        for item, pred in zip(batch, preds):
            if len(pred) > 0:
                item_copy = item.copy()
                item_copy["output"] = pred
                results.append(item_copy)
    
    return results

async def process_gpqa_with_multiple_runs(
    engine: sgl.Engine,
    tokenizer: AutoTokenizer,
    gpqa_data: List[Dict],
    template: Template,
    query_prompt: str,
    max_new_tokens: int,
    temperature: float,
    is_use_chat_template: bool,
    max_tokens: int,
    batch_size: int,
    num_runs: int
) -> List[Dict]:
    """Process GPQA data with multiple runs and majority voting."""
    print(f"\nProcessing {len(gpqa_data)} GPQA examples with {num_runs} runs each...")
    gpqa_results = []
    
    # Run GPQA data multiple times
    for run in range(num_runs):
        print(f"\nGPQA Run {run+1}/{num_runs}")
        run_results = await process_data_batch(
            engine=engine,
            tokenizer=tokenizer,
            input_data=gpqa_data,
            template=template,
            query_prompt=query_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            is_use_chat_template=is_use_chat_template,
            max_tokens=max_tokens,
            batch_size=batch_size
        )
        
        # Store results for this run
        for i, result in enumerate(run_results):
            if run == 0:
                # First run, initialize the result with runs array
                result["runs"] = [{"output": result["output"]}]
                gpqa_results.append(result)
            else:
                # Add this run's output to the existing result
                gpqa_results[i]["runs"].append({"output": result["output"]})
    
    # Process the multiple runs to get averaged results
    for result in gpqa_results:
        # Get all outputs from runs
        outputs = [run["output"] for run in result["runs"]]
        
        # Get all parsed answers and calculate accuracy for each run
        answers = []
        run_accuracies = []
        for output in outputs:
            ans, ans_type = match_choice(output, result["options"])
            last_answer = ans[-1]  # Use the last answer (most likely the final answer)
            answers.append(last_answer)
            # Calculate accuracy for this run (1 if correct, 0 if incorrect)
            is_correct = (last_answer.lower() == result["answer_idx"].lower())
            run_accuracies.append(1 if is_correct else 0)
        
        # Store run accuracies and calculate average
        result["run_accuracies"] = run_accuracies
        result["average_accuracy"] = sum(run_accuracies) / len(run_accuracies)
        
        # Count occurrences of each answer
        answer_counts = Counter(answers)
        
        # Get the most common answer
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Use the output from the first run where the answer matches the most common answer
        for i, run in enumerate(result["runs"]):
            ans, _ = match_choice(run["output"], result["options"])
            if ans[-1] == most_common_answer:
                result["output"] = run["output"]
                result["majority_vote"] = most_common_answer
                result["vote_counts"] = dict(answer_counts)
                result["confidence"] = answer_counts[most_common_answer] / num_runs
                break
    
    return gpqa_results

def get_model_path_from_results(experiment_name: str) -> str:
    """Get the model path from results.json based on experiment name."""
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
        
    with open(results_json, "r") as f:
        results = json.load(f)
    
    if experiment_name not in results.get("experiments", {}):
        raise ValueError(f"Experiment {experiment_name} not found in results.json")
        
    experiment = results["experiments"][experiment_name]
    
    # Get model_path from results.json
    model_path = None
    
    # First try to get model_path from training results
    if experiment.get("results", {}).get("training", {}) and experiment["results"]["training"].get("model_path"):
        base_model_path = experiment["results"]["training"]["model_path"]
        best_model_path = os.path.join(base_model_path, "best_model")
        
        # Check if best_model directory exists, otherwise use the base path
        if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
            model_path = best_model_path
            print(f"Using best_model from training results: {model_path}")
        else:
            model_path = base_model_path
            print(f"Using base model path from training results: {model_path}")
    # If not available, try to get it from model_key in config
    elif experiment.get("config", {}).get("model_key"):
        model_key = experiment["config"]["model_key"]
        print(f"No model_path in training results, using model_key: {model_key}")
        
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
        raise ValueError(f"No model_path or model_key found for experiment {experiment_name}")
    
    return model_path, experiment

def prepare_data(args, experiment):
    """Load and prepare data for evaluation."""
    print("\nLoading evaluation data...")
    input_data = load_file(args.path_to_eval_json)
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

def get_query_prompt(args, experiment=None):
    """Get the appropriate query prompt based on args and experiment config."""
    # Check if experiment config has a prompting approach set to "step"
    step_prompt = False
    if experiment and "config" in experiment:
        step_prompt = experiment["config"].get("prompting", "") == "step"
    
    if args.test_time_scaling or args.strict_prompt:
        base_prompt = "Please answer the following multiple-choice question, ensuring your response concludes with the correct option in the format: 'The answer is BLANK' where BLANK is the correct option. For example, if the correct answer is A, your response should be 'The answer is A.'."
        if step_prompt:
            return f"{base_prompt}\n{{question}}\n{{option_str}}\n\nLet's think step by step."
        else:
            return f"{base_prompt}\n{{question}}\n{{option_str}}"
    else:
        base_prompt = "Please answer the following multiple-choice question:"
        if step_prompt:
            return f"{base_prompt}\n{{question}}\n{{option_str}}\n\nLet's think step by step."
        else:
            return f"{base_prompt}\n{{question}}\n{{option_str}}"

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

def save_and_score_results(args, final_results, model_path):
    """Save results to file and score them."""
    # Save outputs
    model_name = os.path.split(model_path)[-1]
    task_name = model_name + os.path.basename(args.path_to_eval_json).replace('.json','') + \
                ('_strict-prompt' if args.strict_prompt else '') + \
                ('_debug' if args.debug else '')
    file_name = f'{task_name}.json'
    path_to_output = os.path.join(args.path_to_output_dir, file_name)
    with open(path_to_output, 'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)
    
    # Score outputs and get metrics
    if args.test_time_scaling:
        # For test time scaling, score each approach separately
        metrics = {}
        # Get approaches from the results
        approaches = set()
        for item in final_results:
            for result in item['scaling_results']:
                approaches.add(result['approach'])
        
        # Score each approach
        for approach in approaches:
            # Create temporary results with just this approach's outputs
            approach_results = []
            for item in final_results:
                for result in item['scaling_results']:
                    if result['approach'] == approach:
                        # Create copy of item with just this approach's output
                        approach_item = item.copy()
                        approach_item['output'] = result['output']
                        # Add parsed answer using scorer's logic
                        ans, ans_type = match_choice(result['output'], item['options'])
                        approach_item['ans'] = ans
                        approach_item['ans_type'] = ans_type
                        approach_results.append(approach_item)
            
            # Save temporary file for scoring
            approach_file = os.path.join(args.path_to_output_dir, f"{task_name}_{approach}.json")
            with open(approach_file, 'w') as f:
                json.dump(approach_results, f, ensure_ascii=False, indent=2)
            
            # Score this approach and get failures
            metrics[approach] = get_results(approach_file)
            
            # Print failures for this approach
            print(f"\nApproach: {approach}")
            _, wrong_data, _ = score(approach_results)
            if wrong_data:
                print(f"Found {len(wrong_data)} failures. Here are 2 random examples:")
                import random
                for failure in random.sample(wrong_data, min(2, len(wrong_data))):
                    print("\nInput:")
                    print_indented(failure['input_str'])
                    print("\nModel Output:")
                    print_indented(failure['output'])
                    print("\nCorrect Answer:")
                    print_indented(f"Option {failure['answer_idx']}: {failure['answer']}")
                    print(f"Parsed Answer: {failure['ans']}")
                    print("-" * 80)
            else:
                print("No failures found!")
            
            # Clean up temporary file
            os.remove(approach_file)
    else:
        # Original scoring
        metrics = get_results(path_to_output)
        
        # For GPQA, calculate average accuracy across runs
        gpqa_results = [r for r in final_results if r.get('source') == 'GPQA_Medical_test' and 'average_accuracy' in r]
        if gpqa_results:
            # Calculate average accuracy across all GPQA examples
            gpqa_avg_accuracy = sum(r['average_accuracy'] for r in gpqa_results) / len(gpqa_results)
            
            # If GPQA_Medical_test exists in metrics, update its accuracy
            if 'GPQA_Medical_test' in metrics:
                print(f"\nUpdating GPQA_Medical_test accuracy with average of {len(gpqa_results)} examples across {args.gpqa_repeats} runs")
                print(f"Original accuracy (majority vote): {metrics['GPQA_Medical_test']['accuracy']:.4f}")
                print(f"New accuracy (average of runs): {gpqa_avg_accuracy:.4f}")
                
                # Store both accuracies
                metrics['GPQA_Medical_test']['majority_vote_accuracy'] = metrics['GPQA_Medical_test']['accuracy']
                metrics['GPQA_Medical_test']['accuracy'] = gpqa_avg_accuracy
                
                # Recalculate confidence interval for the average accuracy
                # Create binary array for bootstrap
                total_runs = len(gpqa_results) * args.gpqa_repeats
                correct_runs = int(round(gpqa_avg_accuracy * total_runs))
                results_array = np.zeros(total_runs)
                results_array[:correct_runs] = 1
                np.random.shuffle(results_array)
                
                # Calculate CI
                lower, upper = calculate_confidence_interval(results_array)
                metrics['GPQA_Medical_test']['accuracy_ci'] = [float(lower), float(upper)]
    
    # Save detailed metrics with schema
    metrics_file = os.path.join(args.path_to_output_dir, f"{task_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "schema": {
                "accuracy": "float: Best accuracy between first and last answer match",
                "accuracy_ci": "array: [lower, upper] bounds of 95% confidence interval for accuracy",
                "num_correct": "int: Number of correctly answered questions (first match)",
                "total_examples": "int: Total number of examples in dataset",
                "num_answered": "int: Number of questions with correct last answer"
            },
            "metrics": metrics,
            "test_time_scaling": args.test_time_scaling
        }, f, indent=2)
    
    # Print example failures for analysis
    print("\nAnalyzing failures...")
    with open(path_to_output, 'r') as f:
        all_results = json.load(f)
    
    if args.test_time_scaling:
        print("\nAnalyzing failures for each approach...")
        for approach in approaches:
            failures = []
            for result in all_results:
                answer = result['answer_idx']
                # Find this approach's output
                for scaling_result in result['scaling_results']:
                    if scaling_result['approach'] == approach:
                        output = scaling_result['output']
                        ans, ans_type = match_choice(output, result['options'])
                        if ans[-1].lower() != answer.lower():  # Check final answer using scorer's logic
                            failures.append((result, scaling_result))
                            print(f"\nFailure details:")
                            print(f"Output: {output}")
                            print(f"Parsed answer: {ans}")
                            print(f"Expected: {answer}")
            
            # Print example failures for this approach
            if failures:
                import random
                print(f"\nApproach: {approach}")
                print(f"Found {len(failures)} failures. Here are 2 random examples:")
                for result, scaling_result in random.sample(failures, min(2, len(failures))):
                    print("\nInput:")
                    print_indented(result['input_str'])
                    print("\nModel Output:")
                    print_indented(scaling_result['output'])
                    print("\nCorrect Answer:")
                    print_indented(f"Option {result['answer_idx']}: {result['answer']}")
                    print(f"Reasoning tokens: {scaling_result['n_reasoning_tokens']}")
                    print("-" * 80)
            else:
                print(f"\nApproach: {approach} - No failures found!")
    else:
        failures = []
        for result in all_results:
            answer = result['answer_idx']
            output = result['output']
            ans, ans_type = match_choice(output, result['options'])
            if ans[-1].lower() != answer.lower():  # Check final answer using scorer's logic
                failures.append(result)
        
        # Print 3 random failures
        if failures:
            import random
            print(f"\nFound {len(failures)} failures. Here are 3 random examples:")
            for failure in random.sample(failures, min(3, len(failures))):
                print("\nInput:")
                print_indented(failure['input_str'])
                print("\nModel Output:")
                print_indented(failure['output'])
                print("\nCorrect Answer:")
                print_indented(f"Option {failure['answer_idx']}: {failure['answer']}")
                print("-" * 80)
        else:
            print("No failures found!")
    
    return path_to_output, metrics_file, metrics, approaches if args.test_time_scaling else None

def update_results_json(args, path_to_output, metrics_file, metrics, approaches=None):
    """Update results.json with evaluation results."""
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
    
    # Import path_utils for updating results.json
    import sys
    med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
    sys.path.append(os.path.join(med_s1_dir, 'data'))
    from utils.path_utils import update_results_json as update_json
    
    # Create paths dict for update_results_json
    paths = {
        'outputs_path': os.path.abspath(path_to_output),
        'metrics_path': os.path.abspath(metrics_file)
    }
    
    # Create stats dict with summary metrics
    stats = {"summary": metrics}
    
    # Add test time scaling specific data if applicable
    if args.test_time_scaling and approaches:
        # Calculate average reasoning tokens for each approach
        reasoning_tokens = {}
        for approach in approaches:
            tokens = []
            for result in final_results:
                for scaling_result in result['scaling_results']:
                    if scaling_result['approach'] == approach:
                        tokens.append(scaling_result['n_reasoning_tokens'])
            if tokens:
                reasoning_tokens[approach] = sum(tokens) / len(tokens)
        
        # Add to stats
        stats["test_time_scaling"] = True
        stats["reasoning_tokens"] = reasoning_tokens
        
        # Add path to test time scaling plot
        task_name = os.path.basename(path_to_output).replace('.json', '')
        plot_path = os.path.join(args.path_to_output_dir, f"{task_name}_plot.png")
        stats["test_time_scaling_plot"] = os.path.abspath(plot_path)
    
    # Update results.json using the path_utils function
    update_json(
        results_json_path=results_json,
        experiment_name=args.experiment_name,
        stage="eval",
        paths=paths,
        timestamp=datetime.now().isoformat(),
        stats=stats
    )

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

    # Load model
    print("Loading model...")
    engine = sgl.Engine(model_path=model_path)
    print("Model loaded")
    template = Template(tokenizer.chat_template) if args.use_chat_template else None

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
    path_to_output, metrics_file, metrics, approaches = save_and_score_results(args, final_results, model_path)
    
    # Update results.json
    update_results_json(args, path_to_output, metrics_file, metrics, approaches)
    
    print("\nEvaluation complete!")

def main():
    # Install uvloop as event loop policy
    uvloop.install()
    
    # Run the async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
