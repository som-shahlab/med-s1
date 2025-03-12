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
from typing import List, Dict, Tuple, Optional
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment from results.json')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint or model key from config.json')
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
    parser.add_argument('--config_path', type=str, default=None, help='Path to config.json for model lookup')
    parser.add_argument('--gpqa_runs', type=int, default=5, help='Number of times to run GPQA_Medical_test for averaging')
    return parser.parse_args()

def print_indented(text: str):
    """Prints each line of the string with one tab indentation."""
    for line in text.split('\n'):
        print(f'\t{line}')

def postprocess_output(pred: str) -> str:
    """Postprocess the output of the model.
    Args:
        pred (str): The predicted output of the model.
    Returns:
        str: The postprocessed predicted output.
    """
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp: str, gpqa_runs: int = 5) -> List[Dict]:
    """Load the evaluation data from a JSON file.
    Args:
        input_fp (str): The path to the JSON file containing the evaluation data.
        gpqa_runs (int): Number of times to duplicate GPQA_Medical_test examples for averaging.

    Returns:
        list: A list of dictionaries containing the evaluation data.

    Each example in the returned list looks like this:
        {
            'question': 'Which of the following is not true for myelinated nerve fibers:',
            'options': {
                'A': 'Impulse through myelinated fibers is slower than non-myelinated fibers',
                'B': 'Membrane currents are generated at nodes of Ranvier',
                'C': 'Saltatory conduction of impulses is seen',
                'D': 'Local anesthesia is effective only when the nerve is not covered by myelin sheath'
            },
            'answer_idx': 'A',
            'answer': 'Impulse through myelinated fibers is slower than non-myelinated fibers',
            'source': 'MedMCQA_validation'
        }
        
    Count of each example in the HuatuoGPT-O1 evaluation dataset:
        Counter({
            'MedMCQA_validation': 4183,
            'MMLU-Pro_Medical_test': 1535,
            'MedQA_USLME_test': 1273,
            'PubMedQA_test': 1000,
            'GPQA_Medical_test': 390
        })
    """
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    
    # Process each dataset
    for k, v in data.items():
        # Set source for each example
        for da in v:
            da['source'] = k
        
        # For GPQA_Medical_test, duplicate examples for multiple runs
        if k == 'GPQA_Medical_test':
            print(f"Duplicating GPQA_Medical_test examples {gpqa_runs} times for averaging...")
            # Add run_id to each example to track which run it belongs to
            gpqa_examples = []
            for run_id in range(gpqa_runs):
                for da in v:
                    # Create a deep copy of the example
                    example_copy = json.loads(json.dumps(da))
                    example_copy['run_id'] = run_id
                    example_copy['original_id'] = id(da)  # Store original example ID for grouping later
                    gpqa_examples.append(example_copy)
            input_data.extend(gpqa_examples)
        else:
            # For other datasets, add them as is
            input_data.extend(v)
    
    return input_data

from test_time_scaling import evaluate_test_time_scaling

def process_gpqa_results(results: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Process GPQA_Medical_test results from multiple runs.
    
    Args:
        results: List of result dictionaries with run_id and original_id fields
        
    Returns:
        Tuple of (metrics, processed_results)
        - metrics: Dictionary with accuracy and confidence interval
        - processed_results: List of processed results (non-GPQA and one run of GPQA)
    """
    # First, separate GPQA results from other results
    gpqa_results = [r for r in results if r['source'] == 'GPQA_Medical_test' and 'run_id' in r]
    other_results = [r for r in results if r['source'] != 'GPQA_Medical_test' or 'run_id' not in r]
    
    if not gpqa_results:
        print("No GPQA_Medical_test results with multiple runs found.")
        return {}, results
    
    # Group results by run_id
    runs_by_id = {}
    for result in gpqa_results:
        run_id = result['run_id']
        if run_id not in runs_by_id:
            runs_by_id[run_id] = []
        runs_by_id[run_id].append(result)
    
    # Calculate accuracy for each run
    run_accuracies = []
    for run_id, run_results in runs_by_id.items():
        correct = 0
        total = len(run_results)
        
        for item in run_results:
            # Check if the answer is correct
            if 'ans' in item and item['ans'][-1].lower() == item['answer_idx'].lower():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        run_accuracies.append(accuracy)
        print(f"Run {run_id}: Accuracy = {accuracy:.4f} ({correct}/{total})")
    
    # Calculate average accuracy and confidence interval
    avg_accuracy = np.mean(run_accuracies)
    lower_ci, upper_ci = calculate_confidence_interval(np.array(run_accuracies))
    
    # Calculate total correct answers and answered questions across all runs
    total_correct = 0
    total_answered = 0
    examples_per_run = len(runs_by_id[0]) if 0 in runs_by_id else 0
    
    for run_id, run_results in runs_by_id.items():
        run_correct = 0
        run_answered = 0
        for item in run_results:
            if 'ans' in item and item['ans'][-1].lower() == item['answer_idx'].lower():
                run_correct += 1
            if 'ans' in item and item['ans'][-1].lower() in [opt.lower() for opt in item['options'].keys()]:
                run_answered += 1
        total_correct += run_correct
        total_answered += run_answered
    
    # Calculate averages
    avg_correct = total_correct / len(run_accuracies) if run_accuracies else 0
    avg_answered = total_answered / len(run_accuracies) if run_accuracies else 0
    
    metrics = {
        "GPQA_Medical_test": {
            "accuracy": float(avg_accuracy),
            "accuracy_ci": [float(lower_ci), float(upper_ci)],
            "num_correct": int(avg_correct),
            "total_examples": examples_per_run,
            "num_answered": int(avg_answered),
            # Include additional metrics for debugging/analysis
            "run_accuracies": [float(acc) for acc in run_accuracies],
            "num_runs": len(run_accuracies)
        }
    }
    
    print(f"\nGPQA_Medical_test averaged over {len(run_accuracies)} runs:")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
    print(f"  Num correct: {int(avg_correct)}/{examples_per_run}")
    print(f"  Num answered: {int(avg_answered)}/{examples_per_run}")
    print(f"  Individual run accuracies: {[f'{acc:.4f}' for acc in run_accuracies]}")
    
    # Return metrics and only include one run of GPQA results (run_id=0) plus other results
    processed_results = other_results
    if 0 in runs_by_id:
        # Add just the first run's results for GPQA
        for item in runs_by_id[0]:
            # Create a copy without run-specific fields
            processed_item = item.copy()
            if 'run_id' in processed_item:
                del processed_item['run_id']
            if 'original_id' in processed_item:
                del processed_item['original_id']
            processed_results.append(processed_item)
    
    return metrics, processed_results

async def call_model(engine: sgl.Engine, prompts: List[str], tokenizer: AutoTokenizer, template: Template, max_new_tokens: int = 50, temperature: float = 0, is_print_example: bool = False, is_use_chat_template: bool = False, max_tokens: int = -1) -> Tuple[List[str], List[str]]:
    """Call the model to get the predicted output using sglang.
    Args:
        engine (sgl.Engine): The sglang Engine instance
        prompts (List[str]): The prompts to call the model with
        tokenizer (AutoTokenizer): The tokenizer to use
        template (Template): The chat template to use
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Sampling temperature
        is_print_example (bool): Whether to print an example
        is_use_chat_template (bool): Whether to use the chat template
        max_tokens (int): Maximum total tokens (-1 for no limit)

    Returns:
        Tuple[List[str], List[str]]: Tuple of (processed predictions, raw predictions)
    """
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

def main():
    args = parse_args()
    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Get the actual model path, handling both local paths and model keys from config.json
    actual_model_path = args.model_path
    
    # Check if we need to look up the model path from config.json
    if ":" in args.model_path:  # This suggests it's a model key like "llama3.1:8b" or "huatuo:8b"
        # Determine config path
        config_path = args.config_path
        if not config_path:
            # Try to find config.json in the same directory as results.json
            results_json = os.environ.get('RESULTS_JSON')
            if results_json:
                config_path = os.path.join(os.path.dirname(results_json), 'config.json')
            else:
                # Default to looking in the current directory
                config_path = 'config.json'
        
        # Load config.json to get the actual HF path
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Look up the model key in the config
            if args.model_path in config.get('models', {}):
                model_config = config['models'][args.model_path]
                if 'hf_path' in model_config:
                    actual_model_path = model_config['hf_path']
                    print(f"Resolved model key '{args.model_path}' to HuggingFace path: {actual_model_path}")
                else:
                    print(f"Warning: Model key '{args.model_path}' found in config but has no 'hf_path'")
            else:
                print(f"Warning: Model key '{args.model_path}' not found in config.json")
        else:
            print(f"Warning: Config file not found at {config_path}")
    
    # Initialize vllm model
    print(f"\nInitializing vllm with model: {actual_model_path}")
    print(f"Using tensor parallel size: {args.tensor_parallel_size}")
    
    # Set PyTorch memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(actual_model_path, trust_remote_code=True, padding_side='left')
    print("Tokenizer loaded")
    if "Llama" in actual_model_path:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in actual_model_path:
        tokenizer.pad_token = "<|fim_pad|>"

    print("Loading model...")

    # Initialize sglang engine with vllm backend
    # vllm_engine_args = VllmEngineArgs(
    #     tensor_parallel_size=args.tensor_parallel_size,
    #     gpu_memory_utilization=0.9,
    #     max_num_seqs=args.batch_size,
    #     max_num_batched_tokens=4096,
    # )
    
    #     engine_args=vllm_engine_args,
    #     enable_prefix_caching=True,
    #     trust_remote_code=True,

    engine = sgl.Engine(
        model_path=actual_model_path,
    )
    print("Model loaded")
    template = Template(tokenizer.chat_template) if args.use_chat_template else None

    print("\nLoading evaluation data...")
    input_data: List[Dict] = load_file(args.path_to_eval_json, args.gpqa_runs)
    print(f"Loaded {len(input_data)} total examples")
    
    # Get eval_sample from config if test_time_scaling
    eval_sample = None
    if args.test_time_scaling:
        # Get experiment config from results.json
        results_json = os.environ.get('RESULTS_JSON')
        if not results_json:
            raise ValueError("RESULTS_JSON environment variable not set")
            
        with open(results_json, "r") as f:
            results = json.load(f)
        
        config = results["experiments"][args.experiment_name]["config"]
        eval_sample = config.get("eval_sample", None)
        
    # Set random seed for reproducibility
    import random
    random.seed(42)
    
    # If eval_sample is set, randomly sample that many examples
    if eval_sample:
        print(f"\nRandomly sampling {eval_sample} examples (seed 42)...")
        input_data = random.sample(input_data, eval_sample)
    
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
            dataset_items = [item for item in input_data if item['source'] == dataset and "probable diagnosis" in item["question"]]
            if len(dataset_items) > 0:
                # Shuffle with same seed for reproducibility
                random.shuffle(dataset_items)
                debug_data.extend(dataset_items[:samples_per_dataset])
        
        input_data = debug_data
        print(f"\nDebug dataset sizes:")
        for dataset in sorted_datasets:
            count = sum(1 for item in debug_data if item['source'] == dataset)
            print(f"  {dataset}: {count} samples")
    
    final_results: List[Dict] = []
    # Always use strict prompt for test time scaling to ensure consistent answer format
    if args.test_time_scaling:
        query_prompt = "Please answer the following multiple-choice question, ensuring your response concludes with the correct option in the format: 'The answer is BLANK' where BLANK is the correct option. For example, if the correct answer is A, your response should be 'The answer is A.'.\n{question}\n{option_str}"
    else:
        if args.strict_prompt:
            query_prompt = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
        else:
            query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"

    # Process examples
    if args.test_time_scaling:
        print("\nRunning test time scaling evaluation...")
        # Use a larger batch size for test time scaling since we're doing multiple passes
        test_time_batch_size = 48  # Smaller batches since we do multiple passes
        print(f"\nUsing batch size of {test_time_batch_size} for test time scaling evaluation")
        # Install uvloop as event loop policy
        uvloop.install()
        
        # Get reasoning approaches from config if specified
        reasoning_approaches = None
        if "reasoning_approaches" in config:
            reasoning_approaches = config["reasoning_approaches"]
            print(f"\nUsing reasoning approaches from config: {reasoning_approaches}")
        
        # Set environment variable for test_time_scaling.py to access
        os.environ['EXPERIMENT_NAME'] = args.experiment_name
        
        # Run evaluation asynchronously
        final_results = asyncio.run(
            evaluate_test_time_scaling(
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
        )
    else:
        # Install uvloop as event loop policy
        uvloop.install()
        
        # Original batched evaluation
        batch_size = 256  # Match vllm's max_num_seqs
        final_results = []
        total_batches = (len(input_data) + batch_size - 1) // batch_size
        
        print(f"\nProcessing {len(input_data)} examples in {total_batches} batches (size {batch_size})...")
        
        async def process_batches():
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
                    max_new_tokens=args.max_new_tokens,
                    is_print_example=(i == 0),
                    temperature=args.temperature,
                    is_use_chat_template=args.use_chat_template,
                    max_tokens=args.max_tokens
                )
                
                # Store results
                for item, pred in zip(batch, preds):
                    if len(pred) > 0:
                        item["output"] = pred
                        final_results.append(item)
            
            return final_results
        
        # Run batches asynchronously
        final_results = asyncio.run(process_batches())

    # Save outputs
    model_name: str = os.path.split(args.model_path)[-1]
    task_name: str = model_name + os.path.basename(args.path_to_eval_json).replace('.json','') + \
                     ('_strict-prompt' if args.strict_prompt else '') + \
                     ('_debug' if args.debug else '')
    file_name: str = f'{task_name}.json'
    path_to_output: str = os.path.join(args.path_to_output_dir, file_name)
    with open(path_to_output,'w') as fw:
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
            # Use scorer.py's match_choice to properly parse answers
            from scorer import match_choice
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
        
        # Add parsed answers to results for GPQA processing
        with open(path_to_output, 'r') as f:
            scored_results = json.load(f)
        
        # Parse answers for each result
        for result in scored_results:
            if 'output' in result:
                ans, ans_type = match_choice(result['output'], result['options'])
                result['ans'] = ans
                result['ans_type'] = ans_type
        
        # Process GPQA results if we have multiple runs
        if any(r.get('source') == 'GPQA_Medical_test' and 'run_id' in r for r in scored_results):
            print("\nProcessing GPQA_Medical_test results from multiple runs...")
            gpqa_metrics, processed_results = process_gpqa_results(scored_results)
            
            # Update metrics with GPQA averaged results
            metrics.update(gpqa_metrics)
            
            # Save processed results (with only one run of GPQA)
            processed_output_path = os.path.join(args.path_to_output_dir, f'{task_name}_processed.json')
            with open(processed_output_path, 'w') as fw:
                json.dump(processed_results, fw, ensure_ascii=False, indent=2)
            
            print(f"Saved processed results to: {processed_output_path}")
    
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
        from scorer import match_choice
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
    
    # Update results.json with eval results and metrics
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
    
    # Get the model name for display in results
    # If the original argument was a model key, use that
    # Otherwise, use the basename of the model path
    model_display_name = args.model_path if ":" in args.model_path else os.path.basename(args.model_path)
        
    # Create eval results
    if args.test_time_scaling:
        # For test time scaling, include paths, metrics, and token counts for each approach
        eval_results = {
            "outputs_path": os.path.abspath(path_to_output),  # Path to all outputs
            "timestamp": datetime.now().isoformat(),
            "test_time_scaling": True,
            "summary": metrics,  # Contains metrics for each approach
            "reasoning_tokens": {},  # Will store average reasoning tokens per approach
            "model_name": model_display_name  # Add model name for reference
        }
        
        # Calculate average reasoning tokens for each approach
        for approach in approaches:
            tokens = []
            for result in final_results:
                for scaling_result in result['scaling_results']:
                    if scaling_result['approach'] == approach:
                        tokens.append(scaling_result['n_reasoning_tokens'])
            if tokens:
                eval_results["reasoning_tokens"][approach] = sum(tokens) / len(tokens)
        
        # Add path to test time scaling plot
        plot_path = os.path.join(args.path_to_output_dir, f"{task_name}_plot.png")
        eval_results["test_time_scaling_plot"] = os.path.abspath(plot_path)
    else:
        # Original eval results format
        eval_results = {
            "outputs_path": os.path.abspath(path_to_output),
            "metrics_path": os.path.abspath(metrics_file),
            "timestamp": datetime.now().isoformat(),
            "summary": metrics,
            "model_name": model_display_name,  # Add model name for reference
            "gpqa_runs": args.gpqa_runs if "GPQA_Medical_test" in metrics else 1
        }
        
        # If we have processed results, add the path
        if "GPQA_Medical_test" in metrics:
            processed_output_path = os.path.join(args.path_to_output_dir, f'{task_name}_processed.json')
            eval_results["processed_outputs_path"] = os.path.abspath(processed_output_path)
    
    # Update results.json safely by reading latest version before writing
    max_retries = 5
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Read the latest version of results.json
            with open(results_json, "r") as f:
                results = json.load(f)
            
            # Update the eval results
            results["experiments"][args.experiment_name]["results"]["eval"] = eval_results
            
            # Write back to results.json
            with open(results_json, "w") as f:
                json.dump(results, f, indent=2)
                
            break  # Success - exit retry loop
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed to update results.json after {max_retries} attempts: {e}")
                raise
            else:
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                time.sleep(retry_delay)


if __name__ == "__main__":
    main()
