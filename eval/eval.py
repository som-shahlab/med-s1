import argparse
import sglang as sgl
import os
import json
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
from scorer import get_results, score, match_choice
from typing import List, Dict, Tuple
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment from results.json')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
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

def load_file(input_fp: str) -> List[Dict]:
    """Load the evaluation data from a JSON file.
    Args:
        input_fp (str): The path to the JSON file containing the evaluation data.

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
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data

from test_time_scaling import evaluate_test_time_scaling

def call_model(llm: LLM, prompts: List[str], tokenizer: AutoTokenizer, template: Template, max_new_tokens: int = 50, temperature: float = 0, is_print_example: bool = False, is_use_chat_template: bool = False, max_tokens: int = -1) -> Tuple[List[str], List[str]]:
    """Call the model to get the predicted output using vllm.
    Args:
        llm (LLM): The vllm LLM instance
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

    # Set up vllm sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens
    )

    # Generate completions using vllm
    outputs = llm.generate(prompts, sampling_params)
    raw_preds = [output.outputs[0].text for output in outputs]
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

    # Initialize vllm model
    print(f"\nInitializing vllm with model: {args.model_path}")
    print(f"Using tensor parallel size: {args.tensor_parallel_size}")
    
    # Set PyTorch memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    print("Tokenizer loaded")
    if "Llama" in args.model_path:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in args.model_path:
        tokenizer.pad_token = "<|fim_pad|>"

    print("Loading model...")
    # Initialize VLLM with minimal settings for now
    # NOTE: For better performance with large-scale evaluation, we may need to tune:
    # - tensor_parallel_size: For multi-GPU inference
    # - max_num_batched_tokens: For efficient batching
    # - max_num_seqs: For parallel sequence processing
    # - gpu_memory_utilization: For KV cache optimization
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,  # Only needed for custom model code
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,  # Maximum number of sequences to process in parallel
        max_num_batched_tokens=4096,  # Maximum number of tokens across all batched sequences
        enable_prefix_caching=True,  # Enable KV cache prefix optimization
        gpu_memory_utilization=0.9  # Use more GPU memory for KV cache
    )
    print("Model loaded")
    template = Template(tokenizer.chat_template) if args.use_chat_template else None

    print("\nLoading evaluation data...")
    input_data: List[Dict] = load_file(args.path_to_eval_json)
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
            dataset_items = [item for item in input_data if item['source'] == dataset]
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
            query_prompt = "Please answer the following multiple-choice question, ensuring your response concludes with the correct option in the format: 'The answer is BLANK' where BLANK is the correct option. For example, if the correct answer is A, your response should be 'The answer is A.'.\n{question}\n{option_str}"
        else:
            query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"

    # Process examples
    if args.test_time_scaling:
        print("\nRunning test time scaling evaluation...")
        # Use a larger batch size for test time scaling since we're doing multiple passes
        test_time_batch_size = min(args.batch_size // 4, 32)  # Smaller batches since we do multiple passes
        print(f"\nUsing batch size of {test_time_batch_size} for test time scaling evaluation")
        
        final_results = evaluate_test_time_scaling(
            llm=llm,
            tokenizer=tokenizer,
            input_data=input_data,
            template=template,
            temperature=args.temperature,
            debug=args.debug,
            debug_samples=args.debug_samples,
            batch_size=test_time_batch_size
        )
    else:
        # Original batched evaluation
        batch_size = 256  # Match vllm's max_num_seqs
        final_results = []
        total_batches = (len(input_data) + batch_size - 1) // batch_size
        
        print(f"\nProcessing {len(input_data)} examples in {total_batches} batches (size {batch_size})...")
        for i in tqdm(range(0, len(input_data), batch_size), desc="Processing batches"):
            batch = input_data[i:i + batch_size]
            current_batch = i // batch_size + 1
            print(f"\nBatch {current_batch}/{total_batches} ({len(batch)} examples)")
            
            # Format batch
            for item in batch:
                item['option_str'] = '\n'.join([f'{op}. {ans}' for op,ans in item['options'].items()])
                item["input_str"] = query_prompt.format_map(item)
            
            # Process batch
            preds, _ = call_model(
                llm=llm,
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
        for approach in ['immediate', 'reasoning', 'reasoning_2x', 'reasoning_4x']:
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
        for approach in ['immediate', 'reasoning', 'reasoning_2x', 'reasoning_4x']:
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
        
    with open(results_json, "r") as f:
        results = json.load(f)
    
    # Create eval results
    if args.test_time_scaling:
        # For test time scaling, include paths, metrics, and token counts for each approach
        eval_results = {
            "outputs_path": os.path.abspath(path_to_output),  # Path to all outputs
            "timestamp": datetime.now().isoformat(),
            "test_time_scaling": True,
            "summary": metrics,  # Contains metrics for each approach
            "reasoning_tokens": {}  # Will store average reasoning tokens per approach
        }
        
        # Calculate average reasoning tokens for each approach
        for approach in ['immediate', 'reasoning', 'reasoning_2x', 'reasoning_4x']:
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
            "summary": metrics
        }
    
    # Update results.json safely
    results["experiments"][args.experiment_name]["results"]["eval"] = eval_results
    
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
