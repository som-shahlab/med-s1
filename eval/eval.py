import argparse
import json
import asyncio
import uvloop
import os
import sys
import gc
from transformers import AutoTokenizer
import sglang as sgl
from asyncio import TimeoutError
from jinja2 import Template
from tqdm import tqdm
import signal
import atexit

from dataset import load_eval_dataset
from utils import (
    get_query_prompt, print_indented, postprocess_output,
    load_experiment_config
)
from model import process_data_batch
from scoring import save_and_score_results, update_results_json

# Add med-s1 to path for imports
med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
sys.path.insert(0, med_s1_dir)

# Global engine for cleanup
engine = None

def cleanup():
    """Cleanup function to ensure engine and other resources are properly closed."""
    global engine
    if engine:
        try:
            print("\nCleaning up sglang engine...")
            
            # Cancel any pending tasks if we have a running loop
            try:
                loop = asyncio.get_running_loop()
                tasks = [task for task in asyncio.all_tasks(loop)
                        if not task.done() and task != asyncio.current_task()]
                
                if tasks:
                    print(f"Found {len(tasks)} pending tasks:")
                    for task in tasks:
                        print(f"  {task.get_name()}")
                        task.cancel()
                    
                    # Wait briefly for tasks to cancel
                    try:
                        loop.run_until_complete(asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=2.0
                        ))
                        print("All tasks cancelled")
                    except asyncio.TimeoutError:
                        print("Warning: Some tasks did not cancel in time")
            except RuntimeError:
                # No running event loop
                pass
            
            # Close engine and ZMQ context
            try:
                print("Shutting down engine...")
                # Ensure ZMQ context is closed first
                import zmq.asyncio
                ctx = zmq.asyncio.Context.instance()
                ctx.destroy(linger=0)
                print("ZMQ context closed")
                
                # Now safe to shutdown engine
                engine.shutdown()
                engine = None
                gc.collect()
                print("Engine shutdown complete")
            except Exception as e:
                print(f"Error during engine shutdown: {e}")
                engine = None
            
            print("Engine cleanup complete")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            engine = None

# Global event loop for signal handling
loop = None

def signal_handler(signo, frame):
    """Handle signals by scheduling cleanup in the event loop"""
    global loop
    if loop and loop.is_running():
        print(f"\nReceived signal {signo}, scheduling cleanup...")
        # Schedule cleanup in the event loop
        loop.call_soon_threadsafe(lambda: asyncio.create_task(async_cleanup()))
    else:
        print(f"\nReceived signal {signo}, running synchronous cleanup...")
        cleanup()

async def async_cleanup():
    """Async cleanup function for graceful shutdown"""
    print("\nStarting async cleanup...")
    try:
        # Cancel all running tasks except our cleanup
        current_task = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks()
                if task is not current_task and not task.done()]
        
        if tasks:
            print(f"Found {len(tasks)} running tasks:")
            for task in tasks:
                print(f"  {task.get_name()}: {task}")
                task.cancel()
            
            # Wait for tasks to cancel
            print("Waiting for tasks to cancel...")
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
                print("All tasks cancelled successfully")
            except asyncio.TimeoutError:
                print("Warning: Some tasks did not cancel in time")
                # Force close any stuck ZMQ sockets
                try:
                    import zmq.asyncio
                    ctx = zmq.asyncio.Context.instance()
                    ctx.destroy(linger=0)
                    print("Forced ZMQ context shutdown")
                except Exception as e:
                    print(f"Error closing ZMQ context: {e}")
        
        # Run normal cleanup
        cleanup()
        
        # Stop the event loop
        loop = asyncio.get_event_loop()
        loop.stop()
    except Exception as e:
        print(f"Error during async cleanup: {e}")
        # Ensure we still try to cleanup
        cleanup()

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

DEFAULT_REPETITIONS = {
    "GPQA_Medical_test": 5, # only has a few hundred samples
    "MedMCQA_validation": 1,
    "MedQA_USLME_test": 1,
    "PubMedQA_test": 1,
    "MMLU-Pro_Medical_test": 1,
    "MedDS": 1,
    "MedDS_NOTA": 1,
    "NEJMCRMC": 10, # only has 120 samples
}

# Default to 1 repetition, can be overridden in experiment config
DEFAULT_REPETITION_MULTIPLIER = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment from results.json')
    parser.add_argument('--path_to_eval_json', type=str, default=None, required=False, help='Path to the evaluation data')
    parser.add_argument('--path_to_output_dir', type=str, default='./results', help='Path to the output directory')
    parser.add_argument('--max_new_tokens', type=int, default=8192, help='Maximum number of new tokens to generate')
    parser.add_argument('--max_tokens', type=int, default=-1, help='Maximum number of tokens to generate')
    parser.add_argument('--use_chat_template', type=bool, default=True, help='Use chat template')
    parser.add_argument('--strict_prompt', action="store_true", help='Use strict prompt')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for parallel inference (optimal for 7B model on H100)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with subset of data')
    parser.add_argument('--debug_samples', type=int, default=100, help='Number of samples in debug mode')
    return parser.parse_args()

def get_repetitions_config(experiment_config=None):
    """Get repetitions config, merging defaults with experiment-specific config."""
    repetitions = DEFAULT_REPETITIONS.copy()
    
    if experiment_config and 'eval' in experiment_config:
        eval_config = experiment_config['eval']
        
        # Get repetition multiplier (default to 1)
        multiplier = eval_config.get('repetition_multiplier', DEFAULT_REPETITION_MULTIPLIER)
        # Cap at 10
        multiplier = min(10, multiplier)
        
        # Apply multiplier to all sources
        if multiplier > 1:
            for source in repetitions:
                repetitions[source] = repetitions[source] * multiplier
        
        # Allow source-specific overrides
        if 'repetitions' in eval_config:
            # Update defaults with experiment-specific values
            repetitions.update(eval_config['repetitions'])
    
    return repetitions

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
    
    # Check if this is a pre-trained model
    print("\nChecking model path resolution:")
    print(f"Resolved config: {experiment_config}")
    print(f"Training results is None? {experiment.get('results', {}).get('training') is None}")
    print(f"Has model_key? {experiment_config.get('model_key')}")
    
    if experiment.get("results", {}).get("training") is None and experiment_config.get("model_key"):
        model_key = experiment_config["model_key"]
        print(f"Pre-trained model detected, using model_key: {model_key}")
        
        # Load config.json to get the model path
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
        
        # Check if best_model directory exists
        if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
            model_path = best_model_path
            print(f"Using best_model from training results: {model_path}")
        else:
            model_path = base_model_path
            print(f"Using base model path from training results: {model_path}")
    
    # Try model_key as fallback
    elif experiment_config.get("model_key"):
        model_key = experiment_config["model_key"]
        print(f"No training results found, using model_key as fallback: {model_key}")
        
        med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
        with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        if model_key in config.get("models", {}):
            model_path = config["models"][model_key]["hf_path"]
            print(f"Found model path in config.json: {model_path}")
        else:
            raise ValueError(f"Model key {model_key} not found in config.json")
    else:
        if not experiment.get("config"):
            raise ValueError(f"No config found for experiment {experiment_name}")
        elif not experiment.get("config", {}).get("model_key"):
            raise ValueError(f"No model_key found in config for experiment {experiment_name}")
        else:
            raise ValueError(f"Could not determine model path for experiment {experiment_name}")
    
    return model_path, experiment

def prepare_data(args, experiment):
    """Load and prepare data for evaluation."""
    print(f"\nLoading evaluation data with {args=}...")
    input_data = load_eval_dataset(args.experiment_name, args.path_to_eval_json)
    print(f"Loaded {len(input_data)} total examples")
    
    # Get repetitions config
    repetitions_config = get_repetitions_config(experiment.get("config"))
    print("\nRepetitions config:")
    for source, reps in repetitions_config.items():
        print(f"  {source}: {reps} runs")
    
    # Group data by source and repetitions
    data_by_source = {}
    for item in input_data:
        source = item.get('source', 'unknown')
        if source not in data_by_source:
            data_by_source[source] = []
        data_by_source[source].append(item)
    
    # Set random seed for reproducibility
    import random
    random.seed(42)
    
    # Print dataset distribution
    print("\nDataset distribution:")
    for source, items in data_by_source.items():
        print(f"  {source}: {len(items)} examples, {repetitions_config.get(source, 1)} runs each")
    
    if args.debug:
        print(f"\nRunning in debug mode with {args.debug_samples} samples")
        debug_data = []
        samples_per_dataset = max(1, args.debug_samples // len(data_by_source))
        
        for source in sorted(data_by_source.keys()):
            items = data_by_source[source]
            if items:
                random.shuffle(items)
                debug_data.extend(items[:samples_per_dataset])
        
        input_data = debug_data
        
        # Regroup after debug sampling
        data_by_source = {}
        for item in input_data:
            source = item.get('source', 'unknown')
            if source not in data_by_source:
                data_by_source[source] = []
            data_by_source[source].append(item)
        
        print("\nDebug dataset sizes:")
        for source, items in sorted(data_by_source.items()):
            print(f"  {source}: {len(items)} samples")
    
    return input_data, data_by_source, repetitions_config

async def process_with_repetitions(engine, tokenizer, data_by_source, repetitions_config, template, query_prompt, args):
    """Process data with source-specific repetitions."""
    results = []
    error_counts = {
        'timeout': 0,
        'sglang': 0,
        'other': 0
    }
    
    # Calculate batches needed for each source and run
    source_batches = {}
    source_run_batches = {}  # Track batches per run for each source
    total_batches = 0
    
    for source, items in data_by_source.items():
        n_runs = repetitions_config.get(source, 1)
        batches_per_run = (len(items) + args.batch_size - 1) // args.batch_size
        total_source_batches = batches_per_run * n_runs
        
        source_batches[source] = total_source_batches
        source_run_batches[source] = batches_per_run
        total_batches += total_source_batches
    
    print(f"\nProcessing {total_batches} total batches of size {args.batch_size}")
    for source, n_batches in source_batches.items():
        print(f"  {source}: {n_batches} batches")
    
    # Create progress bar
    pbar = tqdm(total=total_batches, desc="Processing batches")
    
    try:
        # Process each source
        for source, items in data_by_source.items():
            n_runs = repetitions_config.get(source, 1)
            n_batches = source_batches[source]
            print(f"\nProcessing {len(items)} {source} examples with {n_runs} runs each")
            
            # Create stable identifiers
            for item in items:
                if 'item_id' not in item:
                    item['item_id'] = f"{source}_{item.get('id', hash(str(item)))}"
            
            # Initialize results tracking
            item_runs = {item['item_id']: [] for item in items}
            processed_items = {item['item_id']: set() for item in items}  # Track which runs are done
            
            # Process each run separately to avoid memory issues with long texts
            for run_idx in range(n_runs):
                # Create batch of items for this run that haven't been processed
                run_items = []
                for item in items:
                    if run_idx not in processed_items[item['item_id']]:
                        run_items.append(item.copy())
                
                if not run_items:
                    continue  # Skip if no items need processing
                
                # Process in smaller batches
                for i in range(0, len(run_items), args.batch_size):
                    batch = run_items[i:i + args.batch_size]
                    
                    try:
                        # Process batch with timeout
                        batch_results = await asyncio.wait_for(
                            process_data_batch(
                                engine=engine,
                                tokenizer=tokenizer,
                                input_data=batch,
                                template=template,
                                query_prompt=query_prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                is_use_chat_template=args.use_chat_template
                            ),
                            timeout=120  # 2 minute timeout per batch since we're using smaller batches
                        )
                        
                        # Store results and mark as processed
                        for item, result in zip(batch, batch_results):
                            if run_idx not in processed_items[item['item_id']]:
                                if result.get('error'):
                                    print(f"Warning: Error processing {item['item_id']}: {result['error']}")
                                    if 'sglang' in result['error'].lower():
                                        error_counts['sglang'] += 1
                                    else:
                                        error_counts['other'] += 1
                                item_runs[item['item_id']].append(result)
                                processed_items[item['item_id']].add(run_idx)
                        
                    except asyncio.TimeoutError:
                        print(f"\nERROR: Batch timeout for {source} run {run_idx+1}, batch {i//args.batch_size + 1}")
                        error_counts['timeout'] += 1
                        # Add empty results for timed out items and mark as processed
                        for item in batch:
                            if run_idx not in processed_items[item['item_id']]:
                                item_runs[item['item_id']].append({
                                    'output': '',
                                    'prompt': '',
                                    'error': 'Batch timeout'
                                })
                                processed_items[item['item_id']].add(run_idx)
                    
                    except Exception as e:
                        print(f"\nERROR: Unexpected error processing {source} run {run_idx+1}, batch {i//args.batch_size + 1}: {str(e)}")
                        error_counts['other'] += 1
                        # Add empty results but mark the specific error
                        for item in batch:
                            if run_idx not in processed_items[item['item_id']]:
                                item_runs[item['item_id']].append({
                                    'output': '',
                                    'prompt': '',
                                    'error': f'Unexpected error: {str(e)}'
                                })
                                processed_items[item['item_id']].add(run_idx)
                    
                    # Update progress
                    batch_num = i//args.batch_size + 1
                    batches_per_run = (len(items) + args.batch_size - 1) // args.batch_size
                    
                    # Get list of sources in order
                    sources = list(data_by_source.keys())
                    current_source_idx = sources.index(source)
                    
                    # Sum batches from completed sources
                    total_batches_done = sum(source_batches[s] for s in sources[:current_source_idx])
                    
                    # Add batches completed in current source
                    source_batches_done = run_idx * batches_per_run + batch_num
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Source': source,
                        'Run': f"{run_idx+1}/{n_runs}",
                        'Source Progress': f"{batch_num}/{batches_per_run}",
                        'Total Progress': f"{total_batches_done + source_batches_done}/{total_batches}"
                    })
                
                # GC after each run to prevent memory accumulation
                gc.collect()
            
            # Store results for this source
            for item in items:
                runs = item_runs[item['item_id']]
                if len(runs) != n_runs:
                    print(f"Warning: Item {item['item_id']} has {len(runs)} runs, expected {n_runs}")
                results.append({
                    'sample': item,
                    'runs': runs
                })
            
            # GC after each source to prevent memory accumulation
            gc.collect()
        
        pbar.close()
        
        # Print error summary
        if sum(error_counts.values()) > 0:
            print("\nError Summary:")
            print(f"  Timeouts: {error_counts['timeout']}")
            print(f"  SGLang errors: {error_counts['sglang']}")
            print(f"  Other errors: {error_counts['other']}")
        
        # Validate results
        print("\nValidating results...")
        total_expected_runs = sum(len(items) * repetitions_config.get(source, 1)
                                for source, items in data_by_source.items())
        total_actual_runs = sum(len(result['runs']) for result in results)
        
        print(f"Total items: {len(results)}")
        print(f"Expected total runs: {total_expected_runs}")
        print(f"Actual total runs: {total_actual_runs}")
        
        if total_actual_runs != total_expected_runs:
            print(f"Warning: Total runs mismatch!")
            
        for result in results:
            source = result['sample'].get('source', 'unknown')
            n_runs = repetitions_config.get(source, 1)
            if len(result['runs']) != n_runs:
                print(f"Warning: {source} item has {len(result['runs'])} runs, expected {n_runs}")
        
        return results
    
    finally:
        # Ensure progress bar is closed
        if pbar:
            pbar.close()

async def main_async():
    global engine
    args = parse_args()
    os.makedirs(args.path_to_output_dir, exist_ok=True)

    try:
        # Get model path and experiment config
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
        
        # Load config
        med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
        with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        # Load model
        print("Loading model...")
        engine = sgl.Engine(model_path=model_path)
        print("Model loaded")
        
        # Set overall timeout
        try:
            # Get model key
            model_key = experiment.get("config", {}).get("model_key")
            
            # Set up chat template
            if args.use_chat_template:
                print("\nSetting up chat template:", flush=True)
                print(f"Model key: {model_key}", flush=True)
                print(f"Model path: {model_path}", flush=True)
                
                first_message = True
                def custom_template(messages):
                    nonlocal first_message
                    
                    # Get model format from config
                    model_config = config["models"].get(model_key, {})
                    format = model_config.get("format", "default")
                    print(f"Using {format} format for {model_key}", flush=True)
                    
                    if format == "qwen":
                        # Handle all Qwen models consistently
                        return tokenizer.apply_chat_template(
                            [messages[0]],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    elif format == "nemotron":
                        # Match training format from formatting.py
                        system_prompt = "detailed thinking on"
                        print(f"Using system prompt: {system_prompt}", flush=True)
                        formatted_messages = [
                            {"role": "system", "content": system_prompt},
                            messages[0]
                        ]
                    elif format == "huatuo":
                        # Use HuatuoGPT specific formatting
                        formatted_messages = [{
                            "role": "user",
                            "content": messages[0]["content"]
                        }]
                    else:
                        # Default format
                        formatted_messages = [messages[0]]
                    
                    formatted_prompt = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                    if first_message:
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
            input_data, data_by_source, repetitions_config = prepare_data(args, experiment)
            
            # Get query prompt
            query_prompt = get_query_prompt(args, experiment)
            
            # Run evaluation
            final_results = await process_with_repetitions(
                engine=engine,
                tokenizer=tokenizer,
                data_by_source=data_by_source,
                repetitions_config=repetitions_config,
                template=template,
                query_prompt=query_prompt,
                args=args
            )
            
            # Save and score results
            if args.path_to_eval_json:
                task_name = os.path.basename(args.path_to_eval_json).replace('.json', '')
            else:
                task_name = input_data[0]['source']
            path_to_output, metrics_file, metrics, approaches = save_and_score_results(
                args, final_results, model_path, task_name
            )
            
            # Update results.json
            print("\nSaving results to results.json...")
            update_results_json(args, path_to_output, metrics_file, metrics, approaches, final_results)
            print("Results saved")
            
            print("\nEvaluation complete!")
            
        except asyncio.TimeoutError:
            print("\nEvaluation timed out!")
            raise
            
    finally:
        # Let main() handle cleanup to avoid redundancy
        print("\nEvaluation finished, cleanup will be handled by main()")

def main():
    global loop
    uvloop.install()
    exit_code = 0
    
    try:
        # Create and set event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run main async function
        loop.run_until_complete(main_async())
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
        exit_code = 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        exit_code = 1
    finally:
        try:
            # First try async cleanup
            if loop and loop.is_running():
                try:
                    loop.run_until_complete(async_cleanup())
                except Exception as e:
                    print(f"Error during async cleanup: {e}")
                    exit_code = 1
            
            # Then ensure all tasks are cancelled
            pending = asyncio.all_tasks(loop)
            if pending:
                print(f"Cancelling {len(pending)} pending tasks")
                for task in pending:
                    task.cancel()
                
                # Wait briefly for tasks to cancel
                try:
                    loop.run_until_complete(asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=2.0
                    ))
                except asyncio.TimeoutError:
                    print("Warning: Some tasks did not cancel in time")
                    exit_code = 1
            
            # Finally close the loop
            if loop:
                loop.close()
                
            # Run synchronous cleanup last
            cleanup()
            
        except Exception as e:
            print(f"Error during final cleanup: {e}")
            exit_code = 1
        
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
