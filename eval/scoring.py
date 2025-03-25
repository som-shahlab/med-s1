import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from scorer import get_results, score, match_choice, calculate_confidence_interval
from utils import print_indented
from datetime import datetime

def score_test_time_scaling(args, final_results, path_to_output):
    """Score results for test time scaling evaluation."""
    metrics = {}
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
        approach_file = os.path.join(args.path_to_output_dir, f"{os.path.basename(path_to_output).replace('.json', '')}_{approach}.json")
        with open(approach_file, 'w') as f:
            json.dump(approach_results, f, ensure_ascii=False, indent=2)
        
        # Score this approach and get failures
        metrics[approach] = get_results(approach_file)
        
        # Print failures for this approach
        print(f"\nApproach: {approach}")
        _, wrong_data, _ = score(approach_results)
        if wrong_data:
            print(f"Found {len(wrong_data)} failures. Here are 2 random examples:")
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
    
    return metrics, approaches

def score_standard_evaluation(path_to_output, final_results, args):
    """Score results for standard evaluation."""
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
    
    return metrics

def analyze_failures(args, path_to_output, final_results, approaches=None):
    """Analyze and print example failures."""
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
        metrics, approaches = score_test_time_scaling(args, final_results, path_to_output)
    else:
        metrics = score_standard_evaluation(path_to_output, final_results, args)
        approaches = None
    
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
    
    # Analyze failures
    analyze_failures(args, path_to_output, final_results, approaches)
    
    return path_to_output, metrics_file, metrics, approaches

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