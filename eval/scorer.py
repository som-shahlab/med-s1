import re
import json
import difflib
import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
from sklearn.utils import resample
from tqdm import tqdm

def str_similarity(str1, str2):
    """Calculate string similarity using sequence matcher."""
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """Find index of most similar string in list."""
    highest_similarity = 0
    most_similar_index = None
    
    for i, str in enumerate(str_list):
        similarity = str_similarity(str, target_str)
        if similarity >= highest_similarity:
            most_similar_index = i
            highest_similarity = similarity

    return most_similar_index

def match_choice(text, options):
    """Extract model's answer choice from output text."""
    # Split on special tokens if present
    if '<|start_header_id|>answer<|end_header_id|>' in text:
        text = text.split('<|start_header_id|>answer<|end_header_id|>')[-1]
    if 'Answer:' in text:
        text = text.split('Answer:')[-1]
    if '## Final Response\n\n' in text:
        text = text.split('## Final Response\n\n')[-1]
    if '</think>' in text:
        text = text.split('</think>')[-1]
    
    # Try strict prompt matching
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first, ans_last], 1

    # Try non-strict matching
    match_options = 'ABCDEFGHIJKLMN'[:len(options)]
    matches = list(re.finditer(
        r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["+match_options+r"])(\W|[\u4e00-\u9fff]|$)", 
        text, re.S
    ))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first, ans_last], 1

    # Try matching option text
    text = text.lower()
    opsindex = [(opt, text.rindex(options[opt].lower())) 
                for opt in options if options[opt].lower() in text]
    if opsindex:
        ans_last = sorted(opsindex, key=lambda x:x[1], reverse=True)[0][0]
        opsindex = [(opt, text.index(options[opt].lower())) 
                    for opt in options if options[opt].lower() in text]
        ans_first = sorted(opsindex, key=lambda x:x[1], reverse=True)[0][0]
        return [ans_first, ans_last], 2
    
    # Fall back to most similar text
    oplabels = [x for x in options]
    opans = [options[x].lower() for x in options]
    ansindex = find_most_similar_index(opans, text.lower())
    return [oplabels[ansindex], oplabels[ansindex]], 3

def match(prediction, ground_truth):
    """Check if prediction matches any ground truth."""
    for gt in ground_truth:
        if re.search(r"(\W|^)("+re.escape(gt)+r")(\W|$)", prediction.lower(), re.S):
            return 1
    return 0

def calculate_confidence_interval(data: List[Dict], confidence_level=0.95, n_iterations=1000):
    """Calculate confidence interval using hierarchical bootstrap.
    
    Uses a two-level bootstrap approach:
    1. Resamples samples with replacement
    2. For each resampled sample, resamples its runs with replacement
    
    This accounts for both:
    - Variance between different samples
    - Variance between different runs of the same sample
    
    Args:
        data: List of dicts, each containing:
            - sample: The original sample dict
            - runs: List of dicts with run results
        confidence_level: Confidence level (default: 0.95)
        n_iterations: Number of bootstrap iterations (default: 1000)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrapped_means = []
    n_samples = len(data)
    
    # Create progress bar for bootstrap iterations
    pbar = tqdm(total=n_iterations, desc="  Bootstrap", file=sys.stderr,
                bar_format='{l_bar}{bar:20}{r_bar}', ncols=80)
    
    try:
        for _ in range(n_iterations):
            # First level: Resample samples
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            resampled_data = [data[i] for i in sample_indices]
            
            # Second level: For each resampled sample, resample its runs
            sample_means = []
            for sample in resampled_data:
                n_runs = len(sample['runs'])
                run_indices = np.random.choice(n_runs, size=n_runs, replace=True)
                resampled_runs = [sample['runs'][i] for i in run_indices]
                
                # Calculate accuracy for this sample's runs
                run_results = []
                for run in resampled_runs:
                    output = run['output']
                    ans, _ = match_choice(output, sample['sample']['options'])
                    run_results.append(ans[0].lower() == sample['sample']['answer_idx'].lower())
                
                sample_means.append(np.mean(run_results))
            
            bootstrapped_means.append(np.mean(sample_means))
            pbar.update(1)
        
        # Calculate CI
        lower = np.percentile(bootstrapped_means, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(bootstrapped_means, (1 + confidence_level) / 2 * 100)
        return float(lower), float(upper)
    finally:
        pbar.close()

def score(data: List[Dict]) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Score evaluation results with multiple runs per sample.
    
    Args:
        data: List of dicts, each containing:
            - sample: The original sample dict
            - runs: List of dicts with run results
    
    Returns:
        Tuple of (metrics_dict, wrong_samples, correct_samples)
    """
    metrics = defaultdict(lambda: {
        'total_samples': 0,
        'correct_samples': 0,
        'total_runs': 0,
        'correct_runs': 0,
        'expected_runs': None  # Will be set once per source
    })
    wrong_samples = []
    correct_samples = []
    
    for sample_data in data:
        sample = sample_data['sample']
        source = sample.get('source', 'unknown')
        metrics[source]['total_samples'] += 1
        
        # Set expected runs once per source
        n_runs = len(sample_data['runs'])
        if metrics[source]['expected_runs'] is None:
            metrics[source]['expected_runs'] = n_runs
        elif metrics[source]['expected_runs'] != n_runs:
            sys.stderr.write(f"Warning: Inconsistent runs for {source}: got {n_runs}, expected {metrics[source]['expected_runs']}\n")
            sys.stderr.flush()
        
        # Add runs for this sample
        metrics[source]['total_runs'] += n_runs
        
        # Process each run
        run_results = []
        for run in sample_data['runs']:
            output = run['output']
            ans, _ = match_choice(output, sample['options'])
            
            if ans[0].lower() == sample['answer_idx'].lower():
                metrics[source]['correct_runs'] += 1
                run_results.append(True)
            else:
                run_results.append(False)
        
        # Use majority voting across runs
        sample_correct = np.mean(run_results) >= 0.5
        if sample_correct:
            metrics[source]['correct_samples'] += 1
            correct_samples.append(sample_data)
        else:
            wrong_samples.append(sample_data)
    
    # Convert metrics to final format
    final_metrics = {}
    for source, source_metrics in metrics.items():
        expected_runs = source_metrics['expected_runs']
        actual_runs_per_sample = source_metrics['total_runs'] / source_metrics['total_samples']
        
        if abs(actual_runs_per_sample - expected_runs) > 0.01:  # Allow small floating point differences
            sys.stderr.write(f"Warning: {source} has {actual_runs_per_sample} runs/sample but expected {expected_runs}\n")
            sys.stderr.flush()
        
        final_metrics[source] = {
            'accuracy': source_metrics['correct_samples'] / source_metrics['total_samples'],
            'run_accuracy': source_metrics['correct_runs'] / source_metrics['total_runs'],
            'total_samples': source_metrics['total_samples'],
            'total_runs': source_metrics['total_runs'],
            'runs_per_sample': expected_runs,  # Use expected instead of calculated
            'actual_runs_per_sample': actual_runs_per_sample  # Keep track of actual for debugging
        }
    
    return final_metrics, wrong_samples, correct_samples

def get_results(res_path: str) -> Dict:
    """Get evaluation results with confidence intervals.
    
    Args:
        res_path: Path to results file
        
    Returns:
        Dict containing metrics for each dataset
    """
    import sys
    
    sys.stderr.write("\nLoading results for scoring...\n")
    sys.stderr.flush()
    with open(res_path) as f:
        data = json.load(f)
    sys.stderr.write(f"Loaded {len(data)} samples\n")
    sys.stderr.flush()
    
    sys.stderr.write("\nCalculating base metrics...\n")
    sys.stderr.flush()
    metrics, _, _ = score(data)
    sys.stderr.write("Base metrics calculated\n")
    sys.stderr.flush()
    
    # Calculate CIs using hierarchical bootstrap
    sys.stderr.write("\nCalculating confidence intervals...\n")
    sys.stderr.flush()
    for source in metrics:
        sys.stderr.write(f"\nProcessing {source}...\n")
        sys.stderr.flush()
        source_data = [
            sample for sample in data
            if sample['sample'].get('source', 'unknown') == source
        ]
        sys.stderr.write(f"  Found {len(source_data)} samples with {len(source_data[0]['runs'])} runs each\n")
        sys.stderr.write(f"  Running bootstrap with 1000 iterations...\n")
        sys.stderr.flush()
        ci_lower, ci_upper = calculate_confidence_interval(source_data)
        sys.stderr.write(f"  CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]\n")
        sys.stderr.flush()
        
        metrics[source]['accuracy_ci'] = [
            float(ci_lower),
            float(ci_upper)
        ]
    
    sys.stderr.write("\nScoring complete!\n")
    sys.stderr.flush()
    return metrics

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        get_results(sys.argv[1])