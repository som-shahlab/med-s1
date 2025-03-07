import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import re

# Global configuration
SHOW_CI_REGION = False  # Toggle for confidence interval shading

def load_results(json_path):
    """Load results from the results.json file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['experiments']

def calculate_weighted_average(metrics):
    """Calculate weighted average accuracy and total examples across datasets"""
    total_examples = 0
    total_correct = 0
    for dataset_metrics in metrics.values():
        if isinstance(dataset_metrics, dict):  # Skip non-dict values
            total_examples += dataset_metrics.get('total_examples', 0)
            total_correct += dataset_metrics.get('num_correct', 0)
    
    if total_examples == 0:
        return None, None, None, 0
    
    # Calculate accuracy and average CI width
    accuracy = (total_correct / total_examples) * 100
    
    # Calculate average CI width
    ci_widths = []
    for dataset_metrics in metrics.values():
        if isinstance(dataset_metrics, dict) and 'accuracy_ci' in dataset_metrics:
            ci = dataset_metrics['accuracy_ci']
            ci_widths.append((ci[1] - ci[0]) * 100)
    
    ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 10.0
    ci_lower = accuracy - ci_width/2
    ci_upper = accuracy + ci_width/2
    
    return accuracy, ci_lower, ci_upper, total_examples

def extract_answer(output, question_data):
    # Split on special tokens if present
    if '<|start_header_id|>answer<|end_header_id|>' in output:
        output = output.split('<|start_header_id|>answer<|end_header_id|>')[-1]
    if 'Answer:' in output:
        output = output.split('Answer:')[-1]
    if '## Final Response\n\n' in output:
        output = output.split('## Final Response\n\n')[-1]
    
    # Try to match letter choice (A, B, C, etc.)
    options = question_data.get('options', {})
    match_options = 'ABCDEFGHIJKLMN'[:len(options)]
    
    # Look for "answer is X" pattern
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", output, re.S))
    if matches:
        return matches[-1].group(2)  # Use last match
    
    # Look for letter with specific surrounding patterns
    matches = list(re.finditer(
        r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["+match_options+r"])(\W|[\u4e00-\u9fff]|$)",
        output, re.S
    ))
    if matches:
        return matches[-1].group(2)  # Use last match
    
    # Try to find the answer text in options
    output = output.lower()
    options_lower = {k: v.lower() for k, v in options.items()}
    opsindex = [(opt, output.rindex(options_lower[opt])) for opt in options if options_lower[opt] in output]
    if opsindex:
        return sorted(opsindex, key=lambda x:x[1], reverse=True)[0][0]
    
    return ''

def find_interesting_failures(eval_data, approaches):
    """Find questions where approach with most tokens fails but others succeed"""
    # Get the approach with most tokens
    max_tokens_approach = max(approaches, key=lambda x: eval_data['reasoning_tokens'][x[0]])[0]
    
    # Load detailed results
    if 'outputs_path' not in eval_data:
        return {}
        
    try:
        with open(eval_data['outputs_path'], 'r') as f:
            detailed_results = json.load(f)
    except:
        return {}
    
    # Find the failure where most other approaches succeeded
    best_failure = None
    max_others_correct = -1
    
    # Iterate through questions
    for question_data in detailed_results:
        # Skip if max tokens approach succeeded or question data incomplete
        if not all(result['approach'] in eval_data['summary'] for result in question_data.get('scaling_results', [])):
            continue
            
        # Find prediction for max tokens approach
        max_tokens_pred = None
        for result in question_data.get('scaling_results', []):
            if result['approach'] == max_tokens_approach:
                max_tokens_pred = result
                break
                
        if not max_tokens_pred:
            continue
            
        # Check if max tokens approach was incorrect
        max_tokens_answer = extract_answer(max_tokens_pred['output'], question_data)
        if max_tokens_answer.strip() == question_data.get('answer_idx', '').strip():
            continue
            
        # Count how many other approaches got it right
        others_correct = 0
        for result in question_data.get('scaling_results', []):
            if result['approach'] != max_tokens_approach:
                answer = extract_answer(result['output'], question_data)
                if answer.strip() == question_data.get('answer_idx', '').strip():
                    others_correct += 1
                               
        if others_correct > len(eval_data['summary'])/2 and others_correct > max_others_correct:
            max_others_correct = others_correct
            best_failure = {
                'dataset': question_data.get('source', 'Unknown'),
                'question': question_data.get('question', ''),
                'correct_answer': question_data.get('answer', ''),
                'max_tokens_answer': max_tokens_answer,
                'others_correct': others_correct,
                'total_approaches': len(eval_data['summary'])
            }
    
    return best_failure if best_failure else {}
            
    return interesting_failures

def plot_test_time_scaling(results_data, output_path):
    """Create line plot comparing performance vs reasoning tokens"""
    # Modern, pleasing color palette (temporarily only showing huatuo-tts)
    colors = {
        # 'med-s1-1k-tuned-tts': '#2D5D7B',  # Deep blue
        # 'random-1k-tts': '#BB4430',  # Rust red
        # 'base-tts': '#6B9080',  # Sage green
        # 'base-tts-wait': '#6B9080',  # Sage green
        'huatuo-tts-wait': '#8E7DBE',  # Purple
        # 'huatuo-tts-wait': '#8E7DBE',  # Purple
        # 'huatuo-2000': '#8E7DBE',  # Same purple for huatuo
        # 'med-s1-5k-tts': '#4C8FBD',  # Soft blue
        # 'med-s1-25k-tts': '#3E8E41',  # Fresh green
        # 'random-5k-tts': '#FFC107',  # Warm orange
    }

    # Model keys mapping (temporarily only showing huatuo-tts)
    model_keys = {
        # 'med-s1-1k-tuned-tts': 'Med-S1 (Ours)',
        # 'random-1k-tts': 'Random Fine-Tune',
        # 'base-tts': 'Base Model',
        'huatuo-tts-wait': 'HuatuoGPT',
        # 'base-tts-wait': 'Base Model (Wait)',
        # 'huatuo-tts-wait': 'HuatuoGPT (Wait)',
        # 'huatuo-2000': 'HuatuoGPT',
        # 'med-s1-5k-tts': 'Med-S1 (5k)',
        # 'med-s1-25k-tts': 'Med-S1 (25k)',
        # 'random-5k-tts': 'Random Fine-Tune (5k)',
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # First plot huatuo-2000 point at x=0
    if 'huatuo-eval-2000' in results_data:
        huatuo_data = results_data['huatuo-eval-2000']
        if huatuo_data.get('results', {}).get('eval', {}).get('summary'):
            metrics = huatuo_data['results']['eval']['summary']
            accuracy, ci_lower, ci_upper, _ = calculate_weighted_average(metrics)
            if accuracy is not None and 'huatuo-2000' in colors:
                # Plot single point at x=0
                ax.plot([0], [accuracy], 'o',
                        color=colors['huatuo-2000'],
                        label='HuatuoGPT',
                        markersize=10,
                        zorder=10)  # Put point on top
                
                if SHOW_CI_REGION:
                    # Add confidence interval shading for single point
                    ax.fill_between([0], [ci_lower], [ci_upper],
                                  color=colors['huatuo-2000'], alpha=0.2)
    
    # Plot each model's performance
    for model_name, exp_data in results_data.items():
        try:
            # Skip huatuo-eval-2000 as it's already plotted
            if model_name == 'huatuo-eval-2000':
                continue
                
            # Get display name, using model name if not in mapping
            display_name = model_keys.get(model_name, model_name)
            
            # Get color, using a default if not in mapping
            color = colors.get(model_name, '#666666')  # Default gray
                
            model_data = results_data[model_name]
            if not model_data or not model_data.get('results') or not model_data['results'].get('eval'):
                continue
                
            eval_data = model_data['results']['eval']
            
            # Check all required fields exist
            if not all(eval_data.get(field) for field in ['test_time_scaling', 'reasoning_tokens', 'summary']):
                continue
            
            # Get x values (average reasoning tokens) and y values (accuracy)
            x_values = []
            y_values = []
            ci_lower = []
            ci_upper = []
            
            # Sort approaches by token count
            approaches = sorted(
                eval_data['reasoning_tokens'].items(),
                key=lambda x: x[1]  # Sort by token count
            )
        except Exception as e:
            print(f"Warning: Failed to process {model_name}: {e}")
            continue
        
        for approach, tokens in approaches:
            try:
                # Get metrics for this approach
                metrics = eval_data['summary'].get(approach)
                if not metrics:
                    print(f"Warning: No metrics for {approach} in {model_name}")
                    continue
                
                # Calculate weighted accuracy and CIs
                accuracy, lower, upper, _ = calculate_weighted_average(metrics)
                if accuracy is None:
                    continue
                
                x_values.append(tokens)
                y_values.append(accuracy)
                ci_lower.append(lower)
                ci_upper.append(upper)
                
            except Exception as e:
                print(f"Warning: Failed to process {approach} in {model_name}: {e}")
                continue
        
        if model_name in colors:
            # Plot line with points
            ax.plot(x_values, y_values, '-o',
                    color=colors[model_name],
                    label=display_name,
                    linewidth=2,
                    markersize=8)
            
            if SHOW_CI_REGION:
                # Add confidence interval shading
                ax.fill_between(x_values, ci_lower, ci_upper,
                            color=colors[model_name], alpha=0.2)
    
    # Customize plot
    ax.set_title('SOTA medical reasoning LLM does not achieve test-time scaling',
                fontsize=14, fontweight='bold', pad=20)
    # ax.set_title('Preliminary Test of Curated Learning to Reason about MedQA',
    #             fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Average Reasoning Tokens', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, labelpad=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Customize legend
    ax.legend(loc='lower right', 
             frameon=True, 
             fancybox=True, 
             shadow=True, 
             fontsize=10)
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    
    # Add subtle spines
    for spine in ax.spines.values():
        spine.set_color('#dddddd')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load results
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        results_json = 'med-s1/results.json'
    results = load_results(results_json)
    
    # Find test time scaling experiments and huatuo-2000
    tts_experiments = {}
    for exp_name, exp_data in results.items():
        # Include both -tts experiments and huatuo-2000
        if '-tts' in exp_name or exp_name == 'huatuo-eval-2000':
            # Skip if no results or eval data
            if not exp_data or not exp_data.get('results') or not exp_data['results'].get('eval'):
                continue
                
            eval_data = exp_data['results']['eval']
            
            # For -tts experiments, check required fields
            if '-tts' in exp_name:
                if not eval_data.get('test_time_scaling') or not eval_data.get('summary') or not eval_data.get('reasoning_tokens'):
                    print(f"Warning: {exp_name} missing required data")
                    continue
            # For huatuo-2000, just check summary
            elif not eval_data.get('summary'):
                print(f"Warning: {exp_name} missing required data")
                continue
                
            print(f"Found results for {exp_name}")
            tts_experiments[exp_name] = exp_data
    
    if not tts_experiments:
        print("No experiments found")
        return
        
    # Plot results
    output_path = 'med-s1/data/test_time_scaling.png'
    plot_test_time_scaling(tts_experiments, output_path)
    print(f"\nPlot saved to: {output_path}")
    
    # Print summary
    print("\nTest Time Scaling Results:")
    print("=" * 40)
    
    for model_name, exp_data in tts_experiments.items():
        eval_data = exp_data['results']['eval']
        # Calculate total examples for experiment
        total_n = 0
        if model_name == 'huatuo-eval-2000':
            accuracy, _, _, n = calculate_weighted_average(eval_data['summary'])
            if accuracy is not None:
                total_n = n
        else:
            for approach in eval_data['summary']:
                _, _, _, n = calculate_weighted_average(eval_data['summary'][approach])
                if n is not None:
                    total_n += n

        # Print experiment name with sample size
        print(f"\n{model_name} (n={total_n}):")
        print("-" * 40)
        
        if model_name == 'huatuo-eval-2000':
            # Print weighted average for huatuo-2000
            accuracy, _, _, n = calculate_weighted_average(eval_data['summary'])
            if accuracy is not None:
                print(f"{'Overall':20} ({0:4.0f} tokens): {accuracy:.1f}%")
            continue
        
        # Sort approaches by token count
        approaches = sorted(
            eval_data['reasoning_tokens'].items(),
            key=lambda x: x[1]
        )
        
        # Find interesting failures
        failures = find_interesting_failures(eval_data, approaches)
        
        for approach, tokens in approaches:
            try:
                metrics = eval_data['summary'].get(approach)
                if not metrics:
                    print(f"{approach:20}: No metrics available")
                    continue
                
                # Calculate weighted accuracy
                accuracy, _, _, n = calculate_weighted_average(metrics)
                if accuracy is None:
                    print(f"{approach:20}: No examples found")
                    continue
                
                print(f"{approach:20} ({tokens:4.0f} tokens): {accuracy:.1f}% (n={n})")
            except Exception as e:
                print(f"{approach:20}: Error calculating metrics - {e}")
                
        # Print most interesting failure
        if failures:
            print(f"\nMost interesting failure (most-token approach failed but ({failures['others_correct']}/{failures['total_approaches']-1} other appraoches succeeded):\n{failures['question']}")

if __name__ == "__main__":
    main()