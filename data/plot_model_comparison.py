import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
sys.path.append('/share/pi/nigam/users/calebwin/med-s1')
from eval.scorer import calculate_confidence_interval

def load_results(json_path):
    """Load results from the results.json file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['experiments']

def get_model_metrics(model_data):
    """Extract accuracies and confidence intervals from model results"""
    if not model_data.get('results', {}).get('eval', {}):
        return None

    eval_data = model_data.get('results', {}).get('eval', {})
    
    # Check if we have a summary directly in results.json
    if 'summary' in eval_data:
        summary = eval_data['summary']
    # Otherwise, check if we have a reference to a summary file
    elif 'summary_file' in eval_data:
        try:
            with open(eval_data['summary_file'], 'r') as f:
                summary_data = json.load(f)
                summary = summary_data.get('summary', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading summary file {eval_data['summary_file']}: {e}")
            return None
    else:
        return None
    
    metrics = {}
    for dataset, data in summary.items():
        if 'total_examples' not in data:
            continue
        dataset_metrics = {}
        if 'accuracy' in data:
            dataset_metrics['accuracy'] = data['accuracy']
        else:
            dataset_metrics['accuracy'] = 0  # Default value if accuracy is missing
        if 'accuracy_ci' in data:
            dataset_metrics['ci_lower'] = data['accuracy_ci'][0]
            dataset_metrics['ci_upper'] = data['accuracy_ci'][1]
        else:
            dataset_metrics['ci_lower'] = 0
            dataset_metrics['ci_upper'] = 0
        dataset_metrics['total_examples'] = data['total_examples']  # Add total examples for weighting
        metrics[dataset] = dataset_metrics
    
    if not metrics:
        return None
    
    # Calculate simple average across datasets (not weighted by size)
    avg_acc = sum(data['accuracy'] for data in metrics.values()) / len(metrics)
    
    # Calculate weighted confidence intervals based on accuracy
    # Create a combined binary array for bootstrap
    all_results = []
    for data in metrics.values():
        # Create binary array for this dataset
        accuracy = data['accuracy']
        total = data['total_examples']
        correct_count = int(round(accuracy * total))
        dataset_results = np.zeros(total)
        dataset_results[:correct_count] = 1
        np.random.shuffle(dataset_results)
        all_results.append(dataset_results)
    
    # Combine all results
    combined_results = np.concatenate(all_results)
    
    # Calculate CI on combined results
    lower, upper = calculate_confidence_interval(combined_results)
    
    # Add overall metrics
    total_examples = sum(data['total_examples'] for data in metrics.values())
    metrics['Overall'] = {
        'accuracy': avg_acc,
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'total_examples': total_examples
    }
    
    return metrics

def calculate_improvements(model_metrics, base_metrics):
    """Calculate percentage improvements over base model"""
    if not base_metrics:
        print("Warning: Base metrics not available for improvement calculation")
        return {}
    
    return {
        dataset: ((model_metrics[dataset]['accuracy'] - base_metrics[dataset]['accuracy']) * 100)
        for dataset in model_metrics.keys() if dataset in base_metrics
    }

def plot_comparison(results_data, output_path):
    """Create bar plot comparing model performances with confidence intervals"""
    # Modern, pleasing color palette - extended for more models
    colors = [
        '#4B88A2',  # Muted blue (base)
        '#BB4430',  # Rust red
        '#2D5D7B',  # Deep blue
        '#6B5B95',  # Purple
        '#88B04B',  # Green
        '#D68C45',  # Warm orange
        '#F7CAC9',  # Pink
        '#92A8D1',  # Light blue
        '#955251',  # Burgundy
        '#B565A7',  # Magenta
        '#009B77',  # Jungle green
        '#DD4124',  # Vermilion
        '#D65076',  # Raspberry
        '#45B8AC',  # Turquoise
    ]
    
    # Find all models with eval results
    models = []
    model_names = []
    
    # Ensure 'base' is first if it exists
    if 'base' in results_data and results_data['base'].get('results', {}).get('eval', {}):
        models.append('base')
        model_names.append('Base')
    
    # Add all other models with eval results
    for model_key in results_data:
        if model_key != 'base' and results_data[model_key].get('results', {}).get('eval', {}):
            models.append(model_key)
            # Create a display name by capitalizing and replacing hyphens with spaces
            display_name = model_key.replace('-', ' ').title()
            model_names.append(display_name)
    
    print(f"Found {len(models)} models with evaluation results: {models}")
    
    if not models:
        print("No models with evaluation results found!")
        return
    
    # Get metrics for each model
    metrics = {}
    for model in models:
        model_metrics = get_model_metrics(results_data[model])
        if model_metrics:
            metrics[model] = model_metrics
    
    # Find common datasets across all models
    all_datasets = set()
    for model_metrics in metrics.values():
        all_datasets.update(model_metrics.keys())
    
    # Ensure we include these important datasets if they exist in any model
    important_datasets = ['MedMCQA_validation', 'MedQA_USLME_test', 'PubMedQA_test',
                         'MMLU-Pro_Medical_test', 'GPQA_Medical_test', 'Overall']
    
    # Filter to datasets that exist in all models, prioritizing important ones
    datasets = [d for d in important_datasets if d in all_datasets]
    
    # Add any other datasets that weren't in our priority list
    other_datasets = sorted([d for d in all_datasets if d not in important_datasets and d != 'Overall'])
    datasets.extend(other_datasets)
    
    # Ensure Overall is last
    if 'Overall' in datasets and datasets[-1] != 'Overall':
        datasets.remove('Overall')
        datasets.append('Overall')
    
    # Create display names for all datasets
    dataset_display_names = {
        'MedMCQA_validation': 'MedMCQA',
        'MedQA_USLME_test': 'USMLE',
        'PubMedQA_test': 'PubMedQA',
        'MMLU-Pro_Medical_test': 'MMLU-Med',
        'GPQA_Medical_test': 'GPQA',
        'Overall': 'Overall'
    }
    
    # Add any missing datasets to the display names dictionary
    for dataset in datasets:
        if dataset not in dataset_display_names:
            # Create a display name by removing underscores and test/validation suffixes
            display_name = dataset.replace('_', ' ')
            display_name = re.sub(r'_(test|validation)$', '', display_name)
            dataset_display_names[dataset] = display_name
    
    # Get base metrics for improvement calculation
    base_metrics = metrics.get('base', None)
    
    # Calculate improvements over base model for all models (if base exists)
    improvements = {}
    if base_metrics:
        for model in models:
            if model != 'base' and model in metrics:
                improvements[model] = calculate_improvements(metrics[model], base_metrics)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(max(16, 10 + len(models) * 0.5), 8))  # Adjust width based on number of models
    
    # Set width of bars and positions of the bars
    bar_width = min(0.15, 0.8 / len(models))  # Adjust bar width based on number of models
    r = np.arange(len(datasets))
    
    # Create bars with confidence intervals
    for i, (model, name, color) in enumerate(zip(models, model_names, colors[:len(models)])):
        if model not in metrics:
            continue
            
        model_metrics = metrics[model]
        
        # Extract values and confidence intervals, handling missing datasets
        values = []
        ci_lower = []
        ci_upper = []
        
        for dataset in datasets:
            if dataset in model_metrics:
                values.append(model_metrics[dataset]['accuracy'] * 100)
                ci_lower.append(model_metrics[dataset]['ci_lower'] * 100)
                ci_upper.append(model_metrics[dataset]['ci_upper'] * 100)
            else:
                # Use NaN for missing datasets
                values.append(float('nan'))
                ci_lower.append(float('nan'))
                ci_upper.append(float('nan'))
        
        # Calculate yerr for error bars
        yerr_lower = [max(0, v - l) if not np.isnan(v) else 0 for v, l in zip(values, ci_lower)]
        yerr_upper = [max(0, u - v) if not np.isnan(v) else 0 for v, u in zip(values, ci_upper)]
        yerr = [yerr_lower, yerr_upper]
        
        # Create bars
        bars = ax.bar(r + i * bar_width, values, bar_width,
                     label=name, color=color, alpha=0.85)
        
        # Add error bars
        ax.errorbar(r + i * bar_width, values, yerr=yerr,
                   fmt='none', color='#333333', capsize=3,
                   capthick=1, linewidth=1, alpha=0.5)
        
        # Add improvement percentages above non-base model bars
        if model != 'base' and model in improvements:
            model_improvements = improvements[model]
            for idx, rect in enumerate(bars):
                dataset = datasets[idx]
                if dataset in model_improvements:
                    height = rect.get_height()
                    improvement = model_improvements[dataset]
                    if not np.isnan(height) and not np.isnan(improvement):
                        ax.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                               f'+{improvement:.1f}%',
                               ha='center', va='bottom',
                               color='#2D5D7B',
                               fontweight='bold',
                               fontsize=10)
    
    # Add vertical line before Overall
    ax.axvline(x=len(datasets)-1.5, color='gray', linestyle='--', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel('Dataset', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, labelpad=10)
    # ax.set_title('MedQA Accuracy By Reasoning LLM Construction Method', 
    #             fontsize=14, pad=20, fontweight='bold')
    ax.set_title('Existing approach to reasoning data curation underperforms on medical reasoning traces', 
                fontsize=14, pad=20, fontweight='bold')
    
    # Customize ticks
    plt.xticks(r + bar_width * 1.5, 
               [dataset_display_names[d] for d in datasets], 
               rotation=45, 
               ha='right')
    
    # Customize legend
    plt.legend(loc='upper right', 
              frameon=True, 
              fancybox=True, 
              shadow=True, 
              fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    
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
    # Load and plot results
    results_path = os.environ.get('RESULTS_JSON', 'med-s1/results.json')
    print(f"Loading results from {results_path}")
    results = load_results(results_path)
    plot_comparison(results, 'med-s1/data/model_comparison.png')
    
    # Find models with evaluation results
    models_with_eval = []
    for model_key, model_data in results.items():
        if model_data.get('results', {}).get('eval', {}):
            models_with_eval.append(model_key)
    
    # Get base metrics if available
    base_metrics = None
    if 'base' in results and results['base'].get('results', {}).get('eval', {}):
        base_metrics = get_model_metrics(results['base'])
    
    # Print detailed improvements for all models compared to base
    if base_metrics:
        print("\nDetailed improvements over base model:")
        
        dataset_display = {
            'MedMCQA_validation': 'MedMCQA',
            'MedQA_USLME_test': 'USMLE',
            'PubMedQA_test': 'PubMedQA',
            'MMLU-Pro_Medical_test': 'MMLU-Med',
            'GPQA_Medical_test': 'GPQA',
            'Overall': 'Overall (Simple Average)'
        }
        
        for model in models_with_eval:
            if model == 'base':
                continue
                
            model_metrics = get_model_metrics(results[model])
            if not model_metrics:
                continue
                
            improvements = calculate_improvements(model_metrics, base_metrics)
            
            print(f"\n{model} improvements:")
            for dataset, improvement in improvements.items():
                display_name = dataset_display.get(dataset, dataset)
                base = base_metrics[dataset]['accuracy'] * 100
                model_acc = model_metrics[dataset]['accuracy'] * 100
                print(f"{display_name:15}: {base:.1f}% → {model_acc:.1f}% ({improvement:+.1f}%)")

    # Print average accuracy and CI for all models
    print("\nAverage accuracy and CI for all models:")
    for model in sorted(models_with_eval):
        metrics = get_model_metrics(results[model])
        if metrics and 'Overall' in metrics:
            overall_metrics = metrics['Overall']
            if 'accuracy' in overall_metrics:
                accuracy = overall_metrics['accuracy'] * 100
                ci_lower = overall_metrics['ci_lower'] * 100
                ci_upper = overall_metrics['ci_upper'] * 100
                print(f"{model:20}: Accuracy = {accuracy:.1f}%, CI = [{ci_lower:.1f}%, {ci_upper:.1f}%]")
            else:
                print(f"{model:20}: Overall accuracy not available")
if __name__ == "__main__":
    main()