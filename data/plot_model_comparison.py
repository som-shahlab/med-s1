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

def aggregate_confidence_intervals(metrics):
    """
    Aggregate confidence intervals using error propagation theory for unweighted means.
    
    For unweighted mean μ = (x₁ + x₂ + ... + xₙ)/n where each xᵢ has CI [xᵢ ± δxᵢ],
    the standard error of μ is δμ = √(Σ(δxᵢ)²)/n
    """
    # Get standard errors from CIs (CI = mean ± 1.96*SE for 95% CI)
    standard_errors = []
    for data in metrics.values():
        if 'ci_upper' in data and 'ci_lower' in data:
            # Convert CI to standard error
            ci_range = data['ci_upper'] - data['ci_lower']
            se = ci_range / (2 * 1.96)  # 1.96 for 95% CI
            standard_errors.append(se)
    
    if not standard_errors:
        return 0, 0
    
    # Calculate aggregated standard error
    n = len(standard_errors)
    aggregated_se = np.sqrt(sum(se**2 for se in standard_errors)) / n
    
    # Convert back to CI
    if 'Overall' not in metrics:
        return 0, 0
    ci_lower = max(0, metrics['Overall']['accuracy'] - (1.96 * aggregated_se))
    ci_upper = min(1, metrics['Overall']['accuracy'] + (1.96 * aggregated_se))
    
    return float(ci_lower), float(ci_upper)

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
    dataset_accuracies = []
    dataset_ses = []
    total_examples = 0

    for dataset, data in summary.items():
        if 'total_examples' not in data:
            continue

        accuracy = data.get('accuracy', 0)
        ci_lower = data.get('accuracy_ci', [0, 0])[0]
        ci_upper = data.get('accuracy_ci', [0, 0])[1]
        total_examples += data['total_examples']

        metrics[dataset] = {
            'accuracy': accuracy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'total_examples': data['total_examples']
        }

        dataset_accuracies.append(accuracy)
        if ci_lower != 0 or ci_upper != 0:
            se = (ci_upper - ci_lower) / (2 * 1.96)
        else:
            se = 0
        dataset_ses.append(se)

    if not dataset_accuracies:
        return None

    # Calculate simple average across datasets (not weighted by size)
    avg_acc = sum(dataset_accuracies) / len(dataset_accuracies)

    # Calculate aggregated standard error
    n = len(dataset_ses)
    aggregated_se = np.sqrt(sum(se**2 for se in dataset_ses)) / n

    # Calculate confidence interval
    ci_lower = max(0, avg_acc - (1.96 * aggregated_se))
    ci_upper = min(1, avg_acc + (1.96 * aggregated_se))

    # Add overall metrics
    metrics['Overall'] = {
        'accuracy': avg_acc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
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

def plot_reasoning_syntax_impact(results_data, output_path, with_ci=True):
    """Create bar plot comparing model performances with different reasoning syntaxes."""
    models = ['base', 'medqa-1k-random', 'medqa-1k-random-list-extract', 'medqa-1k-random-markdown-extract',
              'medqa-1k-random-decision-tree-extract', 'medqa-1k-random-qa-extract',
              'medqa-1k-random-socratic-extract', 'medqa-1k-random-note-extract', 'medqa-1k-random-step-extract']
    model_names = ['Base', 'HuaTuo', 'List', 'Markdown', 'Decision Tree', 'QA', 'Socratic', 'SOAP', 'Steps']
    
    metrics = {}
    for model in models:
        if model in results_data:
            model_metrics = get_model_metrics(results_data[model])
            if model_metrics:
                metrics[model] = model_metrics['Overall']['accuracy'] * 100
                if with_ci:
                    metrics[model] = (metrics[model],
                                      model_metrics['Overall']['ci_lower'] * 100,
                                      model_metrics['Overall']['ci_upper'] * 100)
    
    x = np.arange(len(model_names))
    width = 0.7
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if with_ci:
        accuracies = [metrics.get(model, (0, 0, 0))[0] for model in models if model in metrics]
        ci_lower = [metrics.get(model, (0, 0, 0))[1] for model in models if model in metrics]
        ci_upper = [metrics.get(model, (0, 0, 0))[2] for model in models if model in metrics]
        yerr_lower = [acc - ci_l for acc, ci_l in zip(accuracies, ci_lower)]
        yerr_upper = [ci_u - acc for acc, ci_u in zip(accuracies, ci_upper)]
        ax.bar(x, accuracies, width, yerr=[yerr_lower, yerr_upper], capsize=5)
    else:
        accuracies = [metrics.get(model, 0) for model in models if model in metrics]
        ax.bar(x, accuracies, width)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Reasoning Syntax')
    ax.set_title('Impact of Reasoning Syntax on Performance')
    ax.set_ylim(35, 65)
    ax.set_xticks(x)
    ax.set_xticklabels([name for model, name in zip(models, model_names) if model in metrics], rotation=45, ha="right")
    # ax.legend() # Removed legend
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sample_efficiency(results_data, output_path, with_ci=True):
    """Create line plot comparing sample efficiency of different reasoning syntaxes."""
    no_reasoning_models = ['medqa-1k-random-no-cot', 'medqa-5k-random-no-cot', 'medqa-10k-random-no-cot']
    huatuo_model = 'medqa-1k-random'
    steps_model = 'medqa-1k-random-step-extract'

    no_reasoning_accuracies = []
    no_reasoning_cis = []
    for model in no_reasoning_models:
        if model in results_data:
            model_metrics = get_model_metrics(results_data[model])
            if model_metrics and 'Overall' in model_metrics:
                accuracy = model_metrics['Overall']['accuracy'] * 100
                if with_ci:
                    ci_lower = model_metrics['Overall']['ci_lower'] * 100
                    ci_upper = model_metrics['Overall']['ci_upper'] * 100
                    no_reasoning_accuracies.append((accuracy, ci_lower, ci_upper))
                else:
                    no_reasoning_accuracies.append(accuracy)

    huatuo_metrics = {}
    if huatuo_model in results_data:
        model_metrics = get_model_metrics(results_data[huatuo_model])
        if model_metrics and 'Overall' in model_metrics:
            accuracy = model_metrics['Overall']['accuracy'] * 100
            if with_ci:
                ci_lower = model_metrics['Overall']['ci_lower'] * 100
                ci_upper = model_metrics['Overall']['ci_upper'] * 100
                huatuo_metrics['accuracy'] = (accuracy, ci_lower, ci_upper)
            else:
                huatuo_metrics['accuracy'] = accuracy

    steps_metrics = {}
    if steps_model in results_data:
        model_metrics = get_model_metrics(results_data[steps_model])
        if model_metrics and 'Overall' in model_metrics:
            accuracy = model_metrics['Overall']['accuracy'] * 100
            if with_ci:
                ci_lower = model_metrics['Overall']['ci_lower'] * 100
                ci_upper = model_metrics['Overall']['ci_upper'] * 100
                steps_metrics['accuracy'] = (accuracy, ci_lower, ci_upper)
            else:
                steps_metrics['accuracy'] = accuracy

    x_no_reasoning = [1000, 5000, 10000]
    x_huatuo = [1000]
    x_steps = [1000]

    fig, ax = plt.subplots(figsize=(12, 6))

    if with_ci:
        # No Reasoning
        accuracies, ci_lowers, ci_uppers = zip(*no_reasoning_accuracies)
        yerr_lower = [acc - ci_l for acc, ci_l in zip(accuracies, ci_lowers)]
        yerr_upper = [ci_u - acc for acc, ci_u in zip(accuracies, ci_uppers)]
        ax.errorbar(x_no_reasoning, accuracies, yerr=[yerr_lower, yerr_upper], capsize=5, fmt='-o', label='No Reasoning')

        # HuaTuo
        if huatuo_metrics:
            accuracy, ci_lower, ci_upper = huatuo_metrics['accuracy']
            yerr_lower = accuracy - ci_lower
            yerr_upper = ci_upper - accuracy
            ax.errorbar(x_huatuo, [accuracy], yerr=[[yerr_lower], [yerr_upper]], capsize=5, fmt='-o', label='HuaTuo Reasoning')

        # Steps
        if steps_metrics:
            accuracy, ci_lower, ci_upper = steps_metrics['accuracy']
            yerr_lower = accuracy - ci_lower
            yerr_upper = ci_upper - accuracy
            ax.errorbar(x_steps, [accuracy], yerr=[[yerr_lower], [yerr_upper]], capsize=5, fmt='-o', label='Step Reasoning')
    else:
        # No Reasoning
        ax.plot(x_no_reasoning, no_reasoning_accuracies, '-o', label='No Reasoning')

        # HuaTuo
        if huatuo_metrics:
            ax.plot(x_huatuo, [huatuo_metrics['accuracy']], '-o', label='HuaTuo Reasoning')

        # Steps
        if steps_metrics:
            ax.plot(x_steps, [steps_metrics['accuracy']], '-o', label='Step Reasoning')

    ax.set_xlabel('Reasoning Trace Dataset Size')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Sample Efficiency of Reasoning Syntax')
    ax.set_xscale('log')
    ax.set_xticks([1000, 5000, 10000])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.legend()
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_perturbation_impact(results_data, output_path, with_ci=True):
    """Create line plot showing impact of different perturbation rates."""
    # Base model (x=0 point)
    base_model = 'medqa-1k-random-step-extract'
    
    # Perturbation types and their corresponding model prefixes
    perturbation_types = {
        'Collapse': 'medqa-1k-random-collapse-',
        'Skip': 'medqa-1k-random-skip-',
        'Shuffle': 'medqa-1k-random-shuffle-',
        'Irrelevant': 'medqa-1k-random-irrelevant-',
        'Incorrect': 'medqa-1k-random-wrong-answer-'
    }
    
    # Perturbation rates
    rates = ['33', '66', '100']
    x_values = [0, 33, 66, 100]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors for different perturbation types
    colors = ['#4B88A2', '#BB4430', '#2D5D7B', '#6B5B95', '#88B04B']  # Added a new color (green)
    
    # Get base accuracy (x=0 point)
    base_accuracy = None
    base_ci_lower = None
    base_ci_upper = None
    if base_model in results_data:
        base_metrics = get_model_metrics(results_data[base_model])
        if base_metrics and 'Overall' in base_metrics:
            base_accuracy = base_metrics['Overall']['accuracy'] * 100
            if with_ci:
                base_ci_lower = base_metrics['Overall']['ci_lower'] * 100
                base_ci_upper = base_metrics['Overall']['ci_upper'] * 100
    
    # Plot each perturbation type
    for (perturbation_name, model_prefix), color in zip(perturbation_types.items(), colors):
        accuracies = [base_accuracy] if base_accuracy is not None else []
        ci_lowers = [base_ci_lower] if base_ci_lower is not None else []
        ci_uppers = [base_ci_upper] if base_ci_upper is not None else []
        
        # Get points for each rate
        for rate in rates:
            model_key = f"{model_prefix}{rate}"
            if model_key in results_data:
                metrics = get_model_metrics(results_data[model_key])
                if metrics and 'Overall' in metrics:
                    accuracies.append(metrics['Overall']['accuracy'] * 100)
                    if with_ci:
                        ci_lowers.append(metrics['Overall']['ci_lower'] * 100)
                        ci_uppers.append(metrics['Overall']['ci_upper'] * 100)

        if len(accuracies) == len(x_values):  # Only plot if we have all points
            if with_ci:
                yerr_lower = [acc - ci_l for acc, ci_l in zip(accuracies, ci_lowers)]
                yerr_upper = [ci_u - acc for acc, ci_u in zip(accuracies, ci_uppers)]
                ax.errorbar(x_values, accuracies,
                          yerr=[yerr_lower, yerr_upper],
                          capsize=5, fmt='-o',
                          label=perturbation_name,
                          color=color,
                          markersize=8,
                          linewidth=2)
            else:
                ax.plot(x_values, accuracies, '-o',
                       label=perturbation_name,
                       color=color,
                       markersize=8,
                       linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Percentage of Steps Perturbed', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Impact of Step Perturbations on Performance', fontsize=14, pad=20)
    
    # Set x-axis ticks
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'{x}%' for x in x_values])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Customize legend
    ax.legend(loc='center left',
             bbox_to_anchor=(1, 0.5),
             frameon=True,
             fancybox=True,
             shadow=True)
    
    # Set background colors
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
    
    plot_reasoning_syntax_impact(results, 'med-s1/data/reasoning_syntax_impact_with_ci.png', with_ci=True)
    plot_reasoning_syntax_impact(results, 'med-s1/data/reasoning_syntax_impact_without_ci.png', with_ci=False)
    
    plot_sample_efficiency(results, 'med-s1/data/sample_efficiency_with_ci.png', with_ci=True)
    plot_sample_efficiency(results, 'med-s1/data/sample_efficiency_without_ci.png', with_ci=False)
    
    plot_perturbation_impact(results, 'med-s1/data/perturbation_impact_with_ci.png', with_ci=True)
    plot_perturbation_impact(results, 'med-s1/data/perturbation_impact_without_ci.png', with_ci=False)

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