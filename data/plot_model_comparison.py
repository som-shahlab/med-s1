import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(json_path):
    """Load results from the results.json file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['experiments']

def get_model_metrics(model_data):
    """Extract accuracies and confidence intervals from model results"""
    if not model_data.get('results', {}).get('eval', {}).get('summary'):
        return None
    
    summary = model_data['results']['eval']['summary']
    metrics = {}
    for dataset, data in summary.items():
        metrics[dataset] = {
            'accuracy': data['accuracy'],
            'ci_lower': data['accuracy_ci'][0],
            'ci_upper': data['accuracy_ci'][1]
        }
    return metrics

def calculate_improvements(tuned_metrics, base_metrics):
    """Calculate percentage improvements over base model"""
    return {
        dataset: ((tuned_metrics[dataset]['accuracy'] - base_metrics[dataset]['accuracy']) * 100)
        for dataset in tuned_metrics.keys()
    }

def plot_comparison(results_data, output_path):
    """Create bar plot comparing model performances with confidence intervals"""
    # Modern, pleasing color palette
    colors = [
        '#4B88A2',  # Muted blue
        '#BB4430',  # Rust red
        '#2D5D7B',  # Deep blue
        '#D68C45'   # Warm orange
    ]
    
    # Models to compare
    models = ['base', 'random-1k', 'med-s1-1k-tuned', 'huatuo']
    model_names = ['Base', 'Random-1k', 'Med-S1-1k', 'HuatuoGPT']
    
    # Get metrics for each model
    metrics = {}
    for model in models:
        if model in results_data:
            metrics[model] = get_model_metrics(results_data[model])
    
    # Datasets to plot
    datasets = list(metrics['base'].keys())
    dataset_display_names = {
        'MedMCQA_validation': 'MedMCQA',
        'MedQA_USLME_test': 'USMLE',
        'PubMedQA_test': 'PubMedQA',
        'MMLU-Pro_Medical_test': 'MMLU-Med',
        'GPQA_Medical_test': 'GPQA'
    }
    
    # Calculate improvements over base model for med-s1-1k-tuned
    improvements = calculate_improvements(metrics['med-s1-1k-tuned'], metrics['base'])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set width of bars and positions of the bars
    bar_width = 0.15
    r = np.arange(len(datasets))
    
    # Create bars with confidence intervals
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        model_metrics = metrics[model]
        
        # Extract values and confidence intervals
        values = [model_metrics[dataset]['accuracy'] * 100 for dataset in datasets]
        ci_lower = [model_metrics[dataset]['ci_lower'] * 100 for dataset in datasets]
        ci_upper = [model_metrics[dataset]['ci_upper'] * 100 for dataset in datasets]
        
        # Calculate yerr for error bars
        yerr_lower = [v - l for v, l in zip(values, ci_lower)]
        yerr_upper = [u - v for v, u in zip(values, ci_upper)]
        yerr = [yerr_lower, yerr_upper]
        
        # Create bars
        bars = ax.bar(r + i * bar_width, values, bar_width, 
                     label=name, color=color, alpha=0.85)
        
        # Add error bars
        ax.errorbar(r + i * bar_width, values, yerr=yerr,
                   fmt='none', color='#333333', capsize=3, 
                   capthick=1, linewidth=1, alpha=0.5)
        
        # Add improvement percentages above med-s1-1k-tuned bars
        if model == 'med-s1-1k-tuned':
            for idx, rect in enumerate(bars):
                height = rect.get_height()
                improvement = improvements[datasets[idx]]
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                       f'+{improvement:.1f}%',
                       ha='center', va='bottom',
                       color='#2D5D7B',
                       fontweight='bold',
                       fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Dataset', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, labelpad=10)
    ax.set_title('MedQA Accuracy By Reasoning LLM Construction Method', 
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
    results = load_results('med-s1/results.json')
    plot_comparison(results, 'med-s1/data/model_comparison.png')
    
    # Print detailed improvements
    print("\nDetailed improvements of Med-S1-1k over base model:")
    base_metrics = get_model_metrics(results['base'])
    tuned_metrics = get_model_metrics(results['med-s1-1k-tuned'])
    improvements = calculate_improvements(tuned_metrics, base_metrics)
    
    dataset_display = {
        'MedMCQA_validation': 'MedMCQA',
        'MedQA_USLME_test': 'USMLE',
        'PubMedQA_test': 'PubMedQA',
        'MMLU-Pro_Medical_test': 'MMLU-Med',
        'GPQA_Medical_test': 'GPQA'
    }
    
    for dataset, improvement in improvements.items():
        display_name = dataset_display[dataset]
        base = base_metrics[dataset]['accuracy'] * 100
        tuned = tuned_metrics[dataset]['accuracy'] * 100
        print(f"{display_name:10}: {base:.1f}% → {tuned:.1f}% ({improvement:+.1f}%)")

if __name__ == "__main__":
    main()