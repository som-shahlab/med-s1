import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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
            'ci_upper': data['accuracy_ci'][1],
            'total_examples': data['total_examples']
        }
    return metrics

def calculate_weighted_metrics(metrics):
    """Calculate weighted average accuracy and confidence intervals"""
    total_weight = sum(data['total_examples'] for data in metrics.values())
    
    # Calculate weighted average accuracy
    weighted_acc = sum(data['accuracy'] * data['total_examples'] 
                      for data in metrics.values()) / total_weight
    
    # Calculate weighted standard error
    # Using delta method to combine confidence intervals
    z_score = norm.ppf(0.975)  # 95% CI
    weighted_variance = sum(
        ((data['ci_upper'] - data['ci_lower']) / (2 * z_score))**2 
        * (data['total_examples'] / total_weight)**2
        for data in metrics.values()
    )
    weighted_se = np.sqrt(weighted_variance)
    
    return {
        'accuracy': weighted_acc,
        'ci_lower': weighted_acc - z_score * weighted_se,
        'ci_upper': weighted_acc + z_score * weighted_se
    }

def plot_scaling(results_data, output_path):
    """Create line plot comparing performance scaling with dataset size"""
    # Modern, pleasing color palette
    colors = {
        'med-s1': '#2D5D7B',  # Deep blue
        'random': '#BB4430'   # Rust red
    }
    
    # Model keys mapping
    model_keys = {
        'med-s1': {
            1000: 'med-s1-1k-tuned',
            5000: 'med-s1-5k'
        },
        'random': {
            1000: 'random-1k',
            5000: 'random-5k'
        }
    }
    
    # Get metrics and calculate weighted averages
    weighted_metrics = {
        method: {
            size: calculate_weighted_metrics(get_model_metrics(results_data[model_keys[method][size]]))
            for size in [1000, 5000]
        }
        for method in ['med-s1', 'random']
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_values = [1000, 5000]
    
    for method in ['med-s1', 'random']:
        # Get values and confidence intervals
        y_values = [weighted_metrics[method][size]['accuracy'] * 100 
                   for size in x_values]
        ci_lower = [weighted_metrics[method][size]['ci_lower'] * 100 
                   for size in x_values]
        ci_upper = [weighted_metrics[method][size]['ci_upper'] * 100 
                   for size in x_values]
        
        # Plot line
        ax.plot(x_values, y_values, '-o', 
               color=colors[method], 
               label=f"{'Med-S1' if method == 'med-s1' else 'Random'} Selection",
               linewidth=2,
               markersize=8)
        
        # Add confidence interval shading
        ax.fill_between(x_values, ci_lower, ci_upper, 
                       color=colors[method], alpha=0.2)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            ax.annotate(f'{y:.1f}%', 
                       (x, y), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       color=colors[method],
                       fontweight='bold')
    
    # Customize plot
    ax.set_title('Sample Efficiency for Learning Reasoning', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Training Samples', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, labelpad=10)
    
    # Set x-axis
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'{x:,}' for x in x_values])
    
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
    # Load and plot results
    results = load_results('med-s1/results.json')
    plot_scaling(results, 'med-s1/data/sample_scaling.png')
    
    # Print weighted accuracies
    print("\nWeighted accuracies across all datasets:")
    
    model_keys = {
        'med-s1': {
            1000: 'med-s1-1k-tuned',
            5000: 'med-s1-5k'
        },
        'random': {
            1000: 'random-1k',
            5000: 'random-5k'
        }
    }
    
    methods = {'med-s1': 'Med-S1', 'random': 'Random'}
    sizes = [1000, 5000]
    
    for method, display_name in methods.items():
        print(f"\n{display_name} Selection:")
        for size in sizes:
            metrics = calculate_weighted_metrics(
                get_model_metrics(results[model_keys[method][size]])
            )
            acc = metrics['accuracy'] * 100
            ci_lower = metrics['ci_lower'] * 100
            ci_upper = metrics['ci_upper'] * 100
            print(f"{size:,} samples: {acc:.1f}% ({ci_lower:.1f}% - {ci_upper:.1f}%)")

if __name__ == "__main__":
    main()