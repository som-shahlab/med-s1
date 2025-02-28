import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os

def load_results(json_path):
    """Load results from the results.json file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['experiments']

def plot_test_time_scaling(results_data, output_path):
    """Create line plot comparing performance vs reasoning tokens"""
    # Modern, pleasing color palette
    colors = {
        'med-s1-1k-tuned-tts': '#2D5D7B',  # Deep blue
        'random-1k-tts': '#BB4430',  # Rust red
        'base-tts': '#6B9080',  # Sage green
        'huatuo-tts': '#8E7DBE'  # Purple
    }
    
    # Model keys mapping
    model_keys = {
        'med-s1-1k-tuned-tts': 'Med-S1 (Ours)',
        'random-1k-tts': 'Random Fine-Tune',
        'base-tts': 'Base Model',
        'huatuo-tts': 'HuatuoGPT'
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model's performance
    for model_name, exp_data in results_data.items():
        try:
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
                
                # Calculate weighted accuracy across datasets
                total_examples = 0
                total_correct = 0
                for dataset_metrics in metrics.values():
                    if isinstance(dataset_metrics, dict):  # Skip non-dict values
                        total_examples += dataset_metrics.get('total_examples', 0)
                        total_correct += dataset_metrics.get('num_correct', 0)
                
                if total_examples == 0:
                    print(f"Warning: No examples for {approach} in {model_name}")
                    continue
                
                # Calculate accuracy and confidence intervals
                accuracy = (total_correct / total_examples) * 100
                x_values.append(tokens)
                y_values.append(accuracy)
                
                # Use average CI width from datasets
                ci_widths = []
                for dataset_metrics in metrics.values():
                    if isinstance(dataset_metrics, dict) and 'accuracy_ci' in dataset_metrics:
                        ci = dataset_metrics['accuracy_ci']
                        ci_widths.append((ci[1] - ci[0]) * 100)
                
                ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 10.0
                ci_lower.append(accuracy - ci_width/2)
                ci_upper.append(accuracy + ci_width/2)
                
            except Exception as e:
                print(f"Warning: Failed to process {approach} in {model_name}: {e}")
                continue
        
        # Plot line with error bars
        ax.plot(x_values, y_values, '-o',
                color=colors[model_name],
                label=display_name,
                linewidth=2,
                markersize=8)
        
        # Add confidence interval shading
        ax.fill_between(x_values, ci_lower, ci_upper,
                       color=colors[model_name], alpha=0.2)
    
    # Customize plot
    ax.set_title('Preliminary Test of Curated Learning to Reason about MedQA',
                fontsize=14, fontweight='bold', pad=20)
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
    
    # Find test time scaling experiments
    tts_experiments = {}
    for exp_name, exp_data in results.items():
        # Only look at -tts experiments
        if not exp_name.endswith('-tts'):
            continue
            
        # Skip if no results or eval data
        if not exp_data or not exp_data.get('results') or not exp_data['results'].get('eval'):
            continue
            
        eval_data = exp_data['results']['eval']
        
        # Skip if missing required data
        if not eval_data.get('test_time_scaling') or not eval_data.get('summary') or not eval_data.get('reasoning_tokens'):
            print(f"Warning: {exp_name} missing required data")
            continue
            
        print(f"Found test time scaling results for {exp_name}")
        tts_experiments[exp_name] = exp_data
    
    if not tts_experiments:
        print("No test time scaling results found")
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
        print(f"\n{model_name}:")
        print("-" * 20)
        
        # Sort approaches by token count
        approaches = sorted(
            eval_data['reasoning_tokens'].items(),
            key=lambda x: x[1]
        )
        
        for approach, tokens in approaches:
            try:
                metrics = eval_data['summary'].get(approach)
                if not metrics:
                    print(f"{approach:12}: No metrics available")
                    continue
                
                # Calculate weighted accuracy across datasets
                total_examples = 0
                total_correct = 0
                for dataset_metrics in metrics.values():
                    if isinstance(dataset_metrics, dict):
                        total_examples += dataset_metrics.get('total_examples', 0)
                        total_correct += dataset_metrics.get('num_correct', 0)
                
                if total_examples == 0:
                    print(f"{approach:12}: No examples found")
                    continue
                
                # Calculate accuracy
                accuracy = (total_correct / total_examples) * 100
                print(f"{approach:12} ({tokens:4.0f} tokens): {accuracy:.1f}% ({total_correct}/{total_examples} correct)")
            except Exception as e:
                print(f"{approach:12}: Error calculating metrics - {e}")

if __name__ == "__main__":
    main()