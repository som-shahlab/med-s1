import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_percentage(value):
    """Parse percentage string with optional confidence interval."""
    if value == 'N/A':
        return None, None, None
    
    if '±' in value:
        # Has confidence interval
        base = float(value.split('±')[0].strip().rstrip('%'))
        error = float(value.split('±')[1].strip().rstrip('%'))
        return base, base - error, base + error
    else:
        # Just percentage
        base = float(value.rstrip('%'))
        return base, None, None

# Read data
df = pd.read_csv('med-s1/data/model_comparison.csv')

# Select models and metrics we want
models = ['base-qwen', 'huatuo']
metrics = ['MedQA', 'MedDS', 'MedDS_NOTA', 'NEJMCRMC']
model_display_names = ['Qwen2.5-7B', 'HuatuoGPT-o1-8B']

# Extract data
data = []
errors = []
for model in models:
    model_data = []
    model_errors = []
    row = df[df['Model'] == model].iloc[0]
    for metric in metrics:
        value, ci_lower, ci_upper = parse_percentage(row[metric])
        model_data.append(value if value is not None else 0)
        if ci_lower is not None and ci_upper is not None:
            model_errors.append([value - ci_lower, ci_upper - value])
        else:
            model_errors.append([0, 0])
    data.append(model_data)
    errors.append(model_errors)

# Convert to numpy arrays
data = np.array(data)
errors = np.array(errors)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Set width of bars and positions
bar_width = 0.15
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]

# Create bars
colors = ['#4B88A2', '#BB4430']  # Blue for Qwen, Red for HuaTuo
for i, (model_data, model_errors, color) in enumerate(zip(data, errors, colors)):
    positions = [x + i * bar_width for x in r1]
    ax.bar(positions, model_data, bar_width, label=model_display_names[i], color=color, alpha=0.85)
    # Add error bars
    ax.errorbar(positions, model_data, yerr=model_errors.T, fmt='none', color='black', capsize=5)

# Customize the plot
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks([r + bar_width/2 for r in r1])
ax.set_xticklabels(metrics)
ax.legend()

# Add grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Set background colors
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#ffffff')

# Add subtle spines
for spine in ax.spines.values():
    spine.set_color('#dddddd')

# Adjust layout and save
plt.tight_layout()
plt.savefig('med-s1/data/model_comparison_subset.png', dpi=300, bbox_inches='tight')
plt.close()