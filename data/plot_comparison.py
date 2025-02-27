import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Random\nSampling', 'Difficulty-Weighted\nDiversity Sampling']
subspecialties = [105, 141]  # Number of subspecialties
trace_lengths = [421, 612]  # Average trace lengths

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('white')

# Add main title
fig.suptitle('Preliminary Method for Curating Reasoning Trace Data', 
            fontsize=14, 
            fontweight='bold',
            y=1.1)

# Colors
random_color = '#4682B4'  # Steel blue for Random Sampling
weighted_color = '#8B0000'  # Dark red for Difficulty-Weighted
colors = [random_color, weighted_color]
edge_colors = ['#27496D', '#580000']  # Darker versions for edges

# Plot subspecialties (left subplot)
bars1 = ax1.bar(methods, subspecialties, color=colors, edgecolor=edge_colors, linewidth=1.5)
ax1.set_title('Diversity\n(# of Subspecialties)', fontsize=12, pad=15)
ax1.set_ylabel('Count', fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

# Plot trace lengths (right subplot)
bars2 = ax2.bar(methods, trace_lengths, color=colors, edgecolor=edge_colors, linewidth=1.5)
ax2.set_title('Difficulty\n(Avg. Reasoning Trace Length)', fontsize=12, pad=15)
ax2.set_ylabel('Number of Tokens', fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom')

# Adjust layout
plt.tight_layout()

# Add legend at the top
fig.legend([bars1[0], bars1[1]], 
          ['Random Sampling', 'Difficulty-Weighted Diversity'],
          loc='upper center', 
          bbox_to_anchor=(0.5, 1.05),
          ncol=2,
          frameon=False)

# Save figure
plt.savefig('sampling_comparison.png', 
            bbox_inches='tight', 
            dpi=300,
            facecolor='white')