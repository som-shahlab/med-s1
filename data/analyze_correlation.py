import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load the data
data_dir = "/share/pi/nigam/users/calebwin/hf_cache/med-s1k/plumbing_test_001_20250219_145607"
df = pd.read_parquet(f"{data_dir}/med_s1k_filtered.parquet")

# Calculate CoT lengths
df['cot_length_chars'] = df['Complex_CoT'].str.len()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Box plot of CoT lengths by correctness
sns.boxplot(x='base_model_correct', y='cot_length_chars', data=df, ax=ax1)
ax1.set_title('Distribution of CoT Lengths by Base Model Correctness')
ax1.set_xlabel('Base Model Correct')
ax1.set_ylabel('CoT Length (characters)')

# Calculate mean lengths for correct/incorrect
mean_correct = df[df['base_model_correct']]['cot_length_chars'].mean()
mean_incorrect = df[~df['base_model_correct']]['cot_length_chars'].mean()

# Add mean values as text
ax1.text(0, ax1.get_ylim()[1], f'Mean: {mean_incorrect:.0f}', 
         horizontalalignment='center', verticalalignment='bottom')
ax1.text(1, ax1.get_ylim()[1], f'Mean: {mean_correct:.0f}', 
         horizontalalignment='center', verticalalignment='bottom')

# 2. Histogram of CoT lengths by correctness
sns.histplot(data=df, x='cot_length_chars', hue='base_model_correct', 
             multiple="layer", alpha=0.6, ax=ax2)
ax2.set_title('Distribution of CoT Lengths by Base Model Correctness')
ax2.set_xlabel('CoT Length (characters)')
ax2.set_ylabel('Count')

# Perform statistical test
t_stat, p_value = stats.ttest_ind(
    df[df['base_model_correct']]['cot_length_chars'],
    df[~df['base_model_correct']]['cot_length_chars']
)

# Add statistical test results
plt.figtext(0.5, -0.05, 
            f't-statistic: {t_stat:.2f}\np-value: {p_value:.4f}',
            ha='center', va='center')

# Calculate correlation coefficient
correlation = df['cot_length_chars'].corr(df['base_model_correct'])
plt.figtext(0.5, -0.1,
            f'Correlation coefficient: {correlation:.3f}',
            ha='center', va='center')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Save the plot
plt.savefig('cot_length_analysis.png', bbox_inches='tight', dpi=300)

# Print summary statistics
print("\nSummary Statistics:")
print("\nMean CoT Length by Correctness:")
print(df.groupby('base_model_correct')['cot_length_chars'].mean())

print("\nMedian CoT Length by Correctness:")
print(df.groupby('base_model_correct')['cot_length_chars'].median())

print("\nStandard Deviation of CoT Length by Correctness:")
print(df.groupby('base_model_correct')['cot_length_chars'].std())

# Calculate percentage of correct responses for different CoT length quartiles
quartiles = pd.qcut(df['cot_length_chars'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
accuracy_by_quartile = df.groupby(quartiles)['base_model_correct'].mean()

print("\nAccuracy by CoT Length Quartile:")
print(accuracy_by_quartile)