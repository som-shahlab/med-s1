import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = [
    "base", "medqa-1k-random", "medqa-1k-random-1-sentence-extract", "medqa-1k-random-step-extract",
    "medqa-1k-random-no-cot", "medqa-5k-random-no-cot", "medqa-10k-random-no-cot"
]

tokens = np.array([0, 555355, 207328, 652554, 135694, 668755, 1330109])  # Token counts
accuracy = np.array([58.4, 59.8, 60.6, 62.4, 58.2, 58.5, 58.6])  # Accuracy percentages

# Select groups
trace_quality = [0, 555355, 207328, 652554]  # Base, 1k-random, 1-sentence, step
trace_acc = [58.4, 59.8, 60.6, 62.4]

dataset_size = [0, 135694, 668755, 1330109]  # Base, -no-cot (1k, 5k, 10k)
size_acc = [58.4, 58.2, 58.5, 58.6]

# Create the figure
# Create the figure
plt.figure(figsize=(8, 6))

# Scaling lines (plotted first so dots appear above them)
plt.plot([0, 207328, 652554], [58.4, 60.6, 62.4], linestyle="dashed", color="red", label="Scaling Trace Quality", zorder=1)
plt.plot(dataset_size, size_acc, linestyle="dashed", color="gray", alpha=0.6, label="* Scaling Dataset Size", zorder=1)

# Scatter plot for key data points (zorder=2 to ensure they appear above lines)
plt.scatter(0, 58.4, color="black", label="Base (Llama-3.1-8b)", marker='o', zorder=2)
plt.scatter(207328, 60.6, color="purple", label="1-sentence CoT", marker='o', zorder=2)
plt.scatter(555355, 59.8, color="blue", label="Standard CoT", marker='o', zorder=2)
plt.scatter(652554, 62.4, color="green", label="Step-Extracted CoT", marker='o', zorder=2)

# Gray out base and -no-cot points (zorder=2 for visibility)
plt.scatter([0, 135694, 668755, 1330109], [58.4, 58.2, 58.5, 58.6], color="gray", alpha=0.6, marker='o', zorder=2)

# Labels and title
plt.xlabel("# Tokens in Training Data")
plt.ylabel("Accuracy (%)")
plt.title("Scaling MedQA Performance with Reasoning Trace Quality")

# Legend without title
plt.legend(loc="lower right")

# Save plot
plt.savefig("med-s1/data/scaling.png", dpi=300, bbox_inches='tight')
plt.close()
