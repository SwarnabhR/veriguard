import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('./results/evaluation_results.csv')

# Extract confusion matrix values
tp = df['True Positives'].values[0]
tn = df['True Negatives'].values[0]
fp = df['False Positives'].values[0]
fn = df['False Negatives'].values[0]

# Create confusion matrix array
confusion_matrix = np.array([[tn, fp],
                             [fn, tp]])

# Calculate total for percentages
total = tp + tn + fp + fn

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Count'}, ax=ax, 
            linewidths=2, linecolor='white',
            square=True, vmin=0)

# Add percentage annotations
for i in range(2):
    for j in range(2):
        percentage = (confusion_matrix[i, j] / total) * 100
        text = ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                      ha="center", va="center", color="gray", fontsize=11)

# Set labels
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

# Set tick labels
ax.set_xticklabels(['Negative', 'Positive'], fontsize=12)
ax.set_yticklabels(['Negative', 'Positive'], fontsize=12, rotation=0)

# Add metrics text box
metrics_text = f"""Model Performance Metrics:
Accuracy:    {df['Test Accuracy'].values[0]:.4f} ({df['Test Accuracy'].values[0]*100:.2f}%)
Precision:   {df['Precision'].values[0]:.4f} ({df['Precision'].values[0]*100:.2f}%)
Recall:      {df['Recall'].values[0]:.4f} ({df['Recall'].values[0]*100:.2f}%)
F1-Score:    {df['F1-Score'].values[0]:.4f} ({df['F1-Score'].values[0]*100:.2f}%)
Specificity: {df['Specificity'].values[0]:.4f} ({df['Specificity'].values[0]*100:.2f}%)
Test Loss:   {df['Test Loss'].values[0]:.4f}"""

plt.text(0.02, 0.98, metrics_text, 
         transform=fig.transFigure, 
         fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         family='monospace')

# Adjust layout to make room for text
plt.tight_layout()
plt.subplots_adjust(left=0.35)

# Save the figure
plt.savefig('./results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to './results/confusion_matrix.png'")

# Display the plot
plt.show()

# Print summary
print("\nConfusion Matrix Summary:")
print(f"True Negatives:  {tn:,} ({(tn/total)*100:.2f}%)")
print(f"False Positives: {fp:,} ({(fp/total)*100:.2f}%)")
print(f"False Negatives: {fn:,} ({(fn/total)*100:.2f}%)")
print(f"True Positives:  {tp:,} ({(tp/total)*100:.2f}%)")
print(f"\nTotal Samples: {total:,}")