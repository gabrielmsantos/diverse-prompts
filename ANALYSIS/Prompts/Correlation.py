import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Example data: Prompt structure metrics
data = {
    'Has Context %': [75, 68.9, 57.5, 52, 56.3, 63.9, 59.2],
    '0-shot %': [75, 100, 18.7, 94, 26.4, 99.6, 20.4], 
    'Few-shot %': [25, 0, 53.4, 0, 52.9, 0.4, 53.1],
    'CoT-3+ %': [75, 55.6, 40.2, 52, 40.2, 46, 34.7],
    'Mean TTR': [0.639, 0.638, 0.761, 0.714, 0.669, 0.718, 0.875],
}
df = pd.DataFrame(data, index=['formal', 'logical', 'strange', 'strategyqa', 'winowhy', 'known', 'play'])

# Calculate correlation matrix using Spearman correlation
corr = df.corr(method='spearman')

# Create heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title("Correlation Matrix of Prompt Structure Features")
plt.tight_layout()
plt.show()

# Print correlation values and p-values
features = df.columns
for i in range(len(features)):
    for j in range(i+1, len(features)):
        corr, p_value = spearmanr(df[features[i]], df[features[j]])
        print(f"\n{features[i]} vs {features[j]}:")
        print(f"Correlation: {corr:.3f}")
        print(f"P-value: {p_value:.3f}")