import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
data = {
    'Dataset': ['formal_fallacies', 'logical-deduction', 'strange_stories', 'strategyqa', 'winowhy', 'known_unknowns', 'play_dialog'],
    'Has Context %': [46.2, 62.5, 57.9, 60, 54, 63.6, 57.4],
    '0-shot %': [65.4, 100, 21.6, 90, 24.9, 95.1, 21.8],
    'Few-shot %': [26.9, 0, 51.5, 6, 50.7, 3.1, 50.8],
    'CoT-0 %': [30.8, 33.3, 35.3, 32, 39.2, 35.3, 31.2],
    'CoT-1+ %': [69.2, 66.7, 64.7, 68, 60.8, 64.7, 68.8],
    'Mean TTR': [0.661, 0.626, 0.76, 0.702, 0.67, 0.711, 0.884]
}

df = pd.DataFrame(data)

# Perform binomial test for percentage features
features_to_test = ['Has Context %', '0-shot %', 'Few-shot %', 'CoT-0 %', 'CoT-1+ %']
all_p_values = {}
all_significant = {}

for feature in features_to_test:
    # Perform binomial test for each value
    p_values = [stats.binomtest(int(x * 100 / 100), 100, p=0.5).pvalue for x in df[feature]]
    significant = [p < 0.05 for p in p_values]
    
    all_p_values[feature] = p_values
    all_significant[feature] = significant
    
    # Print results for each feature
    print(f"\nStatistical Analysis for {feature}:")
    for dataset, percentage, p_value, is_sig in zip(df['Dataset'], df[feature], p_values, significant):
        print(f"{dataset}: {percentage}% (p={p_value:.3f}) {'¹' if is_sig else ''}")

# For Mean TTR, perform one-sample t-test against the mean
ttr_mean = df['Mean TTR'].mean()
t_stat, ttr_p_value = stats.ttest_1samp(df['Mean TTR'], ttr_mean)
ttr_significant = ttr_p_value < 0.05

print(f"\nStatistical Analysis for Mean TTR:")
for dataset, value in zip(df['Dataset'], df['Mean TTR']):
    print(f"{dataset}: {value:.3f} (t-test p={ttr_p_value:.3f}) {'¹' if ttr_significant else ''}")

# Modified heatmap with significance markers for all features
plt.figure(figsize=(12, 7))
df_viz = df.copy()
df_viz.set_index('Dataset', inplace=True)

# Create annotation matrix with significance markers
annot_matrix = df_viz.copy()
for feature in features_to_test:
    annot_matrix[feature] = [f"{val:.1f}{'¹' if sig else ''}" 
                            for val, sig in zip(df_viz[feature], all_significant[feature])]
annot_matrix['Mean TTR'] = [f"{val:.3f}{'¹' if ttr_significant else ''}" 
                           for val in df_viz['Mean TTR']]

sns.heatmap(df_viz, annot=annot_matrix, cmap="YlGnBu", fmt='')
plt.title("Prompt Features Across Datasets\n¹ p < 0.05")
plt.tight_layout()
plt.show()