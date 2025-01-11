import pandas as pd
import numpy as np
import os
from scipy import stats

MAPELITES_PATH = 'FEATURES/MAPELITES/INDIVIDUALS/'

def compute_z_test(df, feature_name, threshold=0.55):
    # Get overall population (performance >= 0)
    total_pop = df[df['Performance'] >= 0]
    n1 = len(total_pop)
    p1 = (total_pop[feature_name] == True).sum() / n1
    
    # Get high performers (performance >= threshold) 
    high_perf = df[df['Performance'] >= threshold]
    n2 = len(high_perf)
    p2 = (high_perf[feature_name] == True).sum() / n2
    
    # Pooled proportion
    p_pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Z-score
    z_score = (p2 - p1) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Log statistics
    log_str = f"""
Feature: {feature_name}
Total Population (n1): {n1}
High Performers (n2): {n2}
Population proportion (p1): {p1:.4f}
High performers proportion (p2): {p2:.4f}
Pooled proportion: {p_pooled:.4f}
Standard Error: {se:.4f}
Z-score: {z_score:.4f}
P-value: {p_value:.4f}
"""
    return p_value, log_str

# Define datasets and models (same as BestStructures.py)
datasets = [
    'formal_fallacies_syllogisms_negation',
    'logical-deduction-3', 
    'strange_stories_boolean',
    'strategyqa',
    'winowhy',
    'known_unknowns',
    'play_dialog_same_or_different'
]

models = [
    'starling-lm-7b-alpha-bii',
    'llama-3-1-8b-instruct-lvt', 
    'phi-3-5-mini-instruct-ivr',
    'qwen2-5-7b-instruct-mln'
]

# Store results
results_data = []
all_logs = ""

for dataset in datasets:
    for model in models:
        file_path = os.path.join(MAPELITES_PATH, f'individual_metrics_{model}_{dataset}.parquet')
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            
            # Create binary features for testing
            df['is_0shot'] = df['Number of Shots'] == 0
            df['is_fewshot'] = (df['Number of Shots'] > 0) & (df['Number of Shots'] <= 2)
            df['is_manyshot'] = df['Number of Shots'] > 2
            df['is_nocot'] = df['Depth of Reasoning'] == 0
            df['is_cot1'] = df['Depth of Reasoning'] == 1
            df['is_cot2plus'] = df['Depth of Reasoning'] >= 2
            
            # Features to test
            features = {
                'Has Context': 'Has Context',
                '0-shot': 'is_0shot',
                'few-shot': 'is_fewshot', 
                'many-shot': 'is_manyshot',
                'No-CoT': 'is_nocot',
                'CoT-1': 'is_cot1',
                'CoT-2+': 'is_cot2plus'
            }
            
            # Compute p-values and collect logs
            p_values = {}
            log_block = f"\n{'='*50}\nModel: {model}\nDataset: {dataset}\n{'='*50}\n"
            
            for feature_label, feature_name in features.items():
                p_value, log_str = compute_z_test(df, feature_name)
                p_values[feature_label] = p_value
                log_block += log_str
            
            all_logs += log_block
            
            # Add row to results
            row = [model, dataset] + [p_values[f] for f in features.keys()]
            results_data.append(row)
        else:
            print(f"File not found: {file_path}")

# Save logs
with open('ztest_computation_logs.txt', 'w') as f:
    f.write(all_logs)
print("Computation logs saved to ztest_computation_logs.txt")

# Create and save results DataFrame
columns = ['Model', 'Dataset'] + list(features.keys())
results_df = pd.DataFrame(results_data, columns=columns)
results_df.to_excel('ztest_results.xlsx', index=False)
print("Results saved to ztest_results.xlsx")
