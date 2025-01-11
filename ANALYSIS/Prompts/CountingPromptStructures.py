import pandas as pd
import numpy as np
import glob
from collections import defaultdict

MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'

def count_prompt_structures(df):
    """
    Count different prompt structures found in the dataset
    """
    total = len(df)
    
    results = {
        'count': total,
        'has_context': sum(1 for _, row in df.iterrows() if row['includes_context']) / total * 100,
        '0-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 0) / total * 100,
        '1-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 1) / total * 100,
        '2-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 2) / total * 100,
        '3-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 3) / total * 100,
        '4-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 4) / total * 100,
        '5-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 5) / total * 100,
        '6-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 6) / total * 100,
        '7-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 7) / total * 100,
        '8-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] == 8) / total * 100,
        '9+-shot': sum(1 for _, row in df.iterrows() if row['num_examples'] >= 9) / total * 100,
        'cot-0': sum(1 for _, row in df.iterrows() if row['num_steps'] == 0) / total * 100,
        'cot-1': sum(1 for _, row in df.iterrows() if row['num_steps'] == 1) / total * 100,
        'cot-2': sum(1 for _, row in df.iterrows() if row['num_steps'] == 2) / total * 100,
        'cot-3': sum(1 for _, row in df.iterrows() if row['num_steps'] == 3) / total * 100,
        'cot-4': sum(1 for _, row in df.iterrows() if row['num_steps'] == 4) / total * 100,
        'cot-5': sum(1 for _, row in df.iterrows() if row['num_steps'] == 5) / total * 100,
        'cot-6': sum(1 for _, row in df.iterrows() if row['num_steps'] == 6) / total * 100,
        'cot-7': sum(1 for _, row in df.iterrows() if row['num_steps'] == 7) / total * 100,
        'cot-8': sum(1 for _, row in df.iterrows() if row['num_steps'] == 8) / total * 100,
        'cot-9+': sum(1 for _, row in df.iterrows() if row['num_steps'] >= 9) / total * 100,
        'mean_ttr': np.mean([row['prompt_type_token_ratio'] for _, row in df.iterrows()])
    }
    
    return results

# Define datasets to analyze
datasets = [
    'formal_fallacies_syllogisms_negation',
    'logical-deduction-3', 
    'strange_stories_boolean',
    'strategyqa',
    'winowhy',
    'known_unknowns',
    'play_dialog_same_or_different'
]

# Define models to analyze
models = [
    'starling-lm-7b-alpha-bii',
    'llama-3-1-8b-instruct-lvt', 
    'phi-3-5-mini-instruct-ivr',
    'qwen2-5-7b-instruct-mln'
]

# Analyze each dataset and store results
results_data = []
columns = ['Model', 'Dataset', 'Total Structures', 'Has Context %', '0-shot %', '1-shot %', 
           '2-shot %', '3-shot %', '4-shot %', '5-shot %', '6-shot %', '7-shot %',
           '8-shot %', '9+-shot %', 'CoT-0 %', 'CoT-1 %', 'CoT-2 %', 'CoT-3 %', 
           'CoT-4 %', 'CoT-5 %', 'CoT-6 %', 'CoT-7 %', 'CoT-8 %', 'CoT-9+ %', 'Mean TTR']

for dataset in datasets:
    for model in models:
        file_path = glob.glob(MAPELITES_PATH + f'extracted_features_{model}_{dataset}.parquet')
        if file_path:
            df = pd.read_parquet(file_path[0])
            results = count_prompt_structures(df)
            
            row = [
                model,
                dataset,
                results['count'],
                f"{results['has_context']:.1f}",
                f"{results['0-shot']:.1f}",
                f"{results['1-shot']:.1f}",
                f"{results['2-shot']:.1f}",
                f"{results['3-shot']:.1f}",
                f"{results['4-shot']:.1f}",
                f"{results['5-shot']:.1f}",
                f"{results['6-shot']:.1f}",
                f"{results['7-shot']:.1f}",
                f"{results['8-shot']:.1f}",
                f"{results['9+-shot']:.1f}",
                f"{results['cot-0']:.1f}",
                f"{results['cot-1']:.1f}",
                f"{results['cot-2']:.1f}",
                f"{results['cot-3']:.1f}",
                f"{results['cot-4']:.1f}",
                f"{results['cot-5']:.1f}",
                f"{results['cot-6']:.1f}",
                f"{results['cot-7']:.1f}",
                f"{results['cot-8']:.1f}",
                f"{results['cot-9+']:.1f}",
                f"{results['mean_ttr']:.3f}"
            ]
            results_data.append(row)
        else:
            print(f"Skipping {model} on {dataset} due to missing file")

# Create DataFrame and save to Excel
results_df = pd.DataFrame(results_data, columns=columns)
results_df.to_excel('prompt_structures_counts.xlsx', index=False)
print("Results saved to prompt_structures_counts.xlsx")
