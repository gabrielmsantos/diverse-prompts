import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import os

MAPELITES_PATH = 'FEATURES/MAPELITES/INDIVIDUALS/'

def analyze_prompt_structures(df):
    # Use fixed performance threshold of 0.55
    performance_threshold = 0.7
    
    # Filter for high-performing prompts
    high_performers = df[df['Performance'] >= performance_threshold]
    total = len(high_performers)
    
    # Calculate percentages for different prompt characteristics
    results = {
        'count': total,
        '0-shot': (high_performers['Number of Shots'] == 0).sum() / total * 100,
        'few-shot': ((high_performers['Number of Shots'] > 0) & (high_performers['Number of Shots'] <= 2)).sum() / total * 100,
        'many-shot': (high_performers['Number of Shots'] > 2).sum() / total * 100,
        'has_context': (high_performers['Has Context'] == True).sum() / total * 100,
        'no-cot': (high_performers['Depth of Reasoning'] == 0).sum() / total * 100,
        'cot-1': (high_performers['Depth of Reasoning'] == 1).sum() / total * 100,
        'cot-2+': (high_performers['Depth of Reasoning'] >= 2).sum() / total * 100
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
columns = ['Model', 'Dataset', 'High Performers', 'Has Context %', '0-shot %',
           'Few-shot %', 'Many-shots %', 'No CoT %', 'CoT-1 %', 'CoT 2+ %']

for dataset in datasets:
    for model in models:
        expected_file = os.path.join(MAPELITES_PATH, f'individual_metrics_{model}_{dataset}.parquet')
        if os.path.exists(expected_file):
            df = pd.read_parquet(expected_file)
            results = analyze_prompt_structures(df)
            
            if results:
                row = [
                    model,
                    dataset,
                    results['count'],
                    f"{results['has_context']:.1f}",
                    f"{results['0-shot']:.1f}",
                    f"{results['few-shot']:.1f}",
                    f"{results['many-shot']:.1f}",
                    f"{results['no-cot']:.1f}",
                    f"{results['cot-1']:.1f}",
                    f"{results['cot-2+']:.1f}"
                ]
                results_data.append(row)
        else:
            print(f"File not found: {expected_file}")

# Create DataFrame and save to Excel
results_df = pd.DataFrame(results_data, columns=columns)
results_df.to_excel('NEW_prompt_structures_analysis_70.xlsx', index=False)
print("Results saved to NEW_prompt_structures_analysis_70.xlsx")
