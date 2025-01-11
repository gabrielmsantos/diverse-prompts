import pandas as pd
import numpy as np
import glob
from collections import defaultdict

MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'

def analyze_prompt_structures(df, performance_threshold=0.55):
    """
    Analyze prompt structures of high-performing individuals (performance > threshold)
    """
    # Dictionary to store performance per genotype
    genotype_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'last_row': None})
    
    # Calculate performance for each genotype
    for idx, row in df.iterrows():
        genotype = tuple(row['genotype'])
        genotype_performance[genotype]['correct'] += int(row['model_formatted_output'] == row['correct'])
        genotype_performance[genotype]['total'] += 1
        genotype_performance[genotype]['last_row'] = row
    
    # Filter high-performing individuals
    high_performers = []
    for genotype, stats in genotype_performance.items():
        performance = stats['correct'] / stats['total']
        if performance > performance_threshold:
            high_performers.append(stats['last_row'])
    
    if not high_performers:
        return None
        
    # Calculate percentages
    total = len(high_performers)
    results = {
        'count': total,
        'has_context': sum(1 for ind in high_performers if ind['includes_context']) / total * 100,
        '0-shot': sum(1 for ind in high_performers if ind['num_examples'] == 0) / total * 100,
        'few-shot': sum(1 for ind in high_performers if 1 <= ind['num_examples'] <= 2) / total * 100,
        'many-shots': sum(1 for ind in high_performers if ind['num_examples'] >= 3) / total * 100,
        'no-cot': sum(1 for ind in high_performers if ind['num_steps'] == 0) / total * 100,
        'cot-1': sum(1 for ind in high_performers if ind['num_steps'] == 1) / total * 100,
        'cot-2+': sum(1 for ind in high_performers if ind['num_steps'] >= 2) / total * 100,
        'mean_ttr': np.mean([ind['prompt_type_token_ratio'] for ind in high_performers])
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
           'Few-shot %', 'Many-shots %', 'No CoT %', 'CoT-1 %', 'CoT 2+ %', 'Mean TTR']

for dataset in datasets:
    for model in models:
        file_path = glob.glob(MAPELITES_PATH + f'extracted_features_{model}_{dataset}.parquet')
        if file_path:
            df = pd.read_parquet(file_path[0])
            results = analyze_prompt_structures(df)
            
            if results:
                row = [
                    model,
                    dataset,
                    results['count'],
                    f"{results['has_context']:.1f}",
                    f"{results['0-shot']:.1f}",
                    f"{results['few-shot']:.1f}",
                    f"{results['many-shots']:.1f}",
                    f"{results['no-cot']:.1f}",
                    f"{results['cot-1']:.1f}",
                    f"{results['cot-2+']:.1f}",
                    f"{results['mean_ttr']:.3f}"
                ]
                results_data.append(row)
        else:
            print(f"Skipping {model} on {dataset} due to missing file")

# Create DataFrame and save to Excel
results_df = pd.DataFrame(results_data, columns=columns)
results_df.to_excel('prompt_structures_analysis+models+few-shot.xlsx', index=False)
print("Results saved to prompt_structures_analysis+models+few-shot.xlsx")
