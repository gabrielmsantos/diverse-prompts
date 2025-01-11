import pandas as pd
import numpy as np
import os

MAPELITES_PATH = 'FEATURES/MAPELITES/INDIVIDUALS/'

def analyze_feature_success_rates(df, threshold=0.7):
    # Create binary features
    df['is_0shot'] = df['Number of Shots'] == 0
    df['is_fewshot'] = (df['Number of Shots'] > 0) & (df['Number of Shots'] <= 2)
    df['is_manyshot'] = df['Number of Shots'] > 2
    df['is_nocot'] = df['Depth of Reasoning'] == 0
    df['is_cot1'] = df['Depth of Reasoning'] == 1
    df['is_cot2plus'] = df['Depth of Reasoning'] >= 2

    # Dictionary to store results
    results = {}
    
    # Features to analyze
    features = {
        'Has Context': 'Has Context',
        '0-shot': 'is_0shot',
        'few-shot': 'is_fewshot',
        'many-shot': 'is_manyshot',
        'no-cot': 'is_nocot',
        'cot-1': 'is_cot1',
        'cot-2+': 'is_cot2plus'
    }

    # Calculate success rate for each feature
    for feature_label, feature_name in features.items():
        # Get individuals with this feature
        feature_individuals = df[df[feature_name] == True]
        total_with_feature = len(feature_individuals)
        
        if total_with_feature > 0:
            # Count high performers among those with the feature
            high_perf_with_feature = len(feature_individuals[feature_individuals['Performance'] >= threshold])
            success_rate = (high_perf_with_feature / total_with_feature) * 100
        else:
            success_rate = 0
            
        results[feature_label] = success_rate
        
    return results

# Define datasets and models
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

# Analyze each dataset and store results
results_data = []
columns = ['Model', 'Dataset', 'Has Context %', '0-shot %',
           'Few-shot %', 'Many-shots %', 'No CoT %', 'CoT-1 %', 'CoT 2+ %']

for dataset in datasets:
    for model in models:
        expected_file = os.path.join(MAPELITES_PATH, f'individual_metrics_{model}_{dataset}.parquet')
        if os.path.exists(expected_file):
            df = pd.read_parquet(expected_file)
            results = analyze_feature_success_rates(df)
            
            if results:
                row = [
                    model,
                    dataset,
                    f"{results['Has Context']:.1f}",
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
results_df.to_excel('features_success_rates_analysis_70.xlsx', index=False)
print("Results saved to features_success_rates_analysis_70.xlsx")
