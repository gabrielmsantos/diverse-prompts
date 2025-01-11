import pandas as pd
import numpy as np
import glob
from scipy import stats

# Load all parquet files from directory
PARQUET_PATH = 'FEATURES/PARQUET/'
parquet_files = glob.glob(PARQUET_PATH + '*.parquet')

def analyze_shot_performance(df):
    """
    Analyze model performance across different numbers of shots/examples
    Returns a dictionary with the analysis results
    """
    results = {}
    
    # Calculate 0-shot stats
    zero_shot_mask = df['num_examples'] == 0
    results['0-shot'] = {
        'count': len(df[zero_shot_mask]),
        'accuracy': df[zero_shot_mask]['performance'].mean()
    }
    
    # Calculate stats for different shot ranges
    shot_ranges = {
        '1-shot': df[df['num_examples'] == 1],
        '2-shots': df[df['num_examples'] == 2],
        '3+shots': df[df['num_examples'] > 2]
    }
    
    for range_name, data in shot_ranges.items():
        results[range_name] = {
            'count': len(data),
            'accuracy': data['performance'].mean() if len(data) > 0 else 0
        }
    
    # Calculate overall few-shot stats
    few_shot_mask = df['num_examples'] > 0
    few_shot_data = df[few_shot_mask]
    results['few-shot'] = {
        'count': len(few_shot_data),
        'accuracy': few_shot_data['performance'].mean() if len(few_shot_data) > 0 else 0
    }
    
    # Calculate relative improvement
    if results['0-shot']['accuracy'] > 0:
        rel_improvement = ((results['few-shot']['accuracy'] - results['0-shot']['accuracy']) / 
                         results['0-shot']['accuracy']) * 100
    else:
        rel_improvement = 0
    results['improvement'] = rel_improvement
    
    # Calculate statistical significance
    if len(df[zero_shot_mask]) > 0 and len(df[few_shot_mask]) > 0:
        t_stat, p_value = stats.ttest_ind(
            df[zero_shot_mask]['performance'],
            df[few_shot_mask]['performance']
        )
        results['t_stat'] = t_stat
        results['p_value'] = p_value
    else:
        results['t_stat'] = 0
        results['p_value'] = 1
        
    return results

# Analyze each file and store results
results_data = []
columns = ['Model', 'Dataset', '0-shot count', '0-shot acc', '1-shot count', '1-shot acc',
           '2-shot count', '2-shot acc', '3+ shot count', '3+ shot acc',
           'Total few-shot count', 'Few-shot acc', 'Improvement %', 'p-value']

model_names = {
    'starling-lm-7b-alpha-bii',
    'llama-3-1-8b-instruct-lvt',
    'phi-3-5-mini-instruct-ivr',
    'qwen2-5-7b-instruct-mln'
}

for file_path in parquet_files:
    # Extract model/dataset name from filename
    file_name = file_path.split('/')[-1].replace('extracted_features_', '').replace('.parquet', '')
    
    # Split into model and dataset
    model = None
    for model_name in model_names:
        if model_name in file_name:
            model = model_name
            dataset = file_name.replace(model_name + '_', '')
            break
    
    if not model:
        continue
        
    # Load and analyze file
    df = pd.read_parquet(file_path)
    df['performance'] = (df['model_formatted_output'] == df['correct']).astype(int)
    results = analyze_shot_performance(df)
    
    # Format row data
    row = [
        model,
        dataset,
        results['0-shot']['count'],
        f"{results['0-shot']['accuracy']:.3f}",
        results['1-shot']['count'],
        f"{results['1-shot']['accuracy']:.3f}",
        results['2-shots']['count'], 
        f"{results['2-shots']['accuracy']:.3f}",
        results['3+shots']['count'],
        f"{results['3+shots']['accuracy']:.3f}",
        results['few-shot']['count'],
        f"{results['few-shot']['accuracy']:.3f}",
        f"{results['improvement']:.1f}",
        f"{results['p_value']:.3e}"
    ]
    results_data.append(row)

# Create DataFrame and save to Excel
results_df = pd.DataFrame(results_data, columns=columns)
results_df.to_excel('shot_analysis_results.xlsx', index=False)
print("Results saved to shot_analysis_results.xlsx")
