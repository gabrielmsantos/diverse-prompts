import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob

MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
RANDOM_PATH = 'FEATURES/RANDOM/PARQUET/'

def analyze_performance_correlations(datasets, path=RANDOM_PATH):
    """
    Analyze and plot performance correlations across multiple datasets
    
    Args:
        datasets: List of dataset names to analyze
        path: Path to parquet files
    """
    # Features to correlate with performance
    features = ['num_examples', 'instance_word_count', 'instance_char_count',
               'prompt_word_count', 'prompt_char_count', 'num_steps',
               'num_special_tokens', 'prompt_type_token_ratio', 'includes_context']
    
    # Create correlation and p-value DataFrames
    correlation_df = pd.DataFrame(index=datasets, columns=features)
    p_value_df = pd.DataFrame(index=datasets, columns=features)
    
    for dataset in datasets:
        # Load dataset
        parquet_files = glob.glob(path + f'*{dataset}*.parquet')
        if not parquet_files:
            print(f"No parquet files found for dataset: {dataset}")
            continue
            
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        
        # Calculate performance
        df['performance'] = (df['model_formatted_output'] == df['correct']).astype(int)
        
        # Calculate correlations and p-values with performance
        for feature in features:
            # r, p = stats.pearsonr(df[feature], df['performance'])
            r, p = stats.spearmanr(df[feature], df['performance'])
            correlation_df.loc[dataset, feature] = r
            p_value_df.loc[dataset, feature] = p
    
    # Convert correlation_df to numeric values, replacing any non-numeric values with NaN
    correlation_df = correlation_df.apply(pd.to_numeric, errors='coerce')
    
    # Create heatmap with significance markers
    plt.figure(figsize=(12, 8))
    
    # Create annotation matrix with significance markers
    annot_matrix = correlation_df.round(3).astype(str)
    for i in range(len(datasets)):
        for j in range(len(features)):
            if p_value_df.iloc[i,j] < 0.05:
                annot_matrix.iloc[i,j] = annot_matrix.iloc[i,j] + '\u2731'  # Bold star symbol
    
    sns.heatmap(correlation_df, annot=annot_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                fmt='', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Performance Correlations Across Datasets\n\u2731 indicates p < 0.05')
    plt.tight_layout()
    plt.show()
    
    # Print significant correlations
    print("\nSignificant correlations (p < 0.05):")
    for dataset in datasets:
        print(f"\n{dataset}:")
        for feature in features:
            if p_value_df.loc[dataset, feature] < 0.05:
                print(f"{feature}: r={correlation_df.loc[dataset, feature]:.3f}, p={p_value_df.loc[dataset, feature]:.3e}")

# Example usage:
datasets = [
    'formal_fallacies_syllogisms_negation',
    'logical-deduction-3', 
    'strange_stories_boolean',
    'strategyqa',
    'winowhy',
    'known_unknowns',
    'play_dialog_same_or_different'
]

analyze_performance_correlations(datasets, MAPELITES_PATH)
