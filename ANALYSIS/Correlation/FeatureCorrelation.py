import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob

MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
RANDOM_PATH = 'FEATURES/RANDOM/PARQUET/'

def analyze_performance_correlations(datasets):
    """
    Analyze and plot performance correlations across multiple datasets
    
    Args:
        datasets: List of dataset names to analyze
    """
    # Features to correlate with performance
    features = ['num_examples', 'instance_word_count', 'num_steps',
               'includes_context', 'prompt_type_token_ratio']
    
    # Feature labels for plotting
    feature_labels = {
        'num_examples': 'Number of Shots',
        'instance_word_count': 'Prompt Length (Words)',
        'num_steps': 'Depth of CoT',
        'includes_context': 'Context Included',
        'prompt_type_token_ratio': 'Type-Token Ratio'
    }
    
    # Create correlation and p-value DataFrames
    correlation_df = pd.DataFrame(index=datasets, columns=features)
    p_value_df = pd.DataFrame(index=datasets, columns=features)
    
    for dataset in datasets:
        # Load dataset from both paths
        mapelites_files = glob.glob(MAPELITES_PATH + f'*{dataset}*.parquet')
        random_files = glob.glob(RANDOM_PATH + f'*{dataset}*.parquet')
        
        all_files = mapelites_files + random_files
        if not all_files:
            print(f"No parquet files found for dataset: {dataset}")
            continue
            
        df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
        
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
    
    # Create mapping for dataset labels
    dataset_labels = {
        'formal_fallacies_syllogisms_negation': 'FFSN',
        'logical-deduction-3': 'LD3',
        'strange_stories_boolean': 'SSB', 
        'strategyqa': 'SQA',
        'winowhy': 'winowhy',
        'known_unknowns': 'KU',
        'play_dialog_same_or_different': 'PDSD'
    }
    
    # Create heatmap with significance markers
    plt.figure(figsize=(12, 8))
    
    # Create annotation matrix with significance markers
    annot_matrix = correlation_df.round(3).astype(str)
    for i in range(len(datasets)):
        for j in range(len(features)):
            if p_value_df.iloc[i,j] < 0.05:
                # Use superscript star
                annot_matrix.iloc[i,j] = annot_matrix.iloc[i,j] + '\u00B9'
    
    # Rename index with short labels before plotting
    correlation_df.index = [dataset_labels[d] for d in correlation_df.index]
    annot_matrix.index = [dataset_labels[d] for d in annot_matrix.index]
    
    # Rename columns with feature labels
    correlation_df.columns = [feature_labels[f] for f in features]
    annot_matrix.columns = [feature_labels[f] for f in features]
    
    sns.heatmap(correlation_df, annot=annot_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                fmt='', cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={'size': 18})
    plt.title('Performance Correlations Across Datasets\n¹ indicates p-Value < 0.05', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print significant correlations
    print("\nSignificant correlations (p < 0.05):")
    for dataset in datasets:
        print(f"\n{dataset}:")
        for feature in features:
            if p_value_df.loc[dataset, feature] < 0.05:
                print(f"{feature_labels[feature]}: r={correlation_df.loc[dataset_labels[dataset], feature_labels[feature]]:.3f}, p={p_value_df.loc[dataset, feature]:.3e}")

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

analyze_performance_correlations(datasets)
