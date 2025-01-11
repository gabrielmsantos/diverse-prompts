import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
RANDOM_PATH = 'FEATURES/RANDOM/PARQUET/'

def analyze_correlations(df):
    """
    Analyze correlations between phenotype characteristics using both Pearson and Spearman methods,
    including p-value significance testing
    
    Args:
        df: Pandas DataFrame containing the data
    """
    # Calculate performance
    df['performance'] = (df['model_formatted_output'] == df['correct']).astype(int)
    
    # Select numeric columns for correlation analysis
    numeric_cols = ['num_examples', 'instance_word_count', 'instance_char_count', 
                   'prompt_word_count', 'prompt_char_count', 'num_steps',
                   'num_special_tokens', 'prompt_type_token_ratio', 'performance', 'includes_context']
    
    # Calculate correlations and p-values
    pearson_corr = df[numeric_cols].corr(method='pearson')
    spearman_corr = df[numeric_cols].corr(method='spearman')
    
    # Calculate p-values
    pearson_p = pd.DataFrame(np.zeros_like(pearson_corr), columns=numeric_cols, index=numeric_cols)
    spearman_p = pd.DataFrame(np.zeros_like(spearman_corr), columns=numeric_cols, index=numeric_cols)
    
    for i in numeric_cols:
        for j in numeric_cols:
            pearson_stat, pearson_p.loc[i,j] = stats.pearsonr(df[i], df[j])
            spearman_stat, spearman_p.loc[i,j] = stats.spearmanr(df[i], df[j])
    
    # Create subplots for correlation matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Pearson correlation heatmap
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax1)
    ax1.set_title('Pearson Correlation Matrix')
    
    # Plot Spearman correlation heatmap
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax2)
    ax2.set_title('Spearman Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Print significant correlations with p-values
    alpha = 0.05  # significance level
    print("\nSignificant correlations (|r| > 0.5 and p < 0.05):")
    print("\nPearson correlations:")
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(pearson_corr.iloc[i,j]) > 0.5 and pearson_p.iloc[i,j] < alpha:
                print(f"{numeric_cols[i]} vs {numeric_cols[j]}: r={pearson_corr.iloc[i,j]:.3f}, p={pearson_p.iloc[i,j]:.3e}")
                
    print("\nSpearman correlations:")
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(spearman_corr.iloc[i,j]) > 0.5 and spearman_p.iloc[i,j] < alpha:
                print(f"{numeric_cols[i]} vs {numeric_cols[j]}: ρ={spearman_corr.iloc[i,j]:.3f}, p={spearman_p.iloc[i,j]:.3e}")

    # Print all performance correlations
    print("\nAll performance correlations:")
    print("\nPearson correlations with performance:")
    for col in numeric_cols:
        if col != 'performance':
            r = pearson_corr.loc['performance', col]
            p = pearson_p.loc['performance', col]
            sig = "NOT SIGNIFICANT" if p > 0.05 else "significant"
            print(f"performance vs {col}: r={r:.3f}, p={p:.3e} ({sig})")
            
    print("\nSpearman correlations with performance:")
    for col in numeric_cols:
        if col != 'performance':
            rho = spearman_corr.loc['performance', col]
            p = spearman_p.loc['performance', col]
            sig = "NOT SIGNIFICANT" if p > 0.05 else "significant" 
            print(f"performance vs {col}: ρ={rho:.3f}, p={p:.3e} ({sig})")

# Example usage
# Combine all parquet files for a specific dataset
import glob
def analyze_dataset_correlations(dataset_name, path=RANDOM_PATH):
    parquet_files = glob.glob(path + f'*{dataset_name}*.parquet')
    if not parquet_files:
        print(f"No parquet files found for dataset: {dataset_name}")
        return
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    analyze_correlations(df)

# Example: Analyze correlations for 'winowhy' dataset
analyze_dataset_correlations('play_dialog_same_or_different', MAPELITES_PATH)

# Datasets:
# - formal_fallacies_syllogisms_negation
# - logical-deduction-3
# - strange_stories_boolean 
# - strategyqa
# - winowhy
# - known_unknowns
# - play_dialog_same_or_different
# - winowhy