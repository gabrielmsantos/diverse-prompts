import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ALGORITHM = 'MAPELITES'
PARQUET_PATH = f'FEATURES/{ALGORITHM}/PARQUET/'

def create_2d_density_heatmap(df, feature1, feature2, num_bins_continuous=20):
    """
    Create a 2D density heatmap showing performance distribution across two features,
    with special handling for discrete x-axis values
    
    Args:
        df: Pandas DataFrame containing the data
        feature1: Name of first feature column (discrete)
        feature2: Name of second feature column (continuous) 
        num_bins_continuous: Number of bins for the continuous feature
    """
    # Get unique values for discrete feature
    discrete_values = sorted(df[feature1].unique(), reverse=True)  # Reverse sort for y-axis
    
    # Create bins for continuous feature
    continuous_bins = np.linspace(df[feature2].min(), df[feature2].max(), num_bins_continuous)
    
    # Initialize array to store performance
    heatmap = np.zeros((len(discrete_values), num_bins_continuous-1))
    counts = np.zeros((len(discrete_values), num_bins_continuous-1))
    
    # Calculate average performance for each bin
    for i, val in enumerate(discrete_values):
        mask = df[feature1] == val
        if mask.any():
            for j in range(num_bins_continuous-1):
                bin_mask = (df[feature2][mask] >= continuous_bins[j]) & \
                          (df[feature2][mask] < continuous_bins[j+1])
                if bin_mask.any():
                    heatmap[i,j] = (df['model_formatted_output'][mask][bin_mask] == 
                                  df['correct'][mask][bin_mask]).mean()
                    counts[i,j] = bin_mask.sum()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap,
                xticklabels=[f"{x:.0f}" for x in continuous_bins[:-1]], 
                yticklabels=[f"{x:.0f}" for x in discrete_values],
                cmap='coolwarm',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Average Performance'})

    plt.xlabel(feature2)
    plt.ylabel(feature1)
    plt.title(f'Performance Distribution: {feature1} vs {feature2}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
df = pd.read_parquet(PARQUET_PATH + 'extracted_features_starling-lm-7b-alpha-bii_winowhy.parquet')
create_2d_density_heatmap(df, 'num_steps', 'num_examples')