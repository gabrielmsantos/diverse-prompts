import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PARQUET_PATH = 'FEATURES/PARQUET/'

def create_density_map(df, feature1, feature2):
    """
    Create a line plot showing distribution of correct/incorrect answers across two features
    
    Args:
        df: Pandas DataFrame containing the data
        feature1: Name of first feature column
        feature2: Name of second feature column
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Sort dataframe by feature1 to ensure proper line plotting
    df = df.sort_values(by=feature1)
    
    # Plot each line with color based on correctness
    for _, row in df.iterrows():
        color = 'red' if row['model_formatted_output'] == row['correct'] else 'blue'
        plt.plot([row[feature1]], [row[feature2]], 
                color=color,
                marker='|',
                alpha=0.5)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Answer Distribution: {feature1} vs {feature2}')
    plt.legend(['Correct (Red)', 'Incorrect (Blue)'])
    plt.tight_layout()
    plt.show()

# Example usage
df = pd.read_parquet(PARQUET_PATH + 'extracted_features_starling-lm-7b-alpha-bii_winowhy.parquet')
create_density_map(df, 'num_examples', 'num_steps')
