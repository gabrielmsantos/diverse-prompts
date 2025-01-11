# genotype_id
# genotype
# prompt_instance
# prompt
# model_text_answer
# model_formatted_output
# correct
# num_examples
# instance_word_count
# instance_char_count
# prompt_word_count
# prompt_char_count
# num_steps
# includes_context
# num_special_tokens
# prompt_type_token_ratio

ALGORITHM = 'RANDOM'
#ALGORITHM = 'MAPELITES'
PARQUET_PATH = f'FEATURES/{ALGORITHM}/PARQUET/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_map_elites(df, phenotype1_col, phenotype2_col, num_bins1=10, num_bins2=10):
    """
    Analyze MAP-Elites results using two specified phenotype dimensions
    
    Args:
        df: Pandas DataFrame containing the data
        phenotype1_col: Name of first phenotype column
        phenotype2_col: Name of second phenotype column
        num_bins1: Number of bins for first phenotype dimension
        num_bins2: Number of bins for second phenotype dimension
    """
    # Define phenotype ranges for binning
    phenotype1_bins = np.linspace(df[phenotype1_col].min(), df[phenotype1_col].max(), num_bins1)
    phenotype2_bins = np.linspace(df[phenotype2_col].min(), df[phenotype2_col].max(), num_bins2)

    # Dictionary to store best performing individual per bin
    elite_map = defaultdict(lambda: {'performance': -1, 'individual': None})

    current_genotype = None
    current_correct = 0
    current_tests = 0

    # Process each row
    for idx, row in df.iterrows():
        if not np.array_equal(current_genotype, row['genotype']):
            # New genotype encountered - evaluate previous one if exists
            if current_genotype is not None and current_tests > 0:
                performance = current_correct / current_tests
                
                # Find bins for this individual
                phenotype1_bin = np.digitize(df.loc[idx-1, phenotype1_col], phenotype1_bins) - 1
                phenotype2_bin = np.digitize(df.loc[idx-1, phenotype2_col], phenotype2_bins) - 1
                bin_key = (phenotype1_bin, phenotype2_bin)
                
                # Update elite map if better performance
                if performance > elite_map[bin_key]['performance']:
                    elite_map[bin_key] = {
                        'performance': performance,
                        'individual': df.loc[idx-1].to_dict(),
                    }
            
            # Reset counters for new genotype
            current_genotype = row['genotype']
            current_correct = 0
            current_tests = 0
        
        # Count correct answers
        if row['model_formatted_output'] == row['correct']:
            current_correct += 1
        current_tests += 1
        
        # Handle last genotype
        if idx == len(df)-1 and current_tests > 0:
            performance = current_correct / current_tests
            phenotype1_bin = np.digitize(row[phenotype1_col], phenotype1_bins) - 1
            phenotype2_bin = np.digitize(row[phenotype2_col], phenotype2_bins) - 1
            bin_key = (phenotype1_bin, phenotype2_bin)
            if performance > elite_map[bin_key]['performance']:
                elite_map[bin_key] = {
                    'performance': performance,
                    'individual': row.to_dict(),
                }

    # Calculate bin sizes
    phenotype1_bin_size = (phenotype1_bins[-1] - phenotype1_bins[0]) / (len(phenotype1_bins) - 1)
    phenotype2_bin_size = (phenotype2_bins[-1] - phenotype2_bins[0]) / (len(phenotype2_bins) - 1)
    print(f"Bin size for {phenotype1_col}: {phenotype1_bin_size:.2f}")
    print(f"Bin size for {phenotype2_col}: {phenotype2_bin_size:.2f}")

    # Calculate and print bin statistics (excluding edge bins)
    total_bins = (len(phenotype1_bins)-1) * (len(phenotype2_bins)-1) 
    filled_bins = sum(1 for bin_key in elite_map if 0 <= bin_key[0] < len(phenotype1_bins)-1 
                                                and 0 <= bin_key[1] < len(phenotype2_bins)-1)
    print(f"Total number of bins (excluding edges): {total_bins}")
    print(f"Number of filled bins (excluding edges): {filled_bins}")
    print(f"Percentage of bins filled (excluding edges): {(filled_bins/total_bins)*100:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 8))

    # Draw grid lines for bins
    for bin_edge in phenotype1_bins:
        plt.axvline(x=bin_edge, color='gray', linestyle='--', alpha=0.3)
    for bin_edge in phenotype2_bins:
        plt.axhline(y=bin_edge, color='gray', linestyle='--', alpha=0.3)

    for bin_key, data in elite_map.items():
        x = data['individual'][phenotype1_col]
        y = data['individual'][phenotype2_col]
        performance = data['performance']
        includes_context = data['individual']['includes_context']
        
        if includes_context:
            marker = 's'  # square
        else:
            marker = 'o'  # circle
            
        plt.scatter(x, y, c=[performance], cmap='coolwarm', 
                    marker=marker, s=200, vmin=0, vmax=1)  

    plt.colorbar(label='Performance')
    plt.xlabel(phenotype1_col)
    plt.ylabel(phenotype2_col)
    plt.title(f'{ALGORITHM} Performance Distribution: {(filled_bins/total_bins)*100:.2f}% bins filled')

    # Add legend with bigger markers
    plt.scatter([], [], marker='s', c='gray', s=200, label='Includes Context')  
    plt.scatter([], [], marker='o', c='gray', s=200, label='No Context')  
    plt.legend()

    plt.show()

# Example usage
def analyze_dataset(dataset_name, phenotype1='num_examples', phenotype2='num_steps', 
                   num_bins1=6, num_bins2=6):
    import glob
    parquet_files = glob.glob(PARQUET_PATH + f'*{dataset_name}*.parquet')
    if not parquet_files:
        raise ValueError(f"No parquet files found for dataset: {dataset_name}")
    print(f"Importing files: {parquet_files}")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    analyze_map_elites(df, phenotype1, phenotype2, num_bins1, num_bins2)

# Example call:
analyze_dataset('winowhy')
