import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats
import glob
from tabulate import tabulate

# Constants
MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
RANDOM_PATH = 'FEATURES/RANDOM/PARQUET/'

def get_coverage_data(parquet_file, phenotype1_col, phenotype2_col, phenotype3_col, num_bins1=10, num_bins2=10, num_bins3=25, min_performance=0.0):
    """Get coverage data for a specific algorithm"""
    # Load single parquet file
    df = pd.read_parquet(parquet_file)
    
    # Define phenotype ranges for binning
    phenotype1_bins = np.linspace(df[phenotype1_col].min(), df[phenotype1_col].max(), num_bins1)
    phenotype2_bins = np.linspace(df[phenotype2_col].min(), df[phenotype2_col].max(), num_bins2)
    phenotype3_bins = np.linspace(df[phenotype3_col].min(), df[phenotype3_col].max(), num_bins3)

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
                
                # Only consider individuals that meet minimum performance threshold
                if performance >= min_performance:
                    # Find bins for this individual
                    phenotype1_bin = np.digitize(df.loc[idx-1, phenotype1_col], phenotype1_bins) - 1
                    phenotype2_bin = np.digitize(df.loc[idx-1, phenotype2_col], phenotype2_bins) - 1
                    phenotype3_bin = np.digitize(df.loc[idx-1, phenotype3_col], phenotype3_bins) - 1
                    bin_key = (phenotype1_bin, phenotype2_bin, phenotype3_bin)
                    
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
    if current_tests > 0:
        performance = current_correct / current_tests
        if performance >= min_performance:
            phenotype1_bin = np.digitize(df.iloc[-1][phenotype1_col], phenotype1_bins) - 1
            phenotype2_bin = np.digitize(df.iloc[-1][phenotype2_col], phenotype2_bins) - 1
            phenotype3_bin = np.digitize(df.iloc[-1][phenotype3_col], phenotype3_bins) - 1
            bin_key = (phenotype1_bin, phenotype2_bin, phenotype3_bin)
            if performance > elite_map[bin_key]['performance']:
                elite_map[bin_key] = {
                    'performance': performance,
                    'individual': df.iloc[-1].to_dict(),
                }

    return elite_map, phenotype1_bins, phenotype2_bins, phenotype3_bins

def compare_coverage(dataset_name, dictionary_labels=None, phenotype1_col='num_examples', phenotype2_col='num_steps', phenotype3_col='instance_word_count', num_bins1=6, num_bins2=6, num_bins3=25, min_performance=0.0):
    """Compare coverage between MAP-Elites and Random search for a specific dataset"""
    
    # Calculate num_bins3 based on max prompt length divided by 25
    df = pd.read_parquet(mapelites_files[0])  # Read first file to get max length
    max_prompt_length = df[phenotype3_col].max()
    num_bins3 = int(max_prompt_length / 25)
    
    # Get all parquet files for the dataset
    mapelites_files = glob.glob(f"{MAPELITES_PATH}*{dataset_name}*.parquet")
    random_files = glob.glob(f"{RANDOM_PATH}*{dataset_name}*.parquet")
    
    if not mapelites_files or not random_files:
        print(f"No files found for dataset: {dataset_name}")
        return

    # Store results for table
    results = []
    
    # Get unique LLM names from filenames
    llm_names = set()
    for f in mapelites_files + random_files:
        filename = f.split('/')[-1]
        if 'extracted_features_' in filename:
            llm_name = filename.split('extracted_features_')[1].split(f'_{dataset_name}')[0]
            llm_names.add(llm_name)
    
    # Process each LLM
    for llm_name in llm_names:
        # Get corresponding files for this LLM
        mapelites_file = next((f for f in mapelites_files if f'extracted_features_{llm_name}_{dataset_name}' in f), None)
        random_file = next((f for f in random_files if f'extracted_features_{llm_name}_{dataset_name}' in f), None)
        
        if not mapelites_file or not random_file:
            continue

        # Get data for both algorithms
        mapelites_data, p1_bins, p2_bins, p3_bins = get_coverage_data(mapelites_file, phenotype1_col, phenotype2_col, phenotype3_col, num_bins1, num_bins2, num_bins3, min_performance)
        random_data, _, _, _ = get_coverage_data(random_file, phenotype1_col, phenotype2_col, phenotype3_col, num_bins1, num_bins2, num_bins3, min_performance)

        # Calculate coverage percentages
        total_bins = (len(p1_bins)-1) * (len(p2_bins)-1) * (len(p3_bins)-1)
        mapelites_filled = sum(1 for bin_key in mapelites_data if 0 <= bin_key[0] < len(p1_bins)-1 
                              and 0 <= bin_key[1] < len(p2_bins)-1
                              and 0 <= bin_key[2] < len(p3_bins)-1)
        random_filled = sum(1 for bin_key in random_data if 0 <= bin_key[0] < len(p1_bins)-1 
                           and 0 <= bin_key[1] < len(p2_bins)-1
                           and 0 <= bin_key[2] < len(p3_bins)-1)
        mapelites_coverage = (mapelites_filled/total_bins)*100
        random_coverage = (random_filled/total_bins)*100

        # Chi-square test
        contingency = np.array([[mapelites_filled, total_bins - mapelites_filled],
                               [random_filled, total_bins - random_filled]])
        chi2, p_value = stats.chi2_contingency(contingency)[:2]

        # Effect size
        n = np.sum(contingency)
        min_dim = min(contingency.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim))

        # Statistical power
        dof = 1
        ncp = chi2 * (1 - 1/n)
        power = stats.ncx2.sf(stats.chi2.ppf(0.95, dof), dof, ncp)

        # Add results to table
        results.append([
            llm_name,
            f"{mapelites_coverage:.1f}%",
            f"{random_coverage:.1f}%",
            f"{chi2:.2f}",
            f"{p_value:.4f}",
            f"{cramer_v:.4f}",
            f"{power:.4f}",
            "Yes" if p_value < 0.05 else "No",
            "small" if cramer_v < 0.1 else "medium" if cramer_v < 0.3 else "large" if cramer_v < 0.5 else "very large"
        ])

    # Print results table
    headers = ["LLM", "MAP-Elites Coverage", "Random Coverage", "Chi-square", "p-value", 
              "Effect Size", "Power", "Significant", "Effect Magnitude"]
    print(f"\nCoverage Analysis Results (min performance: {min_performance}):")
    print(tabulate(results, headers=headers, tablefmt="grid"))

# Dictionary mapping feature names to their chart labels
FEATURE_LABELS = {
    'num_examples': 'Number of Shots (Examples)',
    'instance_word_count': 'Prompt Length (Words)', 
    'instance_char_count': 'Prompt Length (Characters)',
    'num_steps': 'Depth of CoT',
    'includes_context': 'Context Included',
    'num_special_tokens': 'Number of Special Tokens',
    'prompt_type_token_ratio': 'Type-Token Ratio'
}

# Example usage:
compare_coverage('winowhy', FEATURE_LABELS, min_performance=0.6)
