import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats
import glob

# Constants
MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
RANDOM_PATH = 'FEATURES/RANDOM/PARQUET/'

def get_coverage_data(parquet_files, phenotype1_col, phenotype2_col, num_bins1=10, num_bins2=10):
    """Get coverage data for a specific algorithm"""
    # Load and concatenate all parquet files
    dfs = []
    for f in parquet_files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    
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
    if current_tests > 0:
        performance = current_correct / current_tests
        phenotype1_bin = np.digitize(df.iloc[-1][phenotype1_col], phenotype1_bins) - 1
        phenotype2_bin = np.digitize(df.iloc[-1][phenotype2_col], phenotype2_bins) - 1
        bin_key = (phenotype1_bin, phenotype2_bin)
        if performance > elite_map[bin_key]['performance']:
            elite_map[bin_key] = {
                'performance': performance,
                'individual': df.iloc[-1].to_dict(),
            }

    return elite_map, phenotype1_bins, phenotype2_bins

def compare_coverage(dataset_name, dictionary_labels=None, phenotype1_col='num_examples', phenotype2_col='num_steps', num_bins1=7, num_bins2=6):
    """Compare coverage between MAP-Elites and Random search for a specific dataset"""
    
    # Get all parquet files for the dataset
    mapelites_files = glob.glob(f"{MAPELITES_PATH}*{dataset_name}*.parquet")
    random_files = glob.glob(f"{RANDOM_PATH}*{dataset_name}*.parquet")
    
    if not mapelites_files or not random_files:
        print(f"No files found for dataset: {dataset_name}")
        return
    
    # Get data for both algorithms
    mapelites_data, p1_bins, p2_bins = get_coverage_data(mapelites_files, phenotype1_col, phenotype2_col, num_bins1, num_bins2)
    random_data, _, _ = get_coverage_data(random_files, phenotype1_col, phenotype2_col, num_bins1, num_bins2)

    # Calculate coverage percentages
    total_bins = (len(p1_bins)-1) * (len(p2_bins)-1)
    mapelites_filled = sum(1 for bin_key in mapelites_data if 0 <= bin_key[0] < len(p1_bins)-1 
                          and 0 <= bin_key[1] < len(p2_bins)-1)
    random_filled = sum(1 for bin_key in random_data if 0 <= bin_key[0] < len(p1_bins)-1 
                       and 0 <= bin_key[1] < len(p2_bins)-1)
    mapelites_coverage = (mapelites_filled/total_bins)*100
    random_coverage = (random_filled/total_bins)*100

    # Create figure with proper spacing
    fig = plt.figure(figsize=(18, 10))  # Increased figure size
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)  # Increased spacing
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Plot MAP-Elites
    scatter1 = None
    for bin_key, data in mapelites_data.items():
        x = data['individual'][phenotype1_col]
        y = data['individual'][phenotype2_col]
        performance = data['performance']
        includes_context = data['individual']['includes_context']
        
        marker = 's' if includes_context else 'o'
        scatter1 = ax1.scatter(x, y, c=[performance], cmap='coolwarm', 
                             marker=marker, s=100, vmin=0, vmax=1)

    # ax1.set_title(f'MAP-Elites Coverage for {dataset_name}\n({mapelites_coverage:.1f}% coverage)', fontsize=12, pad=20)  # Added padding
    ax1.set_title(f'MAP-Elites Coverage for {dataset_name}', fontsize=12, pad=20)  # Added padding
    ax1.set_xlabel(dictionary_labels[phenotype1_col], fontsize=10, labelpad=10)  # Added padding
    ax1.set_ylabel(dictionary_labels[phenotype2_col], fontsize=10, labelpad=10)  # Added padding

    # Plot Random Search
    for bin_key, data in random_data.items():
        x = data['individual'][phenotype1_col]
        y = data['individual'][phenotype2_col]
        performance = data['performance']
        includes_context = data['individual']['includes_context']
        
        marker = 's' if includes_context else 'o'
        ax2.scatter(x, y, c=[performance], cmap='coolwarm', 
                   marker=marker, s=100, vmin=0, vmax=1)

    # ax2.set_title(f'Random Search Coverage\n({random_coverage:.1f}% coverage)', fontsize=12, pad=20)  # Added padding
    ax2.set_title(f'Random Search Coverage for {dataset_name}', fontsize=12, pad=20)  # Added padding
    ax2.set_xlabel(dictionary_labels[phenotype1_col], fontsize=10, labelpad=10)  # Added padding
    ax2.set_ylabel(dictionary_labels[phenotype2_col], fontsize=10, labelpad=10)  # Added padding

    # Add colorbars with padding
    plt.colorbar(scatter1, ax=ax1, label='Performance', pad=0.02)
    plt.colorbar(scatter1, ax=ax2, label='Performance', pad=0.02)

    # Add grid lines
    for ax in [ax1, ax2]:
        for bin_edge in p1_bins:
            ax.axvline(x=bin_edge, color='gray', linestyle='--', alpha=0.3)
        for bin_edge in p2_bins:
            ax.axhline(y=bin_edge, color='gray', linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)

        # Set consistent axis limits with padding
        padding = (p1_bins[-1] - p1_bins[0]) * 0.05  # 5% padding
        ax.set_xlim(p1_bins[0] - padding, p1_bins[-1] + padding)
        padding = (p2_bins[-1] - p2_bins[0]) * 0.05  # 5% padding
        ax.set_ylim(p2_bins[0] - padding, p2_bins[-1] + padding)

    # Add legend
    for ax in [ax1, ax2]:
        ax.scatter([], [], marker='s', c='gray', s=100, label='Includes Role Context')
        ax.scatter([], [], marker='o', c='gray', s=100, label='No Role Context')
        ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    #plt.suptitle(f'Coverage Comparison - {dataset_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Statistical analysis
    print("\nCoverage Statistics:")
    print(f"Total possible bins: {total_bins}")
    print(f"MAP-Elites filled bins: {mapelites_filled} ({mapelites_coverage:.2f}%)")
    print(f"Random Search filled bins: {random_filled} ({random_coverage:.2f}%)")

    # Chi-square test for independence
    contingency = np.array([[mapelites_filled, total_bins - mapelites_filled],
                           [random_filled, total_bins - random_filled]])
    chi2, p_value = stats.chi2_contingency(contingency)[:2]

    # Calculate effect size using Cramer's V
    n = np.sum(contingency)
    min_dim = min(contingency.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))

    # Calculate statistical power
    dof = 1  # degrees of freedom for 2x2 contingency table
    ncp = chi2 * (1 - 1/n)  # non-centrality parameter
    power = stats.ncx2.sf(stats.chi2.ppf(0.95, dof), dof, ncp)

    print("\nStatistical Test Results:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Effect size (Cramer's V): {cramer_v:.4f}")
    print(f"Statistical power: {power:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if p_value < 0.05:
        print("- Statistically significant difference found")
    else:
        print("- No statistically significant difference found")
        
    print(f"- Effect size is " + 
          ("small" if cramer_v < 0.1 else 
           "medium" if cramer_v < 0.3 else 
           "large" if cramer_v < 0.5 else 
           "very large"))
    
    if power < 0.8:
        print("- Warning: Statistical power is below recommended 0.8 threshold")
        sample_size = int(np.ceil((power/0.8) * n))
        print(f"- Recommended minimum total bins for adequate power: {sample_size}")

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
compare_coverage('logical-deduction', FEATURE_LABELS)
