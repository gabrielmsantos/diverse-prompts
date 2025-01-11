import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import numpy as np

# Constants
MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
RANDOM_PATH = 'FEATURES/RANDOM/PARQUET/'

def compute_coverage(df, feature_cols, num_bins_list):
    """Calculate percentage of phenotypic space covered, excluding edge bins
    
    Args:
        df: DataFrame containing the features
        feature_cols: List of feature column names 
        num_bins_list: List of number of bins for each feature
    """
    if len(feature_cols) != len(num_bins_list):
        raise ValueError("Length of feature_cols must match length of num_bins_list")
    
    # Create bins for each feature
    bins_per_feature = []
    for col, num_bins in zip(feature_cols, num_bins_list):
        bins = np.linspace(df[col].min(), df[col].max(), num_bins)
        bins_per_feature.append(bins)
    
    # Track filled bins
    filled_bins = set()
    
    # Process each row
    for _, row in df.iterrows():
        bin_key = []
        for col, bins in zip(feature_cols, bins_per_feature):
            # Get bin index for this feature
            bin_idx = np.digitize(row[col], bins) - 1
            # Only consider non-edge bins
            if 0 <= bin_idx < len(bins)-1:
                bin_key.append(bin_idx)
            else:
                bin_key = None
                break
        
        if bin_key is not None:
            filled_bins.add(tuple(bin_key))
    
    # Calculate total possible bins (excluding edges)
    total_bins = np.prod([(n-1) for n in num_bins_list])
    
    # Calculate coverage as fraction
    coverage = len(filled_bins) / total_bins if total_bins > 0 else 0
    
    return coverage, filled_bins, bins_per_feature

def compute_shannon_entropy(df, feature_cols, num_bins=10):
    """Calculate Shannon entropy of phenotypic distribution"""
    # Discretize features into bins
    binned_data = pd.DataFrame()
    for col in feature_cols:
        binned_data[col] = pd.qcut(df[col], q=num_bins, labels=False, duplicates='drop')
    
    # Get bin counts
    bin_counts = binned_data.value_counts()
    probabilities = bin_counts / len(df)
    
    # Calculate entropy
    return entropy(probabilities)

def compute_pairwise_distances(df, feature_cols):
    """Calculate mean pairwise distance between individuals"""
    # Convert boolean column to numeric
    features = df[feature_cols].copy()
    for col in feature_cols:
        if features[col].dtype == bool:
            features[col] = features[col].astype(float)
    
    # Normalize features
    normalized_features = (features - features.mean()) / features.std()
    
    # Compute pairwise distances
    distances = pdist(normalized_features, metric='euclidean')
    return np.mean(distances)

def compute_elite_pairwise_distances(df, feature_cols, num_bins_list):
    """Calculate mean pairwise distance between best performing individuals per bin"""
    # Create bins for each feature
    bins_per_feature = []
    for col, num_bins in zip(feature_cols, num_bins_list):
        bins = np.linspace(df[col].min(), df[col].max(), num_bins)
        bins_per_feature.append(bins)
    
    # Dictionary to store best performing individual per bin
    elite_map = {}
    
    # Track performance per genotype
    genotype_performance = {}
    current_genotype = None
    current_correct = 0
    current_tests = 0
    
    # Calculate performance for each genotype
    for idx, row in df.iterrows():
        if not np.array_equal(current_genotype, row['genotype']):
            # Evaluate previous genotype
            if current_genotype is not None and current_tests > 0:
                genotype_performance[idx-1] = current_correct / current_tests
            
            # Reset for new genotype
            current_genotype = row['genotype']
            current_correct = 0
            current_tests = 0
        
        if row['model_formatted_output'] == row['correct']:
            current_correct += 1
        current_tests += 1
    
    # Handle last genotype
    if current_tests > 0:
        genotype_performance[len(df)-1] = current_correct / current_tests
    
    # Find best individual per bin
    for idx, row in df.iterrows():
        if idx not in genotype_performance:
            continue
            
        bin_key = []
        for col, bins in zip(feature_cols, bins_per_feature):
            bin_idx = np.digitize(row[col], bins) - 1
            if 0 <= bin_idx < len(bins)-1:
                bin_key.append(bin_idx)
            else:
                bin_key = None
                break
                
        if bin_key is not None:
            bin_key = tuple(bin_key)
            performance = genotype_performance[idx]
            
            # Convert features to float array
            features = row[feature_cols].copy()
            for i, col in enumerate(feature_cols):
                if isinstance(features[col], bool):
                    features[col] = float(features[col])
            
            if bin_key not in elite_map or performance > elite_map[bin_key]['performance']:
                elite_map[bin_key] = {
                    'performance': performance,
                    'features': features.values.astype(float)
                }
    
    # Extract features of elite individuals
    elite_features = np.array([data['features'] for data in elite_map.values()])
    
    if len(elite_features) < 2:  # Need at least 2 points for pairwise distance
        return 0
    
    # Normalize features
    mean = np.mean(elite_features, axis=0)
    std = np.std(elite_features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    normalized_features = (elite_features - mean) / std
    
    # Compute pairwise distances between elite individuals
    distances = pdist(normalized_features, metric='euclidean')
    return np.mean(distances)

def compute_feature_variance(df, feature_cols):
    """Calculate variance of each phenotypic feature"""
    # Convert boolean column to numeric
    features = df[feature_cols].copy()
    for col in feature_cols:
        if features[col].dtype == bool:
            features[col] = features[col].astype(float)
    return features.var()

def compare_diversity_metrics(mapelites_path, random_path, feature_cols, num_bins_list):
    """Compare diversity metrics between MAP-Elites and random sampling"""
    
    # Load data
    df_mapelites = pd.read_parquet(mapelites_path)
    df_random = pd.read_parquet(random_path)
    
    # Initialize results dictionary
    results = {
        'method': ['MAP-Elites', 'Random'],
        'coverage': [],
        'entropy': [],
        'mean_pairwise_dist': [],
        'elite_pairwise_dist': [],
        'feature_variance': []
    }
    
    # Compute metrics for each method
    for df, method in [(df_mapelites, 'MAP-Elites'), (df_random, 'Random')]:
        # Coverage
        coverage, filled_bins, bins_per_feature = compute_coverage(df, feature_cols, num_bins_list)
        results['coverage'].append(coverage)
        
        # Calculate and print coverage statistics for both methods
        bin_sizes = []
        for bins in bins_per_feature:
            bin_size = (bins[-1] - bins[0]) / (len(bins) - 1)
            bin_sizes.append(bin_size)
        
        total_bins = np.prod([(n-1) for n in num_bins_list])
        num_filled = len(filled_bins)
        percent_filled = (num_filled/total_bins)*100
        
        print(f"\nCoverage Statistics for {method}:")
        for col, bin_size in zip(feature_cols, bin_sizes):
            print(f"Bin size for {col}: {bin_size:.2f}")
        print(f"Total number of bins (excluding edges): {total_bins}")
        print(f"Number of filled bins (excluding edges): {num_filled}")
        print(f"Percentage of bins filled (excluding edges): {percent_filled:.2f}%")
        
        # Shannon Entropy
        entropy_val = compute_shannon_entropy(df, feature_cols)
        results['entropy'].append(entropy_val)
        
        # Mean Pairwise Distance
        mean_dist = compute_pairwise_distances(df, feature_cols)
        results['mean_pairwise_dist'].append(mean_dist)
        
        # Elite Pairwise Distance
        elite_dist = compute_elite_pairwise_distances(df, feature_cols, num_bins_list)
        results['elite_pairwise_dist'].append(elite_dist)
        
        # Feature Variance
        variance = compute_feature_variance(df, feature_cols)
        results['feature_variance'].append(variance.mean())  # Average variance across features
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# Example usage:
feature_cols = ['num_examples', 
                'num_steps', 
                'includes_context']
num_bins_list = [11, 6, 2]  # Specific number of bins for each feature

results = compare_diversity_metrics(MAPELITES_PATH + 'extracted_features_starling-lm-7b-alpha-bii_winowhy.parquet',
                                  RANDOM_PATH + 'extracted_features_starling-lm-7b-alpha-bii_winowhy.parquet',
                                  feature_cols,
                                  num_bins_list)
print(results)
