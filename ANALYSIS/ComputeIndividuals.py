import pandas as pd
import numpy as np
from collections import defaultdict
import sys

def compute_individuals(file_path):
    """
    Compute performance metrics for each individual as they appear sequentially in the dataset
    
    Args:
        file_path (str): Path to the parquet file containing MAP-Elites data
    """
    # Read parquet file
    df = pd.read_parquet(file_path)
    
    # Track current genotype and metrics
    current_genotype = None
    current_metrics = {
        'correct': 0,
        'total': 0,
        'has_context': None,
        'num_shots': None, 
        'depth': None,
        'trials': 0
    }
    
    # Store results for each individual
    results = []
    
    # Process rows sequentially
    for idx, row in df.iterrows():
        genotype = tuple(row['genotype'])
        
        # If genotype changes or we've reached 50 trials, save current individual and reset metrics
        if (current_genotype is not None and 
            (genotype != current_genotype or current_metrics['trials'] >= 50)):
            performance = current_metrics['correct'] / current_metrics['total']
            results.append({
                'Genotype': list(current_genotype),
                'Has Context': current_metrics['has_context'],
                'Number of Shots': current_metrics['num_shots'],
                'Depth of Reasoning': current_metrics['depth'],
                'Performance': performance,
                'Num Trials': current_metrics['trials']
            })
            
            # Reset metrics for new individual
            current_metrics = {
                'correct': 0,
                'total': 0,
                'has_context': None,
                'num_shots': None,
                'depth': None,
                'trials': 0
            }
        
        # Update metrics for current genotype
        current_genotype = genotype
        current_metrics['correct'] += int(row['model_formatted_output'] == row['correct'])
        current_metrics['total'] += 1
        current_metrics['has_context'] = row['includes_context']
        current_metrics['num_shots'] = row['num_examples']
        current_metrics['depth'] = row['num_steps']
        current_metrics['trials'] += 1
    
    # Add final individual
    if current_genotype is not None:
        performance = current_metrics['correct'] / current_metrics['total']
        results.append({
            'Genotype': list(current_genotype),
            'Has Context': current_metrics['has_context'],
            'Number of Shots': current_metrics['num_shots'],
            'Depth of Reasoning': current_metrics['depth'],
            'Performance': performance,
            'Num Trials': current_metrics['trials']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Extract filename components for saving
    base_name = file_path.split('/')[-1].replace('extracted_features_', '').replace('.parquet', '')
    
    # Save results
    results_df.to_excel(f'individual_metrics_{base_name}.xlsx', index=False)
    results_df.to_parquet(f'individual_metrics_{base_name}.parquet', index=False)
    print(f"Results saved as individual_metrics_{base_name}.xlsx and .parquet")
    
    return results_df

MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
SAVE_TO_PATH = 'FEATURES/MAPELITES/INDIVIDUALS/'

# Get all parquet files in MAPELITES_PATH
import glob
import os

# Create save directory if it doesn't exist
os.makedirs(SAVE_TO_PATH, exist_ok=True)

# Process each parquet file
for file_path in glob.glob(MAPELITES_PATH + "*.parquet"):
    print(f"Processing {file_path}")
    try:
        # Compute individuals and save to SAVE_TO_PATH
        results_df = compute_individuals(file_path)
        
        # Extract filename components for saving
        base_name = file_path.split('/')[-1].replace('extracted_features_', '').replace('.parquet', '')
        
        # Save results to SAVE_TO_PATH
        results_df.to_excel(SAVE_TO_PATH + f'individual_metrics_{base_name}.xlsx', index=False)
        results_df.to_parquet(SAVE_TO_PATH + f'individual_metrics_{base_name}.parquet', index=False)
        print(f"Saved results for {base_name}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
