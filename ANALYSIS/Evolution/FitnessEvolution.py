import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

MAPELITES_PATH = 'FEATURES/MAPELITES/INDIVIDUALS/'

datasets = [
    'formal_fallacies_syllogisms_negation',
    'logical-deduction-3', 
    'strange_stories_boolean',
    'strategyqa',
    'winowhy',
    'known_unknowns',
    'play_dialog_same_or_different'
]

def plot_fitness_evolution_by_dataset(dataset_files, dataset_name):
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'red', 'green', 'purple']
    
    for idx, (model, file) in enumerate(dataset_files.items()):
        # Read parquet file
        df = pd.read_parquet(file)
        
        # Calculate number of iterations (every 50 individuals)
        total_individuals = len(df)
        iterations = total_individuals // 50
        
        # Initialize arrays to store metrics
        best_fitness = []
        mean_fitness = []
        fitness_variance = []
        
        # Calculate metrics for each iteration
        for i in range(iterations):
            start_idx = i * 50
            end_idx = start_idx + 50
            iteration_data = df.iloc[start_idx:end_idx]
            
            best_fitness.append(iteration_data['Performance'].max())
            mean_fitness.append(iteration_data['Performance'].mean())
            fitness_variance.append(iteration_data['Performance'].std())

        # Create x-axis points
        x = range(iterations)
        
        # Plot best fitness
        plt.plot(x, best_fitness, '-', label=f'{model} (best)', color=colors[idx])
        
        # Plot mean fitness with variance bands
        plt.plot(x, mean_fitness, '--', label=f'{model} (mean)', color=colors[idx], alpha=0.5)
        plt.fill_between(x, 
                        [m - v for m, v in zip(mean_fitness, fitness_variance)],
                        [m + v for m, v in zip(mean_fitness, fitness_variance)],
                        alpha=0.1, color=colors[idx])
    
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Evolution - {dataset_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'fitness_evolution_{dataset_name}.png', bbox_inches='tight')
    plt.close()

# Group files by dataset
dataset_groups = defaultdict(dict)
for file in os.listdir(MAPELITES_PATH):
    if file.endswith('.parquet'):
        for dataset in datasets:
            if dataset in file:
                # Extract model name from filename
                model = file.split('_')[2]
                dataset_groups[dataset][model] = os.path.join(MAPELITES_PATH, file)

# Create one plot per dataset
for dataset, files in dataset_groups.items():
    plot_fitness_evolution_by_dataset(files, dataset)
