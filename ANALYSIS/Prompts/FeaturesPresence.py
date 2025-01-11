import pandas as pd
import numpy as np
import glob
from collections import defaultdict

MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'

def analyze_features_presence(df, bin_sizes=(2, 25, 2)):
    """
    Analyze presence of prompt features in best individuals found by MAP-Elites
    """
    # Define phenotype ranges for binning
    num_examples_bins = np.linspace(df['num_examples'].min(), df['num_examples'].max(), bin_sizes[0])
    word_count_bins = np.linspace(df['prompt_word_count'].min(), df['prompt_word_count'].max(), bin_sizes[1]) 
    steps_bins = np.linspace(df['num_steps'].min(), df['num_steps'].max(), bin_sizes[2])

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
                examples_bin = np.digitize(df.loc[idx-1, 'num_examples'], num_examples_bins) - 1
                words_bin = np.digitize(df.loc[idx-1, 'prompt_word_count'], word_count_bins) - 1
                steps_bin = np.digitize(df.loc[idx-1, 'num_steps'], steps_bins) - 1
                bin_key = (examples_bin, words_bin, steps_bin)
                
                # Update elite map if better performance
                if performance > elite_map[bin_key]['performance']:
                    elite_map[bin_key] = {
                        'performance': performance,
                        'individual': df.loc[idx-1].to_dict(),
                    }
            
            # Reset counters for new genotype
            current_genotype = row['genotype']
            current_correct = int(row['model_formatted_output'] == row['correct'])
            current_tests = 1
        else:
            # Continue counting for current genotype
            current_correct += int(row['model_formatted_output'] == row['correct'])
            current_tests += 1

    # Get best individuals
    best_individuals = [elite['individual'] for elite in elite_map.values() if elite['individual'] is not None]
    
    if not best_individuals:
        return None

    # Calculate feature presence percentages
    has_context = sum(1 for ind in best_individuals if ind['includes_context']) / len(best_individuals)
    has_examples = sum(1 for ind in best_individuals if ind['num_examples'] >= 1) / len(best_individuals)
    has_steps = sum(1 for ind in best_individuals if ind['num_steps'] > 1) / len(best_individuals)

    return {
        'has_context': has_context,
        'has_examples': has_examples, 
        'has_steps': has_steps
    }

# Get all unique datasets and models
results_data = []
columns = ['Model', 'Dataset', 'Has Context %', 'Has Examples %', 'Has Steps %']
parquet_files = glob.glob(MAPELITES_PATH + '*.parquet')

for file_path in parquet_files:
    # Extract dataset and model name from filename
    file_name = file_path.split('/')[-1].replace('extracted_features_', '').replace('.parquet', '')
    model = file_name.split('_')[0]
    dataset = '_'.join(file_name.split('_')[1:])
    
    # Load and analyze file
    df = pd.read_parquet(file_path)
    results = analyze_features_presence(df)
    
    if results:
        row = [
            model,
            dataset,
            f"{results['has_context']*100:.1f}",
            f"{results['has_examples']*100:.1f}",
            f"{results['has_steps']*100:.1f}"
        ]
        results_data.append(row)

# Create DataFrame and save to Excel
results_df = pd.DataFrame(results_data, columns=columns)
results_df.to_excel('features_presence_analysis.xlsx', index=False)
print("Results saved to features_presence_analysis.xlsx")
