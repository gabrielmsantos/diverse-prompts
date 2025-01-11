import matplotlib.pyplot as plt
import json
from collections import defaultdict
import ast  # Safer alternative to eval for parsing literals
from Individual import Individual  # Ensure you import the Individual class
from IndividualFactory import IndividualFactory



def save_chart_bins_4d(result_bins, filename='plot.png'):  # Added filename parameter
    # Prepare data for plotting
    x_values = []
    y_values = []
    z_values = []
    fitness_values = []
    has_context_values = []

    for bin_index, bin_data in result_bins.items():
        if bin_data['individual'] is not None:
            # Assuming phenotype is a dictionary with keys 'n_examples', 'size_w', 'n_steps', and 'has_context'
            phenotype = bin_data['individual'].phenotype
            x_values.append(phenotype['n_examples'])  # First dimension
            y_values.append(phenotype['size_w'])      # Second dimension
            z_values.append(phenotype['n_steps'])      # Third dimension
            fitness_values.append(bin_data['fitness'])
            has_context_values.append(phenotype['has_context'])  # Fourth dimension

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize fitness values for color mapping
    norm = plt.Normalize(vmin=min(fitness_values), vmax=max(fitness_values))
    cmap = plt.get_cmap('coolwarm')

    # Scatter plot with different markers
    for i in range(len(x_values)):
        marker = 'o' if not has_context_values[i] else 's'  # Circle for False, Square for True
        ax.scatter(x_values[i], y_values[i], z_values[i], 
                   c=fitness_values[i], cmap=cmap, norm=norm, 
                   edgecolor='black', s=75, marker=marker)

    # Add color bar
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Fitness')

    # Set labels
    ax.set_xlabel('Number of Examples (n_examples)')
    ax.set_ylabel('Number of Words (size_w)')
    ax.set_zlabel('Number of Steps (n_steps)')
    ax.set_title('3D Fitness Distribution Across Phenotype Dimensions with Context')

    plt.savefig(filename)  # Save the plot to a file


def plot_bins_3d(result_bins):
    # Prepare data for plotting
    x_values = []
    y_values = []
    z_values = []
    fitness_values = []

    for bin_index, bin_data in result_bins.items():
        if bin_data['individual'] is not None:
            # Assuming phenotype is a dictionary with keys 'n_examples', 'size_w', 'n_steps'
            phenotype = bin_data['individual'].phenotype
            x_values.append(phenotype['n_examples'])  # First dimension
            y_values.append(phenotype['size_w'])      # Second dimension
            z_values.append(phenotype['n_steps'])      # Third dimension
            fitness_values.append(bin_data['fitness'])

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize fitness values for color mapping
    norm = plt.Normalize(vmin=min(fitness_values), vmax=max(fitness_values))
    cmap = plt.get_cmap('coolwarm')

    # Scatter plot
    scatter = ax.scatter(x_values, y_values, z_values, c=fitness_values, cmap=cmap, norm=norm, edgecolor='black', s=75)

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness')

    # Set labels
    ax.set_xlabel('Number of Examples (n_examples)')
    ax.set_ylabel('Number of Words (size_w)')
    ax.set_zlabel('Number of Steps (n_steps)')
    ax.set_title('3D Fitness Distribution Across Phenotype Dimensions')

    plt.show()

    # Save result_bins to a JSON file
def save_result_bins_to_json(result_bins, filename='result_bins.json'):
    try:
        result_bins = {str(key): value for key, value in result_bins.items()}  # Convert tuple keys to strings
        with open(filename, 'w') as json_file:
            json.dump(result_bins, json_file, default=lambda o: o.__dict__, indent=4)
    except Exception as e:
        print(f"Error saving result_bins to JSON: {e}")

# Print results
def save_result_bins_to_file(result_bins, filename='result_bins.txt'):
    try:
        with open(filename, 'w') as f:
            for bin_index, bin_data in result_bins.items():
                f.write(f"Bin {bin_index}:\n")
                f.write(f"  Fitness: {bin_data['fitness']}\n")
                f.write(f"  Phenotype: {bin_data['individual'].phenotype}\n")
                f.write(f"  Prompt Instance: {bin_data['individual'].prompt_instance}\n")
                f.write(f"  Genotype: {bin_data['individual'].genotype}\n")
                f.write("\n")
    except Exception as e:
        print(f"Error saving result_bins to file: {e}")

def create_bins_from_file(file_path, ind_factory, bin_sizes):
    # Initialize the defaultdict
    bins = defaultdict(lambda: {'individual': None, 'fitness': float('-inf')})
    
    with open(file_path, 'r') as f:
        current_bin_index = None
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            if line.startswith("Bin"):
                try:
                    # Extract bin index as a tuple
                    index_part = line.split(' ', 1)[1].strip(':').strip('()')
                    current_bin_index = tuple(map(int, index_part.split(',')))  # Create the tuple directly from the split
                    bins[current_bin_index]['fitness'] = None  # Initialize fitness
                    bins[current_bin_index]['individual'] = ind_factory.create_empty_individual()  # Initialize individual data
                except ValueError as e:
                    print(f"Error processing line: {line}. Exception: {e}")
                    continue  # Skip this line if there's an error
            
            elif line.startswith("Fitness:"):
                if current_bin_index is not None:
                    bins[current_bin_index]['fitness'] = float(line.split(":")[1].strip())  # Extract fitness
            
            elif line.startswith("Phenotype:"):
                if current_bin_index is not None:
                    phenotype_str = line.split(":", 1)[1].strip()
                    phenotype = ast.literal_eval(phenotype_str)  # Use ast.literal_eval for safety
                    bins[current_bin_index]['individual'].phenotype = phenotype  # Store phenotype
                    
                    # Calculate the bin index using the get_bin_index logic
                    # bin_index = get_bin_index(phenotype, bin_sizes)
                    #bins[current_bin_index]['individual'] = {'phenotype': phenotype}  # Store the individual in the correct bin
            
            elif line.startswith("Prompt Instance:"):
                if current_bin_index is not None:
                    bins[current_bin_index]['individual'].prompt_instance = line.split(":", 1)[1].strip()  # Extract prompt instance
            
            elif line.startswith("Genotype:"):
                if current_bin_index is not None:
                    genotype_str = line.split(":")[1].strip()
                    bins[current_bin_index]['individual'].genotype = ast.literal_eval(genotype_str)  # Use ast.literal_eval for safety
                    bins[current_bin_index]['individual'].prompt_structure = ind_factory.create_prompt_structure_from_genotype(bins[current_bin_index]['individual'].genotype)

    return bins

def get_bin_index(phenotype, bin_sizes):
    """
    Determine the bin index for a phenotype based on specified bin sizes for each dimension.
    
    :param phenotype: A dictionary containing the dimensions of the phenotype.
    :param bin_sizes: A tuple representing the sizes of the bins for each dimension.
    :return: A tuple representing the bin index for each dimension.
    """
    # Calculate the bin index for each dimension based on the specified bin sizes
    bin_indices = []
    for dimension, bin_size in zip(phenotype.values(), bin_sizes):
        bin_index = min(int(dimension) // bin_size, bin_size - 1)
        bin_indices.append(bin_index)

    return tuple(bin_indices)  # Return as a tuple for unique identification


def extract_answer_multiple_choice(response, prefix, correct_answer):
    if response is None:
        return None
    
    # Extract the first word after " ### Answer:"
    response_parts = response.split(prefix)
    if len(response_parts) > 1:
        stripped_response = response_parts[1].strip()
        response = stripped_response.split()[0] if stripped_response else None

    # Check if the response is valid
    if response is not None:
        digit_str = ''
        for char in response:
            if char.isdigit():
                digit_str += char
            elif digit_str:
                break

        try:
            response = int(digit_str) if digit_str else None
        except ValueError:
            print(f"Invalid response: {response}. Expected a number.")
            response = None

    return response

def extract_answer_single_choice(response, prefix, correct_answer):
    if response is None:
        return None
    
    # Extract the first word after " ### Answer:"
    response_parts = response.split(prefix)
    if len(response_parts) > 1:
        response = response_parts[1].strip()

    import re

    patterns = [
        r"= (\d+)",
        r"is (\d+)",
        r"results in (\d+)",
        r"result: (\d+)",
        r"result of (\d+)",
        r"yields (\d+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            #response = match.group(1)
            return correct_answer
  
    return None


def get_extractor_function(extractor_name):
    if extractor_name == 'single':
        print('Using single choice extractor')
        return extract_answer_single_choice
    else:
        print('Using multiple choice extractor')
        return extract_answer_multiple_choice

