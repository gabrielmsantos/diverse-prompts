import os
from dotenv import load_dotenv
from TaskLoader import TaskLoader
from LangEvaluator import APIModelEvaluator
from ALGORITHMS.MAPElites import MAPElite
from ALGORITHMS.RandomSampling import RandomSampling
from Utils import save_chart_bins_4d, save_result_bins_to_file, save_result_bins_to_json, get_extractor_function

load_dotenv()  # Load environment variables from .env file

def main(model_name, dataset_name, extractor, max_tokens=3):  
    cfg_path = 'CFG/cfg.yaml'
    task_path = f'DATASETS/{dataset_name}.json'

    # Set your API key here
    api_key = os.getenv('HUGGINGFACE_TOKEN')

    # Define the endpoint based on the model name
    model_urls = {
        'starling-lm-7b-alpha-bii': 'https://j7tcbfc9xu94illk.us-east-1.aws.endpoints.huggingface.cloud',
        'llama-3-1-8b-instruct-lvt': 'https://v13wnb8a27o2m59g.us-east-1.aws.endpoints.huggingface.cloud',
        'phi-3-5-mini-instruct-ivr': 'https://r4q42u9sqyx8lt30.us-east-1.aws.endpoints.huggingface.cloud',
        'qwen2-5-7b-instruct-mln': 'https://ml5lrnuncwvqnfl7.us-east-1.aws.endpoints.huggingface.cloud'
    }
    url = model_urls.get(model_name, 'https://r4q42u9sqyx8lt30.us-east-1.aws.endpoints.huggingface.cloud')  # Default to Phi-3.5-Mini if not found

    data_template = {
        "inputs": "",
        "stream": False,
        "parameters": {
            "max_new_tokens": max_tokens
        }
    }

    # Create an instance of APIModelEvaluator
    api_evaluator = APIModelEvaluator(api_key, url, data_template)

    # Create a default task_prefix
    default_task_prefix = "Read the given context and determine the most appropriate answer to the question based on the information provided."
    # Create an instance of TaskLoader
    task_loader = TaskLoader(task_path, default_task_prefix)

    # Run MAP-Elites algorithm
    map_elites_results = None
    # map_elites_results = run_map_elites(model_name, dataset_name, extractor, max_tokens, cfg_path, task_loader, api_evaluator)

    # Run Random Sampling algorithm 
    random_sampling_results = run_random_sampling(model_name, dataset_name, extractor, cfg_path, task_loader, api_evaluator)

    return map_elites_results, random_sampling_results


def run_map_elites(model_name, dataset_name, extractor, cfg_path, task_loader, api_evaluator):
    """
    Test the MAP-Elites algorithm with given parameters
    """
    population_size = 50
    num_iterations = 10
    mutation_percentage = 0.4
    num_evaluations = 50
    bin_sizes = (2, 25, 2)  # Example bin sizes for n_examples, size_w, and n_steps
    
    # Create a common file name based on model and dataset
    common_filename = f'results_mapelites_{model_name}_{dataset_name}'

    map_elite = MAPElite(
        cfg_path, 
        task_loader, 
        population_size, 
        num_iterations, 
        evaluator=api_evaluator,
        bin_sizes=bin_sizes,
        mutation_percentage=mutation_percentage,
        num_evaluations=num_evaluations,
        prompt_log_filename=f'{common_filename}_prompt_log.txt',
        extractor=get_extractor_function(extractor)
    )
    
    result_bins = map_elite.run()

    # Save results
    save_chart_bins_4d(result_bins, filename=f'{common_filename}_plot.png')
    save_result_bins_to_file(result_bins, filename=f'{common_filename}_bins.txt')
    save_result_bins_to_json(result_bins, filename=f'{common_filename}_bins.json')

    print(f"Number of individuals in result_bins: {len(result_bins)}")
    return result_bins

def run_random_sampling(model_name, dataset_name, extractor, cfg_path, task_loader, api_evaluator):
    """
    Test the Random Sampling algorithm with given parameters
    """
    population_size = 50
    num_iterations = 10
    num_evaluations = 50

    # Create a common file name based on model and dataset
    common_filename = f'results_random_{model_name}_{dataset_name}'

    random_sampling = RandomSampling(
        cfg_path,
        task_loader,
        population_size,
        num_iterations,
        evaluator=api_evaluator,
        num_evaluations=num_evaluations,
        prompt_log_filename=f'{common_filename}_prompt_log.txt',
        extractor=get_extractor_function(extractor)
    )

    results = random_sampling.run()

    # Save results
    with open(f'{common_filename}_results.txt', 'w') as f:
        for idx, result in enumerate(results):
            f.write(f"Individual {idx}:\n")
            f.write(f"Fitness: {result['fitness']}\n")
            f.write(f"Genotype: {result['individual'].genotype}\n")
            f.write("-" * 50 + "\n")

    # Get the best individual
    best_result = results[0] if results else None
    if best_result:
        print(f"Best fitness achieved: {best_result['fitness']}")
        print(f"Best individual genotype: {best_result['individual'].genotype}")
    
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python Main.py <ModelName> <DatasetName> [Extractor] [MaxTokens]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    extractor = sys.argv[3] if len(sys.argv) > 3 else None
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    main(model_name, dataset_name, extractor, max_tokens)