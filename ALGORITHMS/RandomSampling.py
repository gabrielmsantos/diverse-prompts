from IndividualFactory import IndividualFactory
from ALGORITHMS.BaseOptimizer import BaseOptimizer

class RandomSampling(BaseOptimizer):
    def __init__(self, cfg_path, task_loader, population_size, num_iterations, evaluator, num_evaluations, prompt_log_filename='random_sampling_log.txt', extractor=None):
        super().__init__(task_loader, evaluator, num_evaluations, prompt_log_filename, extractor)
        self.factory = IndividualFactory(cfg_path)
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.results = []

    def run(self):
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}")
            
            # Generate random population
            population = self.factory.create_new_population(self.population_size)
            
            # Evaluate each individual
            for individual in population:
                fitness = self._get_individual_fitness(individual)
                self.results.append({
                    'individual': individual,
                    'fitness': fitness
                })
                
        # Sort results by fitness
        self.results.sort(key=lambda x: x['fitness'], reverse=True)
        return self.results
