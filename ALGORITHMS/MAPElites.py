import random
from collections import defaultdict
from IndividualFactory import IndividualFactory
from ALGORITHMS.BaseOptimizer import BaseOptimizer


class MAPElite(BaseOptimizer):
    def __init__(self, cfg_path, task_loader, population_size, num_iterations, evaluator, bin_sizes, mutation_percentage, num_evaluations, prompt_log_filename='prompt_log.txt', extractor=None):
        super().__init__(cfg_path, task_loader, evaluator, num_evaluations, prompt_log_filename, extractor)
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.bins = defaultdict(lambda: {'individual': None, 'fitness': float('-inf')})
        self.bin_sizes = bin_sizes
        self.mutation_percentage = mutation_percentage
        self.factory = IndividualFactory(cfg_path)
        self.mutation_chance = 0.4

    def run(self):
        # Initialize population
        population = self.factory.create_new_population(self.population_size)

        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}")
            for individual in population:
                fitness = self._get_individual_fitness(individual)
                phenotype = individual.compute_phenotype()
                bin_index = self.get_bin_index(phenotype)

                # Fill the archive
                if fitness > self.bins[bin_index]['fitness']:
                    self.bins[bin_index] = {'individual': individual, 'fitness': fitness}

            # Generate new individuals
            new_population = []
            num_mutated = int(self.population_size * self.mutation_percentage)
            num_new = self.population_size - num_mutated

            # Generate mutated individuals from the archive
            for _ in range(num_mutated):
                parent = random.choice(list(self.bins.values()))['individual']
                mutated_genotype = self.factory.mutate_genotype(parent.genotype, self.mutation_chance, self.mutation_chance)
                child = self.factory.create_new_from_genotype(mutated_genotype)
                print(f"Child: {child}")
                new_population.append(child)

            # Generate new individuals from the factory
            new_population.extend(self.factory.create_new_population(num_new))

            population = new_population

        return self.bins

    def get_bin_index(self, phenotype):
        """
        Determine the bin index for a phenotype based on specified bin sizes for each dimension.
        
        :param phenotype: A dictionary containing the dimensions of the phenotype.
        :return: A tuple representing the bin index for each dimension.
        """
        # Calculate the bin index for each dimension based on the specified bin sizes
        bin_indices = []
        for dimension, bin_size in zip(phenotype.values(), self.bin_sizes):
            bin_index = min(int(dimension) // bin_size, bin_size - 1)
            bin_indices.append(bin_index)

        return tuple(bin_indices)  # Return as a tuple for unique identification

