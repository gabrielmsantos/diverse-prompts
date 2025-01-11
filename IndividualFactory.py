# The factory creates individuals based on the CFG
import random
from CFGLoader import ContextFreeGrammar
from Individual import Individual
from TaskLoader import TaskLoader


class IndividualFactory:
        
    def __init__(self, yaml_file):
        self.cfg = ContextFreeGrammar()
        self.cfg.load_cfg(yaml_file)

    def create_empty_individual(self):
        return Individual(None, None)
    
    def create_new_individual(self):
        """
        Generate a random individual from the CFG based on the given symbol.
        Returns the genotype as a list of symbols+number where number is the index of the rule used to derive,
        and the prompt structure (the result of derivation) without non-terminal symbols.
        """
        genotype = []
        prompt_structure = []

        def derive(current_symbol):
            if current_symbol not in self.cfg.cfg_rules:
                # If the symbol is not in the CFG, it must be a terminal, so add it to the prompt structure
                prompt_structure.append(current_symbol)
                return

            # Choose a random rule for the symbol
            rule_index = random.randint(0, len(self.cfg.cfg_rules[current_symbol]) - 1)
            
            # Add the current symbol and rule index to the genotype
            genotype.append(f"{current_symbol}{rule_index}")

            rule = self.cfg.cfg_rules[current_symbol][rule_index]

            # Derive each part of the rule recursively
            for part in rule.split():
                if part.startswith('<') and part.endswith('>'):
                    derive(part)
                else:
                    prompt_structure.append(part)

        # Start the derivation process
        derive(self.cfg.start_symbol)
        
        return Individual(' '.join(prompt_structure), genotype)
    
    def create_prompt_structure_from_genotype(self, genotype):
        """
        Create a prompt structure from a given genotype.
        Returns the generated prompt structure.
        """
        result = ''

        for gene in genotype:
            symbol, rule_index = gene.split('>')
            symbol = '<' + symbol.strip('<') + '>'
            
            rule = self.cfg.cfg_rules[symbol][int(rule_index)]
            
            if not rule.startswith('<'):
                if symbol in result:
                    result = result.replace(symbol, rule, 1)
                else:
                    result += rule + ' '

        return result.strip()
    
    def create_new_from_genotype(self, genotype):
        """
        Create a new individual from a given genotype.
        Returns the generated individual.
        """
        prompt_structure = self.create_prompt_structure_from_genotype(genotype)
        # Return the prompt structure
        return Individual(prompt_structure, genotype)
    
    def mutate_genotype(self, genotype, mutation_rate_examples, mutation_rate_nsbs):
        # Decide whether to mutate the number of examples based on the mutation rate
        if random.uniform(0, 1) < mutation_rate_examples:
            # Count the current number of examples
            current_examples = sum(1 for gene in genotype if gene.startswith('<example'))
            # Choose a new number of examples between 0 and 10
            new_examples = random.randint(0, 10)

            # Find the index of '<examples>' gene
            examples_index = next((i for i, gene in enumerate(genotype) if gene.startswith('<examples>')), None)
            print(f"Examples index: {examples_index}")
            # If examples_index is None, find the index for <req>
            if examples_index is None:
                req_index = next((i for i, gene in enumerate(genotype) if gene.startswith('<req>')), None)
                if req_index is not None:
                    # Randomly choose whether to insert before or after <req>
                    examples_index = req_index if random.choice([True, False]) else req_index + 1
                else:
                    # If <req> is not found, append to the end
                    examples_index = len(genotype)
            # Remove all existing example genes
            genotype = [gene for gene in genotype if not gene.startswith('<example')]
            print(f"Current examples: {current_examples}, New examples: {new_examples}")
            # If new_examples is 0, remove the '<examples>' gene
            if new_examples == 0:
                genotype = [gene for gene in genotype if not gene.startswith('<examples>')]
            else:
                # Update the '<examples>' gene
                genotype.insert(examples_index, '<examples>1' if new_examples > 1 else '<examples>0')
                # Add new example genes
                for i in range(new_examples - 1):
                    genotype.insert(examples_index + 1 + i, '<example>1')
                if new_examples > 1:
                    genotype.insert(examples_index + new_examples, '<example>0')


        # mutate the number of steps (COT depth)
        if random.uniform(0, 1) < mutation_rate_nsbs:
            # Get the index of <nsbs>, if there is no nsbs get the index of <base>
            nsbs_index = next((i for i, gene in enumerate(genotype) if gene.startswith('<nsbs>')), None)
            if nsbs_index is None:
                nsbs_index = next((i for i, gene in enumerate(genotype) if gene.startswith('<base>')), None)
            
            # If neither <nsbs> nor <base> is found, set index to the end of the genotype
            if nsbs_index is None:
                nsbs_index = len(genotype)
            
            # Remove all <nsbs> and <number> genes
            genotype = [gene for gene in genotype if not gene.startswith('<nsbs>') and not '<number>' in gene]

            # Add a new <nsbs> gene, or just keep it without inserting anything
            if random.uniform(0, 1) < 0.9:
                # Add a new <nsbs> gene
                nsbs_options = self.cfg.cfg_rules['<nsbs>']
                new_nsbs = '<nsbs>' + str(random.randint(0, len(nsbs_options) - 1))
                # Get the value of the new nsbs from cfg
                nsbs_number = int(new_nsbs.split('>')[1])
                new_nsbs_value = self.cfg.cfg_rules['<nsbs>'][nsbs_number]
                
                genotype.insert(nsbs_index, new_nsbs)

                if '<number>' in new_nsbs_value:
                    number_options = self.cfg.cfg_rules['<number>']
                    number = random.randint(0, len(number_options) - 1)
                    genotype.insert(nsbs_index+1, '<number>' + str(number))


        return genotype
    
    # create a list of genotypes receiving a number of individuals
    def create_new_population(self, num_individuals):
        return [self.create_new_individual() for _ in range(num_individuals)]

def main():
    # Example usage:
    # Create an instance of the IndividualFactory class
    factory = IndividualFactory('CFG/cfg.yaml')

    # Generate 10 individuals
    individuals = [factory.create_new_individual() for _ in range(2000)]

    # Example usage:
    file_path = 'DATASETS/winowhy.json'
    task_loader = TaskLoader(file_path)



    for i, individual in enumerate(individuals):
        print(f"Individual {i+1}:")
        print(f"Original Genotype: {individual.genotype}")
        mutated_genotype = factory.mutate_genotype(individual.genotype, 0.9, 0.9)

        print(f"Mutated Genotype: {mutated_genotype}")
        individual_from_genotype = factory.create_new_from_genotype(mutated_genotype)

        # Check if the last element of mutated_genotype is not '<input>0'
        if mutated_genotype[-1] != '<input>0':
            print("******* Warning: Last element of mutated genotype is not '<input>0'")
            print(f"Last element: {mutated_genotype[-1]}")
            exit()
        
        # Check if the last word in mutated_prompt_instance is '[[INPUT]]'
        if not individual_from_genotype.prompt_structure.strip().endswith('[[INPUT]]'):
            print("******* Warning: Last word in mutated prompt instance is not '[[INPUT]]'")
            print()
            print(f"Mutated Genotype: {individual_from_genotype.genotype}")
            print(f"Mutated prompt structure: {individual_from_genotype.prompt_structure}")
            print(f"Last word: {individual_from_genotype.prompt_structure.strip().split()[-1]}")
            exit()

        print()  # Add an empty line for better readability

    # Get one individual randomly
    random_individual = random.choice(individuals)

    # Get a random input from the TaskLoader matrix
    random_input = random.choice(task_loader.matrix)[1] # Get a random row
    random_input_question = random_input[0]

    # Create prompt instance for the random individual
    prompt_instance = random_individual.create_prompt_instance(task_loader)

    # Substitute [[INPUT]] with the random input
    prompt_with_input = prompt_instance.replace('[[INPUT]]', random_input_question + '<eos>### Answer:', 1)

    print("Randomly selected individual:")
    print(f"Genotype: {random_individual.genotype}")
    print(f"Phenotype: {random_individual.phenotype}")
    print(f"Prompt structure: {random_individual.prompt_structure}")
    print(f"Random input:")
    print(random_input)
    print(f"Prompt instance with random input:")
    print(prompt_with_input)
    print()
    print("END")


if __name__ == "__main__":
    main()
