from TaskLoader import TaskLoader  # Import TaskLoader at the beginning of the file

# Create an individual class that will hold the genotype, phenotype, and score
class Individual:
    def __init__(self, prompt_structure, genotype):
        self.prompt_instance = None
        self.genotype = genotype
        self.prompt_structure = prompt_structure
        self.phenotype = None
        self.score = 0

    #def compute_phenotype(self):
        # n_examples = sum(1 for gene in self.genotype if gene.startswith('<example'))
        # size_w = len(self.prompt_instance.split())
        # size_c = len(self.prompt_instance)
        # n_steps = 0  # Default value if there's no <nsbs>
        # for gene in self.genotype:
        #     if gene.startswith('<nsbs>'):
        #         n_steps = 1
        #         break
        # for gene in self.genotype:
        #     if '<number>' in gene:
        #         number_str = gene.split('<number>')[1].split(',')[0]
        #         n_steps = max(n_steps, int(number_str) + 1)
        # has_context = any(gene.startswith('<context>') for gene in self.genotype)

        # return {
        #     'n_examples': n_examples,
        #     'size_w': size_w,
        #     'n_steps': n_steps,
        #     'has_context': has_context
        #     #'size_c': size_c,
        # }
        #pass

    # New method to substitute [[EXAMPLE]] with a random example from TaskLoader
    def create_prompt_instance(self, task_loader):
        """
        Substitute all [[EXAMPLE]]s in the prompt with random examples from TaskLoader.
        """
        self.prompt_instance = self.prompt_structure
        while '[[EXAMPLE]]' in self.prompt_instance:
            random_example = task_loader.get_random_example()  # Get a random example from TaskLoader
            self.prompt_instance = self.prompt_instance.replace('[[EXAMPLE]]', random_example, 1)
        
        # self.phenotype = self.compute_phenotype()

        return self.prompt_instance
    
    def _compute_ttr(self, text):
        # Tokenize the text by splitting on whitespace
        tokens = text.split()
        
        # Count total tokens
        total_tokens = len(tokens)
        
        # Count unique tokens using a set
        unique_tokens = len(set(tokens))
        
        # Calculate Type-Token Ratio
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        return ttr

    def compute_phenotype(self):
        """
        Create the phenotype from the genotype and prompt_instance.
        """

        if self.prompt_instance is None:
            return None
        # Basic features
        n_examples = sum(1 for gene in self.genotype if gene.startswith('<example'))
        
        # Size features for prompt_instance
        instance_size_w = len(self.prompt_instance.split())
        instance_size_c = len(self.prompt_instance)
        
        # Step-related features
        n_steps = 0  # Default value if there's no <nsbs>
        for gene in self.genotype:
            if gene.startswith('<nsbs>'):
                n_steps = 1
                break
        for gene in self.genotype:
            if '<number>' in gene:
                number_str = gene.split('<number>')[1].split(',')[0]
                n_steps = max(n_steps, int(number_str) + 1)
                
        # Context features        
        has_context = any(gene.startswith('<context>') for gene in self.genotype)
        
        # Count special tokens
        n_special_tokens = sum(1 for gene in self.genotype if gene.startswith('<'))
        
        # Calculate Type-Token Ratio for prompt and instance
        prompt_ttr = self._compute_ttr(self.prompt_instance)

        return {
            'num_examples': n_examples,
            'instance_word_count': instance_size_w,
            'instance_char_count': instance_size_c,
            'num_steps': n_steps,
            'includes_context': has_context,
            'num_special_tokens': n_special_tokens,
            'prompt_type_token_ratio': prompt_ttr
        }
