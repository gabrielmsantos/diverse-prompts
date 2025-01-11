import json
import random


class TaskLoader:
    def __init__(self, file_path, task_prefix='', answer_prefix='### ANSWER:'):
        self.answer_prefix = answer_prefix
        # Load JSON data from a file
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        self.input_matrix, self.instruction_matrix = self._create_matrixes()
        self._set_ask()
        self.task_prefix = self.data.get("task_prefix", task_prefix)


    def get_task_prefix(self):
        # Access task_prefix
        return self.task_prefix

    def get_random_example(self):
        # Randomly select an example from the input matrix
        random_index = random.randint(0, len(self.input_matrix) - 1)
        index, (example_input, options, correct_score) = self.input_matrix[random_index]

        # Get the corresponding instruction from the instruction matrix
        instruction = self.instruction_matrix[random_index][1]

        # Format the example as requested
        return f"EXAMPLE:Input: {example_input} {instruction}{self.answer_prefix} {correct_score}."
    
    def _create_instruction(self, targets):
        """
        Create an instruction for a language model to answer by transforming the answer using the indexes in the array.
        
        :param targets: List of possible targets (e.g., ["A", "B", "C"])
        :return: Instruction string for the language model
        """
        instruction = "Please determine the correct answer from the following options: "
        for index, target in enumerate(targets):
            instruction += f"{index}: {target}; "
        #instruction += ". If you think the correct answer is the first option, output 0. If you think it is the second option, output 1, and so on. Only output the number corresponding to your choice (e.g., 0, 1, 2)."
        return instruction 

    def _set_ask(self):
        self.ask = ''
        if self.input_matrix and self.input_matrix[0][1][1]:
            self.ask = ". If you think the correct answer is the first option, output 0. If you think it is the second option, output 1, and so on. Only output the number corresponding to your choice (e.g., 0, 1, 2). "
        self.ask += self.answer_prefix

    def _create_matrixes(self):
        # Create a matrix with (index, (input, answer)) and an instruction matrix
        matrix = []
        instruction_matrix = []
        for index, example in enumerate(self.data['examples']):
            input_text = example['input']
            try:
                target = example['target_scores']
            except KeyError:
                target = example['target']
            
            if isinstance(target, dict):
                # Vectorize the options and find the index of the answer with value 1
                options = list(target.keys())
                answer = options.index(next(key for key, value in target.items() if value == 1))
                instruction = self._create_instruction(options)
            else:
                # If target is not a dictionary, the options list should be empty and answer is the target
                options = []
                answer = target
                instruction = '.'
                
            instruction_matrix.append((index, instruction))
            matrix.append((index, (input_text, options, answer)))

        
        assert len(matrix) == len(instruction_matrix), "Matrix and instruction matrix must be of the same length"
        return matrix, instruction_matrix


# ==== Tests ====

def test_matrix_creation():
    # Path to the dataset file
    file_path = 'DATASETS/formal_fallacies_syllogisms_negation.json'
    
    # Create an instance of TaskLoader
    task_loader = TaskLoader(file_path)
    
    # Print 10 random items from the matrix to verify its creation
    random_items = random.sample(task_loader.matrix, 10)
    for item in random_items:
        index, (input_text, options, answer) = item
        instruction = task_loader._create_instruction(options)
        concatenated_output = f"{input_text} {instruction}"
        print(f"Index: {index}")
        print(f"Concatenated Input and Instruction: {concatenated_output}")
        print(f"Options: {options}")
        print(f"Answer: {answer}")
        print("#" * 56)

def test_get_random_example():
    # Path to the dataset file
    file_path = 'DATASETS/operators.json'
    
    # Create an instance of TaskLoader
    task_loader = TaskLoader(file_path)
    
    # Print 10 random examples to verify the method
    for _ in range(10):
        example = task_loader.get_random_example()
        print('-' * 56)
        print(example + '\n')

# Call the test function
# test_get_random_example()





