import random

class BaseOptimizer:
    def __init__(self, task_loader, evaluator, num_evaluations, prompt_log_filename, extractor):
        self.task_loader = task_loader
        self.evaluator = evaluator
        self.num_evaluations = num_evaluations
        self.prompt_log_filename = prompt_log_filename
        self.extractor = extractor

    def _get_individual_fitness(self, individual):
        correct_answers = 0
        total_questions = min(self.num_evaluations, len(self.task_loader.input_matrix))

        prompt_instance = individual.create_prompt_instance(self.task_loader)
        error_count = 0
        # Open the file outside the loop
        with open(self.prompt_log_filename, 'a') as f:
            for _ in range(total_questions):
                # Randomly select an index from the input matrix
                input_index = random.randint(0, len(self.task_loader.input_matrix) - 1)
                input_text, options, correct_answer = self.task_loader.input_matrix[input_index][1]
                instruction = self.task_loader.instruction_matrix[input_index][1]
                input_text = f"QUESTION: {input_text}"
                
                f.write(f"G: {individual.genotype}\n") #GENOTYPE
                f.write(f"I: {prompt_instance}\n") #PROMPT INSTANCE
                
                prompt = prompt_instance.replace('[[INPUT]]', input_text, 1)
                prompt = prompt.replace('[[INSTRUCTION]]', instruction + self.task_loader.ask, 1)
                prompt = prompt.replace('[[REQUEST]]', self.task_loader.get_task_prefix(), 1)

                f.write(f"P: {prompt}\n") #PROMPT
                
                try:
                    response_data = self.evaluator.generate_response(prompt)
                    response = response_data[0]['generated_text']
                    response = response.replace('\n', '')
                    error_count = 0  # Reset error count on success
                except Exception as e:
                    print(f"Error generating response: {e}")
                    error_count += 1
                    if error_count == 3:
                        return -1
                    continue

                
                f.write(f"A: {response}\n") #RAW ANSWER

                response = self.extractor(response, self.task_loader.answer_prefix, correct_answer) if response is not None else None

                f.write(f"F: {response}\n") #FORMATTED RESPONSE
                f.write(f"C: {correct_answer}\n") # CORRECT ANSWER
                f.write("#" * 56 + "\n\n")

                if response == correct_answer:
                    correct_answers += 1
        
        fitness = correct_answers / total_questions
        print(f"Fitness: {fitness}")
        return fitness