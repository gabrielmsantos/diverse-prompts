import transformers
from IndividualFactory import IndividualFactory  
from TaskLoader import TaskLoader
import torch
import random
import requests
import time  

from abc import ABC, abstractmethod
import torch

class EvalInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt):
        pass

class LangEvaluator(EvalInterface):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        else:
            self.model = self.model.to('cpu')  # Ensure it's on CPU if no GPU is available

    def generate_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Ensure input_ids is on the correct device (CPU or GPU)
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        # Create attention mask: 1 for actual tokens, 0 for padding tokens
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()  # Create attention mask
        
        # Move attention_mask to GPU if available
        if torch.cuda.is_available():
            attention_mask = attention_mask.to('cuda')

        outputs = self.model.generate(
            input_ids,
            max_length=512,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,  # Pass the attention mask
            num_return_sequences=1  # Ensure only one sequence is returned
        )
        
        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text

class RandomEvaluator(EvalInterface):
    def generate_response(self, prompt):
        # return random response between 0 and 1
        return str(random.randint(0, 1))
    

class APIModelEvaluator(EvalInterface):
    def __init__(self, api_key, url, data_template):
        self.api_key = api_key
        self.url = url
        self.data_template = data_template
        self.headers = {
            "Accept" : "application/json",
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    def _prepare_data(self, prompt):
        # Substitute the input in data_template
        data = self.data_template
        data['inputs'] = prompt
        return data

    def generate_response(self, prompt, c_headers=None):
        # Define the message you want to send
        data = self._prepare_data(prompt)

        if c_headers is None:
            c_headers = self.headers

        # Make the API request with a timeout and retry logic
        max_retries = 3
        retry_delay = 5  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, headers=c_headers, json=data, timeout=30)  # 30 seconds timeout
                response.raise_for_status()  # Raise an error for bad responses
                break  # If successful, break out of the retry loop
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                if attempt < max_retries - 1:  # If it's not the last attempt
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    #print("Headers:", c_headers)
                    #print("Data:", data)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"All {max_retries} attempts failed. Last error: {e}")
                    raise Exception(f"All {max_retries} attempts failed. Last error: {e}")  # Handle the failure after all retries

        # Check the response
        if response.status_code == 200:
            response_data = response.json()
            return response_data
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        

