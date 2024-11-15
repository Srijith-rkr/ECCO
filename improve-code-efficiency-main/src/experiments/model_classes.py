import pandas as pd
import argparse 
import tqdm 
import os
from vllm import LLM, SamplingParams
import torch
from typing import List
import re
# import openai
import requests
import json
from transformers import AutoTokenizer

from fastchat.conversation import get_conv_template

CODER_PROMPT_FORMAT = '{}\n\n## Program:\n{}\n\n## Optimized (Runtime and Space) version of Program above:'
CODER_FEW_SHOT_FORMAT = '## Program:\n{}\n\n## Optimized (Runtime and Space) version of Program above:\n{}'
# The first {} corresponds to the start of the prompt (includes few shot), {} corresponds to slow code

FEEDBACK_PROMPT_FORMAT = '{}\n\n## Candidate solution:\n{}\n\n## Feedback for incorrectnes/inefficiency and how it can be improved:'
FEEDBACK_FEW_SHOT_FORMAT = '## Candidate solution:\n{}\n\n##  Feedback for incorrectnes/inefficiency and how it can be improved:{}'

REFLECT_PROMPT_FORMAT = '{}\n\n## Execution Results:\n{}\n\n## Reflection on incorrectnes/inefficiency and how it can be improved:'
REFLECT_FEW_SHOT_FORMAT = '## Execution Results:\n{}\n\n##  Reflection on incorrectnes/inefficiency and how it can be improved:{}'

CANNONICAL_FEEDBACK_PROMPT_FORMAT = '{}\n\n## Candidate solution:\n{}\n\n## Feedback for incorrectnes/inefficiency and how it can be improved:'

REFINE_PROMPT_FORMAT = '{}\n\n## Feedback to improve the code:\n{}\n\n## Refined code that includes optimizations specified in feedback:'
REFINE_FEW_SHOT_FORMAT = '## Candidate solution:\n{}\n\n##  Feedback for incorrectnes/inefficiency and how it can be improved:{}'

NL2CODE_PROMPT_FORMAT = '{}\n\n## Details:\n{}\n\n## Solution:'
NL2CODE_FEW_SHOT_FORMAT = '## Details:\n{}\n\n## Solution:{}'
FEEDBACK_NL2CODE_PROMPT_FORMAT = '{}\n\n## Candidate solution:\n{}\n\n## Feedback for incorrectnes/inefficiency and how it can be improved:'

# The first {} corresponds to the start of the prompt (includes few shot), {} corresponds to slow code

class BaseModel(object):
    def __init__(self):
        pass

    def get_model(self):
        return self.llm # Assuming llm is defined
        
    # Only used privately
    def _build_prompt(self, prompt_details: dict, code: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        if few_shots > 0 and train_data is not None: # Added few shot examples if required
            prompt_example_rows = train_data.sample(few_shots)

            prompt_example_rows['prompt'] = prompt_example_rows.apply(
                lambda row: prompt_details['few_shot_format'].format(
                    row['input'], 
                    self._wrap_in_context_output(row['target']) # Wrap in context output wraps it in the format to extract later
                ),
                axis=1
            ) # Format the examples

            few_shot_examples = prompt_example_rows['prompt'].str.cat(sep='\n\n')

            if instruct:
                prompt_details['prompt_start'] += 'Here are a few examples:\n\n{}'.format(few_shot_examples)
            else:
                prompt_details['prompt_start'] += few_shot_examples

        prompt = prompt_details['prompt_format'].format(prompt_details['prompt_start'], code)

        return prompt
    
    def _build_nl_prompt(self, prompt_details: dict, code: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        if few_shots > 0 and train_data is not None: # Added few shot examples if required
            prompt_example_rows = train_data.sample(few_shots)

            prompt_example_rows['prompt'] = prompt_example_rows.apply(
                lambda row: prompt_details['few_shot_format'].format(
                    row['problem_description'], 
                    self._wrap_in_context_output(row['fastest_code']) # Wrap in context output wraps it in the format to extract later
                ),
                axis=1
            ) # Format the examples

            few_shot_examples = prompt_example_rows['prompt'].str.cat(sep='\n\n')

            if instruct:
                prompt_details['prompt_start'] += 'Here are a few examples:\n\n{}'.format(few_shot_examples)
            else:
                prompt_details['prompt_start'] += few_shot_examples

        prompt = prompt_details['prompt_format'].format(prompt_details['prompt_start'], code)

        return prompt 
    
    def _build_coder_prompt(self, slow_code: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        prompt_start = '' if not instruct else \
            'Optimize the python program below to be functionally equivalent but run faster and use less memory.\
            Wrap the optimized code in a block of 3 backticks (```).\n\n'
        
        prompt_details = {
            'prompt_start': prompt_start,
            'few_shot_format': CODER_FEW_SHOT_FORMAT,
            'prompt_format': CODER_PROMPT_FORMAT
        }

        return self._build_prompt(prompt_details, slow_code, few_shots, train_data, instruct)
    
    def _build_feedback_prompt(self, code: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        prompt_start = '' if not instruct else \
            'Give feedback in english for why the code solution below is incorrect or inefficient and how the program can be fixed.\n\n'
        
        prompt_details = {
            'prompt_start': prompt_start,
            'few_shot_format': FEEDBACK_FEW_SHOT_FORMAT,
            'prompt_format': FEEDBACK_PROMPT_FORMAT
        }

        return self._build_prompt(prompt_details, code, few_shots, train_data, instruct)
    
    # Refine prompt is public method as it requires 2 arguments, and can't be clubbed into get_prompt
    def build_refine_prompt(self, prev_try: str, feedback: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        """ 
            Refine prompt is public method as it requires 2 arguments, and can't be clubbed into get_prompt
        """
        prompt_start = '' if not instruct else \
        'Refine the given incorrect or sub-optimal code solution based on the feedback specified below. Wrap the refined code in a block of 3 backticks (```)\n\n## Sub-optimal soliution:\n{}'

        prompt_start = prompt_start.format(prev_try) # Add prev try into prompt_start

        prompt_details = {
            'prompt_start': prompt_start,
            'few_shot_format': REFINE_FEW_SHOT_FORMAT,
            'prompt_format': REFINE_PROMPT_FORMAT
        }

        return self._build_prompt(prompt_details, feedback, few_shots, train_data, instruct)
    
    def _build_nl2code_prompt(self,problem_description: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        prompt_start = 'Write a python code which is efficient in terms of runtime and memory usage for the following problem description\
            Wrap the optimized code in a block of 3 backticks (```).\n\n'
        prompt_details = {
            'prompt_start': prompt_start,
            'few_shot_format': NL2CODE_FEW_SHOT_FORMAT,
            'prompt_format': NL2CODE_PROMPT_FORMAT
        }
        return self._build_nl_prompt(prompt_details, code=problem_description, few_shots=few_shots, train_data=train_data, instruct=instruct)
    def _build_nl2code_feedback_prompt(self, problem_description: str,code: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        prompt_start = '' if not instruct else \
            'Give feedback in english for why the code solution below is incorrect or inefficient and how the program can be fixed based on the problem description.\n{}'
        prompt_start = prompt_start.format(problem_description)
        prompt_details = {
            'prompt_start': prompt_start,
            'few_shot_format': NL2CODE_FEW_SHOT_FORMAT,
            'prompt_format': FEEDBACK_NL2CODE_PROMPT_FORMAT
        }

        return self._build_prompt(prompt_details, code=code)
    
    def build_nl2code_refine_prompt(self, prev_try: str, feedback: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        prompt_start = '' if not instruct else \
        'Refine the given incorrect or sub-optimal code solution based on the feedback specified below. Wrap the refined code in a block of 3 backticks (```)\n\n## Sub-optimal soliution:\n{}'

        prompt_start = prompt_start.format(prev_try) # Add prev try into prompt_start

        prompt_details = {
            'prompt_start': prompt_start,
            'prompt_format': REFINE_PROMPT_FORMAT
        }

        return self._build_prompt(prompt_details, feedback)

    def build_reflect_prompt(self, prev_try: str, exec_results: str, few_shots: int = 0, train_data: str = None, instruct: bool = True):
        prompt_start = '' if not instruct else \
            'Based on the execution results, reflect on why the code solution below was incorrect or inefficient and how the program can be fixed.\n\n{}'
        
        prompt_start = prompt_start.format(prev_try)

        prompt_details = {
            'prompt_start': prompt_start,
            'few_shot_format': REFLECT_FEW_SHOT_FORMAT,
            'prompt_format': REFLECT_PROMPT_FORMAT
        }

        return self._build_prompt(prompt_details, exec_results, few_shots, train_data, instruct)
    
    def get_prompt(self, code: str, few_shots: int = 0, train_data: str = None, instruct: bool = True, mode: str = 'coder'):
        prompt_builder_map = {
            'coder': self._build_coder_prompt,
            'feedback': self._build_feedback_prompt,
            'nl2code': self._build_nl2code_prompt,
            'nl2code_feedback': self._build_nl2code_feedback_prompt
        }

        # Call the corresponding function based on the mode
        return prompt_builder_map[mode](code, few_shots, train_data, instruct) 
    

    # def generate(self, prompts: list, temp: int, max_tokens: int, n_samples: int):
    #     # TODO: Replace with for-loop for better memory utilization
    #     samples = self.llm.generate(prompts, 
    #                     SamplingParams(
    #                         max_tokens=max_tokens, 
    #                         temperature=temp,
    #                         n=n_samples
    #                     ))         
    #     return samples # (num_samples, prompts)
    
    # TODO: Override generate when tested to work
    def generate(self, prompts: list, temp: int, max_tokens: int, n_samples: int):
        sample_outputs = []

        outputs = None
        for i in range(n_samples):
            print(f'\n==== Generating sample number {i} ===\n')
            s_i = self.llm.generate(prompts, 
                          SamplingParams(
                                max_tokens=max_tokens, 
                                temperature=temp,
                                n=1
                          ))
            # print(s_i) # s_i has the fields (request_id, prompt, prompt_token_ids and outputs). Thie first three remain the same. 
            sample_outputs.append(s_i)

            if outputs is None: # Add the first sample output
                outputs = s_i

        # first output is of shape (num_prompts)
        for prompt_idx in range(len(sample_outputs[0])):
            for s_i in sample_outputs[1:]: # Num samples (Skip the first one as it is already present)
                outputs[prompt_idx].outputs.append(s_i[prompt_idx].outputs[0]) # Append into the outputs of the first one for each prompt
        
        # print(len(outputs), len(outputs[0].outputs))
        return outputs
    
    def extract_text_output(self, generations):
        # print(generations)
        return [[output.text for output in samples.outputs] for samples in generations] # Only return text fields
    
    # Wrap in context is related to extract code. The patterns has to match. If you override one, please override both
    def _wrap_in_context_output(self, target_code:str):
        return f"```\n{target_code}```"
    
    def extract_codes(self, output_texts):
        extracted_codes = []
        pattern = r'```.*?```'

        for samples in output_texts:
            extracted_samples = [] # For each sample
            for output_text in samples:
                matches = re.findall(pattern, output_text, re.DOTALL)

                if len(matches) != 0: # If there is a code wrapped in ````
                    biggest_match = max(matches, key = len) # Pick max based on length of string
                    extracted_samples.append(biggest_match[3:-3]) # 3 and -3 are to parse the ``` out

                else: # TODO: Investigate when/why this case occurs
                    extracted_samples.append(output_text)

            extracted_codes.append(extracted_samples)

        return extracted_codes    
    
    def generate_api(self, prompts: list, temp: int = 0.7, max_tokens: int = 1024, n_samples: int = 1, \
                    endpoint: str = None, model: str = None, api_key_env: str = None):
        
        assert endpoint is not None and model is not None # Assertion Error if either are None

        headers = {
            'Content-Type': 'application/json',
        }

        if api_key_env:
            headers.update({'Authorization': f'Bearer {os.environ.get(api_key_env)}'}) # Add API if OpenAI

        samples = []

        # 1 request per prompt as OpenAI API is unreliable with batched prompts
        for prompt in tqdm.tqdm(prompts):
            req_body = {
                'model': model,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': max_tokens,
                'n': n_samples,
                'temperature': temp
            }

            res = requests.post(endpoint, json.dumps(req_body), headers=headers)
            res_body = res.json()

            samples.append(res_body)

        return samples

class CodeLLaMa(BaseModel):
    def __init__(
        self, 
        size: str = '7b', 
        instruct: bool = True,
        python: bool = False,
        path: str = None,
        dtype: str = torch.bfloat16,
        finetuned_weights: str=None
    ):
        super(CodeLLaMa, self).__init__() 

        self.size = size 
        instruct = instruct

        # If path is None use cached codellama on babel
        if finetuned_weights:
            print("Loading finetuned weights")
            path = finetuned_weights
        elif not path:
            assert size in ['7b', '13b', '34b', '70b']
            instruct_str = 'Instruct-' if instruct else '' 
            python_str = 'Python-' if python else '' 

            # TODO: Maybe assert that instruct is false when base python version used?

            path = f"/data/models/huggingface/meta-llama/CodeLlama-{size}-{instruct_str}hf" if not python \
                else f"/data/models/huggingface/meta-llama/CodeLlama-{size}-{python_str}hf"
            
            if size == '70b' or (not instruct and not python): #base (Not on babel)
                path = f"meta-llama/CodeLlama-{size}-{instruct_str}hf"


        num_gpus = torch.cuda.device_count()
        print("#####Number of GPUS available:",num_gpus)
        hugging_cache = os.environ.get('HF_HOME')

        self.llm = LLM(
            path,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            download_dir=hugging_cache
        )

    def wrap_prompt_chat_template(self, prompt):
        template = get_conv_template("llama-2")
        template.append_message(template.roles[0], prompt) 
        template.append_message(template.roles[1], None) # Adding [\INST]
        return template.get_prompt()
    
class Mistral(BaseModel):
    def __init__(
        self, 
        version: str = '0.2',
        instruct: bool = True,
        path: str = None,
        dtype: str = torch.bfloat16
    ):
        super(Mistral, self).__init__() 

        instruct = instruct

        # If path is None use cached codellama on babel
        if not path:
            # assert size in ['7B']
            instruct_str = 'Instruct-' if instruct else '' 
            path = f"mistralai/Mistral-7B-{instruct_str}v{version}"
        
        num_gpus = torch.cuda.device_count()
        hugging_cache = os.environ.get('HF_HOME')

        self.llm = LLM(
            path,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            download_dir=hugging_cache
        )

    def wrap_prompt_chat_template(self, prompt):
        template = get_conv_template("mistral")
        template.append_message(template.roles[0], prompt) 
        template.append_message(template.roles[1], None) # Adding [\INST]
        return template.get_prompt()

class MarkdownCoderModel(BaseModel):
    """BaseModel for MarkDown formated coding models (```python ``` used for wrapping) 
    """
    def __init__(self):
        super(MarkdownCoderModel, self).__init__()

    # Overriding base methods 
    def _wrap_in_context_output(self, target_code:str):
        return f"```python\n{target_code}```"
    
    def extract_codes(self, output_texts: List[List[str]]):
        # Deepseek puts its codes in ```python triple quotes
        extracted_codes = []
        python_pattern = r'```python.*?```'
        pattern = r'```.*?```'

        for samples in output_texts:
            extracted_samples = [] # For each sample
            for output_text in samples:
                # Try python pattern
                matches = re.findall(python_pattern, output_text, re.DOTALL)
                if len(matches) != 0: # If there is a code wrapped in ```python 
                    biggest_match = max(matches, key = len) # Pick min based on length of string. Max doesn't work if there are multiple in which case it will take the ends
                    extracted_samples.append(biggest_match[9:-3]) # 9 and -3 are to parse the ```python and ``` out

                else: # TODO: Investigate when/why this case occurs
                    matches = re.findall(pattern, output_text, re.DOTALL)
                    if len(matches) != 0: # If there is a code wrapped in ````
                        biggest_match = max(matches, key = len) # Pick max based on length of string
                        extracted_samples.append(biggest_match[3:-3]) # 3 and -3 are to parse the ``` out
                    else:
                        extracted_samples.append(output_text)

            extracted_codes.append(extracted_samples)

        return extracted_codes 
    
    # Both markdown coder models (Deepseek and Wizard) use the same instruction format
    def wrap_prompt_chat_template(self, prompt):
        template = get_conv_template("deepseek-coder") # According to model card of wizard the format is the same as deepseek ### Instruction: ### Response
        template.append_message(template.roles[0], prompt) 
        template.append_message(template.roles[1], None) # Adding [\INST]
        return template.get_prompt() 
 
class DeepSeekCoder(MarkdownCoderModel):
    def __init__(
        self, 
        size: str = '7b', 
        version: str = '1.5',
        instruct: bool = True,
        path: str = None,
        dtype: str = torch.bfloat16,
        finetuned_weights: str=None
    ):
        super(DeepSeekCoder, self).__init__() 

        self.size = size 
        instruct = instruct

        # If path is None use cached codellama on babel
        if finetuned_weights:
            print("Loading finetuned weights")
            path = finetuned_weights
        elif not path:
            assert size in ['7b', '33b']
            instruct_str = 'instruct' if instruct else 'base' 
            path = f"deepseek-ai/deepseek-coder-{size}-{instruct_str}-v{version}"
        
        num_gpus = torch.cuda.device_count()
        hugging_cache = os.environ.get('HF_HOME')

        self.llm = LLM(
            path,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            download_dir=hugging_cache
        )

class WizardCoder(MarkdownCoderModel):
    def __init__(
        self, 
        size: str = '13B', 
        instruct: bool = True,
        version: str = '1.0',
        path: str = None,
        dtype: str = torch.float16
    ):
        super(WizardCoder, self).__init__() 

        instruct = instruct

        # If path is None use cached codellama on babel
        if not path:
            # assert size in ['7B']
            instruct_str = 'Instruct-' if instruct else '' 
            # path = f"WizardLM/WizardCoder-Python-{size}-V{version}"
            path = f"/scratch/vveerend/WizardCoder/5ac6748b1f5a4c282107ddc7d3b69fdc4a686d75/"
        
        num_gpus = torch.cuda.device_count()
        hugging_cache = os.environ.get('HF_HOME')

        self.llm = LLM(
            path,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            download_dir=hugging_cache
        )

class CodeGemma(MarkdownCoderModel):
    def __init__(
        self, 
        size: str = '7b', 
        instruct: bool = True,
        path: str = None,
        dtype: str = torch.float16
    ):
        super(CodeGemma, self).__init__() 

        instruct = instruct

        # If path is None use cached codellama on babel
        if not path:
            assert size in ['7b']
            instruct_str = '-it' if instruct else '' 
            path = f"google/codegemma-{size}{instruct_str}" # codegemma-7b-it
        
        num_gpus = torch.cuda.device_count()
        hugging_cache = os.environ.get('HF_HOME')

        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.llm = LLM(
            path,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            download_dir=hugging_cache
        )

    def wrap_prompt_chat_template(self, prompt):
        chat = [
            {"role": "user", "content": prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return wrapped_prompt

class StarCoder2(MarkdownCoderModel):
    def __init__(
        self, 
        size: str = '15B', 
        instruct: bool = True,
        version: str = '1.0',
        path: str = None,
        dtype: str = torch.float16
    ):
        super(StarCoder2, self).__init__() 

        instruct = instruct

        # If path is None use cached codellama on babel
        if not path:
            # assert size in ['7B']
            instruct_str = '-instruct-v0.1' if instruct else '' 
            path = f"bigcode/starcoder2-15b{instruct_str}"
        
        num_gpus = torch.cuda.device_count()
        hugging_cache = os.environ.get('HF_HOME')

        self.llm = LLM(
            path,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            download_dir=hugging_cache
        )
    
class OpenAI(MarkdownCoderModel):
    def __init__(
        self,
        model = 'gpt-4o'
    ):
        super(OpenAI, self).__init__()
        self.model = model  
    
    # Overriding
    def get_model(self):
        return None 
    
    # Override generate to not use self.llm
    def generate(self, prompts: list, temp: int = 0.7, max_tokens: int = 1024, n_samples: int = 1): # TODO: Test
        return self.generate_api(prompts, temp, max_tokens, n_samples, 
            'https://api.openai.com/v1/chat/completions', self.model, 'OPENAI_API_KEY') # OpenAI arguments for the API call
    
    # Overriding the extraction for OpenAI response format
    # TODO: Verify how the response for vllm's openAI server is different and make modifications
    def extract_text_output(self, generations):
        outputs = []
        for samples in generations:
            try:
                if 'choices' in samples:
                    texts = [output['message']['content'] for output in samples['choices']]
                elif 'message' in samples:
                    texts = [samples['message']['content']] # Singleton list
            except Exception as e:
                texts = f'Unsuccessful in calling GPT-4 due to reason {e} try again!'
            outputs.append(texts)
        return outputs
    
    # Override
    def wrap_prompt_chat_template(self, prompt):
        return prompt