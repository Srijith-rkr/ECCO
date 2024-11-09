import os
import sys
import json
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import *
import argparse
import datetime


models = {
    'codellama_7b' : "/data/datasets/models/huggingface/meta-llama/CodeLlama-7b-Instruct-hf",
    'deepseek' : "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    'wizard': "WizardLM/WizardCoder-Python-13B-V1.0",
    'codellama_13b':  "/data/datasets/models/huggingface/meta-llama/CodeLlama-13b-Instruct-hf",
    'codellama_python_13b': '/data/datasets/models/huggingface/meta-llama/CodeLlama-13b-Python-hf',
    'mistral_7b': 'mistralai/Mistral-7B-Instruct-v0.2'
}


parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', default='./data/filtered_codenet-python-val.jsonl')
parser.add_argument('--test_data_path', default='./data/test.jsonl')
parser.add_argument('--max_new_tokens', default=1024,type=int)
parser.add_argument('--temperature', default=0.4,type=float)
parser.add_argument('--model', default=None, choices=list(models.keys()))
parser.add_argument('--output_path', default='./vllm_results/generated_codes/')
parser.add_argument('--nrows', default=None,type=int)
parser.add_argument('--num_samples', default=1,type=int)
parser.add_argument('--num_gpus',type=int)
args = parser.parse_args()

train = pd.read_json(args.train_data_path, orient='records', lines = True)
test = pd.read_json(args.test_data_path, orient='records', lines = True, nrows=args.nrows)

model_path = models[args.model]

hugging_cache = os.environ.get('HF_HOME')
llm = LLM(model_path, tensor_parallel_size=args.num_gpus, dtype='float16', download_dir=hugging_cache)

def get_prompt(row):
    num_examples = 4

    # prompt = ''
    prompt = 'Optimize the program below to be functionally equivalent but run faster and use less memory. Only output the optimized code without any reasoning steps. If you need to output your reasoning make it a comment (start with #) for every line.\n'
    # prompt = '[INST] Optimize the program below to be functionally equivalent but run faster and use less memory. Only output the optimized code. If you have any reasoning steps add it as a comment in code. [\INST]\n'
    # prompt = '[INST] Optimize the program below to be functionally equivalent but run faster and use less memory. [\INST]\n'
    # prompt = ''
#     prompt = '[INST]\n\
# Your task is to optimize a Python program given below.\n\
# The objective is to write a program that is functionally equivalent to the provided program, but one that runs faster and uses less memory\n\
# You can use different data structures and optimized algorithms to do so.\n\
# However, make sure that the optimized program does not break any unit test cases.\n\
# Output only the optimized program.[/INST]\n'

    # prompt_example_rows = train.sample(num_examples)
    # prompt_example_rows['prompt'] = '# slower version:\n\n' + prompt_example_rows['input'] + '\n\n# optimized version of the same code:\n\n' + prompt_example_rows['target']
    # prompt += prompt_example_rows['prompt'].str.cat(sep='\n\n')

    prompt += '\n\n### Program:\n'
    prompt += row['input'] + '\n\n### Optimized (Runtime and Space) version of Program above:'
    slow_code = row['input']

    row['prompt'] = prompt

    return prompt, slow_code, row['problem_id']

# Generate prompts
input_details = test.apply(get_prompt, axis=1)
prompts_all = list(input_details.apply(lambda x: x[0]))


# Generate faster codes
generated_codes = llm.generate(
    prompts_all,
    SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_new_tokens, 
        temperature=args.temperature
    )
)

# Create JSONL format
jsonl_data = []
for (prompt, slow_code, problem_id), output in zip(input_details, generated_codes):
    generated_samples=[]
    for out in output.outputs:
        generated_text = out.text.strip()
        generated_samples.append(generated_text)
    pair = {'input': slow_code, 'target': generated_samples, 'problem_id': problem_id, 'prompt': prompt}
    jsonl_data.append(pair)

# Save to JSONL file
# with open("", 'w') as jsonl_file:
#     for line in jsonl_data:
#         jsonl_file.write(json.dumps(line) + '\n')
        
jsonl_final=pd.DataFrame(jsonl_data)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# filename = f'./{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl'

filename = f"./{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

path = os.path.join(args.output_path, filename)
jsonl_final.to_json(path, orient='records', lines=True)