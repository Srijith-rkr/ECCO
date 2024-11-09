import os
import sys
import json
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import *
import argparse
import datetime
import requests

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
    prompt = 'Optimize the program below to be functionally equivalent but run faster and use less memory. Only output the optimized code with NO REASONING steps.\n'
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

    feedback_prompt_start = 'Given below are two programs that are functionally equivalent. Give instructional feedback in natural language as to what stragey was used to improve the second version from the first one in terms of runtime and space efficiency.\n\n'
    # feedback_prompt_start = 'What changed between the first and second versions of the program?\n\n'
    feedback_prompt = feedback_prompt_start + '\n\n### First version:\n' + row['input'] + '\n\n\n\n## Second version:\n' + row['target']

    row['feedback_prompt'] = feedback_prompt

    return prompt, slow_code, row['problem_id'], feedback_prompt



# Generate prompts
input_details = test.apply(get_prompt, axis=1)
prompts_all = list(input_details.apply(lambda x: x[0]))

feedback_prompts = list(input_details.apply(lambda x: x[3]))

# Generate faster codes
generated_codes = llm.generate(
    prompts_all,
    SamplingParams(
        n=1,
        max_tokens=args.max_new_tokens, 
        temperature=args.temperature
    )
)

# feedback_prompt_instr = 'Reason about why the python program that you generated above is inefficient in terms of runtime or space usage. Then provide feedback for how the inefficiency can be improved.\n'
# feedback_prompts = []
# for prompt, generated_code in zip(prompts_all, generated_codes):
#     feedback_prompt = f'{generated_code.outputs[0].text.strip()}\n\n{feedback_prompt_instr}\n'
#     feedback_prompts.append(feedback_prompt)

max_tokens = 1024
temp = 0.4
n_samples = 1
api_key_env = 'OPENAI_API_KEY'
model = 'gpt-3.5-turbo'
endpoint = 'https://api.openai.com/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',
}
if api_key_env:
    headers.update({'Authorization': f'Bearer {os.environ.get(api_key_env)}'}) # Add API if OpenAI


feedbacks = []
total_tokens = 0
input_tokens = 0
output_tokens = 0

for prompt in feedback_prompts:
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

    print(res_body)

    gen_samples = [choice['message']['content'] for choice in res_body['choices']]

    total_tokens += res_body['usage']['total_tokens']
    input_tokens += res_body['usage']['prompt_tokens']
    output_tokens += res_body['usage']['completion_tokens']

    feedbacks.append(gen_samples)

    # print(res_body['choices'], len(res_body['choices']))

# print(generated_code)
    
print('AVG tokens', total_tokens, total_tokens/len(feedback_prompts))
print('AVG inp tokens', input_tokens, input_tokens/len(feedback_prompts))
print('AVG output tokens', output_tokens, output_tokens/len(feedback_prompts))

# feedbacks = llm.generate(
#     feedback_prompts,
#     SamplingParams(
#         n=1,
#         max_tokens=args.max_new_tokens, # Feedback has to be concise 
#         temperature=args.temperature
#     )
# )

refine_prompt_instr = 'Based on the feedback on how the program can be improved, improve the runtime and space efficiency of the code. ONLY OUTPUT THE OPTIMIZED CODE.\n'
refine_prompts = []
for generated_code, feedback in zip(generated_codes, feedbacks):
    refine_prompt = f'{generated_code.outputs[0].text.strip()}\n\n{feedback.outputs[0].text.strip()}\n\n{refine_prompt_instr}\n'
    refine_prompts.append(refine_prompt)

refined_codes = llm.generate(
    refine_prompts,
    SamplingParams(
        n=1,
        max_tokens=args.max_new_tokens, 
        temperature=args.temperature
    )
)

# Create JSONL format
jsonl_data = []
for (prompt, slow_code, problem_id, fb_prompt), first_try, feedback, ref_prompt, second_try  in zip(
    input_details,
    generated_codes, 
    # feedback_prompts, 
    feedbacks, 
    refine_prompts, 
    refined_codes
):
    # generated_samples=[]
    # for out in first_try.outputs: # For multi samples
    # generated_text = out.text.strip()
    # generated_samples.append(generated_text)

    # refined_samples = []
    # for out in second_try.outputs: # For multi samples
    #     refined_text = out.text.strip()
    #     refined_samples.append(refined_text)

    pair = {
        'input': slow_code, 
        'first_gen': first_try.outputs[0].text.strip(), 
        'feedback': feedback.outputs[0].text.strip(),
        'second_try': second_try.outputs[0].text.strip(),
        'problem_id': problem_id, 
        'prompt': prompt,
        'feedback_prompt': fb_prompt,
        'ref_prompt': ref_prompt
        }
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

path = os.path.join(args.output_path, 'cannonical_refine', filename)
jsonl_final.to_json(path, orient='records', lines=True)