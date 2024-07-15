import os
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from typing import *
import argparse
import datetime

from src.evaluation.utils import judge_submit
from src.experiments.utils import (
    get_execution_feedback,
    build_coder_prompts,
    build_feedback_prompts,
    build_refine_prompts,
    build_reflect_prompts,
    build_nl2code_prompt,
    build_nl2code_feedback_prompt,
    build_nl2code_refine_prompts,
)
from src.experiments.model_classes import * 

model_classes = {
    'codellama_7b' : CodeLLaMa,
    'deepseek' : DeepSeekCoder,
    'wizard': WizardCoder,
    'codellama_13b':  CodeLLaMa,
    'codegemma': CodeGemma,
    'starcoder2': StarCoder2,
    'gpt-4o': OpenAI,
    'gpt-4-turbo': OpenAI 
}

model_kwargs = {
    'codellama_7b' : {'size': '7b'},
    'deepseek' : {},
    'wizard': {},
    'codellama_13b':  {'size': '13b'},
    'codellama_70b':  {'size': '70b'},
    'codegemma': {},
    'starcoder2': {},
    'gpt-4o': {'model': 'gpt-4o'},
    'gpt-4-turbo': {'model': 'gpt-4-turbo'}
}


parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, choices=list(model_classes.keys()))
parser.add_argument('--temperature', default=0.4,type=float)
parser.add_argument('--max_new_tokens', default=1024,type=int)
parser.add_argument('--few_shot_examples', default=0,type=int)
parser.add_argument('--instruct_version', type=str, choices=['base', 'instruct'], default='instruct')
parser.add_argument('--python_version', action='store_true', default=False) 
parser.add_argument('--output_path', default='./inference_generations/generated_codes/')
parser.add_argument('--num_samples', default=1,type=int)
parser.add_argument('--num_refinements', default=1,type=int)
parser.add_argument('--judge_url', default='http://<YOUR_URL>:PORT')
parser.add_argument('--test_cases_path', default='./data/codenet/public_test_cases', help='Path to public test cases')
parser.add_argument('--nrows', default=None,type=int)
parser.add_argument('--num_gpus',type=int, default=1)
parser.add_argument('--finetuned_weights',type=str, default=None)
parser.add_argument('--eval_mode',type=str, choices=['optim', 'nl2code', 'self-refine', 'exec-refine','nl2code-refine', 'reflexion', 'nl2code-exec-refine', 'nl2code-reflexion'], default='optim') 
args = parser.parse_args()

if 'nl2code' in args.eval_mode:
    dataset = load_dataset('EfficientCode/ECCO', 'generate')
    train = dataset['train'].to_pandas()
    problem_description = dataset['test'].to_pandas()
else: # All of optimization setting
    dataset = load_dataset('EfficientCode/ECCO', 'edit')
    train = dataset['train'].to_pandas()
    test = dataset['test'].to_pandas()

model_class = model_classes[args.model]
model_kwarg = model_kwargs[args.model]

if args.instruct_version == 'base':
    model_kwarg.update({'instruct': False}) # Switch off instruct
if args.python_version:
    model_kwarg.update({'python': True}) # Switch on Python
if args.finetuned_weights:
    model_kwarg.update({'finetuned_weights': args.finetuned_weights})

engine = model_class(**model_kwarg) # Instantiate model
llm = engine.get_model()

# Generate prompts

if args.eval_mode == 'optim':
    test = test.apply(build_coder_prompts, axis=1, engine=engine, train=train) # Added prompts to test

    prompts = list(test['prompt'])

    # Generate faster codes
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, args.num_samples) # prompts, num_samples
    generated_text = engine.extract_text_output(raw_generations)

    test['full_generations'] = generated_text # Add column
    generated_codes = engine.extract_codes(generated_text)

    test['generated_codes'] = generated_codes # Add column

    # Create JSONL format
    out_file = test[['input', 'generated_codes', 'full_generations', 'target', 'problem_id', 'prompt']]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    filename = f"{args.model}_{args.instruct_version}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_fewshotex{args.few_shot_examples}_samples{args.num_samples}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)

    print('Written to', path)

elif args.eval_mode == 'self-refine':
    # ====== Step 1. Generate Faster Codes (First Try) =============
    print('\n=== Generating Codes: Try 0 =====\n')
    test = test.apply(build_coder_prompts, axis=1, engine=engine, train=train) # Added prompts to test    
    prompts = list(test['prompt'])

    # For now doing self-refine with 1 sample
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, n_samples=1) # prompts, num_samples 
    generated_text = engine.extract_text_output(raw_generations)
    test['full_generations_0'] = generated_text
    generated_codes = engine.extract_codes(generated_text)
    test['generated_codes_0'] = generated_codes

    for iteration in range(args.num_refinements):
        # ========== Step 2. Feedback ============
        print(f'\n=== Feedback {iteration} =====\n')
        # Get feedbacks prompts
        test = test.apply(build_feedback_prompts, axis=1, try_col_name=f'generated_codes_{iteration}', engine=engine, train=train)
        feedback_prompts = list(test['feedback_prompt'])

        # Get feeedbacks
        feedbacks = engine.generate(feedback_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        feedback_text = engine.extract_text_output(feedbacks)
        test[f'feedback_{iteration}'] = feedback_text

        # ========== Step 3. Refine =============
        print(f'\n=== Generating refinement: {iteration} =====\n')
        # Get refinement prompts 
        test = test.apply(build_refine_prompts, axis=1, prev_try_col_name=f'generated_codes_{iteration}', feedback_col_name=f'feedback_{iteration}', engine=engine, train=train)
        refine_prompts = list(test['refine_prompt'])
        
        # Get refined codes
        refine_raw = engine.generate(refine_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        refine_text = engine.extract_text_output(refine_raw)
        test[f'full_generations_{iteration+1}'] = refine_text
        refined_codes = engine.extract_codes(refine_text)
        test[f'generated_codes_{iteration+1}'] = refined_codes

    # Create JSONL format
    refinement_cols = []
    # Add all the col names
    for i in range(args.num_refinements+1):
        if i != args.num_refinements: # Feedback not present for last one
            refinement_cols.append(f'feedback_{i}')
        refinement_cols.append(f'generated_codes_{i}')
        refinement_cols.append(f'full_generations_{i}')

    cols = ['input', 
            'target', 'problem_id', # Metadata
            'prompt', 'feedback_prompt', 'refine_prompt']
    
    cols.extend(refinement_cols)
    out_file = test[cols] 


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    filename = f"{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_numrefine{args.num_refinements}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)

    print('Written to', path)

elif args.eval_mode == 'exec-refine':
    # ====== Step 1. Generate Faster Codes (First Try) =============
    print('\n=== Generating Codes: Try 0 =====\n')
    test = test.apply(build_coder_prompts, axis=1, engine=engine, train=train) # Added prompts to test    
    prompts = list(test['prompt'])

    # For now doing self-refine with 1 sample
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, n_samples=1) # prompts, num_samples 
    generated_text = engine.extract_text_output(raw_generations)
    test['full_generations_0'] = generated_text
    generated_codes = engine.extract_codes(generated_text)
    test['generated_codes_0'] = generated_codes

    for iteration in range(args.num_refinements):
        print(f'\n=== Execute {iteration} =====\n')
        # ========== Step 2. Execute  ============
        # Run the codes 
        exec_feedbacks = []
        # print(len(generated_codes))

        for i, gen_code in enumerate(tqdm(generated_codes)):
            # judge res is tuple of (accept, pass_tests, errors, run_times, memory)
            judge_res = judge_submit(gen_code, test.iloc[i]['problem_id'], 
                        'args.test_cases_path', number_of_runs=1,
                        judge_url=args.judge_url)
            
            exec_feedbacks.append([get_execution_feedback(*judge_res)]) # Expand the tuple to pass args
            # Wrapping it in a lest as the refine prompt expects it for every sample

        test[f'exec_feedback_{iteration}'] = exec_feedbacks

        # ========== Step 3. Refine =============
        print(f'\n=== Generating refinement: {iteration} =====\n')
        # Get refinement prompts 
        test = test.apply(build_refine_prompts, axis=1, prev_try_col_name=f'generated_codes_{iteration}', feedback_col_name=f'exec_feedback_{iteration}', engine=engine, train=train)
        refine_prompts = list(test['refine_prompt'])
        
        # Get refined codes
        refine_raw = engine.generate(refine_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        refine_text = engine.extract_text_output(refine_raw)
        test[f'full_generations_{iteration+1}'] = refine_text
        refined_codes = engine.extract_codes(refine_text)

        generated_codes = refined_codes # Update generated codes
        test[f'generated_codes_{iteration+1}'] = refined_codes

    # Create JSONL format
    refinement_cols = []
    # Add all the col names
    for i in range(args.num_refinements+1):
        if i != args.num_refinements: # Feedback not present for last one
            refinement_cols.append(f'exec_feedback_{i}')
        refinement_cols.append(f'generated_codes_{i}')
        refinement_cols.append(f'full_generations_{i}')

    cols = ['input', 
            'target', 'problem_id', # Metadata
            'prompt', 'refine_prompt']
    
    cols.extend(refinement_cols)
    out_file = test[cols] 

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    filename = f"{args.eval_mode}_{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_numrefine{args.num_refinements}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)

    print('Written to', path)

elif args.eval_mode == 'reflexion':
    # ====== Step 1. Generate Faster Codes (First Try) =============
    print('\n=== Generating Codes: Try 0 =====\n')
    test = test.apply(build_coder_prompts, axis=1, engine=engine, train=train) # Added prompts to test    
    prompts = list(test['prompt'])

    # For now doing self-refine with 1 sample
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, n_samples=1) # prompts, num_samples 
    generated_text = engine.extract_text_output(raw_generations)
    test['full_generations_0'] = generated_text
    generated_codes = engine.extract_codes(generated_text)
    test['generated_codes_0'] = generated_codes

    for iteration in range(args.num_refinements):
        # ========== Step 2. Execute  ============
        print(f'\n=== Executing: Try {iteration} =====\n')
        # Run the codes 
        exec_feedbacks = []
        # print(len(generated_codes))

        for i, gen_code in enumerate(tqdm(generated_codes)):
            # judge res is tuple of (accept, pass_tests, errors, run_times, memory)
            judge_res = judge_submit(gen_code, test.iloc[i]['problem_id'], 
                        'args.test_cases_path', number_of_runs=1, 
                        judge_url=args.judge_url)
            
            exec_feedbacks.append([get_execution_feedback(*judge_res)]) # Expand the tuple to pass args
            # Wrapping it in a lest as the refine prompt expects it for every sample

        test[f'exec_feedback_{iteration}'] = exec_feedbacks

        # ========== Step 3. Reflect =============
        print(f'\n=== Reflecting {iteration} =====\n')
        test = test.apply(build_reflect_prompts, axis=1, prev_try_col_name=f'generated_codes_{iteration}', exec_col_name=f'exec_feedback_{iteration}', engine=engine, train=train)
        reflect_prompts = list(test['reflect_prompt'])

        # Get feeedbacks
        feedbacks = engine.generate(reflect_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        feedback_text = engine.extract_text_output(feedbacks)
        test[f'reflect_{iteration}'] = feedback_text

        # =========== Step 4: Refine ============
        print(f'\n=== Generating refined: Try {iteration+1} =====\n')
        # Get refinement prompts 
        test = test.apply(build_refine_prompts, axis=1, prev_try_col_name=f'generated_codes_{iteration}', feedback_col_name=f'reflect_{iteration}', engine=engine, train=train)
        refine_prompts = list(test['refine_prompt'])
        
        # Get refined codes
        refine_raw = engine.generate(refine_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        refine_text = engine.extract_text_output(refine_raw)
        test[f'full_generations_{iteration+1}'] = refine_text
        refined_codes = engine.extract_codes(refine_text)
        test[f'generated_codes_{iteration+1}'] = refined_codes


    # Create JSONL format
    refinement_cols = []
    # Add all the col names
    for i in range(args.num_refinements+1):
        if i != args.num_refinements: # Feedback not present for last one
            refinement_cols.append(f'exec_feedback_{i}')
            refinement_cols.append(f'reflect_{i}')
        refinement_cols.append(f'generated_codes_{i}')
        refinement_cols.append(f'full_generations_{i}')

    cols = ['input', 
            'target', 'problem_id', # Metadata
            'prompt', 'refine_prompt', 'reflect_prompt']
    
    cols.extend(refinement_cols)
    out_file = test[cols] 

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    filename = f"{args.eval_mode}_{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)

    print('Written to', path)

elif args.eval_mode == 'nl2code':
    test_problem_ids = list(test['problem_id'].unique())
    problem_description = problem_description[problem_description['problem_id'].isin(test_problem_ids)]
    problem_description = problem_description.apply(build_nl2code_prompt, axis=1, engine=engine, train=train) # Added prompts to test
    prompts = list(problem_description['prompt'])
    print('Generating for', len(prompts), prompts)
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, args.num_samples) # prompts, num_samples
    # print('Generated', len(raw_generations), raw_generations)
    generated_text = engine.extract_text_output(raw_generations)

    problem_description['full_generations'] = generated_text # Add column
    generated_codes = engine.extract_codes(generated_text)

    problem_description['generated_codes'] = generated_codes # Add column

    # Create JSONL format
    out_file = problem_description[[ 'problem_id', 'problem_description','prompt','full_generations', 'generated_codes']]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    filename = f"NL2Code_{args.model}_fewshot_{args.few_shot_examples}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)

    print('Written to', path)

elif args.eval_mode == 'nl2code-refine':
    # ====== Step 1. Generate Faster Codes (First Try) ============= 
    test_problem_ids = list(test['problem_id'].unique())
    problem_description = problem_description[problem_description['problem_id'].isin(test_problem_ids)]
    problem_description = problem_description.apply(build_nl2code_prompt, axis=1, engine=engine, train=train)
    prompts = list(problem_description['prompt'])
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, n_samples=1)
    generated_text = engine.extract_text_output(raw_generations)
    problem_description['full_generations_0'] = generated_text
    generated_codes = engine.extract_codes(generated_text)
    problem_description['generated_codes_0'] = generated_codes

    # ========== Step 2. Feedback ============
    problem_description = problem_description.apply(build_nl2code_feedback_prompt, axis=1, try_col_name='generated_codes_0', engine=engine, train=train)
    feedback_prompts = list(problem_description['feedback_prompt'])
    feedbacks = engine.generate(feedback_prompts, args.temperature, args.max_new_tokens, n_samples=1)
    feedback_text = engine.extract_text_output(feedbacks)
    problem_description['feedback_0'] = feedback_text

    # ========== Step 3. Refine ============= 
    problem_description = problem_description.apply(build_nl2code_refine_prompts, axis=1, prev_try_col_name='generated_codes_0', feedback_col_name='feedback_0', engine=engine, train=train)
    refine_prompts = list(problem_description['refine_prompt'])
    refine_raw = engine.generate(refine_prompts, args.temperature, args.max_new_tokens, n_samples=1)
    refine_text = engine.extract_text_output(refine_raw)
    problem_description['full_generations_1'] = refine_text
    refined_codes = engine.extract_codes(refine_text)
    problem_description['generated_codes_1'] = refined_codes

    # Create JSONL format
    out_file = problem_description[['problem_id', 'problem_description', 'prompt', 'full_generations_0', 'generated_codes_0', 'feedback_0', 'full_generations_1', 'generated_codes_1']]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    filename = f"NL2Code-Refine_{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"
    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)
    print('Written to', path)

elif args.eval_mode == 'nl2code-exec-refine':
    # ====== Step 1. Generate Faster Codes (First Try) ============= 
    test_problem_ids = list(test['problem_id'].unique())
    problem_description = problem_description[problem_description['problem_id'].isin(test_problem_ids)]
    problem_description = problem_description.apply(build_nl2code_prompt, axis=1, engine=engine, train=train)
    prompts = list(problem_description['prompt'])
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, n_samples=1)
    generated_text = engine.extract_text_output(raw_generations)
    problem_description['full_generations_0'] = generated_text
    generated_codes = engine.extract_codes(generated_text)
    problem_description['generated_codes_0'] = generated_codes

    for iteration in range(args.num_refinements):
        print(f'\n=== Execute {iteration} =====\n')
        # ========== Step 2. Execute  ============
        # Run the codes 
        exec_feedbacks = []
        # print(len(generated_codes))

        for i, gen_code in enumerate(tqdm(generated_codes)):
            # judge res is tuple of (accept, pass_tests, errors, run_times, memory)
            judge_res = judge_submit(gen_code, problem_description.iloc[i]['problem_id'], 
                        'args.test_cases_path', number_of_runs=1,
                        judge_url=args.judge_url)
            
            exec_feedbacks.append([get_execution_feedback(*judge_res)]) # Expand the tuple to pass args
            # Wrapping it in a lest as the refine prompt expects it for every sample

        problem_description[f'exec_feedback_{iteration}'] = exec_feedbacks

        # ========== Step 3. Refine =============
        print(f'\n=== Generating refinement: {iteration} =====\n')
        # Get refinement prompts 
        problem_description = problem_description.apply(build_nl2code_refine_prompts, axis=1, prev_try_col_name=f'generated_codes_{iteration}', feedback_col_name=f'exec_feedback_{iteration}', engine=engine, train=train)
        refine_prompts = list(problem_description['refine_prompt'])
        
        # Get refined codes
        refine_raw = engine.generate(refine_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        refine_text = engine.extract_text_output(refine_raw)
        problem_description[f'full_generations_{iteration+1}'] = refine_text
        refined_codes = engine.extract_codes(refine_text)

        generated_codes = refined_codes # Update generated codes
        problem_description[f'generated_codes_{iteration+1}'] = refined_codes

    # Create JSONL format
    refinement_cols = []
    # Add all the col names
    for i in range(args.num_refinements+1):
        if i != args.num_refinements: # Feedback not present for last one
            refinement_cols.append(f'exec_feedback_{i}')
        refinement_cols.append(f'generated_codes_{i}')
        refinement_cols.append(f'full_generations_{i}')

    cols = ['problem_id', 'problem_description', # Metadata
            'prompt', 'refine_prompt']
    
    cols.extend(refinement_cols)
    out_file = problem_description[cols] 

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    filename = f"{args.eval_mode}_{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_numrefine{args.num_refinements}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)
    print('Written to', path)

elif args.eval_mode == 'nl2code-reflexion':
    # ====== Step 1. Generate Faster Codes (First Try) ============= 
    test_problem_ids = list(test['problem_id'].unique())
    problem_description = problem_description[problem_description['problem_id'].isin(test_problem_ids)]
    problem_description = problem_description.apply(build_nl2code_prompt, axis=1, engine=engine, train=train)
    prompts = list(problem_description['prompt'])
    raw_generations = engine.generate(prompts, args.temperature, args.max_new_tokens, n_samples=1)
    generated_text = engine.extract_text_output(raw_generations)
    problem_description['full_generations_0'] = generated_text
    generated_codes = engine.extract_codes(generated_text)
    problem_description['generated_codes_0'] = generated_codes

    for iteration in range(args.num_refinements):
        print(f'\n=== Execute {iteration} =====\n')
        # ========== Step 2. Execute  ============
        # Run the codes 
        exec_feedbacks = []
        # print(len(generated_codes))

        for i, gen_code in enumerate(tqdm(generated_codes)):
            # judge res is tuple of (accept, pass_tests, errors, run_times, memory)
            judge_res = judge_submit(gen_code, problem_description.iloc[i]['problem_id'], 
                        'args.test_cases_path', number_of_runs=1,
                        judge_url=args.judge_url)
            
            exec_feedbacks.append([get_execution_feedback(*judge_res)]) # Expand the tuple to pass args
            # Wrapping it in a lest as the refine prompt expects it for every sample

        problem_description[f'exec_feedback_{iteration}'] = exec_feedbacks

        # ========== Step 3. Reflect =============
        print(f'\n=== Reflecting {iteration} =====\n')
        problem_description = problem_description.apply(build_reflect_prompts, axis=1, prev_try_col_name=f'generated_codes_{iteration}', exec_col_name=f'exec_feedback_{iteration}', engine=engine, train=train)
        reflect_prompts = list(problem_description['reflect_prompt'])

        # Get feeedbacks
        feedbacks = engine.generate(reflect_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        feedback_text = engine.extract_text_output(feedbacks)
        problem_description[f'reflect_{iteration}'] = feedback_text

        # =========== Step 4: Refine ============
        print(f'\n=== Generating refined: Try {iteration+1} =====\n')
        # Get refinement prompts 
        problem_description = problem_description.apply(build_nl2code_refine_prompts, axis=1, prev_try_col_name=f'generated_codes_{iteration}', feedback_col_name=f'reflect_{iteration}', engine=engine, train=train)
        refine_prompts = list(problem_description['refine_prompt'])
        
        # Get refined codes
        refine_raw = engine.generate(refine_prompts, args.temperature, args.max_new_tokens, n_samples=1)
        refine_text = engine.extract_text_output(refine_raw)
        problem_description[f'full_generations_{iteration+1}'] = refine_text
        refined_codes = engine.extract_codes(refine_text)
        problem_description[f'generated_codes_{iteration+1}'] = refined_codes

    # Create JSONL format
    refinement_cols = []
    # Add all the col names
    for i in range(args.num_refinements+1):
        if i != args.num_refinements: # Feedback not present for last one
            refinement_cols.append(f'exec_feedback_{i}')
            refinement_cols.append(f'reflect_{i}')
        refinement_cols.append(f'generated_codes_{i}')
        refinement_cols.append(f'full_generations_{i}')

    cols = ['problem_id', 'problem_description', # Metadata
            'prompt', 'refine_prompt', 'reflect_prompt']
    
    cols.extend(refinement_cols)
    out_file = problem_description[cols] 

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    filename = f"{args.eval_mode}_{args.model}_nrows{args.nrows}_tokens{args.max_new_tokens}_temp{args.temperature}_samples{args.num_samples}_numrefine{args.num_refinements}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jsonl"

    path = os.path.join(args.output_path, filename)
    out_file.to_json(path, orient='records', lines=True)
    print('Written to', path)

