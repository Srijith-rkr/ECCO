import pandas as pd
import os
from time import time
import json
import argparse 
import asyncio
from tqdm import tqdm

from utils import judge_submit

parser = argparse.ArgumentParser()
parser.add_argument('--test_cases_path', default='/Users/srijith/courses/11711-ANLP/anlp_project/data/codenet/public_test_cases')
parser.add_argument('--judge_url', default='http://ec2-18-220-179-89.us-east-2.compute.amazonaws.com:2358')
parser.add_argument('--input_path', default='/Users/srijith/courses/11711-ANLP/anlp_project/ECCO/outputs/srijith_inference_outputs/edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_08:21:27.jsonl')
parser.add_argument('--code_col_name', default='generated_codes')
parser.add_argument('--num_runs', default=1)
parser.add_argument('--num_tests', default=20, type=int)
parser.add_argument('--out_path', default='./judge_eval/edit/')
parser.add_argument('--store_errors', default=True, action='store_false')
args = parser.parse_args() 

error_list = []

def calculate_speedup(json_data,file_name):
    speedup_data={}
    file_path = os.path.join(args.out_path, args.code_col_name + '_' + args.input_path.split("/")[-1])

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    for index,row in tqdm(json_data.iterrows(), total=len(json_data)):
        slow_code = row['input']
        fast_codes = row[args.code_col_name] # Can be a list of generated samples
        problem_id = row['problem_id']

        if type(fast_codes) == str: # If not a list 
            fast_codes = [fast_codes] # Make it a singular list

        # Get the test cases
        problem_input_folder = os.path.join(args.test_cases_path,problem_id)
        if not os.path.exists(problem_input_folder):
            print(f"Does not exist {problem_input_folder}")
            continue
        data = os.listdir(problem_input_folder)
        data = sorted(data)
        file_count = len(data)
        if file_count<2:
            print(f"Not enough test files for {problem_id}")
            continue

        data = os.listdir(problem_input_folder)
        input_files = [file for file in data if file.startswith("input")]
        if args.num_tests: # not None
            if args.num_tests > len(input_files):
                input_files = input_files[:args.num_tests]

        num_tests = len(input_files)
        input_files = sorted(input_files)

        # Stats for each test case
        num_tests = len(input_files)

        num_samples = len(fast_codes)

        output_valid = [True] * num_samples
        output_pass = {} # Dict {sample_id : True/False}
        output_memory = {} # Test sample no : dict{test_id: metric}
        output_run_times = {} # Test sample no : dict{test_id: metric}
        output_errors = {} # Test sample no: dict{test_id: error str}

        judge_url = args.judge_url
        # judge_url = f'{args.judge_url}:{args.judge_port}/'

        # For all test cases for the problem
        input_valid, input_pass, input_errors, input_run_times, input_memory = judge_submit(
            slow_code, problem_id, args.test_cases_path, args.num_runs, synchronous=True, judge_url=judge_url,
            number_of_tests=args.num_tests
        )

        # Get stats for all generated samples
        for sample_id, fast_code in enumerate(fast_codes): # For each sample
            sample_valid, sample_pass, sample_errors, sample_run_times, sample_memory = judge_submit(
                fast_code, problem_id, args.test_cases_path, args.num_runs, synchronous=True, judge_url=judge_url,
                number_of_tests=args.num_tests
            )

            output_valid[sample_id] = sample_valid
            output_pass[sample_id] = sample_pass
            output_run_times[sample_id] = sample_run_times
            output_memory[sample_id] = sample_memory
            output_errors[sample_id] = sample_errors

        # ======= Pick the best generated sample ========
        # Pick the solution that is the most correct first
        most_correct_samples = [] # ids - saves sample id of the code with highest pass rate

        num_passed_test = {x[0]: len(x[1]) for x in output_pass.items()} # x[0] sample id, x[1] is the set of passed test_ids, and get num
        max_pass_rate = max(num_passed_test.values()) # Max pass rate number
        for sample_id, pass_rate in num_passed_test.items():
            if pass_rate == max_pass_rate:
                most_correct_samples.append(sample_id)
        
        # Then pick based on memory and space across all tests
        # This is a dict sample_id: {test_id: dict of {test_id: runtime}} # 2 levels of dict
        filtered_run_times = {sample_id: output_run_times.get(sample_id) for sample_id in most_correct_samples}
        filtered_memory = {sample_id: output_memory.get(sample_id) for sample_id in most_correct_samples}

        best_time_sample_id, best_time = min(filtered_run_times.items(), key=lambda x: sum(x[1].values())) # Tuple of (bestsampleid, best time)
        best_mem_sample_id, best_mem = min(filtered_memory.items(), key=lambda x: sum(x[1].values())) 

        best_sample_id = best_time_sample_id # Currently, picking best time sample as the best solution
        # =====================

        # All statistics
        values = {
            f'problem_id': problem_id,

            f'input_accepted': input_valid,
            f'output_accepted': output_valid, # At least one sample accepted

            f'input_run_time': None, # float
            f'best_run_time': None, # float

            f'input_memory': None, # float
            f'best_memory': None, # float

            f'speedup': None, # float
            f'mem_reduction': None, # float

            f'input_pass_rate': len(input_pass) / num_tests, # float
            f'output_pass_rate': len(output_pass[best_sample_id]) / num_tests, # float

            # Metadata from runs
            f'input_run_time_all': input_run_times, # List (for each sample)
            f'input_memory_all': input_memory, # List
            f'input_pass_all': input_pass, # List of all test ids
            f'output_run_time_all': output_run_times[best_sample_id], # List (for each sample)
            f'output_memory_all': output_memory[best_sample_id], # List
            f'output_pass_all': output_pass[best_sample_id], # List of all test ids
            f'input_errors_all': input_errors, 
            f'output_errors_all': output_errors[best_sample_id]
        }

        # Input passed all tests and The best sample passed at least one test 
        if len(input_pass) == num_tests and len(output_pass[best_sample_id]) > 0: 
            # Get sum of times of for all tests that the best sample passed 
            in_time_passed, out_time_passed = [], []
            in_mem_passed, out_mem_passed = [], []

            for passed_test_id in output_pass[best_sample_id]: # Gives test id
                in_time_passed.append(input_run_times[passed_test_id])
                out_time_passed.append(output_run_times[best_sample_id][passed_test_id])

                in_mem_passed.append(input_memory[passed_test_id])
                out_mem_passed.append(output_memory[best_sample_id][passed_test_id])

            speedup= sum(in_time_passed) / sum(out_time_passed)
            mem_reduction = sum(in_mem_passed) / sum(out_mem_passed) 

            values.update({
                'input_run_time': sum(in_time_passed), 
                'best_run_time': sum(out_time_passed), 
                
                'input_memory': sum(in_mem_passed), 
                'best_memory': sum(out_mem_passed), 
                
                'speedup': speedup, 
                'mem_reduction': mem_reduction
            })
            
        speedup_data[index] = (values)

        with open(file_path, 'w') as json_file:
            #print('Saving evaluation results to', file_path)
            json.dump(speedup_data, json_file, indent=2)

if __name__=='__main__':
      input_file = args.input_path
      json_data = pd.read_json(input_file, orient='records', lines = True)
      calculate_speedup(json_data,input_file)