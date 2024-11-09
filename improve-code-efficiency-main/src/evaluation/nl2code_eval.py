import pandas as pd
import json
import requests
import os
import argparse
from tqdm import tqdm

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test_cases_path', default='/data/tir/projects/tir6/general/vveerend/data/pie/codenet/generated_test_cases/')
parser.add_argument('--judge_url', default='http://13.58.18.211:2358/')
parser.add_argument('--input_path', default='./data/trajectories.csv')
parser.add_argument('--gen_code_path', default='./data/trajectories.csv')
parser.add_argument('--traj_path', default='./data/trajectories.csv')
parser.add_argument('--out_path', default='./vllm_results/nl2code_eval/')
parser.add_argument('--nrows', default=None, type=int)
parser.add_argument('--code_col_name', default='generated_codes')
parser.add_argument('--num_runs', type=int, default=3)
parser.add_argument('--num_tests', type=int, default=None)
args = parser.parse_args()

def calculate_percentile(generated_json_data, submission_details):
    eval_data={}

    for index,row in tqdm(generated_json_data.iterrows(), total=len(generated_json_data)):
        codes = row[args.code_col_name] # Can be a list of generated samples
        problem_id = row['problem_id']
        if problem_id not in submission_details['problem_id'].values:
            print(f"Problem {problem_id} not in submission details")
            continue

        if type(codes) == str: # If not a list 
            fast_codes = [codes] # Make it a singular list
        else:
            fast_codes = codes
        # print("Fast codes\n",fast_codes)

        # Get the test cases
        problem_input_folder = args.test_cases_path + problem_id
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
        input_files = sorted(input_files)


        # Stats for each test case
        num_tests = len(input_files)
        if num_tests > args.num_tests:
            num_tests = args.num_tests
        num_samples = len(fast_codes)

        output_valid = [True] * num_samples
        output_pass = {} # Dict {sample_id : True/False}
        output_memory = {} # Test sample no : dict{test_id: metric}
        output_run_times = {} # Test sample no : dict{test_id: metric}
        output_errors = {} # Test sample no: dict{test_id: error str}

        # judge_url = f'{args.judge_url}:{args.judge_port}/'

        judge_url = args.judge_url
        # For all test cases for the problem
        # input_valid, input_pass, input_errors, input_run_times, input_memory = judge_submit(
        #     slow_code, problem_id, args.test_cases_path, args.num_runs, synchronous=True, judge_url=judge_url
        # )

        # Get stats for all generated samples
        for sample_id, code in enumerate(codes): # For each sample
            sample_valid, sample_pass, sample_errors, sample_run_times, sample_memory = judge_submit(
                code, problem_id, args.test_cases_path, args.num_runs, synchronous=True, judge_url=judge_url, number_of_tests=args.num_tests
            )

            output_valid[sample_id] = sample_valid
            output_pass[sample_id] = sample_pass
            output_run_times[sample_id] = sample_run_times
            output_memory[sample_id] = sample_memory
            output_errors[sample_id] = sample_errors

        # ======= Pick the best generated sample ========
        # Pick the solution that is the most correct first
        most_correct_samples = [] # ids

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

        # TODO: Handle the case where the best time and best mem is not the same sample
        best_sample_id = best_time_sample_id # For now picking best time sample as the best solution
        # =====================

        # All statistics
        values = {
            f'problem_id': problem_id,
            f'accepted': output_valid, # At least one sample accepted

            f'best_run_time': None, # float
            f'best_memory': None, # float
            f'runtime_percentile': None, # float
            f'mem_percentile': None, # float

            f'pass_rate': len(output_pass[best_sample_id]) / num_tests, # float

            # Metadata from runs
            f'run_time_all': output_run_times[best_sample_id], # List (for each sample)
            f'memory_all': output_memory[best_sample_id], # List
            f'pass_all': output_pass[best_sample_id], # List of all test ids
            f'errors_all': output_errors[best_sample_id]
        }

        # ==== Calculate percentile ========
        # submission_runtimes = sorted(submission_details[problem_id]['runtimes'])
        # submission_memory = sorted(submission_details[problem_id]['memories'])
        submission_runtimes = submission_details[submission_details['problem_id']== problem_id]['runtimes']
        # print("Submission runtimes",submission_runtimes)
        
        submission_memory = submission_details[submission_details['problem_id'] == problem_id]['memories']
        # print("Submission memories",submission_memory)
        submission_runtimes = sorted(submission_runtimes.iloc[0],reverse=True)
        submission_memory = sorted(submission_memory.iloc[0],reverse=True)

        if len(output_pass[best_sample_id]) == num_tests: # Passed all tests 
            # Get sum of times of for all tests that the best sample passed 
            out_time_passed = []
            out_mem_passed = []

            # TODO: CALCULATE SPEEDUP ONLY ON THE SAME SUBSET OF CASES THAT BOTH PASS, BUT RECORD RUNTIMES FOR ALL CASES
            for passed_test_id in output_pass[best_sample_id]: # Gives test id
                out_time_passed.append(output_run_times[best_sample_id][passed_test_id])

                out_mem_passed.append(output_memory[best_sample_id][passed_test_id])

            # TODO: Check if we need to take average instead of sum
            total_time = sum(out_time_passed) / len(out_time_passed)
            total_memory = sum(out_mem_passed) / len(out_mem_passed)

            # Runtime percentile 
            # TODO: Replace these with binary search
            # print(f"Submission runtimes {submission_runtimes}")
            # print(f"Total time {total_time}")
            time_percentile = None
            mem_percentile = None
            for i, sub_run in enumerate(submission_runtimes):
                # print(f"Sub run {sub_run}")
                if sub_run < total_time: # First greater
                    time_percentile = i / len(submission_runtimes)
                    break
            if not time_percentile:
                time_percentile = 1
            # print(f"Submission memories {submission_memory}")
            # print(f"Total memory {total_memory}")
            for i, sub_mem in enumerate(submission_memory):
                if sub_mem < total_memory: # First greater
                    mem_percentile = i / len(submission_memory)
                    break
            if not mem_percentile:
                mem_percentile = 1

            values.update({
                'best_run_time': total_time, 
                
                'best_memory': total_memory, 
                
                'runtime_percentile': time_percentile,
                'mem_percentile': mem_percentile
            })
            
        eval_data[index] = (values)

    speedup_data_values = eval_data
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    file_path = os.path.join(args.out_path, args.code_col_name + '_' + args.input_path.split("/")[-1])
    with open(file_path, 'w') as json_file:
        print('Saving evaluation results to', file_path)
        json.dump(speedup_data_values, json_file, indent=2)

# ======

def group_runtimes_by_problem(json_data):
    """
    Running all the submissions in NL2Code dataset and grouping them by problem ids
    """
    problem_ids = set(json_data['problem_id'])
    details = {pid: {'accepted_solutions': [], 'failed_solutions': [], 'runtimes': [], 'memories': []} for pid in problem_ids}

    json_data['accepted_solutions'] = [[] for _ in range(len(json_data))]
    json_data['failed_solutions'] = [[] for _ in range(len(json_data))]
    json_data['runtimes'] = [[] for _ in range(len(json_data))]
    json_data['memories'] = [[] for _ in range(len(json_data))]

    for idx, row in tqdm(json_data.iterrows(), total=len(json_data)):
        problem_id = row['problem_id']
        codes = row[args.code_col_name]
        # submissions = row['submission_id']

        # assert len(codes) == len(submissions)

        problem_input_folder = args.test_cases_path + problem_id

        if not os.path.exists(problem_input_folder):
            print(f"Does not exist {problem_input_folder}")
            continue
        data = os.listdir(problem_input_folder)
        data = sorted(data)
        file_count = len(data)
        if file_count < 2:
            print(f"Not enough test files for {problem_id}")
            continue

        input_files = [file for file in data if file.startswith("input")]
        input_files = sorted(input_files)
        num_tests = len(input_files)

        for i, code in enumerate(codes):
            correct = True
            total_time = 0
            total_memory = 0

            valid, passed_tests, errors, run_times, memory = judge_submit(
                code, problem_id, args.test_cases_path, args.num_runs, synchronous=True, judge_url=args.judge_url
            )

            if not valid:
                details[problem_id]['failed_solutions'].append(i)
                continue

            total_time = float(sum(run_times.values())) / len(run_times)
            total_memory = float(sum(memory.values())) / len(memory)

            if len(passed_tests) == num_tests:
                details[problem_id]['accepted_solutions'].append(i)
                details[problem_id]['runtimes'].append(total_time)
                details[problem_id]['memories'].append(total_memory)
            else:
                details[problem_id]['failed_solutions'].append(i)

        # Update DataFrame columns
        json_data.at[idx, 'accepted_solutions'] = details[problem_id]['accepted_solutions']
        json_data.at[idx, 'failed_solutions'] = details[problem_id]['failed_solutions']
        json_data.at[idx, 'runtimes'] = details[problem_id]['runtimes']
        json_data.at[idx, 'memories'] = details[problem_id]['memories']

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Dump all the stats
    out_file_name = args.input_path.split("/")[-1]
    # json.dump(details, open(os.path.join(args.out_path, 'run_time_details.json'), 'w'))
    json_data.to_json(f"{args.out_path}/runtime_results_{out_file_name}", orient='records', lines=True)


# if __name__ == '__main__':
#     json_data = pd.read_json(args.input_path, nrows=args.nrows, orient='records', lines=True)
#     out_file_name = args.input_path.split("/")[-1]
#     print(out_file_name)

#     group_runtimes_by_problem(json_data)/
if __name__ == '__main__':
    gen_code_data = pd.read_json(args.input_path, nrows=args.nrows, orient='records', lines=True)

    # gen_code_data = pd.read_json(args.gen_code_path, orient='records', lines=True)
    traj_data = pd.read_json(args.traj_path, orient='records', lines=True)

    calculate_percentile(gen_code_data, traj_data)

    # out_file_name = args.input_path.split("/")[-1]
    # print(out_file_name)

    # group_runtimes_by_problem(json_data)
    # group_runtimes_by_problem(json_data)
