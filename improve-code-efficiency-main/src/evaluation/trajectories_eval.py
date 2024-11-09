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
parser.add_argument('--out_path', default='./vllm_results/nl2code_eval/')
parser.add_argument('--nrows', default=None, type=int)
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--num_tests', type=int, default=None)
args = parser.parse_args()

# ======

print('Passing num tests', args.num_tests)

def group_runtimes_by_problem(json_data):
    """
    Running all the submissions in NL2Code dataset and grouping them by problem ids
    """
    problem_ids = set(json_data['problem_id'])
    details = {pid: {'accepted_solutions': [], 'failed_solutions': [], 'runtimes': [], 'memories': [], 'errors': []} for pid in problem_ids}

    json_data['accepted_solutions'] = [[] for _ in range(len(json_data))]
    json_data['failed_solutions'] = [[] for _ in range(len(json_data))]
    json_data['runtimes'] = [[] for _ in range(len(json_data))]
    json_data['memories'] = [[] for _ in range(len(json_data))]
    json_data['errors'] = [[] for _ in range(len(json_data))]

    for idx, row in tqdm(json_data.iterrows(), total=len(json_data)):
        print('Starting index', idx)
        problem_id = row['problem_id']
        codes = row['trajectories']
        submissions = row['submission_id']

        assert len(codes) == len(submissions)

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
        
        if num_tests > args.num_tests:
            num_tests = args.num_tests

        for i, code in enumerate(codes):
            correct = True
            total_time = 0
            total_memory = 0

            valid, passed_tests, errors, run_times, memory = judge_submit(
                code, problem_id, args.test_cases_path, args.num_runs, synchronous=True, judge_url=args.judge_url,
                number_of_tests=args.num_tests
            )

            if not valid:
                details[problem_id]['failed_solutions'].append(submissions[i])
                details[problem_id]['errors'].append('Invalid solution caught in not valid if block')
                continue

            total_time = float(sum(run_times.values())) / len(run_times)
            total_memory = float(sum(memory.values())) / len(memory)

            if len(passed_tests) == num_tests: # Either 20 or lesser
                details[problem_id]['accepted_solutions'].append(submissions[i])
                details[problem_id]['runtimes'].append(total_time)
                details[problem_id]['memories'].append(total_memory)
                details[problem_id]['errors'].append(None)

            else:
                details[problem_id]['failed_solutions'].append(submissions[i])
                details[problem_id]['errors'].append(errors)

        # Update DataFrame columns
        json_data.at[idx, 'accepted_solutions'] = details[problem_id]['accepted_solutions']
        json_data.at[idx, 'failed_solutions'] = details[problem_id]['failed_solutions']
        json_data.at[idx, 'runtimes'] = details[problem_id]['runtimes']
        json_data.at[idx, 'memories'] = details[problem_id]['memories']
        json_data.at[idx, 'errors'] = details[problem_id]['errors']

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Dump all the stats
    out_file_name = args.input_path.split("/")[-1]
    json_data.to_json(f"{args.out_path}/results_{out_file_name}", orient='records', lines=True)
    json.dump(details, open(os.path.join(args.out_path, 'run_time_details.json'), 'w'))


if __name__ == '__main__':
    json_data = pd.read_json(args.input_path, nrows=args.nrows, orient='records', lines=True)
    out_file_name = args.input_path.split("/")[-1]
    print(out_file_name)

    group_runtimes_by_problem(json_data)
