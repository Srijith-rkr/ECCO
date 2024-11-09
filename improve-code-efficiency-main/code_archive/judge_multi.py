import subprocess
import sys
import pandas as pd
import os
from time import time
import json
import psutil
import argparse 
import requests
import asyncio

parser = argparse.ArgumentParser()
parser.add_argument('--test_cases_path', default='./data/codenet/public_test_cases/')
parser.add_argument('--judge_url', default='http://13.58.18.211:2358/')
parser.add_argument('--input_path', default=None)
parser.add_argument('--output_path', default='./vllm_results/judge_eval/')
parser.add_argument('--fast_col_name', default='target')
args = parser.parse_args() 

# base_file_path = '/mnt/c/Users/Sid/Desktop/Directed Study/PIE/pie-perf-new/pie-perf/data/codenet/public_test_cases/'
# base_file_path = '/home/swaghjal/data/codenet/public_test_cases/'
error_list = []


def judge_submit(code, test_file_path, number_of_runs=1, synchronous=True):
    with open(test_file_path, "r") as in_f:
        test_case_input = in_f.read()

    request_body = {
        'source_code': code,
        'language_id': 71, # For python3 (For python 2 it'll be 70)
        'stdin': test_case_input,
        'number_of_runs': number_of_runs,
        'cpu_time_limit': 7,
        'cpu_extra_time': 5
        # 'expected_output':
    }

    if synchronous:
        post_res = requests.post(args.judge_url+'/submissions/?wait=true', request_body) # Just to debug with synchronous request
    else:
        post_res = requests.post(args.judge_url+'/submissions', request_body)

    if 'token' in post_res:
        token = post_res['token']
    else:
        token = ''

    status_url = args.judge_url + '/submissions' + f'/{token}'


    return token, status_url, post_res

def calculate_speedup(json_data,file_name):
    speedup_data={}

    epochs = 5
    for index,row in json_data.iterrows():
        slow_code = row['input']
        fast_codes = row[args.fast_col_name]
        problem_id = row['problem_id']

        print(index, len(fast_codes))

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

        input_file_id = '/'+data[0] # First test
        # print(input_file_id)
        input_file=problem_input_folder + input_file_id

        slow_token, slow_url, slow_res = judge_submit(slow_code, input_file, epochs, synchronous=True)
        
        slow_res = slow_res.json()
        input_run = ('status' in slow_res and slow_res['status']['id'] == 3)

        # print(slow_res)

        if input_run:
        # Currently not asynronous requests 
            input_total_time = float(slow_res['time'])
            input_mem = float(slow_res['memory'])
        else:
            print(f"##Invalid slow code:{slow_res}")

        run_times = []
        memory = []

        for fast_code in fast_codes:
            fast_token, fast_url, fast_res = judge_submit(fast_code, input_file, epochs, synchronous=True)
            fast_res = fast_res.json()
            output_run = ('status' in fast_res and fast_res['status']['id'] == 3)

            if output_run:
                target_total_time = float(fast_res['time'])
                target_mem = float(fast_res['memory'])

                run_times.append(target_total_time)
                memory.append(target_mem)
            else:
                print(f"##Invalid fast code:{fast_res}")


        if input_run and len(run_times) > 0:
            # Picking best
            target_total_time = min(run_times)
            target_mem = min(memory)
            print(target_total_time, target_mem)

            speedup=(input_total_time)/target_total_time
            mem_reduction = (input_mem)/target_mem
            rtr = float(input_total_time-target_total_time)/input_total_time
            rtr=rtr*100

            values = {f'input_run_time':input_total_time,
                        f'target_run_time':target_total_time,
                        f'input_memory': input_mem,
                        f'target_memory': target_mem,
                        f'mem_reduction': mem_reduction,
                        f'speedup':speedup,
                        f'target_times_all': run_times,
                        f'target_memory_all': memory,
                        f'rtr':rtr,
                        f'problem_id':problem_id}
            if index in speedup_data:
                values.update(speedup_data[index])
        else:
            values = {f'input_run_time':None,
                        f'input_memory': None,
                        f'target_run_time':None,
                        f'target_memory': None,
                        f'speedup':None,
                        f'target_times_all': [],
                        f'target_memory_all': [],
                        f'rtr':None,
                        f'problem_id':problem_id}
            
        speedup_data[index]=(values)

    speedup_data_values = list(speedup_data.values())

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(f'{args.output_path}/{args.input_path.split("/")[-1]}', 'w') as json_file:
    # with open(f'./vllm_filtered_judgejudge_avergage_7b_codellama_first_gen_output_20.jsonl', 'w') as json_file:
        json.dump(speedup_data_values, json_file)


if __name__=='__main__':
    #   input_file=sys.argv[1]
      input_file = args.input_path
      json_data = pd.read_json(input_file, orient='records', lines = True)
      calculate_speedup(json_data,input_file)