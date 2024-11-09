import subprocess
import pandas as pd
import os
from time import time
import json
import psutil
import argparse
import requests



parser = argparse.ArgumentParser(description='Performance analysis of generated code.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input data file')
parser.add_argument('--test_case_path', type=str, help='Path to the test cases', default='../data/codenet/public_test_cases/')
parser.add_argument('--judge_url', default='http://localhost:2358/')
args = parser.parse_args()

error_list = []

def run_cmd(code, input_file_path,idx,tag):
    

    try:
        command = ["python", "-c",code]

        def _kill(proc_pid):
            process = psutil.Process(proc_pid)
            for proc in process.children(recursive=True):
                # logging.info(f"Killing {proc}")
                proc.kill()
            # logging.info(f"Killing {process}")
            process.kill()
        with open(input_file_path, "r") as f:
                    proc = subprocess.Popen(
                        command,
                        stdin=f,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                        # preexec_fn=limit_virtual_memory,
                    )
                    stdout, stderr = proc.communicate(timeout=3)

                    # Check the return code to see if the script ran successfully
                    if proc.returncode == 0:
                        print(f"Script executed successfully for {idx}-{tag}!")
                        print("Script output:")
                        print(stdout)
                        return stdout
                    else:
                        print("Script encountered an error!")
                        print(f"Error output for {idx}-{tag}")
                        print(stderr)
                        message = {'error':stderr,'index':idx,'tag':tag}
                        error_list.append(message)
                        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout Error for {idx}:{tag}")
        _kill(proc.pid)
         # type: ignore
        message = {'error':"Timeout",'index':idx,'tag':tag}
        error_list.append(message)
        return None
    except Exception as e:
         print(e)
         return None

# run_cmd(slow_code,test_file)


def filter_invalid_runs(json_data, test_cases_path):
    speedup_data=[]
    invalid_indices = []

    for index,row in json_data.iterrows():
        print(f"For example:{index}")
        slow_code = row['input']
        fast_code = row['target']
        problem_id = row['problem_id']

        problem_input_folder = test_cases_path+problem_id
        if not os.path.exists(problem_input_folder):
            print(f"Does not exist {problem_input_folder}")
            invalid_indices.append(index)
            continue
        data = os.listdir(problem_input_folder)
        data = sorted(data)
        file_count = len(data)
        if file_count<2:
            print(f"Not enough test files for {problem_id}")
            invalid_indices.append(index)
            continue

        input_file_id = '/'+data[0]
        print(input_file_id)
        input_file=problem_input_folder+ input_file_id

        input_start_time=time()
        input_run = run_cmd(code = slow_code,input_file_path=input_file,idx=index,tag='input')
        input_total_time=time()-input_start_time

        target_start_time = time()
        output_run = run_cmd(code = fast_code,input_file_path=input_file,idx=index,tag='target')
        target_total_time = time() -target_start_time

        speedup=(input_total_time)/target_total_time
        rtr = float(input_total_time-target_total_time)/input_total_time
        rtr=rtr*100
        if input_run and output_run and input_run==output_run:
            output_file_id = input_file_id.replace("input","output")
            # print("Output file:",output_file_id)
            output_file = input_file=problem_input_folder+ output_file_id
            # print(output_file)
            with open(output_file, 'r') as file:
                file_contents = file.read()
                print("Output contents:",file_contents)
            if file_contents==input_run==output_run:
                 print("Functionally correct pairs")
            else:
                 print("##Functionally incorrect pairs#########################")

             
            values = {'input_run_time':input_total_time,
                        'target_run_time':target_total_time,
                        'speedup':speedup,
                        'rtr':rtr,
                        'problem_id':problem_id}
        else:
             values = {'input_run_time':None,
                        'target_run_time':None,
                        'speedup':None,
                        'rtr':None,
                        'problem_id':problem_id}
             invalid_indices.append(index)
             
        speedup_data.append(values)
    return invalid_indices


def judge_submit(code, test_file_path, number_of_runs=1, synchronous=True):
    with open(test_file_path, "r") as in_f:
        test_case_input = in_f.read()

    request_body = {
        'source_code': code,
        'language_id': 71, # For python3 (For python 2 it'll be 70)
        'stdin': test_case_input,
        'number_of_runs': number_of_runs
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

def calculate_speedup(json_data, test_cases_path):
    speedup_data={}

    epochs = 5
    # for epoch in range(epochs):
    # print(f"Epoch:{epoch}")
    for index,row in json_data.iterrows():
        print(index)
        slow_code = row['input']
        fast_code = row['target']
        problem_id = row['problem_id']

        problem_input_folder = test_cases_path + problem_id
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

        # input_start_time=time()
        # input_run = run_cmd(code = slow_code,input_file_path=input_file,idx=index,tag='input')
        # input_total_time=time()-input_start_time
        slow_token, slow_url, slow_res = judge_submit(slow_code, input_file, epochs, synchronous=True)

        # target_start_time = time()
        # output_run = run_cmd(code = fast_code,input_file_path=input_file,idx=index,tag='target')
        # target_total_time = time() -target_start_time

        fast_token, fast_url, fast_res = judge_submit(fast_code, input_file, epochs, synchronous=True)

        slow_res = slow_res.json()
        fast_res = fast_res.json()

        input_run = ('status' in slow_res and slow_res['status']['id'] == 3)
        output_run = ('status' in fast_res and fast_res['status']['id'] == 3)

        print(slow_res)
        print(fast_res)

        if input_run:
        # Currently not asynronous requests 
            input_total_time = float(slow_res['time'])
            input_mem = float(slow_res['memory'])

        if output_run:
            target_total_time = float(fast_res['time'])
            target_mem = float(fast_res['memory'])


        if input_run and output_run: 
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
                        f'rtr':rtr,
                        f'problem_id':problem_id}
            if index in speedup_data:
                values.update(speedup_data[index])
        else:
            values = {f'input_run_time':None,
                        f'target_run_time':None,
                        f'speedup':None,
                        f'rtr':None,
                        f'problem_id':problem_id}
            
        speedup_data[index]=(values)

    speedup_data_values = list(speedup_data.values())
    return speedup_data_values
    


    








if __name__=='__main__':

    file_name = args.input_file.split('/')[-1]
    print(file_name)
    

    data= pd.read_json(args.input_file,orient='records', lines = True)

    data= data
    json_data = data[['input', 'target','problem_id']]

    invalid_indices = filter_invalid_runs(json_data, args.test_case_path)
    print("Error pairs",len(error_list))
    print("Functionally incorrect",len(invalid_indices))
    print("Original Data length",len(json_data))
    # json_data.drop(invalid_indices,inplace=True).reset_index(drop=True)
    json_data = json_data.drop(invalid_indices).reset_index(drop=True)
    print("After filtering json length",len(json_data))
    json_data.to_json( f'../vllm_results/vllm_filtered/filtered_{file_name}',orient='records', lines=True)


    speedup_data_values = calculate_speedup(json_data, args.test_case_path)

    with open(f'../vllm_results/vllm_average/average_{file_name}', 'w') as json_file:
        json.dump(speedup_data_values, json_file)

    


      
      



