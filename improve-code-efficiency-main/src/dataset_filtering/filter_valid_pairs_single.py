import subprocess
import pandas as pd
import os
from time import time
import json
import psutil
import argparse
import datetime



# base_file_path = '/mnt/c/Users/Sid/Desktop/Directed Study/PIE/pie-perf-new/pie-perf/data/codenet/public_test_cases/'
# base_file_path = '/home/swaghjal/data/codenet/public_test_cases/'
parser = argparse.ArgumentParser(description='Performance analysis of generated code.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input data file')
parser.add_argument('--test_case_path', type=str, help='Path to the test cases', default='/data/tir/projects/tir6/general/vveerend/improving-code-efficiency/shared/data/codenet/public_test_cases/')
parser.add_argument('--out_path', type=str, help='Path to the filtered output', default='/data/tir/projects/tir6/general/vveerend/improving-code-efficiency/shared/data/')
parser.add_argument('--col_name', type=str, default='target')
args = parser.parse_args()
base_file_path = args.test_case_path
error_list = []
insufficient_input_files = []
invalid_input_runs = set()

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
                    stdout, stderr = proc.communicate(timeout=5)

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
        message = {'error':str(e),'index':idx,'tag':tag}
        error_list.append(message)
        return None

# run_cmd(slow_code,test_file)

def filter_invalid_runs(json_data):
    invalid_indices = []  # Initialize as a set
    for index, row in json_data.iterrows():
        print(f"For example: {index}")
        slow_code = row['input']
        fast_code = row[args.col_name]
        problem_id = row['problem_id']

        problem_input_folder = base_file_path + problem_id
        if not os.path.exists(problem_input_folder):
            print(f"Does not exist {problem_input_folder}")
            insufficient_input_files.append({'index':index,'problem_id':problem_id})
            invalid_indices.append(index)  # Add to set instead of appending to list
            continue
        
        data = os.listdir(problem_input_folder)
        input_files = [file for file in data if file.startswith("input")]
        input_files = sorted(input_files)
        functionally_incorrect= False
        if len(input_files) == 0:
            print(f"Not enough test files for {problem_id}")
            insufficient_input_files.append({'index':index,'problem_id':problem_id})
            invalid_indices.append(index)  # Add to set instead of appending to list
            continue
        for input_file_idx, input_file_id in enumerate(input_files):
            print(input_file_id)
            input_file = os.path.join(problem_input_folder, input_file_id)

            input_run = run_cmd(code=slow_code, input_file_path=input_file, idx=index, tag='input')
            
            if not input_run:
                print(f"##Invalid slow code for {problem_id} and {input_file_id}")
                invalid_input_runs.add(index)
                functionally_incorrect = True
                break
            output_run = run_cmd(code=fast_code, input_file_path=input_file, idx=index, tag=args.col_name)
            if output_run and input_run == output_run:
                output_file_id = input_file_id.replace("input", "output")
                output_file = os.path.join(problem_input_folder, output_file_id)
                with open(output_file, 'r') as file:
                    file_contents = file.read()
                    print(f"Output contents for {problem_id} and {input_file_id} and {output_file}", file_contents)
                if file_contents == input_run == output_run:
                    print(f"Functionally correct pairs for {problem_id} and {input_file_id}")
                else:
                    functionally_incorrect= True
                    print(f"Failed for test case for {input_file_idx}")
                    break

            else:
                print(f"Failed for test case for {input_file_idx}")
                functionally_incorrect = True
                break
        if functionally_incorrect:
            invalid_indices.append(index)

    
    return invalid_indices

if __name__=='__main__':
    
    file_name = args.input_file.split('/')[-1]
    print(file_name)
    

    data= pd.read_json(args.input_file,orient='records', lines = True)
    json_data= data
    # json_data = data[['input', args.col_name,'problem_id']]


    invalid_indices = filter_invalid_runs(json_data)
    # json_data[args.col_name] = updated_second_try
    print("Error pairs",len(error_list))
    print("Functionally incorrect",len(invalid_indices))
    print("Invalid slow codes",len(invalid_input_runs))
    print("Original Data length",len(json_data))
    print("Insufficient Input Files",len(insufficient_input_files))
    json_data = json_data.drop(invalid_indices).reset_index(drop=True)
    print("After filtering json length",len(json_data))
    # json_data.to_json(f'{args.out_path}/filtered_{args.col_name}_{file_name}', orient='records', lines=True)
    json_data.to_json(f'{args.out_path}/filtered_with_all_params_{file_name}', orient='records', lines=True)

    if not os.path.exists('./error_logs'):
        os.makedirs('./error_logs/')
    
    error_file = f"./error_logs/error_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"

    with open(error_file, 'w') as json_file:
        json.dump(error_list, json_file)

    


      
      



