import subprocess
import sys
import pandas as pd
import os
from time import time
import json
import psutil

# slow_code = """
# import sys

# val=input()
# val2=input()
# # print(val)
# val = int(val)

# for i in range(val):
#     print(val2*i)
# """
# Define the command to run your script with input parameters
  # Replace "5" and "7" with your desired input parameters
# test_file = 'input1.txt'
# Run the script

base_file_path = '/mnt/c/Users/Sid/Desktop/Directed Study/PIE/pie-perf-new/pie-perf/data/codenet/public_test_cases/'

def run_cmd(code, input_file_path):
    command = ["python3", "-c",code]

    def _kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            # logging.info(f"Killing {proc}")
            proc.kill()
        # logging.info(f"Killing {process}")
        process.kill()

    try:
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
                        print("Script executed successfully!")
                        print("Script output:")
                        print(stdout)
                        return 1
                    else:
                        print("Script encountered an error!")
                        print("Error output:")
                        print(stderr)
                        return None
    except subprocess.TimeoutExpired:
        # print(f"Timeout for {args}")
        _kill(proc.pid)  # type: ignore
        return None

# run_cmd(slow_code,test_file)


def calculate_speedup(json_data):
    speedup_data=[]

    for index,row in json_data.iterrows():
        slow_code = row['input']
        fast_code = row['target']
        problem_id = row['problem_id']

        problem_input_folder = base_file_path+problem_id
        if not os.path.exists(problem_input_folder):
            continue
        data = os.listdir(problem_input_folder)
        file_count = len(data)
        if file_count<2:
             continue
        # input_files_count = file_count/2

        # for i in range(input_files_count):
            #   input_file_id = f'/input.{i+1}.txt'
            #   input_file=problem_input_folder+ input_file_id

            #   input_start_time=time()
            #   run_cmd(slow_code,input_file_path=input_file)
            #   input_total_time=time()-input_start_time

            #   target_start_time = time()
            #   run_cmd(fast_code,input_file_path=input_file)
            #   target_total_time = time() -target_start_time

        input_file_id = '/'+data[0]
        # print(input_file_id)
        input_file=problem_input_folder+ input_file_id

        input_start_time=time()
        input_run = run_cmd(code = slow_code,input_file_path=input_file)
        input_total_time=time()-input_start_time

        target_start_time = time()
        output_run = run_cmd(code = fast_code,input_file_path=input_file)
        target_total_time = time() -target_start_time

        speedup=(input_total_time)/target_total_time
        rtr = float(input_total_time-target_total_time)/input_total_time
        rtr=rtr*100
        if input_run and output_run:
             
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
             
        speedup_data.append(values)
    with open('result_per_id_large_codellama', 'w') as json_file:
        json.dump(speedup_data, json_file)

    








if __name__=='__main__':
      
    #   json_file=sys.argv[1]
      json_data = pd.read_json('codellama_generated_code.json')
      calculate_speedup(json_data)


      
      



