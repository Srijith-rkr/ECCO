import subprocess
import sys
import pandas as pd
import os
from time import time
import json
import psutil



# base_file_path = '/mnt/c/Users/Sid/Desktop/Directed Study/PIE/pie-perf-new/pie-perf/data/codenet/public_test_cases/'
base_file_path = '/home/swaghjal/data/codenet/public_test_cases/'
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
                        # print("Script executed successfully!")
                        # print("Script output:")
                        # print(stdout)
                        return 1
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


def calculate_speedup(json_data,file_name):
    speedup_data={}

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        for index,row in json_data.iterrows():
            print(index)
            slow_code = row['input']
            fast_code = row['target']
            problem_id = row['problem_id']

            problem_input_folder = base_file_path+problem_id
            if not os.path.exists(problem_input_folder):
                print(f"Does not exist {problem_input_folder}")
                continue
            data = os.listdir(problem_input_folder)
            data = sorted(data)
            file_count = len(data)
            if file_count<2:
                print(f"Not enough test files for {problem_id}")
                continue

            input_file_id = '/'+data[0]
            # print(input_file_id)
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
            if input_run and output_run:

                values = {f'input_run_time_{epoch}':input_total_time,
                            f'target_run_time_{epoch}':target_total_time,
                            f'speedup_{epoch}':speedup,
                            f'rtr_{epoch}':rtr,
                            f'problem_id_{epoch}':problem_id}
                if index in speedup_data:
                    values.update(speedup_data[index])
            else:
                values = {f'input_run_time_{epoch}':None,
                            f'target_run_time_{epoch}':None,
                            f'speedup_{epoch}':None,
                            f'rtr_{epoch}':None,
                            f'problem_id_{epoch}':problem_id}
                
            speedup_data[index]=(values)

    speedup_data_values = list(speedup_data.values())
    with open(f'/home/swaghjal/Llama_Prompting/vllm_results/vllm_average/average_filtered_deepseek_generated_codes_temp_0.4_100_0_1024_7b.jsonl', 'w') as json_file:
        json.dump(speedup_data_values, json_file)

    








if __name__=='__main__':
      
      input_file=sys.argv[1]
      json_data = pd.read_json(input_file,orient='records', lines = True)
      calculate_speedup(json_data,input_file)


      
      



