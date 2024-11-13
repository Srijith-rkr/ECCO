import pandas as pd
import os
import json
import logging
from datasets import load_dataset
import numpy as np

GENERATED_OUTPUTS_DIR = "./judge_eval/generate_leander/"
#GENERATED_OUTPUTS_DIR = "./judge_eval/generate/"
TRAJECTORIES = "./leander_metrics/trajectories.csv"


def get_percentile(problem_values,trajectory_values):
    if len(trajectory_values)==0:
        return 1
    test_vals = list(problem_values.values())
    test_val = sum(test_vals)/len(test_vals)
    more_count = 0
    for val in trajectory_values:
        if val>test_val: 
            more_count+=1
    return more_count/len(trajectory_values)

def aggregate(lists):
    out_list = []
    for list in lists:
        out_list.extend(list)
    return out_list

#trajectories = pd.read_csv(TRAJECTORIES,converters={'cpu_time': pd.eval,'memory': pd.eval})
#print(trajectories.head())
#print(list(map(lambda x:x/1000,aggregate(list(trajectories[trajectories['problem_id']=='p02379']['cpu_time'])))))

submission_dataset = load_dataset('CodeEff/ECCO', 'generate_eval', split='test')
submission_data = submission_dataset.to_pandas()

results={}
for file in os.listdir(GENERATED_OUTPUTS_DIR):
    pass_1=[]
    time=[]
    memory=[]
    pass_rates=[]
    pass_rates_input_denom=[]
    with open(os.path.join(GENERATED_OUTPUTS_DIR,file),'r') as f:
        eval = json.load(f)
        for key in eval:
            ##Pass@1
            assert len(eval[key]['accepted'])==1,"More than one element in accepted list"
            pass_1.append(1 if eval[key]['accepted'][0] and len(eval[key]['pass_all'])==20 else 0) ##CHANGE HERE#####
            #pass_1.append(1 if eval[key]['accepted'][0] and len(eval[key]['errors_all'])==0 else 0)
            
            ##Pass Rates
            pass_rates.append(len(eval[key]['pass_all'])/20)
            pass_rates_input_denom.append(eval[key]['pass_rate'])
                
            if len(submission_data[submission_data['problem_id']==eval[key]['problem_id']])>0:
                if eval[key]['accepted'][0] and len(eval[key]['pass_all'])==20:
                #if eval[key]['accepted'][0] and len(eval[key]['errors_all'])==0: ##CHANGE HERE#####
                    ##Time Percentile
                    #trajectory_cpu_times = list(map(lambda x:x/1000,aggregate(list(trajectories[trajectories['problem_id']==eval[key]['problem_id']]['cpu_time']))))
                    trajectory_cpu_times = submission_data[submission_data['problem_id']==eval[key]['problem_id']]['runtimes'].iloc[0]
                    time.append(get_percentile(eval[key]['run_time_all'],trajectory_cpu_times))
                
                    ##Memory Percentile
                    trajectory_memories = submission_data[submission_data['problem_id']==eval[key]['problem_id']]['memories'].iloc[0]
                    memory.append(get_percentile(eval[key]['memory_all'],trajectory_memories))
                # print(trajectory_memories)
                # print(trajectory_cpu_times)
                # else:
                #     time.append(0.5)
                #     memory.append(0.5)
            else:
                logging.warning(f"Problem {eval[key]['problem_id']} not found in trajectories")
            
    results[file]={'pass_1':np.mean(pass_1)*100,'time':f"{np.mean(time)*100} +- {np.std(time)*100}",'memory':f"{np.mean(memory)*100} +- {np.std(memory)*100}","pass_rates":f"{np.mean(pass_rates)*100} +- {np.std(pass_rates)*100}"}
    with open("results.json",'w') as f:
        json.dump(results,f,indent=2)
    
print(results)

#gen_code_data = pd.read_json(args.input_path, nrows=args.nrows, orient='records', lines=True)

#print(submission_data[submission_data['problem_id']=='p02624']['runtimes'].iloc[0])
# print(submission_data.head())
# print(len(submission_data))
# print(len(submission_data['problem_id'].unique()))
# print(type(list(submission_data['runtimes'])[0][0]))