import json 
import os
from pathlib import Path
import pandas as pd
wd = Path(__file__).parent

# temp = '/home/srijithr/course_hw/anlp_project/ECCO/outputs/srijith_inference_outputs/edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_2024-11-10_08:26:33.jsonl'
# json_data = pd.read_json(temp, orient='records', lines = True)
# x = json_data.head(1).to_dict()

look_into = Path('/home/srijithr/course_hw/anlp_project/edit-public-temp')
files = os.listdir(look_into)


for file in files:

    with open(look_into / file, 'r') as f:
        data = json.load(f)


    num_of_questions = len(data)
    num_of_correct_questions = 0
    speeds = []
    mems = []

    for k,v in data.items():
        in_time_passed, out_time_passed = [], []
        in_mem_passed, out_mem_passed = [], []

        if v['input_accepted'] and len(v['output_pass_all']) ==20:
            num_of_correct_questions += 1

        # if v['output_accepted'] and len(v['output_pass_all']) and v['input_accepted']> 0:
        if v['output_accepted'] and v['input_accepted'] and len(v['output_pass_all']) > 0:

            for test_id_int in v['output_pass_all']:
                test_id = str(test_id_int)
                in_time_passed.append(v['input_run_time_all'][test_id])
                out_time_passed.append(v['output_run_time_all'][test_id])

                in_mem_passed.append(v['input_memory_all'][test_id])
                out_mem_passed.append(v['output_memory_all'][test_id])

            speeds.append( sum(in_time_passed) / sum(out_time_passed))
            mems.append(sum(in_mem_passed) / sum(out_mem_passed))


    print(file)
    print(f'Total num of questions: {num_of_questions}')
    print(f'Num of correct questions: {num_of_correct_questions}')
    print(f'Accuracy: {num_of_correct_questions/num_of_questions*100}%')
    print(f'Speed up: {sum(speeds) / len(speeds)}')
    print(f'Memory reduction: {sum(mems) / len(mems)}')
    print('\n')


