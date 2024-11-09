import pandas as pd
import numpy as np
import sys
import argparse
import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer



class CodeOptimizer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(args.model)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.data = pd.read_csv(args.train_input, sep='\t')
        self.few_shot_count = args.few_shot_count
        self.generated_codes = []
        self.clean_data()

    def clean_data(self):
        self.data=self.data[self.data['language']=='Python']
        self.data = self.data[['code_v0', 'code_v1','problem_id']]
        self.data = self.data.groupby('problem_id').first().reset_index()

        self.data['target']=self.data['code_v1']
        self.data['input']=self.data['code_v0']
        size = len(self.data)
        self.train, self.test = self.data.iloc[:int(0.5*size)], self.data.iloc[int(0.5*size):]
        print("Data cleaned")

    def get_prompt(self,idx):
        row = self.test.iloc[idx]

        prompt = ''
        # prompt = f'###Given below are {num_examples} examples of slow and fast code formatted as (slow code) -> (fast code) with 1 line gap in between the slow code, the arrow -> and also in between the arrow and fast code. There is a gap of 4 lines and 30 hash (#) between every example. Replace in the end <fast_code> and return only replaced code### \n\n'
        prompt_example_rows = self.train.sample(self.few_shot_count)
        prompt_example_rows['prompt'] = '# slower version:\n\n' + prompt_example_rows['code_v0'] + '\n\n# optimized version of the same code:\n\n' + prompt_example_rows['code_v1']
        prompt += prompt_example_rows['prompt'].str.cat(sep='\n\n')


        prompt += '\n\n# slower version:\n'
        prompt += row['code_v0']
        slow_code=row['code_v0']

        return prompt, slow_code,row['problem_id']


    def get_fast_code(self):
        for idx in range(3):
            try:
                prompt, slow_code, problem_id = self.get_prompt(idx)
                input_ids = self.tokenizer(prompt, return_tensors="pt",truncation=True).to(self.device)
                generated_code = self.model.generate(**input_ids ,max_new_tokens=300)
                generated_string = self.tokenizer.decode(generated_code[0])
                # refined_string= generated_string.split("Optimized version of Program above:")[-1]
                pair = {'input': slow_code, 'target': generated_string, 'problem_id': problem_id}
                self.generated_codes.append(pair)
                print(f"#Fast Code Generated for index {idx}")
            except Exception as e:
                print(f"Error:{e} for {idx}")


        with open(args.output_file, 'w') as json_file:
            json.dump(self.generated_codes, json_file)





def main():
    code_optimizer = CodeOptimizer()
    code_optimizer.get_fast_code()
    print("Output File generated")
    





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("few_shot_count",type=int)
    args = parser.parse_args()

    main()
