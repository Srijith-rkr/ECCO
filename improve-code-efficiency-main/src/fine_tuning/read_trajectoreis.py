import pandas as pd 
import json
import ast
path = '/home/srijithr/course_hw/anlp_project/ECCO/improve-code-efficiency-main/src/fine_tuning/trajectories.csv'
df = pd.read_csv(path)

# Using iterrows to iterate over rows
lens = []
for index, row in df.iterrows():
    trajectories = ast.literal_eval(row['trajectories'])
    lens.append(len(trajectories))

from collections import Counter
counts = Counter(lens)
for value, count in counts.items():
    print(f"Value {value} appears {count} times.")

print('debug')
