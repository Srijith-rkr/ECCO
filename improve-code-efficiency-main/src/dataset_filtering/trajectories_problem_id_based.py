import pandas as pd
import os

metadata_folder = '/data/tir/projects/tir6/general/swaghjal/Project_CodeNet/metadata'
python_solutions_folder = '/data/tir/projects/tir6/general/swaghjal/Project_CodeNet/data'


trajectories_dataframe = pd.DataFrame(columns=['user_id', 'problem_id','original_language', 'trajectories', 'submission_id',"status","cpu_time","memory","code_size","accuracy"])
invalid_file_ids = []

from lib2to3 import refactor
avail_fixes = refactor.get_fixers_from_package('lib2to3.fixes')
py_converter = refactor.RefactoringTool(avail_fixes)
def convert_2to3(py_script):
    try:
        # convert python2 to python3
        # taken from https://stackoverflow.com/questions/30340151/using-2to3-on-in-memory-scripts
        # if the script does not end with a newline, add one
        added_newline = False
        if py_script[-1] != '\n':
            py_script += '\n'
            added_newline = True
        ast = py_converter.refactor_string(py_script, '<script>')
        converted_code = str(ast)
        if added_newline:
            converted_code = converted_code[:-1]
        return converted_code
    except Exception as e:  # if 2to3 fails, just return the original code
        return py_script


# Read the CSV file into a pandas DataFrame
def update_trajectories(csv_file_path,trajectories_dataframe):
    df = pd.read_csv(csv_file_path)
    print(f'Processing {csv_file_path}...')

    # Filter submissions only for the Python language
    python_submissions = df[(df['language'] == 'Python') & (df['status'] == 'Accepted')]


    # Create a dictionary to store trajectories for each combination of user_id and problem_id
    trajectories = {}

    # Iterate over each row in the filtered DataFramee
    for index, row in python_submissions.iterrows():
        user_id = row['user_id']
        problem_id = row['problem_id']
        submission_id = row['submission_id']
        filename_ext = row['filename_ext']
        print(f'Processing submission {index+1} with submission_id {submission_id} and problem_id {problem_id}', end='\r')
        
        # Check if the problem_id folder exists, if not create it
        file_path = f'{python_solutions_folder}/{problem_id}/Python/{submission_id}.{filename_ext}'
        if not os.path.exists(file_path):
            invalid_file_ids.append([submission_id,problem_id])
            continue
        if row['status']!='Accepted':
            print("Not Accepted Solution")
        
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Add the submission code to the trajectories dictionary
        trajectory_key = problem_id
        if trajectory_key not in trajectories:
            trajectories[trajectory_key] = {'problem_id': problem_id, 'original_language': row['original_language'],'trajectories': [], 'submission_id': [], 'status': [], 'cpu_time': [], 'memory': [], 'code_size': [], 'accuracy': []}
        trajectories[trajectory_key]['trajectories'].append(convert_2to3(code))
        trajectories[trajectory_key]['submission_id'].append(submission_id)
        trajectories[trajectory_key]['status'].append(row['status'])
        trajectories[trajectory_key]['cpu_time'].append(row['cpu_time'])
        trajectories[trajectory_key]['memory'].append(row['memory'])
        trajectories[trajectory_key]['code_size'].append(row['code_size'])
        trajectories[trajectory_key]['accuracy'].append(row['accuracy'])

    
    new_df = pd.DataFrame.from_dict(trajectories, orient='index')
    trajectories_dataframe = pd.concat([trajectories_dataframe, new_df], ignore_index=True)
    return trajectories_dataframe




    
for filename in os.listdir(metadata_folder):
    if filename.endswith('.csv') and filename != 'problem_list.csv':
        csv_file_path = os.path.join(metadata_folder, filename)
        trajectories_dataframe=update_trajectories(csv_file_path, trajectories_dataframe)

trajectories_dataframe.to_json('/data/tir/projects/tir6/general/swaghjal/trajectories_per_id_full_with_2to3.jsonl', orient='records', lines=True)

print("Submission id path not found:",len(invalid_file_ids))
with open('/data/tir/projects/tir6/general/swaghjal/invalid_file_ids.txt', 'w') as f:
    for item in invalid_file_ids:
        f.write("%s\n" % item)
