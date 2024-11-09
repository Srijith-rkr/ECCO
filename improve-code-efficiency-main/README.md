# Improve Code Efficiency

This repository contains the source code for the Improve Code Efficiency project

The `src/` consists of the primary codebase:
1. `src/dataset_filtering/` consists of the scripts used to filter the PIE dataset based on functional correctness. 
2. `src/evaluation` consists of the script to run evaluation of model generated code on the Judge0 environment server present at the URL `http://13.58.18.211:2358/`
3. `src/experiments` consists of the scripts to run modelling experiment. 
   `model_classes.py` consists of the Inference Engine Classes for each model that is benchmarked 
   `inference.py` is the entrypoint for running the experiments

   NOTE: `self_refine.py` and `cannonical_refine.py` were experiments run independant of the `model_classes.py` structure, but will be integrated into `inference.py` that uses the model classes.

---

# Improve Code Efficiency Python Dataset

A Python subset of the CodeNet/PIE dataset with functionally identical pairs removed. Additionally, it includes curated trajectories for user-problem_id pairs and a file with problem descriptions used in the dataset.

## Dataset Files

All dataset files can be accessed from the following Google Drive link: [Improve Code Efficiency Python Dataset](https://drive.google.com/drive/folders/1g85ZJOlg9erT7qyxpYSnixsB0tLcE-Oz?usp=sharing)

The structure of the repository is as follows:

1. **python_noDuplicates**:
   - This folder contains the Python subset files of the CodeNet data with identical pairs removed. They are divided into training, testing, and validation sets.

2. **python_noDuplicates_functionallyCorrect**:
   - This folder contains the filtered subset of the Python deduplicated dataset where functionally incorrect pairs have been filtered out.

3. **problem_description.jsonl**:
   - This file includes problem descriptions used in the dataset. Each line in the file represents a problem description along with the problem_id in JSON format.

4. **trajectories.csv**:
   - This CSV file contains our curated trajectories for user->problem_id pairs. Each row in the file represents a trajectory for a specific user and problem_id combination, along with additional information such as submission_id, status, CPU time, memory usage, code size, and accuracy.

## Usage

To use the dataset, you can download the required files from the Google Drive link provided above. Depending on your specific needs, you may use the filtered subsets for training machine learning models or analyze the curated trajectories for user-problem_id pairs.

---

