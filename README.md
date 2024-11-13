# ANLP Assignment-3 Project Proposal and Reimplementation of Baselines 

This repository contains the source code for reimplementing the paper "ECCO: Can We Improve Model-Generated Code Efficiency Without Sacrificing Functional Correctness?" and analysing the results.

## Dataset
The dataset is available on Huggingface at: [CodeEff/ECCO](https://huggingface.co/datasets/CodeEff/ECCO).

It consists of 2 subsets `edit` and `generate` each with 3 splits (`train`, `val` and `test`).

### Loading the dataset 
```python
dataset = load_dataset('CodeEff/ECCO', 'edit') # For history-based editing setting
dataset = load_dataset('CodeEff/ECCO', 'generate') # For nl-instructed generation setting
```


## Instructions to reproduce our results
### Environment setup
```bash 
conda env create -f environment.yml
conda activate ecco
```
### Download the test cases 
```sh
mkdir data && cd data
wget https://huggingface.co/datasets/CodeEff/ECCO/resolve/main/test_cases.zip
unzip test_cases.zip
```
### Setup a judge on AWS
   Guide can be found in the evaluation README

### Running inference for NL-based tasks (NL2Code)

#### Pre-Refine 

* DeepSeek
```sh
   python experiments/inference.py --eval_mode nl2code --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```
* CodeLLaMa-13b
```sh
   python experiments/inference.py --eval_mode nl2code --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```
#### Self-Refine (with Natural Language Feedback)

* DeepSeek
```sh
   python experiments/inference.py --eval_mode nl2code-self-refine --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```
* CodeLLaMa-13b
```sh
   python experiments/inference.py --eval_mode nl2code-self-refine --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```
#### Refine with Interpreter Feedback

* DeepSeek
```sh
   python experiments/inference.py --eval_mode nl-exec-refine --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```
* CodeLLaMa-13b
```sh
   python experiments/inference.py --eval_mode nl-exec-refine --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```
#### Refine with Interpreter Feedback and Natural Language

* DeepSeek
```sh
   python experiments/inference.py --eval_mode nl-exec-refine --judge_url http://<PUBLIC_URL>:2358 --model deepseek
```
* CodeLLaMa-13b
```sh
   python experiments/inference.py --eval_mode nl-exec-refine --judge_url http://<PUBLIC_URL>:2358 --model codellama_13b --num_gpus 2
```


## Experiments



### Code structure 
1. `evaluation` consists of scripts to run evaluation of model generated code on the Judge0 environment server hosted on AWS. Please see instructions to setup the evaluation server.
   - `edit_eval.py` is the script for evaluating code generated on the metrics for the history-based editing setting
   - `generate_eval.py` is the script for evaluating code generated on the metrics for the NL-instructed generation setting

   - `calculate_scores.py` is the script to calculate the Pass@1 accuracy with speedup and memory reduction.
   
2. `experiments` consists of the scripts to run modelling experiment. 
   - `model_classes.py` consists of the Inference Engine Classes for each model that is benchmarked.
   - `inference.py` is the entrypoint for running the experiments
   - `prompt_formats.py` and `utils.py` cotains utilities for prompt building and execution feedback formatting

### Starting up the evaluation setup 

![Judge Setup](https://github.com/user-attachments/assets/b3875151-336d-446f-961f-352e0d34ed6a)

Setup the evaluation setup with the guide in the [evaluation README](./evaluation/README.md)

### Running experiments / Generating Code
We run experiments to generate code from the `experiments/inference.py` entrypoint. An example is provided below:
```sh
python experiments/inference.py --model deepseek \
   --temperature 0.4 --num_samples 1 --eval_mode "edit" 
```

Model choices are in [the registry](https://github.com/CodeEff/ECCO/blob/80df5bb9c3145b8d673732fa13c50d9259e5d079/experiments/inference.py#L23)

`--eval_mode` choices are `['edit', 'nl2code', 'self-refine', 'exec-refine','nl2code-self-refine', 'nl-exec-refine', 'nl2code-exec-refine', 'nl2code-nl-exec-refine']` for the different experiments. Modes without the prefix `nl2code` correspond to the *history-based editing* setting and with the prefix refer to the *NL-instructed generation* paradigm.

## Citation 
```bib
@article{waghjale2024ecco,
  title={ECCO: Can We Improve Model-Generated Code Efficiency Without Sacrificing Functional Correctness?},
  author={Waghjale, Siddhant and Veerendranath, Vishruth and Wang, Zora Zhiruo and Fried, Daniel},
  journal={arXiv preprint arXiv:2407.14044},
  year={2024}
}
```