#!/bin/sh 
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=general
#SBATCH --mem=100
#SBATCH --cpus-per-task=8
#SBATCH -t 0-16:00:00              # time limit:  add - for days (D-HH:MM) 
#SBATCH --job-name=ecco_
#SBATCH --error=/home/srijithr/course_hw/anlp_project/job_outputs/%x__%j.err
#SBATCH --output=/home/srijithr/course_hw/anlp_project/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu


source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate ecco

export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir


cd /home/srijithr/course_hw/anlp_project
python ECCO/experiments/inference.py --eval_mode edit
python ECCO/experiments/inference.py --eval_mode edit --few_shot_examples 2
python ECCO/experiments/inference.py --eval_mode self-refine
python ECCO/experiments/inference.py --eval_mode exec-refine
python ECCO/experiments/inference.py --eval_mode nl-exec-refine

