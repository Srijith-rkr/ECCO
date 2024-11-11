#!/bin/sh 
#SBATCH --partition=general
#SBATCH --mem=50Gb
#SBATCH --cpus-per-task=8
#SBATCH -t 0-15:00:00              # time limit:  add - for days (D-HH:MM) 
#SBATCH --job-name=nl-exec-refine 
#SBATCH --error=/home/srijithr/course_hw/anlp_project/eval_jobs/%x__%j.err
#SBATCH --output=/home/srijithr/course_hw/anlp_project/eval_jobs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu


source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate ecco

cd /home/srijithr/course_hw/anlp_project

# running it on cpu
# edit - judge 0 
# python ECCO/evaluation/edit_eval.py \
# --judge_url http://ec2-18-220-179-89.us-east-2.compute.amazonaws.com:2358 \
# --code_col_name generated_codes \
# --input_path /home/srijithr/course_hw/anlp_project/ECCO/outputs/srijith_inference_outputs/edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex0_samples1_2024-11-10_08:21:27.jsonl

# might have failed - lets-see
# few-shot- judge 0-5
# python ECCO/evaluation/edit_eval.py \
# --judge_url http://ec2-18-224-71-117.us-east-2.compute.amazonaws.com:2358  \
# --code_col_name generated_codes \
# --input_path /home/srijithr/course_hw/anlp_project/ECCO/outputs/srijith_inference_outputs/edit_deepseek_instruct_nrowsNone_tokens1024_temp0.4_fewshotex2_samples1_2024-11-10_08:26:33.jsonl


# self-refine - judge 0-2
# python ECCO/evaluation/edit_eval.py \
# --judge_url http://ec2-13-58-251-169.us-east-2.compute.amazonaws.com:2358 \
# --code_col_name generated_codes_1 \
# --input_path /home/srijithr/course_hw/anlp_project/ECCO/outputs/srijith_inference_outputs/self-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_08:31:40.jsonl


# exec-refine - judge 0-3
# python ECCO/evaluation/edit_eval.py \
# --judge_url http://ec2-18-219-231-121.us-east-2.compute.amazonaws.com:2358 \
# --code_col_name generated_codes_1 \
# --input_path /home/srijithr/course_hw/anlp_project/ECCO/outputs/srijith_inference_outputs/exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-10_08:42:21.jsonl

# nl-exec-refine - judge 0-6
python ECCO/evaluation/edit_eval.py \
--judge_url http://ec2-3-15-210-12.us-east-2.compute.amazonaws.com:2358 \
--code_col_name generated_codes_1 \
--input_path /home/srijithr/course_hw/anlp_project/ECCO/outputs/srijith_inference_outputs/nl-exec-refine_deepseek_nrowsNone_tokens1024_temp0.4_samples1_numrefine1_2024-11-11_01:24:24.jsonl



