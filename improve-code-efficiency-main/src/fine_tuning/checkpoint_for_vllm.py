from datetime import datetime
import os
import sys
import argparse

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftConfig, PeftModel,LoraConfig
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from accelerate import Accelerator

# Create an argument parser
parser = argparse.ArgumentParser()

# Add arguments with default values
# parser.add_argument("--train_path", default="/data/tir/projects/tir6/general/vveerend/improving-code-efficiency/shared/data/filtered_sped_up_train.jsonl", help="Path to the training data file")
# parser.add_argument("--eval_path", default="/data/tir/projects/tir6/general/vveerend/improving-code-efficiency/shared/data/filtered_sped_up_val.jsonl", help="Path to the evaluation data file")
parser.add_argument('--model', default='deepseek-ai/deepseek-coder-7b-instruct-v1.5')
parser.add_argument('--lora', default=True)
parser.add_argument('--instruct', default=True, action='store_true') # Check this
parser.add_argument('--markdown_format', default=True, action='store_true') # deepseek specifice adjustment
parser.add_argument('--device_map', default=True, action='store_false')
parser.add_argument('--exec_feedback', default=False, action='store_true')
parser.add_argument('--hub_model_name', default=None)
parser.add_argument('--wandb_run_name', default='DeepSeek-7B-history_based_SFT')
# Train args
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)  # They finetune on 4 gpus with per device batch size of 2
# adjust 
parser.add_argument('--lora_rank', type=int, default=8) 
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--log_interval', type=int, default=10)

# Parse the arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # for single gpu
#  launch script with python for single gpu
#   for multi-gpu do : accelerate launch --config_file sft_cfg.yaml pair_finetuning.py 
# tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-7b-instruct-v1.5')

# tokenizer.push_to_hub('deepseek_SFT_history')

base_model = args.model
model = AutoModelForCausalLM.from_pretrained(
    # base_model,
    'Srijith-rkr/deepseek_SFT_history',

    # load_in_8bit = True,
    torch_dtype=torch.bfloat16,
    # sri
    device_map= 'auto', #None, #"auto" if args.device_map else None  to run with acceleatr
)


model = PeftModel.from_pretrained(model, '/home/srijithr/course_hw/anlp_project/finetuned_checkpoints/checkpoints_2024-11-10_11:07:22/checkpoint-6048') 
merged_model = model.merge_and_unload()

merged_model.push_to_hub('deepseek_SFT_history')

save_path = '/home/srijithr/course_hw/anlp_project/finetuned_checkpoints/checkpoints_2024-11-10_11:07:22/checkpoint-final/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path = os.path.join(save_path, 'merged_model.bin')

print('Saving final model to', save_path)
torch.save(merged_model.state_dict(), save_path)
print('Model saved successfully')
merged_model.push_to_hub(args.hub_model_name)
# 
# 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
