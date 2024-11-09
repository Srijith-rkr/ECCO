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
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from sklearn.model_selection import train_test_split

# Create an argument parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument("--train_path", default="/data/tir/projects/tir6/general/swaghjal/nl_finetuning_traindata.jsonl", help="Path to the training data file")
# parser.add_argument("--eval_path", default="/data/tir/projects/tir6/general/vveerend/improving-code-efficiency/shared/data/filtered_sped_up_val.jsonl", help="Path to the evaluation data file")
parser.add_argument('--model', default='/data/models/huggingface/meta-llama/CodeLlama-7b-Python-hf')
parser.add_argument('--lora', default=True)
parser.add_argument('--instruct', default=False, action='store_true')
parser.add_argument('--markdown_format', default=False, action='store_true')
parser.add_argument('--device_map', default=True, action='store_false')
parser.add_argument('--exec_feedback', default=False, action='store_true')
parser.add_argument('--hub_model_name', default=None)
parser.add_argument('--wandb_run_name', default=None)


# Train args
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lora_rank', type=int, default=8)
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--log_interval', type=int, default=1000)

# Parse the arguments
args = parser.parse_args()

# Assign the argument values to variables
train_file = args.train_path
train_dataset = load_dataset("json", data_files=train_file, split='train')

# Split the dataset into training and evaluation sets
data = train_dataset.train_test_split(test_size=0.2)
train_dataset, eval_dataset = data['train'], data['test']

# Convert to Dataset format
# train_dataset = Dataset.from_dict(train_data)
# eval_dataset = Dataset.from_dict(eval_data)

base_model = args.model
print(f"Using model: {base_model}")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto" if args.device_map else None
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
        return_tensors=None,
    )
    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

def build_prompt(row):
    prompt = f"""'Write a python code which is efficient in terms of runtime and memory usage for the following problem description\
            Wrap the optimized code in a block of 3 backticks (```).\n\n"""
    prompt += f"""##Details:\n{row['problem_description']}\n"""
    prompt += f"""##Solution:\n"""
    return prompt

def generate_and_tokenize_prompt(data_point, instruct=args.instruct, markdown=args.markdown_format):
    wrap_string = "```" if not markdown else "```python"
    response = f"{wrap_string}\n{data_point['fastest_code']}```"
    full_prompt = build_prompt(data_point)

    if not instruct:
        full_seq = full_prompt + response
    else:  # If chat template to be used 
        messages = [
            {'role': 'user', 'content': full_prompt},
            {'role': 'assistant', 'content': response}
        ]
        full_seq = tokenizer.apply_chat_template(messages, tokenize=False)

    return tokenize(full_seq)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.train()  # put model back into training mode
# model = prepare_model_for_int8_training(model)

if args.lora:
    print('LORA HAPPENING')
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

resume_from_checkpoint = ""  # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

wandb_project = "NL2Code-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

batch_size = args.batch_size
per_device_train_batch_size = args.batch_size
# gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = f"/data/tir/projects/tir6/general/swaghjal/checkpoints/nl2code_checkpoints_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    # gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    # max_steps=400,
    num_train_epochs=10,
    learning_rate=1e-3,
    fp16=True,
    optim="adamw_torch",
    evaluation_strategy="steps",  # if val_set_size > 0 else "no",
    save_strategy="steps",
    logging_steps=args.log_interval,
    eval_steps=args.log_interval,
    save_steps=args.log_interval,
    output_dir=output_dir,
    # save_total_limit=3,
    load_best_model_at_end=False,
    # ddp_find_unused_parameters=False if ddp else None,
    group_by_length=True,  # group sequences of roughly the same length together to speed up training
    report_to="wandb",  # if use_wandb else "none",
    run_name=args.wandb_run_name if args.wandb_run_name else f"finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}" # if use_wandb else None,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)

trainer.train()
if trainer.accelerator.is_main_process: 

    if args.lora:
        merged_model = model.merge_and_unload()
    else:
        merged_model = model

    if not os.path.exists(f"{output_dir}/checkpoint-final/"):
        os.makedirs(f"{output_dir}/checkpoint-final/")

    print('Saving final model to', f"{output_dir}/checkpoint-final/merged_model.bin")
    torch.save(merged_model.state_dict(), f"{output_dir}/checkpoint-final/merged_model.bin")

    merged_model.push_to_hub(args.hub_model_name)
    # tokenizer.push_to_hub(args.hub_model_name)
