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
from datasets import load_dataset

# Create an argument parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument("--train_path",type=str, default="/data/tir/projects/tir6/general/vveerend/improving-code-efficiency/shared/data/train_5_filter_spedup.jsonl", help="Path to the training data file")
parser.add_argument("--eval_path",type=str, default="/data/tir/projects/tir6/general/vveerend/improving-code-efficiency/shared/data/filtered_sped_up_val.jsonl", help="Path to the evaluation data file")

# Parse the arguments
args = parser.parse_args()

# Assign the argument values to variables
train_file = args.train_path
eval_file = args.eval_path


train_dataset = load_dataset("json", data_files=train_file)
eval_dataset = load_dataset("json", data_files=eval_file)

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Optimize the python program below to be functionally equivalent but run faster and use less memory.\
            Wrap the optimized code in a block of 3 backticks (```).

    ## Program:
    {data_point["input"]}

    ## Optimized (Runtime and Space) version of Program above:
    {data_point["target"]}

    """
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.train() # put model back into training mode
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=64,
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

resume_from_checkpoint = "" # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

wandb_project = "Optim-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

from dataclasses import dataclass
@dataclass
class CustomDataCollator:
    """
    Data collator that will return the input_ids, attention_mask, and labels from the tokenized dataset.
    """
    def __call__(self, examples):
        # if isinstance(examples):
        #     examples = examples['train']

        batch = {
            'input_ids': torch.stack([torch.tensor(ex['input_ids']) for ex in examples]),
            'attention_mask': torch.stack([torch.tensor(ex['attention_mask']) for ex in examples]),
            'labels': torch.stack([torch.tensor(ex['labels']) for ex in examples])
        }
        return batch
data_collator = CustomDataCollator()

batch_size = 128
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = f"/data/tir/projects/tir6/general/swaghjal/checkpoints/checkpoints_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        # max_steps=400,
        num_train_epochs=1,
        learning_rate=1e-3,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset['train'],
    eval_dataset=tokenized_val_dataset['train'],
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

merged_model = model.merge_and_unload()

torch.save(merged_model.state_dict(), f"{output_dir}/checkpoint-final/merged_model.bin")
