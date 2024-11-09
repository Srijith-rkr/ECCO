from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument("--model_path", required=True, help="Path to the saved model state dictionary")
parser.add_argument("--model_name", required=True, help="Name of the model for AutoModelForCausalLM and AutoTokenizer")
parser.add_argument("--push_to_hub_name", required=True, help="Name to push the model and tokenizer to Hugging Face Hub")

# Parse the arguments
args = parser.parse_args()

# Load the saved model's state dictionary into an AutoModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load(args.model_path, map_location=device)

# Create a new AutoModelForCausalLM instance with the specified config
model = AutoModelForCausalLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Load the fine-tuned model's state dictionary into the new model
model.load_state_dict(model_state_dict)

# Push the model and tokenizer to Hugging Face Hub
model.push_to_hub(args.push_to_hub_name)
print("Model pushed to the Hugging Face Hub successfully!")

print("Pushing Tokenizer now")
tokenizer.push_to_hub(args.push_to_hub_name)
