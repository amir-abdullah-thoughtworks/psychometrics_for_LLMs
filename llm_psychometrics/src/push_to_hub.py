from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# Load the local JSONL into a Dataset
dataset = load_dataset("json", data_files="v12.jsonl", split="train")

# Optionally inspect
print(dataset)
print(dataset[0])

# Push to hub
dataset.push_to_hub("thoughtworks/psychometric_personas", private=True)
