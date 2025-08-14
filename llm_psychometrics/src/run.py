import torch as t
from transformers import AutoTokenizer
from transformers import pipeline
import yaml
import json
import argparse

from utils import generate_combinations
from src.experiment import experiment_setup

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model Id for the experiment")

# defining arguments for the experiment, input from command line
args = parse_args()

model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)


text_generation_pipeline = pipeline(
    task="text-generation", model=model_id, device=device)


with open('../configs/personas.yaml', 'r') as file:
    persona_config = yaml.safe_load(file)

with open('../psychometric_tests/hexapro_100_questions.yaml', 'r') as file:
    question_list = yaml.safe_load(file)

with open('../configs/generation_config.yaml', 'r') as file:
    generation_config = yaml.safe_load(file)


pipeline_config = generation_config['pipeline_config']
base_prompt_config = generation_config['base_prompt_config']
text_generation_config = generation_config['text_generation_config']

base_prompt_config["tokenizer"] = tokenizer
config_combinations = generate_combinations(
    pipeline_config, text_generation_config)


job_title = "customer_service"
experiment_results = experiment_setup(job_title, question_list[:3], [
                                      config_combinations[0],
                                      config_combinations[5]],
                                      text_generation_pipeline,
                                      base_prompt_config)


with open('experiment_results.json', 'w') as f:
    json.dump(experiment_results, f)
