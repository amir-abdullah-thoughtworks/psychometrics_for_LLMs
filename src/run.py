import torch as t
from transformers import AutoTokenizer
from transformers import pipeline
import yaml
import json

from utils import generate_combinations
from src.experiment import experiment_setup

device = t.device("cuda" if t.cuda.is_available() else "cpu")


model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)


text_generation_pipeline = pipeline(
    task="text-generation", model=model_id, device=device)


with open('../configs/personas.yaml', 'r') as file:
    persona_config = yaml.safe_load(file)

with open('../psychometric_tests/hexapro_100_questions.yaml', 'r') as file:
    question_list = yaml.safe_load(file)

with open('../configs/generation_config.yaml', 'r') as file:
    generation_config = yaml.safe_load(file)


pipeline_config = {
    "temperature": [0, 0.1, 0.5, 1, 1.2, 1.5],
    "num_return_sequences": [1, 4, 6]
}

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
