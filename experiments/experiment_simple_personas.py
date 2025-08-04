import llm_psychometrics
import torch as t
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import yaml
import random
import itertools
import json

import yaml
from importlib.resources import files

hexapro_path = files('llm_psychometrics.psychometric_tests').joinpath('hexaco_100_eval.yaml')
personas_path = files('llm_psychometrics.configs').joinpath('personas.yaml')
generation_path = files('llm_psychometrics.configs').joinpath('generation_config.yaml')

with generation_path.open('r') as f:
    generation_config = yaml.safe_load(f)

device = t.device("cuda" if t.cuda.is_available() else "cpu")


model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text_generation_pipeline = pipeline(task="text-generation", model=model_id, device=device)


with persona_path.open('r')  as f:
    persona_config = personas_path.safe_load(f)
    
with open('../psychometric_tests/hexapro_100_questions.yaml', 'r') as file:
    question_list = yaml.safe_load(file)


def persona_curation(base_text, job_persona):
    return f"""{base_text} who is a {job_persona['Summary']}. You are {job_persona['Age']} years old, based out of {job_persona['Location']}.
You have a background as a {job_persona['Background']}, you are {job_persona['Personality Traits']}
and {job_persona['Style']}"""

def prompt_formatting(persona, base_prompt, questions):
    prompt = f"{persona} \n Task: {base_prompt} \n Question: \n {questions} \n Answer:"
    return prompt
