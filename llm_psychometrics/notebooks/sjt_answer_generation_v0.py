import pandas as pd
import os
import yaml
import sys
import json
import random
import hashlib
from itertools import product
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import outlines
from outlines import Generator
from openai import OpenAI
from datasets import Dataset, load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.append("../")
from src.utils import list_to_str, openai_api_call
device = "cpu"


class OpenaiResponse(BaseModel):
    response: str

def write_to_json(file, file_path):
    with open(file_path, 'w') as f:
        json.dump(file, f)
        
def read_json(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file

openai_model_name = "gpt-4.1-mini"
openai_model = outlines.from_openai(OpenAI(), openai_model_name)

synthetic_sjts = read_json('sjt_data/synthetic_generated_sjt_list.json')

hf_persona_dataset = load_dataset("thoughtworks/psychometric_personas")
hf_persona_dataset = hf_persona_dataset['train']


persona_sjt_template = outlines.Template.from_string("""
<|im_start>user
You are a law enforcement officer with following attributes :

{{attributes}}

Task: Answer the below multiple choice questions:

Question: {{ question }}

Choices for these questions are:

{{ answer_options }}

Answer the question using one of the values from [1,2,3,4,5,6].
<|im_end>
<|im_start>assistant
""")


def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def openai_answer(prompt):
    response = openai_model(prompt, OpenaiResponse)
    return json.loads(response)['response']
        
def generate_answers(hf_persona_dataset, sjt_list, batch_size = 5,answer_shuffle=False):
    
    answer_index = [0,1,2,3,4,5]
    if answer_shuffle:
        random.shuffle(answer_index)
    sjt_answers = []
    for index, persona_dataset in enumerate(hf_persona_dataset):
        print(index)
        persona = persona_dataset['persona_text']
        
        persona_answers = []
        question_hashes = []
        for sjt_batch in tqdm(batch_list(sjt_list, batch_size), desc="Batches", position=2):
            prompt_list = []
            hash_list = []
            for sjt in sjt_batch:
                
                answer_options = [sjt[key] for key in sjt.keys() if "_option" in key]
                answer_options = [answer_options[idx] for idx in answer_index]
                question = sjt['question']
                
                prompt = persona_sjt_template(question = question,
                                attributes = persona.split('script_version')[0],
                                answer_options = list_to_str(answer_options))
                prompt_list.append(prompt)
                hash_list.append(sjt['hash_id'])
            
            
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(openai_answer, prompt_list))

            persona_answers.extend(results)
            question_hashes.extend(hash_list)
            
        persona_dict = {}
        persona_dict['script_version'] = "v0"
        persona_dict['model'] = "gpt_41_mini"
        persona_dict['config'] = {}
        persona_dict['config']['persona_id'] = persona_dataset['uuid']
        persona_dict['config']['question_hashes'] = question_hashes
        persona_dict['config']['answer_index'] = answer_index
        persona_dict['answers'] = persona_answers
        sjt_answers.append(persona_dict)
    return sjt_answers


if __name__ == "__main__":
    sjt_answers = generate_answers([hf_persona_dataset[0]],synthetic_sjts[:5])
    write_to_json(sjt_answers,"sjt_answers_sample.json")