import pandas as pd
import os
import yaml
import sys
import json
import random
import hashlib
import argparse
from typing import Literal
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
device = t.device("cuda" if t.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("--generation-method", type=str,
                    help="Generation Method (Local or OpenAI)",
                    default="gpt", choices=["gpt", "local"])
parser.add_argument("--model-name", type=str, help="Model Name",
                    default="gpt-4.1-mini")
parser.add_argument("--hf-personas", type=bool,
                    help="Specify if using personas from HF or local",
                    default=False)


args = parser.parse_args()


class OpenaiResponse(BaseModel):
    response: str


def write_to_json(file, file_path):
    with open(file_path, 'w') as f:
        json.dump(file, f)


def read_json(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file


if args.generation_method == "gpt":
    # print(f"Using GPT Generation. Model Used: {args.model_name}")
    model = outlines.from_openai(OpenAI(), args.model_name)
else:
    # print(f"Using Local Generation. Model Used: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=t.float16
    ).to(device)

    # Create outlines-wrapped model
    model = outlines.from_transformers(hf_model, tokenizer,  temperature=0)

synthetic_sjts = read_json('sjt_data/synthetic_generated_sjt_list.json')
synthetic_sjts_v2 = read_json('sjt_data/synthetic_generated_sjt_list_v2.json')

synthetic_sjts = synthetic_sjts + synthetic_sjts_v2

if args.hf_personas:
    # print("Using Huggingface Personas")
    hf_persona_dataset = load_dataset("thoughtworks/psychometric_personas")
    persona_datasets = hf_persona_dataset['train']
    # print(f"No of Personas: {len(persona_datasets)}")
else:
    # print("Using Local Personas")
    with open('../configs/personas_v2.yaml', 'r') as file:
        local_personas = yaml.safe_load(file)

    job_title = 'law_enforcement'
    persona_datasets = local_personas[job_title]['personas']

    # print(f"No of Personas: {len(persona_datasets)}")


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


def local_answer(prompt, answer_options=[0, 1, 2, 3, 4, 5]):
    response = model(prompt, Literal[*answer_options])
    return response


def openai_answer(prompt):
    response = model(prompt, OpenaiResponse, temperature=0)
    return json.loads(response)['response']


def generate_answers(sjt_list, batch_size=5, answer_shuffle=False):

    answer_index = [0, 1, 2, 3, 4, 5]

    if answer_shuffle:
        random.shuffle(answer_index)
    sjt_answers = []

    print(f"No of personas: {len(persona_datasets)}")
    print(f"No of SJTs: {len(sjt_list)}")

    for persona_dataset in tqdm(persona_datasets, desc="Personas", position=0):
        if args.hf_personas:
            persona = persona_dataset['persona_text']
        else:
            persona = ",\n".join([f"{key} : {persona_dataset[key]}" for key in 
                                  persona_dataset if key != 'uuid'])

        persona_answers = []
        question_hashes = []
        for sjt_batch in tqdm(batch_list(sjt_list, batch_size), desc="Batches", position=1):
            prompt_list = []
            hash_list = []
            for sjt in sjt_batch:

                answer_options = [sjt[key] for key in sjt.keys() if "_option" in key]
                answer_options = [answer_options[idx] for idx in answer_index]
                question = sjt['question']

                prompt = persona_sjt_template(question=question,
                                              attributes=persona.split('script_version')[0],
                                              answer_options=list_to_str(answer_options))
                prompt_list.append(prompt)
                hash_list.append(sjt['hash_id'])

            if args.generation_method == "gpt":
                with ProcessPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(openai_answer, prompt_list))
            else:
                with ProcessPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(local_answer, prompt_list))

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
    sjt_answers = generate_answers(synthetic_sjts)
    write_to_json(sjt_answers, "sjt_answers_handmade_personas_62_sjts.json")