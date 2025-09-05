import torch as t
import outlines
from outlines import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from typing import Literal
import argparse
from enum import Enum
import yaml
import pandas as pd
import numpy as np
import sys
import os
import json
import re
from huggingface_hub import login
from datasets import Dataset, load_dataset
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from src.utils import inverse_likert, list_to_str, batch_list
from src.prompt_templates import persona_hexaco_template, base_hexaco_template
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
parser.add_argument("--hf-token", type=str,
                    help="HuggingFace Authentication Token",
                    default=False)
parser.add_argument("--job-title", type=str,
                    help="Applicable only for Handmade Personas",
                    default="law_enforcement")
parser.add_argument("--hf-persona-path", type=str,
                    help="HuggingFace Path for Personas",
                    default="thoughtworks/psychometric_personas")


args = parser.parse_args()


class OpenaiResponse(BaseModel):
    response: str


login(args.hf_token)


with open('llm_psychometrics/configs/generation_config.yaml', 'r') as file:
    generation_config = yaml.safe_load(file)

with open('llm_psychometrics/psychometric_tests/hexaco_100_questions.yaml', 'r') as file:
    question_list = yaml.safe_load(file)

with open('llm_psychometrics/psychometric_tests/paraphrased_hexaco_100_questions.yaml', 'r') as file:
    paraphrased_question_list = yaml.safe_load(file)

with open('llm_psychometrics/psychometric_tests/hexaco_100_eval.yaml', 'r') as file:
    hexaco_eval = yaml.safe_load(file)

with open('llm_psychometrics/configs/personas_v2.yaml', 'r') as file:
    personas = yaml.safe_load(file)


if args.generation_method == "gpt":
    print(f"Using GPT Generation. Model Used: {args.model_name}")
    model = outlines.from_openai(OpenAI(), args.model_name)
else:
    print(f"Using Local Generation. Model Used: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=t.float16
    ).to(device)

    # Create outlines-wrapped model
    model = outlines.from_transformers(hf_model, tokenizer,  temperature=0)


NO_ANSWER = "Do not wish to answer"
likert_scale = generation_config['likert_scale'].copy()
likert_scale.append(NO_ANSWER)


inverted_likert = inverse_likert(generation_config['likert_scale'].copy())
inverted_likert.append(NO_ANSWER)

likert_scale_without_no = generation_config['likert_scale'].copy()
inverted_likert_without_no = inverse_likert(generation_config['likert_scale'].copy())


if args.hf_personas:
    hf_persona_dataset = load_dataset(args.hf_persona_path)
    persona_dataset = hf_persona_dataset['train']
    base_text = 
else:
    base_text = personas[args.job_title]['base_text']
    persona_dataset = personas[args.job_title]['personas']


def local_generation(hexaco_template, model, question_batch, likert_scale, batching=False, persona_str=None, persona_base_text=None):
    
    if batching:
        generator = Generator(model)
        prompt = hexaco_template(text=list_to_str(question_batch), likert_scale = ", ".join(likert_scale), base_text = persona_base_text, attributes = persona_str)
        answers = generator(prompt).strip().splitlines()
        results = [re.sub(r"[^a-zA-Z]", "", answer).strip() for answer in answers]
    else:
        results = []
        for question in question_batch:
            prompt = hexaco_template(text=question, likert_scale = ", ".join(likert_scale), base_text = persona_base_text, attributes = persona_str)
            
            result = model(prompt,Literal(*likert_scale))
            results.append(result)
        
        return results
    
def openai_generation(hexaco_template, model, question_batch, likert_scale, batching = False,persona_str=None, persona_base_text=None):
    if batching:
        generator = Generator(model)
        prompt = hexaco_template(text=list_to_str(question_batch), likert_scale = ", ".join(likert_scale), base_text = persona_base_text, attributes = persona_str)
        answers = generator(prompt).strip().splitlines()
        results = [re.sub(r"[^a-zA-Z]", "", answer).strip() for answer in answers]
    else:
        prompt_list = []
        for question in question_batch:
            prompt = hexaco_template(text=question, likert_scale = ", ".join(likert_scale), base_text = persona_base_text, attributes = persona_str)
            prompt = f"{prompt}, use the json format."
            prompt_list.append(prompt)
    
        def answer(prompt):
            response = model(prompt, OpenaiResponse)
            return json.loads(response)['response']
    
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(answer, prompt_list))
        return results
    
def persona_answer_generation():
    pass

def base_model_answer_generation(hexaco_template, generation_function, n_times=30,batch_size = 5,batching = False):
    
    repeated_answers = []
    for i in range(n_times):
        persona_answer = []
        for question_batch in batch_list(question_list, batch_size):
            batch_answers = generation_function(hexaco_template = hexaco_template, 
                                                    model = model, 
                                                    question_batch = question_batch,
                                                    likert_scale = likert_scale,
                                                    batching = batching)
            persona_answer.extend(batch_answers)
        repeated_answers.append(persona_answer)
        answer_dict = {}
        answer_dict['config'] = {}
        
        # TO DO: best way to pass below config details
        answer_dict['config']['likert_scale'] = ""
        answer_dict['config']['paraphrase'] = ""
        answer_dict['config']['refusal'] = ""
        answer_dict['config']['model_name'] = ""
        answer_dict['config']['persona'] = "base_model"
        answer_dict['answers'] = repeated_answers
    
    return [answer_dict]

def generate_answers(generation_function, model, question_list, likert_scale, persona_dataset=None, job_title=None, n_times = 30, batch_size = 5, batching = False):
    
    if persona_dataset:
        hexaco_template = persona_hexaco_template
    else:
        hexaco_template = base_hexaco_template
        
    
        
    
        
