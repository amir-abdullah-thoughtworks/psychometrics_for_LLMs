from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import json
import os
import sys
import numpy as np
import pandas as pd
import yaml
from enum import Enum
from typing import Literal
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import outlines
import torch as t
sys.path.append("../")
from src.utils import inverse_likert, list_to_str

device = t.device("cuda" if t.cuda.is_available() else "cpu")


class OpenaiResponse(BaseModel):
    response: str


with open('../configs/generation_config.yaml', 'r') as file:
    generation_config = yaml.safe_load(file)

with open('../psychometric_tests/hexaco_100_questions.yaml', 'r') as file:
    question_list = yaml.safe_load(file)

with open('../psychometric_tests/paraphrased_hexaco_100_questions.yaml', 'r') as file:
    paraphrased_question_list = yaml.safe_load(file)

with open('../psychometric_tests/hexaco_100_eval.yaml', 'r') as file:
    hexaco_eval = yaml.safe_load(file)

with open('../configs/personas.yaml', 'r') as file:
    personas = yaml.safe_load(file)


openai_model_name = "gpt-4.1-mini"
openai_model = outlines.from_openai(OpenAI(), openai_model_name)


NO_ANSWER = "Do not wish to answer"
likert_scale = generation_config['likert_scale'].copy()
likert_scale.append(NO_ANSWER)

inverted_likert = inverse_likert(generation_config['likert_scale'].copy())
inverted_likert.append(NO_ANSWER)

likert_scale_without_no = generation_config['likert_scale'].copy()
inverted_likert_without_no = inverse_likert(
    generation_config['likert_scale'].copy())

base_text = personas['customer_service']['base_text']
persona = personas['customer_service']['personas'][0]


base_hexaco_template = outlines.Template.from_string("""
<|im_start>user
Task: Answer the below questions:

{{ text }}

Answer the question as either {{ likert_scale }} .
<|im_end>
<|im_start>assistant
""")

persona_hexaco_template = outlines.Template.from_string("""
<|im_start>user
{{base_text}} with following attributes :

{{attributes}}

Task: Answer the below questions:

{{ text }}

Answer the question as either {{ likert_scale }}.
<|im_end>
<|im_start>assistant
""")


def local_generation(hexaco_template, model, question_batch, likert_scale, batching=False, persona_str=None, persona_base_text=None):

    if batching:
        raise NotImplementedError("Batching Not Implemented !!!")
    else:
        prompt_list = []
        for question in question_batch:
            prompt = hexaco_template(text=question, likert_scale=", ".join(
                likert_scale), base_text=persona_base_text, attributes=persona_str)
            prompt_list.append(prompt)

        def answer(prompt, likert_scale=likert_scale):
            return model(
                prompt,
                Literal[*likert_scale]
            )
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(answer, prompt_list))

        return results


def openai_generation(hexaco_template, model, question_batch, likert_scale, batching=False, persona_str=None, persona_base_text=None):
    if batching:
        raise NotImplementedError("Batching Not Implemented !!!")
    else:
        prompt_list = []
        for question in question_batch:
            prompt = hexaco_template(text=question, likert_scale=", ".join(
                likert_scale), base_text=persona_base_text, attributes=persona_str)
            prompt = f"{prompt}, use the json format."
            prompt_list.append(prompt)

        def answer(prompt):
            response = model(prompt, OpenaiResponse)
            return json.loads(response)['response']

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(answer, prompt_list))
        return results


def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_answers(generation_function, model, question_list, likert_scale, job_title=None, n_times=30, batch_size=5, batching=False):

    if job_title:
        answers = []
        hexaco_template = persona_hexaco_template
        base_text = personas[job_title]['base_text']
        persona_list = personas[job_title]['personas']
        for persona in tqdm(persona_list, desc="Personas", position=0):
            persona_str = ", ".join(
                [f"{key} : {persona[key]}" for key in persona if key != "Summary"])
            repeated_answers = []
            for i in tqdm(range(n_times), desc="Iterations", position=1):
                persona_answer = []
                for question_batch in tqdm(batch_list(question_list, batch_size), desc="Batches", position=2):
                    batch_answers = generation_function(hexaco_template=hexaco_template,
                                                        model=model,
                                                        question_batch=question_batch,
                                                        likert_scale=likert_scale,
                                                        persona_str=persona_str,
                                                        persona_base_text=base_text,
                                                        batching=batching)
                    persona_answer.extend(batch_answers)
                repeated_answers.append(persona_answer)
            persona_dict = {}
            persona_dict['config'] = {}
            persona_dict['config']['persona'] = persona_str
            persona_dict['answers'] = repeated_answers
            answers.append(persona_dict)

    else:
        hexaco_template = base_hexaco_template
        repeated_answers = []
        for i in range(n_times):
            persona_answer = []
            for question_batch in batch_list(question_list, batch_size):
                batch_answers = generation_function(hexaco_template=hexaco_template,
                                                    model=model,
                                                    question_batch=question_batch,
                                                    likert_scale=likert_scale,
                                                    batching=batching)
                persona_answer.extend(batch_answers)
            repeated_answers.append(persona_answer)
            persona_dict = {}
            persona_dict['config'] = {}
            persona_dict['config']['persona'] = "base_model"
            persona_dict['answers'] = repeated_answers
        answers = [persona_dict]
    return answers


def write_to_json(file, file_path):
    with open(file_path, 'w') as f:
        json.dump(file, f)


def read_json(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file


def run_experiment(data_dir, model_name, job_title=None):

    if "gpt" in model_name:
        print("Running Openai generation")
        model = openai_model
        generation_model = openai_generation
    else:
        print("Running Local generation")
        model = local_model
        generation_model = local_generation

    # normal_hexaco_answers = generate_answers(generation_model, model, question_list, likert_scale, job_title)
    # for persona_dict in normal_hexaco_answers:
    #     persona_dict['config']['likert_scale'] = "normal"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_answers, os.path.join(data_dir,f"normal_hexaco_answers_{model_name}.json"))

    # normal_hexaco_answers_without_no = generate_answers(generation_model, model, question_list, likert_scale_without_no, job_title)
    # for persona_dict in normal_hexaco_answers_without_no:
    #     persona_dict['config']['likert_scale'] = "normal"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "no_refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_answers_without_no, os.path.join(data_dir,f"normal_hexaco_answers_without_no_{model_name}.json"))

    # normal_hexaco_inverted_likert_answers = generate_answers(generation_model, model, question_list, inverted_likert, job_title)
    # for persona_dict in normal_hexaco_inverted_likert_answers:
    #     persona_dict['config']['likert_scale'] = "inverted"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_inverted_likert_answers, os.path.join(data_dir,f"normal_hexaco_inverted_likert_answers_{model_name}.json"))

    # normal_hexaco_inverted_likert_without_no_answers = generate_answers(generation_model, model, question_list, inverted_likert_without_no, job_title)
    # for persona_dict in normal_hexaco_inverted_likert_without_no_answers:
    #     persona_dict['config']['likert_scale'] = "inverted"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "no_refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_inverted_likert_without_no_answers, os.path.join(data_dir,f"normal_hexaco_inverted_likert_without_no_answers_{model_name}.json"))

    # print("################ Paraphrase Questions, Normal Likert, Refusal Allowed ################")
    # paraphrase_hexaco_answers = generate_answers(
    #     generation_model, model, paraphrased_question_list, likert_scale, job_title)
    # for persona_dict in paraphrase_hexaco_answers:
    #     persona_dict['config']['likert_scale'] = "normal"
    #     persona_dict['config']['paraphrase'] = "paraphrase"
    #     persona_dict['config']['refusal'] = "refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(paraphrase_hexaco_answers, os.path.join(
    #     data_dir, f"paraphrase_hexaco_answers_{model_name}.json"))

    print("################ Paraphrase Questions, Normal Likert, Refusal Not Allowed ################")
    paraphrase_hexaco_answers_without_no = generate_answers(
        generation_model, model, paraphrased_question_list, likert_scale_without_no, job_title)
    for persona_dict in paraphrase_hexaco_answers_without_no:
        persona_dict['config']['likert_scale'] = "normal"
        persona_dict['config']['paraphrase'] = "paraphrase"
        persona_dict['config']['refusal'] = "no_refusal"
        persona_dict['config']['model_name'] = model_name
    write_to_json(paraphrase_hexaco_answers_without_no, os.path.join(
        data_dir, f"paraphrase_hexaco_answers_without_no_{model_name}.json"))

    print("################ Paraphrase Questions, Inverted Likert, Refusal Allowed ################")
    paraphrase_hexaco_inverted_likert_answers = generate_answers(
        generation_model, model, paraphrased_question_list, inverted_likert, job_title)
    for persona_dict in paraphrase_hexaco_inverted_likert_answers:
        persona_dict['config']['likert_scale'] = "inverted"
        persona_dict['config']['paraphrase'] = "paraphrase"
        persona_dict['config']['refusal'] = "refusal"
        persona_dict['config']['model_name'] = model_name
    write_to_json(paraphrase_hexaco_inverted_likert_answers, os.path.join(
        data_dir, f"paraphrase_hexaco_inverted_likert_answers_{model_name}.json"))

    print("################ Paraphrase Questions, Inverted Likert, Refusal Not Allowed ################")
    paraphrase_hexaco_inverted_likert_without_no_answers = generate_answers(
        generation_model, model, paraphrased_question_list, inverted_likert_without_no, job_title)
    for persona_dict in paraphrase_hexaco_inverted_likert_without_no_answers:
        persona_dict['config']['likert_scale'] = "inverted"
        persona_dict['config']['paraphrase'] = "paraphrase"
        persona_dict['config']['refusal'] = "no_refusal"
        persona_dict['config']['model_name'] = model_name
    write_to_json(paraphrase_hexaco_inverted_likert_without_no_answers, os.path.join(
        data_dir, f"paraphrase_hexaco_inverted_likert_without_no_answers_{model_name}.json"))


if __name__ == "__main__":

    run_experiment(data_dir="persona_basic_info_experiment_results_v2",
                   model_name="gpt_41_mini", job_title="customer_service")
