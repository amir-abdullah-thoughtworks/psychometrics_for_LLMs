import torch as t
import outlines
from outlines import Generator
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from typing import Literal
import argparse
import yaml
import sys
import os
import re
import json
from huggingface_hub import login
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.append("../")
from src.utils import list_to_str

transformers.logging.set_verbosity_error()

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Device: {device}")


parser = argparse.ArgumentParser()

parser.add_argument("--model-name", type=str, help="Model Name",
                    default="gpt-4.1-mini")

args = parser.parse_args()

print(f"Model Used: {args.model_name}")


class OpenaiResponse(BaseModel):
    response: str


def write_to_json(file, file_path):
    with open(file_path, 'w') as f:
        json.dump(file, f)


def read_json(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file


login("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")

with open('../configs/generation_config.yaml', 'r') as file:
    generation_config = yaml.safe_load(file)

with open('../psychometric_tests/hexaco_100_questions.yaml', 'r') as file:
    question_list = yaml.safe_load(file)

with open('../psychometric_tests/paraphrased_hexaco_100_questions.yaml', 'r') as file:
    paraphrased_question_list = yaml.safe_load(file)

with open('../psychometric_tests/hexaco_100_eval.yaml', 'r') as file:
    hexaco_eval = yaml.safe_load(file)

with open('../configs/personas_v2.yaml', 'r') as file:
    personas = yaml.safe_load(file)

if "gpt" in args.model_name:
    # print(f"Using GPT Generation. Model Used: {args.model_name}")
    model = outlines.from_openai(OpenAI(), args.model_name)
else:
    # print(f"Using Local Generation. Model Used: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=t.float16
    ).to(device)
    
    # tokenizer.pad_token = tokenizer.eos_token
    # hf_model.config.pad_token_id = tokenizer.eos_token_id

    # Create outlines-wrapped model
    model = outlines.from_transformers(hf_model, tokenizer)

NO_ANSWER = "Do not wish to answer"
likert_scale = generation_config['likert_scale'].copy()
likert_scale.append(NO_ANSWER)


hf_persona_dataset = load_dataset("thoughtworks/psychometric_personas")
persona_datasets = hf_persona_dataset['train']


persona_hexaco_template = outlines.Template.from_string("""
<|im_start>user
{{base_text}} with following attributes :

{{attributes}}

Task: Answer the below questions:

{{ text }}

Answer the question as either {{ likert_scale }}. Do not return the question, just return the answer directly.
<|im_end>
<|im_start>assistant
""")


def openai_answer(prompt):
    response = model(prompt, OpenaiResponse, temperature=0)
    return json.loads(response)['response']


def local_answer(prompt, answer_options=likert_scale):
    response = model(prompt, Literal[*answer_options])
    return response


def generation_function(hexaco_template, question_batch, likert_scale,
                      batching=False, persona_str=None, persona_base_text=None):
    if batching:
        generator = Generator(model)
        prompt = hexaco_template(text=list_to_str(question_batch),
                                 likert_scale=", ".join(likert_scale),
                                 base_text=persona_base_text,
                                 attributes=persona_str)
        answers = generator(prompt).strip().replace(">","").splitlines()
        results = [re.sub(r"[^a-zA-Z]", "", answer).strip()
                   for answer in answers]
    else:
        prompt_list = []
        for question in question_batch:
            prompt = hexaco_template(text=question,
                                     likert_scale=", ".join(likert_scale),
                                     base_text=persona_base_text,
                                     attributes=persona_str)
            prompt = f"{prompt}, use the json format."
            prompt_list.append(prompt)

        if "gpt" in args.model_name:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(openai_answer, prompt_list))
        else:
            results = []
            for prompt in prompt_list:
                results.append(local_answer(prompt))
            # with ThreadPoolExecutor(max_workers=4) as executor:
            #     results = list(executor.map(local_answer, prompt_list))

    return results


def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_answers(persona_datasets, generation_function,
                     question_list, likert_scale, n_times=1,
                     batch_size=5, batching=False):

    if batching:
        print(f"Batching {batch_size} Questions together in one prompt")
    else:
        print("Passing One question per prompt")
    answers = {}
    hexaco_template = persona_hexaco_template
    # base_text = personas[job_title]['base_text']
    # persona_list = personas[job_title]['personas']
    base_text = "You are a law enforcement officer."
    for persona_dataset in tqdm(persona_datasets, desc="Personas", position=0):
        persona_str = persona_dataset['persona_text']
        repeated_answers = []
        for i in tqdm(range(n_times), desc="Iterations", position=1):
            persona_answer = []
            for question_batch in tqdm(batch_list(question_list, batch_size), desc="Batches", position=2):
                batch_answers = generation_function(hexaco_template=hexaco_template, 
                                                    question_batch=question_batch,
                                                    likert_scale=likert_scale,
                                                    persona_str=persona_str,
                                                    persona_base_text=base_text,
                                                    batching=batching)
                persona_answer.extend(batch_answers)
            repeated_answers.append(persona_answer)
        persona_dict = {}
        persona_dict['config'] = {}
        persona_dict['config']['persona'] = persona_dataset['uuid']
        persona_dict['config']['likert_scale'] = "normal"
        persona_dict['config']['paraphrase'] = "normal"
        persona_dict['config']['refusal'] = "refusal"
        persona_dict['config']['model_name'] = "gpt_41_mini"
        persona_dict['answers'] = repeated_answers
        answers[persona_dataset['uuid']] = persona_dict
    return answers


batch_size = 10
persona_dataset_df = persona_datasets.to_pandas()
for start in range(0, 30, batch_size):
    print(f"Batch {start}")
    batch = persona_dataset_df.iloc[start:start+batch_size]
    batch = batch.to_dict("records")
    hf_persona_answers = generate_answers(batch, generation_function,
                                          question_list, likert_scale, batching=True)

    write_to_json(hf_persona_answers, os.path.join("hf_persona_batching_vs_individual_questions_results_v0",f"hf_persona_answers_with_batching_v{start}.json"))
