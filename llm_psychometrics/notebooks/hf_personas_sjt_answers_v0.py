import argparse
import json
import os
import re
import sys
from typing import List, Literal
import random
import torch as t
import transformers
import outlines
from outlines import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from huggingface_hub import login
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor

# Custom imports
sys.path.append("../")
from src.utils_v0 import list_to_str

# ----------------------------
# Setup
# ----------------------------
transformers.logging.set_verbosity_error()
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Device: {device}")

sjt_answer_options = ["1", "2", "3", "4", "5", "6"]


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini",
                        help="Model Name")
    parser.add_argument("--persona-source", type=str, default="huggingface",
                        help="Source of Persona Being Used (base_model | huggingface)")
    parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas_temp",
                        help="HF Path for Personas")
    parser.add_argument("--sjt-dir", type=str, default=None,
                        help="Source directory for Synthetic SJTs")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Huggingface token")
    parser.add_argument("--batching", action="store_true",
                        help="Enable batching mode (default: False)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of questions per batch when batching is enabled")
    parser.add_argument("--n-times", type=int, default=1,
                        help="Number of repetitions per persona")
    parser.add_argument("--n-sjtsample", type=int, default=1,
                        help="Number of SJTs to be sampled")
    parser.add_argument("--n-personasample", type=int, default=1,
                        help="Number of Personas to be sampled for each archetype")
    parser.add_argument("--answer-shuffle", action="store_true",
                        help="Enabling Shuffling of answer index for SJTs (default: False)")
    return parser.parse_args()


# ----------------------------
# Utility Functions
# ----------------------------
def write_to_json(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(file, f, indent=2)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ----------------------------
# Model Setup
# ----------------------------
class OpenaiResponse(BaseModel):
    response: str


class LocalResponse(BaseModel):
    response: str


def load_model(model_name: str, hf_token: str = None):
    """Load either OpenAI or HF model wrapped with Outlines."""
    if hf_token:
        login(hf_token)

    if "gpt" in model_name:
        return outlines.from_openai(OpenAI(), model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.float16
    ).to(device)

    return outlines.from_transformers(hf_model, tokenizer)

# ----------------------------
# Data Setup
# ----------------------------


def load_sjt(args):
    sjt_list = []
    for file in os.listdir(args.sjt_dir):
        synthetic_sjt = read_json(os.path.join(args.sjt_dir, file))
        sampled_sjts = random.sample(synthetic_sjt, args.n_sjtsample)
        sjt_list += sampled_sjts

    print(f"{len(sjt_list)} SJTs Loaded")
    return sjt_list


def load_personas(args):
    if args.persona_source == "huggingface":
        print("Using Huggingface Personas")
        print(f"Loading Personas from {args.hf_persona_path}")
        hf_persona_dataset = load_dataset(args.hf_persona_path)
        persona_datasets_total = hf_persona_dataset['train']
        total_persona_df = persona_datasets_total.to_pandas()
        sampled_personas = total_persona_df.groupby("archetype").sample(n=args.n_personasample, random_state=42)
        persona_datasets = Dataset.from_pandas(sampled_personas)
        print(f"No of Personas: {len(persona_datasets)}")
    elif args.persona_source == "base_synthetic_personas":
        raise NotImplementedError("Base Synthetic Personas is not implemented yet")
    elif args.persona_source == "base_model":
        return None
    else:
        print("Using Local Personas")
        with open('../configs/personas_v2.yaml', 'r') as file:
            local_personas = yaml.safe_load(file)

        job_title = 'law_enforcement'
        persona_datasets = local_personas[job_title]['personas']
        print(f"No of Personas: {len(persona_datasets)}")

    return persona_datasets


# ----------------------------
# Answer Generation
# ----------------------------
def openai_answer(model, prompt: str):
    response = model(prompt, OpenaiResponse, temperature=0)
    return json.loads(response)['response']


def local_answer(model, prompt_list: List):
    return [model(prompt, Literal[*sjt_answer_options])
            for prompt in prompt_list]


def generation_function(model, sjt_template, question_batch,answer_index ,batching=False,
                        persona_str=None, answer_shuffle=False):
    """Generate answers for a batch of questions."""
    if batching:
        raise NotImplementedError()
        # generator = Generator(model)
        # prompt = sjt_template(
        #     text=list_to_str(question_batch),
        #     sjt_answer_options=", ".join(sjt_answer_options),
        #     base_text=persona_base_text,
        #     attributes=persona_str
        # )
        # answers = generator(prompt).strip().replace(">", "").splitlines()
        # return [re.sub(r"[^a-zA-Z]", "", ans).strip() for ans in answers]

    # Non-batched
    prompt_list = []
    hash_list = []
    answer_index_list = []
    for sjt_dict in question_batch:
        
        if answer_shuffle:
            random.shuffle(answer_index)
        answer_index_list.append(list(answer_index))

        sjt = sjt_dict['corrected_sjt']

        answer_options = [sjt[key] for key in sjt.keys() if "_option" in key]
        answer_options = [answer_options[idx] for idx in answer_index]
        question = sjt['question']

        prompt = sjt_template(question=question,
                              attributes=persona_str,
                              answer_options=list_to_str(answer_options))
        prompt_list.append(prompt)
        hash_list.append(sjt_dict['hash_id'])


    if "gpt" in args.model_name:
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(lambda p: openai_answer(model, p), prompt_list)), hash_list, answer_index_list
    else:
        return local_answer(model, prompt_list), hash_list, answer_index_list


# ----------------------------
# Experiment Runner
# ----------------------------
def generate_answers(model, args, synthetic_sjts, persona_datasets=None, answer_shuffle=False):

    answer_index = [0, 1, 2, 3, 4, 5]
    
    if answer_shuffle:
        sjt_answer_options = "shuffle"
        print("Shuffling answer options for SJTs")
    else:
        sjt_answer_options = "normal"
        print("Default Ordering of answer options for SJTs")

    if persona_datasets:
        print(f"No of personas: {len(persona_datasets)}")
    else:
        print("Answering the SJTs using the base model, without any personas")
    print(f"No of SJTs: {len(synthetic_sjts)}")
    
    """Run experiments for base model or persona-conditioned runs."""
    if args.batching:
        print(f"Batching {args.batch_size} questions together")
    else:
        print("Passing one question per prompt")

    answers = {}

    if args.persona_source == "base_model":
        print("Running SJTs on Base Model without Personas")
        sjt_template = base_sjt_template
        repeated_answers = []

        for _ in tqdm(range(args.n_times), desc="Iterations"):
            persona_answer = []
            question_hashes = []
            answer_indexes = []
            for q_batch in tqdm(batch_list(synthetic_sjts, args.batch_size), desc="SJT Batches"):
                batch_answers, batch_hash_list, batch_answer_index_list = generation_function(model, sjt_template, q_batch,answer_index=answer_index,
                                        batching=args.batching,
                                        answer_shuffle=answer_shuffle)
                persona_answer.extend(batch_answers)
                question_hashes.extend(batch_hash_list)
                answer_indexes.extend(batch_answer_index_list)
            repeated_answers.append(persona_answer)

        answers['base_model'] = {
            'config': {
                'persona': "base_model",
                'question_hashes': question_hashes,
                'sjt_answer_options': sjt_answer_options,
                'answer_index': answer_indexes,
                'model_name': args.model_name
            },
            'answers': repeated_answers
        }

    else:
        print("Running SJTs with Personas")
        sjt_template = persona_sjt_template

        for persona_dataset in tqdm(persona_datasets, desc="Personas"):
            persona_str = persona_dataset['persona_string']
            
            repeated_answers = []

            for _ in tqdm(range(args.n_times), desc="Iterations"):
                persona_answer = []
                question_hashes = []
                answer_indexes = []
                for q_batch in tqdm(batch_list(synthetic_sjts, args.batch_size), desc="Batches"):
                    batch_answers, batch_hash_list, batch_answer_index_list = generation_function(model, sjt_template, q_batch,
                                            answer_index=answer_index,
                                            persona_str=persona_str,
                                            batching=args.batching,
                                            answer_shuffle=answer_shuffle)
                    persona_answer.extend(batch_answers)
                    question_hashes.extend(batch_hash_list)
                    answer_indexes.extend(batch_answer_index_list)
                repeated_answers.append(persona_answer)

            answers[persona_dataset['uuid']] = {
                'config': {
                    'persona': persona_dataset['uuid'],
                    'question_hashes': question_hashes,
                    'sjt_answer_options': sjt_answer_options,
                    'answer_index': answer_indexes,
                    'model_name': args.model_name
                },
                'answers': repeated_answers
            }

    return answers


# ----------------------------
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    print(f"Model Used: {args.model_name}")

    # Load model
    model = load_model(args.model_name, args.hf_token)

    # Load SJTs
    synthetic_sjts = load_sjt(args)

    # Load Personas
    persona_datasets = load_personas(args)

    # Define templates (TODO: replace placeholders)
    base_sjt_template = outlines.Template.from_string("""
    <|im_start>user
    Task: Answer the below multiple choice questions:

    Question: {{ question }}

    Choices for these questions are:

    {{ answer_options }}

    Answer the question using one of the values from [1,2,3,4,5,6].
    <|im_end>
    <|im_start>assistant
    """)

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

    # Run
    results = generate_answers(model, args, synthetic_sjts, persona_datasets, args.answer_shuffle)

    # Save results
    model_name = args.model_name.replace(".", "_").split("/")[-1]
    out_dir = "case_study_data"
    out_file = os.path.join(out_dir,
                            f"{args.persona_source}_sjt_answers_{model_name}.json")
    write_to_json(results, out_file)
    print(f"Results saved to {out_file}")
