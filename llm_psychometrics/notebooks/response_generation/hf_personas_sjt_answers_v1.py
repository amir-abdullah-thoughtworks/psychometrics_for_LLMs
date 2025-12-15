import argparse
import json
import os
import re
import sys
from typing import List, Literal
import random
import torch as t
import transformers
from jinja2 import Template
# from outlines import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from huggingface_hub import login
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor


# Custom imports

from utils_v0 import list_to_str
from prompt_templates.sjt_base_prompt_templates import sjt_base_prompt_templates
from prompt_templates.sjt_persona_prompt_templates import sjt_persona_prompt_templates
from utils.vllm_utils import VLLMServerManager

import psutil
import subprocess
import time
import requests
import socket

vllm_client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="-",
)

# ----------------------------
# Setup
# ----------------------------
transformers.logging.set_verbosity_error()
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Device: {device}")

sjt_answer_options = ["1", "2", "3", "4", "5", "6"]
default_answer_option_ordering = ['honesty_humility_option',
                                  'emotionality_option',
                                  'extraversion_option',
                                  'agreeableness_option',
                                  'conscientiousness_option',
                                  'openness_option']


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model Name")
    parser.add_argument("--persona-source", type=str, default="huggingface",
                        help="Source of Persona Being Used (base_model | huggingface)")
    parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas",
                        help="HF Path for Personas")
    parser.add_argument("--hf-sjt-path", type=str, default="thoughtworks/psychometric_SJTs",
                        help="HF Path for SJTs")
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
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "vllm"],
                        help="Backend provider: openai (API), vllm (OpenAI-compatible server), or hf (local Transformers).")
    parser.add_argument("--vllm-base-url", type=str, default="http://127.0.0.1:8000/v1",
                        help="Base URL for vLLM OpenAI-compatible server, e.g., http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key", type=str, default="-",
                        help="API key to send to vLLM (usually ignored but required by the OpenAI client).")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory for Storing output")
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


# class LocalResponse(BaseModel):
#     response: str


def load_model(model_name: str, hf_token: str = None):
    """Load OpenAI or vLLM (OpenAI-compatible) client. Signature unchanged."""
    # Do not use HF; keep hf_token param for compatibility.
    if args.provider == "openai":
        client = OpenAI()  # uses OPENAI_API_KEY from environment
    else:

        client = OpenAI(base_url=args.vllm_base_url.rstrip("/"), api_key=args.vllm_api_key)
    # NOTE: add provider so local_answer can branch to structured outputs for vLLM
    return {"client": client, "model_name": model_name, "provider": args.provider}

# ----------------------------
# Data Setup
# ----------------------------


def load_sjt(args):
    print("Using Huggingface SJTs")
    print(f"Loading SJTs from {args.hf_sjt_path}")
    hf_sjt_dataset = load_dataset(args.hf_sjt_path)
    sjt_datasets_total = hf_sjt_dataset['train']
    total_sjt_df = sjt_datasets_total.to_pandas()
    sampled_sjt = total_sjt_df.groupby("template_no").sample(n=args.n_sjtsample, random_state=42)
    sjt_datasets = sampled_sjt.to_dict("records")
    print(f"No of SJTs: {len(sjt_datasets)}")
    return sjt_datasets

def load_prompt_templates(model_name):
    """
    Use chat-style (OpenAI/vLLM) templates for all models to avoid Outlines/HF.
    Signature unchanged.
    """
    # We ignore non-GPT branches to keep everything in chat format.
    print("Loading Prompt Templates for Chat-style (OpenAI/vLLM) models")
    base_templates = sjt_base_prompt_templates["gpt"]
    persona_templates = sjt_persona_prompt_templates["gpt"]

    # compile GPT messages into Jinja templates
    def compile_message_templates(messages):
        return [
            {"role": msg["role"], "content": Template(msg["content"])}
            for msg in messages
        ]

    base_sjt_template = compile_message_templates(base_templates)
    persona_sjt_template = compile_message_templates(persona_templates)
    return base_sjt_template, persona_sjt_template


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
    elif args.persona_source == "personallm_paper":
        print("Using Persona LLM Paper Personas")
        persona_datasets_total = read_json("../data/persona_llm_paper_seed_combinations.json")
        random.seed(42)
        persona_datasets = random.sample(persona_datasets_total, args.n_personasample)
        print(f"No of Personas: {len(persona_datasets)}")
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
def openai_answer(model, prompt: List[dict], answer_options: List):
    """
    Minimal change: still called openai_answer(model, prompt) but now
    `prompt` is a list of OpenAI chat messages, and `model` is a bundle from load_model.
    Returns the assistant text directly.
    """
    client = model["client"]
    model_name = model["model_name"]
    # Dynamically constrain to one of the SJT Answer Order
    AnswerLiteral = Literal[tuple(answer_options)]

    class SJTAnswer(BaseModel):
        answer: AnswerLiteral
    resp = client.responses.parse(
        model=model_name,
        input=prompt,               # list of messages, unchanged
        text_format=SJTAnswer,
        temperature=0
    )

    parsed = resp.output_parsed    # SJTAnswer instance
    return parsed.answer

def local_answer(model, prompt, answer_options: List[str]):
    """
    Use vLLM guided_choice to constrain the output to one of answer_options.
    Works with an OpenAI-compatible client pointed at your vLLM server.
    """

    model_name = model["model_name"].strip()

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]

    # print(f"Working with {model_name}")
    resp = vllm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=16,  # small cap; response is a single choice
        extra_body={"guided_choice": answer_options},
    )

    text = resp.choices[0].message.content.strip()

    # Safety: normalize and map back to the exact canonical option
    norm_map = {opt.strip().lower(): opt for opt in answer_options}
    return norm_map.get(text.lower(), text)

def render_openai_messages(template_messages, **kwargs):
    rendered =  [
        {"role": msg["role"], "content": msg["content"].render(**kwargs)}
        for msg in template_messages
    ]
    return rendered

def generation_function(model, sjt_template, question_batch,answer_index ,batching=False,
                        persona_str=None, answer_shuffle=False):
    """Generate answers for a batch of questions."""
    if batching:
        raise NotImplementedError("Batching not implemented")

    # Non-batched
    prompt_list = []
    hash_list = []
    answer_index_list = []
    for sjt_dict in question_batch:
        
        if answer_shuffle:
            random.shuffle(answer_index)
        answer_index_list.append(list(answer_index))

        sjt = sjt_dict['corrected_sjt']
        # print("raw sjt", sjt)
        answer_options = [sjt[key] for key in default_answer_option_ordering if "_option" in key]
        # print("answer options after default ordering ", answer_options)
        answer_options = [answer_options[idx] for idx in answer_index]
        # print("final answer options", answer_options)
        question = sjt['question']
        
        prompt = render_openai_messages(
                            sjt_template,
                            attributes=persona_str,
                            question=question,
                            answer_options=list_to_str(answer_options)
                )
        # print("final prompt",prompt)
        prompt_list.append(prompt)
        hash_list.append(sjt_dict['hash_id'])


    if "gpt" in args.model_name:
        with ThreadPoolExecutor(max_workers=12) as executor:
            return list(executor.map(lambda p: openai_answer(model, p, sjt_answer_options), prompt_list)), hash_list, answer_index_list
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(lambda p: local_answer(model, p, sjt_answer_options), prompt_list)), hash_list, answer_index_list


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
        repeated_answer_indexes = []
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
            repeated_answer_indexes.append(answer_indexes)
        answers['base_model'] = {
            'config': {
                'persona': "base_model",
                'question_hashes': question_hashes,
                'sjt_answer_options': sjt_answer_options,
                'answer_index': repeated_answer_indexes,
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
            repeated_answer_indexes = []

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
                repeated_answer_indexes.append(answer_indexes)

            answers[persona_dataset['uuid']] = {
                'config': {
                    'persona': persona_dataset['uuid'],
                    'question_hashes': question_hashes,
                    'sjt_answer_options': sjt_answer_options,
                    'answer_index': repeated_answer_indexes,
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
    print(f"Writing Output in: {args.out_dir}")
    
    # --- NEW: auto-boot vLLM server when provider==vllm ---
    if args.provider == "vllm":
        from urllib.parse import urlparse
        u = urlparse(args.vllm_base_url)
        host = u.hostname or "127.0.0.1"
        port = u.port or 8000

        # Optional: pass extra CLI flags via env var VLLM_SERVER_ARGS="--gpu-memory-utilization 0.9 --dtype bfloat16"
        server_extra_args = os.environ.get("VLLM_SERVER_ARGS", "").strip().split()
        vllm_mgr = VLLMServerManager(
            model=args.model_name,
            host=host,
            port=port,
            timeout_s=180,
            kill_existing=True,
            server_extra_args=server_extra_args,
        )
        vllm_mgr.ensure_fresh_server()
        
    base_sjt_template, persona_sjt_template = load_prompt_templates(args.model_name)

    # Load model
    model = load_model(args.model_name, args.hf_token)

    # Load SJTs
    synthetic_sjts = load_sjt(args)

    # Load Personas
    persona_datasets = load_personas(args)

    # Run
    results = generate_answers(model, args, synthetic_sjts, persona_datasets, args.answer_shuffle)

    # Save results
    model_name = args.model_name.replace(".", "_").split("/")[-1]
    # out_dir = "../experiment_results/reliability_experiments/vllm_experiment_6"
    out_file = os.path.join(args.out_dir,
                            f"{args.persona_source}_sjt_answers_{model_name}.json")
    write_to_json(results, out_file)
    print(f"Results saved to {out_file}")
