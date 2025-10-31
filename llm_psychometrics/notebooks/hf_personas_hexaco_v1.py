import argparse
import json
import os
import sys
from typing import List, Literal
import transformers
# from outlines import Generator
from jinja2 import Template
from pydantic import BaseModel
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor
import random

# Custom imports
sys.path.append("../")
from src.utils_v0 import list_to_str, inverse_likert
from src.utils.vllm_utils import VLLMServerManager
from src.prompt_templates.hexaco_base_prompt_templates import hexaco_base_prompt_templates
from src.prompt_templates.hexaco_persona_prompt_templates import hexaco_persona_prompt_templates

import psutil

vllm_client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="-",
)

# ----------------------------
# Setup
# ----------------------------
transformers.logging.set_verbosity_error()

NO_ANSWER = "Do not wish to answer"


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model Name")
    parser.add_argument("--persona-source", type=str, default="huggingface",
                        help="Source of Persona (huggingface | base_model | personallm_paper)")
    parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas",
                        help="HF Path for Personas")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Huggingface token")
    parser.add_argument("--batching", action="store_true",
                        help="Enable batching mode (default: False)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of questions per batch")
    parser.add_argument("--n-times", type=int, default=1,
                        help="Number of repetitions per persona")
    parser.add_argument("--paraphrase", action="store_true",
                        help="Use paraphrased versions of HEXACO (default: False)")
    parser.add_argument("--n-personasample", type=int, default=1,
                        help="Number of Personas to be sampled for each archetype")
    parser.add_argument("--inverted-likert", action="store_true",
                        help="Whether Likert Scale needs to be inverted or not")
    parser.add_argument("--no-refusal", action="store_true",
                        help="Whether Refusal is allowed or not")
    parser.add_argument("--likert-shuffle", action="store_true",
                        help="Enabling Shuffling of likert scale for hexaco (default: False)")
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
def load_data(args):
    with open('../configs/generation_config.yaml', 'r') as file:
        generation_config = yaml.safe_load(file)

    with open('../psychometric_tests/hexaco_100_questions.yaml', 'r') as file:
        question_list = yaml.safe_load(file)

    with open('../psychometric_tests/paraphrased_hexaco_100_questions.yaml', 'r') as file:
        paraphrased_question_list = yaml.safe_load(file)

    # with open('../psychometric_tests/hexaco_100_eval.yaml', 'r') as file:
    #     hexaco_eval = yaml.safe_load(file)

    # with open('../configs/personas_v2.yaml', 'r') as file:
    #     personas = yaml.safe_load(file)

    # Scale
    if args.inverted_likert:
        print("Inverting Likert")
        likert_scale = inverse_likert(generation_config['likert_scale'].copy())
    else:
        print("Normal Likert")
        likert_scale = generation_config['likert_scale'].copy()

    if args.no_refusal:
        print("Refusal Not Allowed")
    else:
        print("Refusal Allowed")
        likert_scale.append(NO_ANSWER)

    if args.paraphrase:
        print("Using Paraphrased Questions")
        question_list = paraphrased_question_list
    else:
        print("Using Normal Questions")

    return question_list, likert_scale

def load_prompt_templates(model_name):
    """
    Use chat-style (OpenAI/vLLM) templates for all models to avoid Outlines/HF.
    Signature unchanged.
    """
    # We ignore non-GPT branches to keep everything in chat format.
    print("Loading Prompt Templates for Chat-style (OpenAI/vLLM) models")
    base_templates = hexaco_base_prompt_templates["gpt"]
    persona_templates = hexaco_persona_prompt_templates["gpt"]

    # compile GPT messages into Jinja templates
    def compile_message_templates(messages):
        return [
            {"role": msg["role"], "content": Template(msg["content"])}
            for msg in messages
        ]

    base_hexaco_template = compile_message_templates(base_templates)
    persona_hexaco_template = compile_message_templates(persona_templates)
    return base_hexaco_template, persona_hexaco_template


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


def openai_answer(model, prompt: List[dict], answer_options: List):
    """
    Minimal change: still called openai_answer(model, prompt) but now
    `prompt` is a list of OpenAI chat messages, and `model` is a bundle from load_model.
    Returns the assistant text directly.
    """
    client = model["client"]
    model_name = model["model_name"]
    # Dynamically constrain to one of the Likert strings
    AnswerLiteral = Literal[tuple(answer_options)]

    class LikertAnswer(BaseModel):
        answer: AnswerLiteral
    resp = client.responses.parse(
        model=model_name,
        input=prompt,               # list of messages, unchanged
        text_format=LikertAnswer,
        temperature=0
    )

    parsed = resp.output_parsed    # LikertAnswer instance
    return parsed.answer

    content = (resp.choices[0].message.content or "").strip()
    return content

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

def generation_function(model, hexaco_template, question_batch, likert_scale,
                        batching=False, persona_str=None, persona_base_text=None, likert_shuffle=False):

    if batching:
        raise NotImplementedError("Batching not implemented")

    # Non-batched
    prompt_list = []
    for question in question_batch:
        if likert_shuffle:
            random.shuffle(likert_scale)


        prompt = render_openai_messages(
                            hexaco_template,
                            attributes=persona_str,
                            text=question,
                            likert_scale=", ".join(likert_scale),
                )
        prompt_list.append(prompt)

    if "gpt" in args.model_name:
        with ThreadPoolExecutor(max_workers=12) as executor:
            return list(executor.map(lambda p: openai_answer(model, p, likert_scale), prompt_list))
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(lambda p: local_answer(model, p, likert_scale), prompt_list))

# ----------------------------
# Experiment Runner
# ----------------------------
def generate_answers(model, args, persona_datasets, question_list,
                     likert_scale):
    answers = {}
    hexaco_template = persona_hexaco_template
    base_text = "You are a law enforcement officer."

    if persona_datasets is None:
        print("Running HEXACO on Base Model without Personas")
        persona_datasets = [{"uuid": "base_model", "persona_text": ""}]
        hexaco_template = base_hexaco_template

    if args.batching:
        print(f"Batching {args.batch_size} questions together")
    else:
        print("Passing one question per prompt")

    if args.likert_shuffle:
        print("Shuffling likert scale for every persona and question combination")
    else:
        print("Using the default likert scale order without shuffling")

    for persona_dataset in tqdm(persona_datasets, desc="Personas"):
        persona_str = persona_dataset.get('persona_string', "")
        persona_id = persona_dataset.get('uuid', "base_model")

        repeated_answers = []
        for _ in tqdm(range(args.n_times), desc="Iterations"):
            persona_answer = []
            for question_batch in tqdm(batch_list(question_list, args.batch_size), desc="Batches"):
                batch_answers = generation_function(model, hexaco_template,
                                                    question_batch, likert_scale,
                                                    persona_str=persona_str,
                                                    persona_base_text=base_text,
                                                    batching=args.batching,
                                                    likert_shuffle=args.likert_shuffle)
                persona_answer.extend(batch_answers)
            repeated_answers.append(persona_answer)

        if args.inverted_likert:
            likert = "inverted"
        elif args.likert_shuffle:
            likert = "shuffle"
        else:
            likert = "normal"
        
        answers[persona_id] = {
            'config': {
                'persona': persona_id,
                'paraphrase': "paraphrased" if args.paraphrase else "normal",
                "likert_scale": likert,
                "refusal_allowed": "no refusal" if args.no_refusal else "refusal",
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
    
    base_hexaco_template, persona_hexaco_template = load_prompt_templates(args.model_name)

    # Load model
    model = load_model(args.model_name, args.hf_token)

    # Load data
    question_list, likert_scale = load_data(args)

    # Load persona
    persona_datasets = load_personas(args)

    print(f"model name is {args.model_name}.")
    # Run
    results = generate_answers(model, args, persona_datasets, question_list, likert_scale)

    # Save results
    model_name = args.model_name.replace(".", "_").split("/")[-1]
    # out_dir = "../experiment_results/zero_compute_analysis_hexaco"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.persona_source}_hexaco_answers_{model_name}.json")
    write_to_json(results, out_file)
    print(f"Results saved to {out_file}")
