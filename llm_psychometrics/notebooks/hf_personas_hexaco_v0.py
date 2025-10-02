import argparse
import json
import os
import re
import sys
from typing import List, Literal
import torch as t
import transformers
from jinja2 import Template
import outlines
from outlines import Generator
from outlines.inputs import Chat
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from huggingface_hub import login
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor
import random

# Custom imports
sys.path.append("../")
from src.utils_v0 import list_to_str, inverse_likert
from src.prompt_templates.hexaco_base_prompt_templates import hexaco_base_prompt_templates
from src.prompt_templates.hexaco_persona_prompt_templates import hexaco_persona_prompt_templates

# ----------------------------
# Setup
# ----------------------------
transformers.logging.set_verbosity_error()
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Device: {device}")

NO_ANSWER = "Do not wish to answer"


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini",
                        help="Model Name")
    parser.add_argument("--persona-source", type=str, default="huggingface",
                        help="Source of Persona (huggingface | base_model | personallm_paper)")
    parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas_temp",
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
    
    if "gpt" in model_name.lower():
        print("Loading Prompt Templates for GPT models")
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
    
    elif "llama" in model_name.lower():
        print("Loading Prompt Templates for Llama models")
        base_hexaco_template_str = hexaco_base_prompt_templates['llama']
        persona_hexaco_template_str = hexaco_persona_prompt_templates['llama']
        
        base_hexaco_template = outlines.Template.from_string(base_hexaco_template_str)
        persona_hexaco_template = outlines.Template.from_string(persona_hexaco_template_str)
        
    elif "qwen" in model_name.lower():
        print("Loading Prompt Templates for Qwen models")
        base_hexaco_template_str = hexaco_base_prompt_templates['qwen']
        persona_hexaco_template_str = hexaco_persona_prompt_templates['qwen']
        
        base_hexaco_template = outlines.Template.from_string(base_hexaco_template_str)
        persona_hexaco_template = outlines.Template.from_string(persona_hexaco_template_str)
    else:
        raise NotImplementedError("Use Models from GPT, Llama or Qwen Families")
    
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


# ----------------------------
# Answer Generation
# ----------------------------


def openai_answer(model, prompt: str):
    response = model(prompt, OpenaiResponse, temperature=0)
    return json.loads(response)['response']


def local_answer(model, prompt, answer_options: List):
    # print(f"Model loaded on: {model.model.device}")
    return model(prompt, Literal[*answer_options])

def render_openai_messages(template_messages, **kwargs):
    rendered =  [
        {"role": msg["role"], "content": msg["content"].render(**kwargs)}
        for msg in template_messages
    ]
    
    return Chat(rendered)

def generation_function(model, hexaco_template, question_batch, likert_scale,
                        batching=False, persona_str=None, persona_base_text=None, likert_shuffle=False):

    if batching:
        generator = Generator(model)
        prompt = hexaco_template(text=list_to_str(question_batch),
                                 likert_scale=", ".join(likert_scale),
                                 base_text=persona_base_text,
                                 attributes=persona_str)
        answers = generator(prompt).strip().replace(">", "").splitlines()
        return [re.sub(r"[^a-zA-Z]", "", ans).strip() for ans in answers]

    # Non-batched
    prompt_list = []
    for question in question_batch:
        if likert_shuffle:
            random.shuffle(likert_scale)
        
        if "gpt" in args.model_name.lower():
            prompt = render_openai_messages(
                                hexaco_template,
                                attributes=persona_str,
                                text=question,
                                likert_scale=", ".join(likert_scale),
                            )
        else:
            
            prompt = hexaco_template(text=question,
                                    likert_scale=", ".join(likert_scale),
                                    attributes=persona_str)
        prompt_list.append(prompt)

    if "gpt" in args.model_name:
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(lambda p: openai_answer(model, p), prompt_list))
    else:
        return [local_answer(model, p, likert_scale) for p in prompt_list]

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
    
    base_hexaco_template, persona_hexaco_template = load_prompt_templates(args.model_name)

    # Load model
    model = load_model(args.model_name, args.hf_token)
    if "gpt" not in args.model_name:
        print(f"Model loaded on: {model.model.device}")

    
    print(t.cuda.is_available())  # True means GPU is visible
    print(t.cuda.current_device())  
    print(t.cuda.get_device_name(0))

    # Load data
    question_list, likert_scale = load_data(args)

    # Load personas
    persona_datasets = load_personas(args)

    # Run
    results = generate_answers(model, args, persona_datasets, question_list, likert_scale)

    # Save results
    model_name = args.model_name.replace(".", "_").split("/")[-1]
    out_dir = "../experiment_results/reliability_experiments/experiment_2"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.persona_source}_hexaco_answers_{model_name}.json")
    write_to_json(results, out_file)
    print(f"Results saved to {out_file}")
