from huggingface_hub import login
hf_token = "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"
login(token=hf_token, add_to_git_credential=True)

import json
import os
import sys
import pickle
sys.path.append('../')
import dataclasses
import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from datasets import load_dataset
from methods.control_vectors.control import ControlModel
from methods.control_vectors.extract import ControlVector, DatasetEntry 
from langchain.chat_models import ChatOpenAI
os.environ['OPENAI_API_KEY'] = "sk-proj-RhpNuVTcfW3mEoSyMQwNT3BlbkFJrlHIHpVZCAiqHWw9ZyDr"

import re
import pdb
from tqdm.notebook import trange, tqdm

from enum import Enum

from model_utils import create_and_prepare_model
from peft import LoraConfig
from peft import PeftModelForCausalLM, PeftConfig


model_path = "/home/vithu-wand/extra-storage/wandx_sandbox/results/mistral-7b-instruct-v0p2-peft-sft-lora-personachat-multigpu/"


templates = {
    "mistral-single-grading": """
        [INST] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant 
        to the user question displayed below. Your evaluation should consider factors such as the helpfulness, 
        relevance, accuracy, depth, creativity, and level of detail of the response. The scoring range is from 1 to 10.
        
        Response for Evaluation:
        
        #CONTEXT
        {context}
        #ENDCONTEXT
        
        #TASK
        Begin your evaluation by 
            providing a short explanation. Be as objective as possible. After providing your explanation, you must 
            rate the response on a scale of 1 to 10 by strictly following this format: "The score is: [score]"
        
        #OUTPUT FORMAT
        "The score is: [score]" [/INST]
        """,
    "mistral-pairwise" : """
        [INST] Please act as an impartial judge and evaluate the quality of the response provided by two AI assistants
        to the user question displayed below. You should choose the assistant that follows the user's instructions and 
        answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, 
        accuracy, depth, creativity, and level of detail of their responses.
        
        Response A for Evaluation:
        #CONTEXTA
        {text_a}
        #ENDCONTEXTA
        
        Response B for Evaluation:
        #CONTEXTB
        {text_b}
        #ENDCONTEXTB
        
        #TASK
        Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and 
        ensure that the order in which the responses were presented does not influence your decision. Do not allow the length 
        of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
        After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is 
        better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie." [/INST]
        """,
    "mistral-perso-pairwise" : """
        [INST] Please act as an impartial judge and evaluate the quality of the response provided by two AI assistants
        to the user question displayed below. You should choose the assistant that follows the user's instructions and 
        answers the user's in a more personalized manner. Your evaluation should consider factors such as 
        the helpfulness, relevance, accuracy, depth, creativity, level of detail and personalization of their responses.
        
        Response A for Evaluation:
        #CONTEXTA
        {text_a}
        #ENDCONTEXTA
        
        Response B for Evaluation:
        #CONTEXTB
        {text_b}
        #ENDCONTEXTB
        
        #TASK
        Begin your evaluation by comparing the two responses and provide a short explanation on which one is more personalized. 
        Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
        Do not allow the length  of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as 
        objective as possible. After providing your explanation, output your final verdict by strictly following this format: 
        "Final Verdict: [[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie." [/INST]
        """
}


# Mistral Instruct doesn't allow system prompts, so we append it to the user message.
DEFAULT_MISTRAL_CHAT_TEMPLATE = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{% if messages[1]['role'] == 'user' %}"\
            "{{ '[INST] ' + messages[0]['content'] + ' ' + messages[1]['content'] + ' [/INST]' }}"\
            "{% set loop_messages = messages[2:] %}"\
        "{% else %}"\
            "{{ '[INST] ' + messages[0]['content'] + ' [/INST]' }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% endif %}"\
    "{% else %}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"

class MistralSpecialTokens(str, Enum):
    # Define the roles and tokens for Mistral chat formatting
    user = "user"
    assistant = "assistant"
    system = "system"
    eos_token = "eos_token"  # End of sentence token
    bos_token = "bos_token"    # Beginning of sentence token
    pad_token = "<pad>"
    inst_token = "[INST]"  # Instruction start token
    inst_end_token = "[/INST]"  # Instruction end token

    @classmethod
    def list(cls):
        return [c.value for c in cls]  # List all values for convenience


def remove_persona_label(text):
    if "Persona A: " in text:
        return text.replace("Persona A: ", "")
    elif "Persona B: " in text:
        return text.replace("Persona B: ", "")
    else:
        return text

def create_conversation_synthetic_persona_dataset(batch):
    samples = []
    
    for dialogue_sequence, reference_statement in zip(batch["dialogue"], batch["reference"]):
        messages = []
        
        # Add each turn to the dialogue history
        for index, turn in enumerate(dialogue_sequence):
            role = MistralSpecialTokens.user if index % 2 == 0 else MistralSpecialTokens.assistant
            processed_turn = remove_persona_label(turn)
            messages.append({"role": role, "content": processed_turn})
        
        dialogue_instance = tokenizer.apply_chat_template(messages, tokenize=False)
        samples.append(dialogue_instance)
        # pdb.set_trace()
    return {"content": samples}


if __name__ == "__main__":
    # Load the model and tokenizer
    baseline_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token

    # Adjust the embeddings and output layers
    baseline_model.resize_token_embeddings(32008)

    # Load the PEFT configuration and model
    peft_config = PeftConfig.from_pretrained(model_path)
    lora_finetuned_model = PeftModelForCausalLM.from_pretrained(baseline_model, model_path)
    lora_finetuned_model.to('cuda')

    model_name = 'gpt-4'
    api  = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key = "")

    prompt_template = "mistral-perso-pairwise"

    settings = {
        "pad_token_id": tokenizer.eos_token_id, # silence warning
        "do_sample": False, # temperature=0
        "max_new_tokens": 128,
        "repetition_penalty": 1.1, # reduce control jank
    }

    # Put baseline model onto CUDA
    baseline_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    baseline_model.to('cuda')

    dataset_name = "nazlicanto/persona-based-chat"
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.map(create_conversation_synthetic_persona_dataset, batched=True)

    total_len = 0
    baseline_win_counter = 0
    personalized_win_counter = 0
    tie_counter = 0

    eval_stats = {'total_len': total_len,
                'baseline_win_counter': baseline_win_counter,
                'personalized_win_counter': personalized_win_counter,
                'tie_counter': tie_counter
                }

    if prompt_template == "mistral-perso-pairwise":
        for input_context in tqdm(dataset['content']):
            input_ids = tokenizer(input_context, return_tensors="pt").to(lora_finetuned_model.device)

            baseline_response = tokenizer.decode(baseline_model.generate(**input_ids, **settings).squeeze())
            # print("Generated Baseline Response:", baseline_response)

            lora_response = tokenizer.decode(lora_finetuned_model.generate(**input_ids, **settings).squeeze())
            # print("\nGenerated Controlled Response:", controlled_response)

            pairwise_prompt = templates[prompt_template].format(text_a=baseline_response, text_b=lora_response)

            response = api.invoke(pairwise_prompt).content

            verdict = re.search(r'Final Verdict:\s*\[\[([A-Z])\]\]', response).group(1)

            total_len += 1
            eval_stats['total_len'] = total_len 
            
            if verdict == "A":
                eval_stats['baseline_win_counter'] += 1
            elif verdict == "B":
                eval_stats['personalized_win_counter'] += 1
            elif verdict == "C":
                eval_stats['tie_counter'] += 1

            eval_stats['baseline_win_rate'] = eval_stats['baseline_win_counter']/total_len
            eval_stats['personalized_win_rate'] = eval_stats['personalized_win_counter']/total_len
            eval_stats['tie_win_rate'] = eval_stats['tie_counter']/total_len

            print(eval_stats)

        savename = 'lora_finetuned_personachat_mistral7b_perso_pairwise_eval.pt'
        torch.save(eval_stats, os.path.join(save_path, savename))


# baseline_response = tokenizer.decode(baseline_model.generate(**input_ids, **settings).squeeze())
# print("Generated Baseline Response:", strip_after_sentiment(baseline_response))