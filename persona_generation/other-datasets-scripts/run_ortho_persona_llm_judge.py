from huggingface_hub import login
hf_token = "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"
login(token=hf_token, add_to_git_credential=True)

import json
import os
import re
import sys
sys.path.append('../')
import dataclasses
import numpy as np
import transformers

import torch
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

#from methods.control_vectors.control import ControlModel
#from methods.control_vectors.extract import ControlVector, DatasetEntry 


from langchain.chat_models import ChatOpenAI

from tqdm import tqdm

import pdb

os.environ['OPENAI_API_KEY'] = "sk-proj-RhpNuVTcfW3mEoSyMQwNT3BlbkFJrlHIHpVZCAiqHWw9ZyDr"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
        Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must 
        rate the response on a scale of 1 to 10 by strictly following this format: "The score is: [score]"
        
        #OUTPUT FORMAT
        "The score is: [score]" [/INST]
        """,
    "mistral-perso-single-grading": """
        [INST] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant 
        to the user question displayed below. Your main evaluation should consider the level of 
        personalization in the response based on past dialogue history. Also, consider factors such as helpfulness, 
        relevance, accuracy, depth, creativity, and level of detail of the response. The scoring range is from 1 to 10.
        
        Response for Evaluation:
        
        #CONTEXT
        {context}
        #ENDCONTEXT
        
        #TASK
        Begin your evaluation by providing a short explanation on the level of personalization in the response. Be as objective as possible. 
        After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "The score is: [score]"
        
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
        After providing your explanation, output your final verdict by strictly following this format: "Final Verdict: [[A]]\" if 
        assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie." [/INST]
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
        """,
    "mistral_perso_strict_pairwise" : """
        [INST] Please act as an impartial judge and evaluate the quality of the response provided by two AI assistants
        to the user question displayed below. You should choose the assistant that follows the user's instructions and 
        answers the user's in a more personalized manner considering the past dialogue history, personal interests, 
        communication style and tone.
        
        Response A for Evaluation:
        #CONTEXTA
        {text_a}
        #ENDCONTEXTA
        
        Response B for Evaluation:
        #CONTEXTB
        {text_b}
        #ENDCONTEXTB
        
        #TASK
        Begin your evaluation by comparing the two responses and provide a short explanation on which one is more personalized considering 
        the past dialogue history, personal interests, communication style and tone. 
        Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
        Do not allow the length  of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as 
        objective as possible. After providing your explanation, output your final verdict by strictly following this format: 
        "Final Verdict: [[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie." [/INST]
        """
}


def create_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created")
        except FileExistsError:
            print(f"Directory '{directory_path}' already exists")
            

# def create_conversation_personachat_dataset(batch):
#     global dataset
#     dataset = []
#     for personality, candidates, history, conv_id, utterance_idx in zip(batch['personality'], batch['candidates'], batch['history'], batch['conv_id'], batch['utterance_idx']):
#         # Combine personality traits into a single string
#         personality_description = " ".join(personality)
        
#         # Create a full context by appending the history to the personality description
#         full_context = f"{user_tag} Personality: {personality_description} History: {' '.join(history)} {asst_tag}"

#         # Iterate over each candidate response and create a dataset entry
#         for response in candidates[:-1]:
#             # Construct a complete dialogue instance with the response appended
#             negative_dialogue_instance = f"{full_context} {response}"
#             positive_dialogue_instance = f"{full_context} {candidates[-1]}"
            
#             # In this example, every candidate response is treated as a potential correct response
#             # There's no explicit negative example in the provided data structure, so we might just replicate the context or use a different mechanism
#             dataset.append(
#                 DatasetEntry(
#                     positive=positive_dialogue_instance,
#                     negative=negative_dialogue_instance  # Consider varying this for actual training scenarios
#                 )
#             )

# def create_conversation_personachat_dataset(batch):
#     samples = []

#     for personality, candidates, history, conv_id, utterance_idx in zip(
#         batch['personality'], batch['candidates'], batch['history'], batch['conv_id'], batch['utterance_idx']
#     ):
#         # Combine personality traits into a single string
#         personality_description = " ".join(personality)

#         # Build the dialogue history with alternating roles
#         messages = [{"role": "user", "content": f"Personality: {personality_description}"}]
#         for index, turn in enumerate(history):
#             role = "user" if index % 2 == 0 else "assistant"
#             messages.append({"role": role, "content": turn})

#         # Create a positive dialogue instance
#         positive_messages = messages + [{"role": "assistant", "content": candidates[-1]}]
#         positive_dialogue_instance = tokenizer.apply_chat_template(positive_messages, tokenize=False)
        
#         # Create negative dialogue instances
#         for candidate in candidates[:-1]:
#             negative_messages = messages + [{"role": "assistant", "content": candidate}]
#             negative_dialogue_instance = tokenizer.apply_chat_template(negative_messages, tokenize=False)
            
#             samples.append({"positive": positive_dialogue_instance, "negative": negative_dialogue_instance})

#         pdb.set_trace()

#     return {"content": samples}




# def create_conversation_personachat_dataset(batch):
#     global dataset
#     dataset = []

#     for personality, candidates, history, conv_id, utterance_idx in zip(
#         batch['personality'], batch['candidates'], batch['history'], batch['conv_id'], batch['utterance_idx']
#     ):
#         # Combine personality traits into a single string
#         personality_description = " ".join(personality)

#         # Build the dialogue history with alternating roles
#         messages = [{"role": "user", "content": f"Personality: {personality_description}"}]
#         for index, turn in enumerate(history):
#             role = "user" if index % 2 == 0 else "assistant"
#             messages.append({"role": role, "content": turn})

#         # Create a positive dialogue instance
#         # positive_messages = messages + [{"role": "assistant", "content": candidates[-1]}]
#         context = messages[0]['content']

#         positive_dialogue_instance = f"""
#         Context:
#         {context}
        
#         [INST]Please review the dialogue histroy and please generate a personalized response based on the 
#         conversation context, tone, user's interests, communication style.[/INST]
#         Output: {candidates[-1]} </s>"""
#         # positive_dialogue_instance = tokenizer.apply_chat_template(positive_messages, tokenize=False)
        
#         # Create negative dialogue instances
#         for candidate in candidates[:-1]:
#             # negative_messages = messages + [{"role": "assistant", "content": candidate}]
#             # negative_dialogue_instance = tokenizer.apply_chat_template(negative_messages, tokenize=False)
#             negative_dialogue_instance = f"""
#             Context:
#             {context}
            
#             [INST]Please review the dialogue histroy and please generate a personalized response based on the 
#             conversation context, tone, user's interests, communication style.[/INST]
#             Output: {candidate} </s>"""

#             dataset.append(
#                 DatasetEntry(
#                     positive=positive_dialogue_instance,
#                     negative=negative_dialogue_instance  # Consider varying this for actual training scenarios
#                 )
#             )
#             # samples.append({"positive": positive_dialogue_instance, "negative": negative_dialogue_instance})

#             # pdb.set_trace()

#     # return {"content": samples}


def create_conversation_personachat_dataset(batch):
    global dataset
    dataset = []

    for personality, candidates, history, conv_id, utterance_idx in zip(
        batch['personality'], batch['candidates'], batch['history'], batch['conv_id'], batch['utterance_idx']
    ):
        # Combine personality traits into a single string
        personality_description = " ".join(personality)

        # Build the dialogue history with alternating roles
        messages = [{"role": "user", "content": f"Personality: {personality_description}"}]

        for index, turn in enumerate(history):
            role = "user" if index % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})

        # pdb.set_trace()
        # Create a positive dialogue instance
        # positive_messages = messages + [{"role": "assistant", "content": candidates[-1]}]
        context = messages[0]['content']
        user_query = messages[1]['content']

        positive_dialogue_instance = f"""
        Context:
        {context}
        
        [INST]Please review the dialogue history and please generate a personalized response for user's query based on the 
        conversation context, tone, user's interests, communication style.\n{user_query}[/INST]
        Output: {candidates[-1]} </s>"""
        # positive_dialogue_instance = tokenizer.apply_chat_template(positive_messages, tokenize=False)
        # pdb.set_trace()
        # Create negative dialogue instances
        for candidate in candidates[:-1]:
            # negative_messages = messages + [{"role": "assistant", "content": candidate}]
            # negative_dialogue_instance = tokenizer.apply_chat_template(negative_messages, tokenize=False)
            negative_dialogue_instance = f"""
            Context:
            {context}
            
            [INST]Please review the dialogue history and please generate a personalized response for user's query. Factor the  
        conversation context and user's tone, interests, and communication style into the response.\n user: {user_query}[/INST]
            Output: {candidate} </s>"""

            dataset.append(
                [positive_dialogue_instance, negative_dialogue_instance]
            )
            # samples.append({"positive": positive_dialogue_instance, "negative": negative_dialogue_instance})

            # pdb.set_trace()

    # return {"content": samples}


def remove_persona_label(text):
    if "Persona A: " in text:
        return text.replace("Persona A: ", "")
    elif "Persona B: " in text:
        return text.replace("Persona B: ", "")
    else:
        return text


# def create_conversation_synthetic_persona_dataset(batch):
#     global dataset
#     dataset = []
#     for persona_b_statements, dialogue_sequence, reference_statement in zip(batch["persona_b"], batch["dialogue"], batch["reference"]):
#         # Combine persona descriptions into a single string
#         persona_description = " ".join(persona_b_statements)

#         # # Start with the persona description to contextualize the dialogue
#         # dialogue_history = f"{user_tag} {persona_description} {asst_tag} "

#         dialogue_history = ""
        
#         # Add each turn to the dialogue history
#         for turn in dialogue_sequence:
#             processed_turn = remove_persona_label(turn)
#             dialogue_history += f"{user_tag} {processed_turn} {asst_tag} "

#         dataset.append(dialogue_history)
        
#         # The reference response is what we want the model to learn to predict
#         reference_response = reference_statement

def create_conversation_synthetic_persona_dataset(batch):
    global dataset
    dataset = []
    for persona_b_statements, dialogue_sequence, reference_statement in zip(batch["persona_b"], batch["dialogue"], batch["reference"]):
        # Combine persona descriptions into a single string
        persona_description = " ".join(persona_b_statements)

        # Initialize the dialogue history with persona description as context
        dialogue_history = []

        for index, turn in enumerate(dialogue_sequence):
            role = "user" if index % 2 == 0 else "assistant"
            dialogue_history.append({"role": role, "content": remove_persona_label(turn)})

        # for index, turn in enumerate(dialogue_sequence):
        #     role = "user" if index % 2 == 0 else "assistant"
        #     role = "user" if dialogue_history else "assistant"  # The first turn is user's, subsequent alternate
        #     dialogue_history.append({"role": role, "content": remove_persona_label(turn)})

        # Format context with persona and dialogue history
        # context = f"Persona Description: {persona_description}\n"
        context = f""
        for message in dialogue_history[:-1]:
            context += f"{message['role']}: {message['content']}\n"

        user_query = dialogue_history[-1]['content']

        # The reference response in a structured output format
        dialogue_instance = f"""
        Context:
        {context}

        [INST]Please review the dialogue history and please generate a personalized response for user's query. Factor the  
        conversation context and user's tone, interests, and communication style into the response.\n user: {user_query}[/INST]"""
        
        dataset.append(dialogue_instance)


QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 1228,
    batch_size: int = 1,
) -> List[str]:

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations        

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

if __name__ == "__main__":
    

    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    save_path = "./results/control_vectors/Mistral-7B-Instruct-v0.1/synthetic_persona_chat/instruct_template"
    #config = AutoConfig.from_pretrained(model_name, max_len = 2048)

    baseline_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    baseline_model = baseline_model.to("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")

    #tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True, padding_side="right", model_max_length = 2048) 
    #tokenizer.pad_token = tokenizer.unk_token
    #tokenizer.padding_side = "right"

    model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    device="cuda:0",
    dtype=torch.float16,
    default_padding_side='left'
    )

    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = '<|extra_0|>'
    tokenizer = model.tokenizer

    #Configure the pad token in the model
    #model.config.pad_token_id = tokenizer.pad_token_id
    tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)


    # user_tag, asst_tag = "[INST]", "[/INST]"

    # Assuming the dataset is loaded and includes 'personality', 'candidates', 'history', 'conv_id', 'utterance_idx'
    dataset_name = "bavard/personachat_truecased"
    dataset = load_dataset(dataset_name, split='train')
    #dataset = dataset.select(range(10))
    dataset.map(create_conversation_personachat_dataset, batched=True)
    prompt_no_input_template = """%s"""
    positive_examples = [example[0] for example in dataset[0:10]]
    print(positive_examples[0:2])
    negative_examples = [example[1] for example in dataset[0:10]]
    harmful_toks = tokenize_instructions_fn(instructions=positive_examples)
    harmless_toks = tokenize_instructions_fn(instructions=negative_examples)
    harmful_logits, harmful_cache = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'resid' in hook_name)
    harmless_logits, harmless_cache = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'resid' in hook_name)
    pos = -1
    layer = 14

    harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
    harmless_mean_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0)

    refusal_dir = harmful_mean_act - harmless_mean_act
    refusal_dir = refusal_dir / refusal_dir.norm()
    del harmful_cache, harmless_cache, harmful_logits, harmless_logits
    gc.collect(); torch.cuda.empty_cache()
    N_INST_TEST = 32
    intervention_dir = refusal_dir
    intervention_layers = list(range(model.cfg.n_layers)) # all layers

    hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

    #intervention_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=fwd_hooks)
    #baseline_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=[])

    # Initialize LLM Judge
    model_name = 'gpt-4'
    api  = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key = "")

    # Initialize eval dataset
    dataset_name = "nazlicanto/persona-based-chat"
    dataset = load_dataset(dataset_name, split='train')
    dataset.map(create_conversation_synthetic_persona_dataset, batched=True)


    # Initialize settings for evaluation loop
    prompt_template = "mistral-perso-pairwise"
    # prompt_template = "mistral_perso_strict_pairwise"
    # prompt_template = 'mistral-single-grading'
    # prompt_template = 'mistral-perso-single-grading'
    # prompt_template = "mistral-pairwise"
    settings = {
        "pad_token_id": tokenizer.eos_token_id, # silence warning
        "do_sample": False, # temperature=0
        "max_new_tokens": 128,
        "repetition_penalty": 1.1, # reduce control jank
    }

    total_len = 0
    baseline_win_counter = 0
    personalized_win_counter = 0
    tie_counter = 0
    baseline_raw_scores = []
    controlled_raw_scores = []

    eval_stats = {'total_len': total_len,
                'baseline_win_counter': baseline_win_counter,
                'personalized_win_counter': personalized_win_counter,
                'tie_counter': tie_counter
                }

    cv_temperature = 1.0
    filename = "ortho_personachat_unsupervisedPCA_mistral7b_mistralchat_newtemplate_10epochs"
    savename = filename + prompt_template + '_' + str(cv_temperature) + '.pt'

    if prompt_template in ["mistral-perso-pairwise", "mistral_perso_strict_pairwise"]:
        for input_context in tqdm(dataset):
            input_ids = tokenizer(input_context, return_tensors="pt").to("cuda:0")
            toks = tokenize_instructions_fn(instructions=input_context)

            #model.reset()
            baseline_response = tokenizer.decode(baseline_model.generate(**input_ids, **settings).squeeze())
            #print("Generated Baseline Response:", baseline_response)

            #model.set_control(control_vector, cv_temperature)

            controlled_response = get_generations(model, input_context, tokenize_instructions_fn, max_tokens_generated=128, fwd_hooks=fwd_hooks)
            #print("\nGenerated Controlled Response:", controlled_response)

            pairwise_prompt = templates[prompt_template].format(text_a=baseline_response, text_b=controlled_response)

            response = api.invoke(pairwise_prompt).content

            verdict = re.search(r'Final Verdict:\s*\[\[([A-Z])\]\]', response).group(1)

            if verdict is not None:
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

        # savename = 'control_vector_personachat_mistral7b_perso_pairwise_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_perso_pairwise_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_perso_pairwise_eval_neg.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_perso_pairwise_eval_pos1p5.pt'

        torch.save(eval_stats, os.path.join(save_path, savename))

    elif prompt_template == 'mistral-pairwise':
        for input_context in tqdm(dataset):
            input_ids = tokenizer(input_context, return_tensors="pt").to(model.device)

            model.reset()
            baseline_response = tokenizer.decode(model.generate(**input_ids, **settings).squeeze())
            # print("Generated Baseline Response:", baseline_response)

            model.set_control(control_vector, cv_temperature)
            controlled_response = tokenizer.decode(model.generate(**input_ids, **settings).squeeze())
            # print("\nGenerated Controlled Response:", controlled_response)

            pairwise_prompt = templates[prompt_template].format(text_a=baseline_response, text_b=controlled_response)

            response = api.invoke(pairwise_prompt).content

            verdict = re.search(r'Final Verdict:\s*\[\[([A-Z])\]\]', response).group(1)

            if verdict is not None:
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
        
        # savename = 'control_vector_personachat_mistral7b_pairwise_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_pairwise_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_pairwise_eval_neg.pt'

        torch.save(eval_stats, os.path.join(save_path, savename))

    elif prompt_template == 'mistral-single-grading':
        baseline_score_avg = 0
        controlled_score_avg = 0

        for input_context in tqdm(dataset):
            input_ids = tokenizer(input_context, return_tensors="pt").to(model.device)

            # baseline
            model.reset()
            baseline_response = tokenizer.decode(model.generate(**input_ids, **settings).squeeze())
            baseline_judge_prompt = templates[prompt_template].format(context=baseline_response)
            baseline_judge_response = api.invoke(baseline_judge_prompt).content
            match = re.search(r"\d+", baseline_judge_response)
            new_baseline_score = int(match.group(0)) if match else None
    
            # control vector response
            model.set_control(control_vector, cv_temperature)
            controlled_response = tokenizer.decode(model.generate(**input_ids, **settings).squeeze())
            controlled_judge_prompt = templates[prompt_template].format(context=controlled_response)
            controlled_judge_response = api.invoke(controlled_judge_prompt).content
            match = re.search(r"\d+", controlled_judge_response)
            new_controlled_score = int(match.group(0)) if match else None

            # Update the running average of baseline score
            if new_baseline_score is not None:
                baseline_score_avg = (baseline_score_avg * total_len + new_baseline_score) / (total_len + 1)

            # Update the running average of controlled model score
            if new_controlled_score is not None:
                controlled_score_avg = (controlled_score_avg * total_len + new_controlled_score) / (total_len + 1)

            if new_baseline_score is not None and new_controlled_score is not None:
                total_len += 1
                eval_stats['total_len'] = total_len
                
                baseline_raw_scores.append(new_baseline_score)
                eval_stats['baseline_raw_scores'] = baseline_raw_scores
                controlled_raw_scores.append(new_controlled_score)
                eval_stats['controlled__raw_scores'] = controlled_raw_scores

                if new_baseline_score > new_controlled_score:
                    eval_stats['baseline_win_counter'] += 1
                elif new_baseline_score < new_controlled_score:
                    eval_stats['personalized_win_counter'] += 1
                elif new_baseline_score == new_controlled_score:
                    eval_stats['tie_counter'] += 1

            eval_stats['baseline_win_rate'] = eval_stats['baseline_win_counter']/total_len
            eval_stats['baseline_score_avg'] = baseline_score_avg
            eval_stats['personalized_win_rate'] = eval_stats['personalized_win_counter']/total_len
            eval_stats['controlled_score_avg'] = controlled_score_avg
            eval_stats['tie_win_rate'] = eval_stats['tie_counter']/total_len

            print(eval_stats)
    
        # savename = 'control_vector_personachat_mistral7b_single_grade_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_single_grade_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_single_grade_eval_neg.pt'

        # savename = 'control_vector_differencevectors_personachat_mistral7b_single_grade_eval_temp1p0.pt'
        # savename = 'control_vector_personachat_mistral7b_single_grade_eval_temp1p0.pt'

        torch.save(eval_stats, os.path.join(save_path, savename))

    elif prompt_template == 'mistral-perso-single-grading':
        baseline_score_avg = 0
        controlled_score_avg = 0

        for input_context in tqdm(dataset):
            input_ids = tokenizer(input_context, return_tensors="pt").to(model.device)

            # baseline
            model.reset()
            baseline_response = tokenizer.decode(model.generate(**input_ids, **settings).squeeze())
            baseline_judge_prompt = templates[prompt_template].format(context=baseline_response)
            baseline_judge_response = api.invoke(baseline_judge_prompt).content
            match = re.search(r"\d+", baseline_judge_response)
            new_baseline_score = int(match.group(0)) if match else None
    
            # control vector response
            model.set_control(control_vector, cv_temperature)
            controlled_response = tokenizer.decode(model.generate(**input_ids, **settings).squeeze())
            controlled_judge_prompt = templates[prompt_template].format(context=controlled_response)
            controlled_judge_response = api.invoke(controlled_judge_prompt).content
            match = re.search(r"\d+", controlled_judge_response)
            new_controlled_score = int(match.group(0)) if match else None

            # Update the running average of baseline score
            if new_baseline_score is not None:
                baseline_score_avg = (baseline_score_avg * total_len + new_baseline_score) / (total_len + 1)

            # Update the running average of controlled model score
            if new_controlled_score is not None:
                controlled_score_avg = (controlled_score_avg * total_len + new_controlled_score) / (total_len + 1)

            if new_baseline_score is not None and new_controlled_score is not None:
                total_len += 1
                eval_stats['total_len'] = total_len

                baseline_raw_scores.append(new_baseline_score)
                eval_stats['baseline_raw_scores'] = baseline_raw_scores
                controlled_raw_scores.append(new_controlled_score)
                eval_stats['controlled__raw_scores'] = controlled_raw_scores

                if new_baseline_score > new_controlled_score:
                    eval_stats['baseline_win_counter'] += 1
                elif new_baseline_score < new_controlled_score:
                    eval_stats['personalized_win_counter'] += 1
                elif new_baseline_score == new_controlled_score:
                    eval_stats['tie_counter'] += 1

            eval_stats['baseline_win_rate'] = eval_stats['baseline_win_counter']/total_len
            eval_stats['baseline_score_avg'] = baseline_score_avg
            eval_stats['personalized_win_rate'] = eval_stats['personalized_win_counter']/total_len
            eval_stats['controlled_score_avg'] = controlled_score_avg
            eval_stats['tie_win_rate'] = eval_stats['tie_counter']/total_len

            print(eval_stats)
    
        # savename = 'control_vector_personachat_mistral7b_single_grade_perso_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_single_grade_perso_eval.pt'
        # savename = 'control_vector_differencevectors_personachat_mistral7b_single_grade_perso_eval_neg.pt'

        # savename = 'control_vector_differencevectors_personachat_mistral7b_single_grade_perso_eval_temp1p0.pt'
        # savename = 'control_vector_personachat_mistral7b_single_grade_perso_eval_temp1p0.pt'

        torch.save(eval_stats, os.path.join(save_path, savename))