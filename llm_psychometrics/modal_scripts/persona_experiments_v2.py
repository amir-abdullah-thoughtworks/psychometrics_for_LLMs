import torch as t
import outlines
from outlines import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from typing import Literal
from enum import Enum
import yaml
import sys
import os
import json
import re
from tqdm import tqdm
# from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# sys.path.append("../")
from utils import inverse_likert, list_to_str
device = t.device("cuda" if t.cuda.is_available() else "cpu")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class OpenaiResponse(BaseModel):
    response: str
    
    
with open('generation_config.yaml', 'r') as file:
    generation_config = yaml.safe_load(file)

with open('hexaco_100_questions.yaml', 'r') as file:
    question_list = yaml.safe_load(file)

with open('paraphrased_hexaco_100_questions.yaml', 'r') as file:
    paraphrased_question_list = yaml.safe_load(file)

with open('hexaco_100_eval.yaml', 'r') as file:
    hexaco_eval = yaml.safe_load(file)

with open('personas_v2.yaml', 'r') as file:
    personas = yaml.safe_load(file)
    

from huggingface_hub import login
login("")

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=t.float16
).to(device)

# Create outlines-wrapped model
local_model = outlines.from_transformers(hf_model, tokenizer)

NO_ANSWER = "Do not wish to answer"
likert_scale = generation_config['likert_scale'].copy()
likert_scale.append(NO_ANSWER)

inverted_likert = inverse_likert(generation_config['likert_scale'].copy())
inverted_likert.append(NO_ANSWER)

likert_scale_without_no = generation_config['likert_scale'].copy()
inverted_likert_without_no = inverse_likert(generation_config['likert_scale'].copy())

base_text = personas['law_enforcement']['base_text']
persona = personas['law_enforcement']['personas'][0]

print(",\n".join([f"{key} : {persona[key]}" for key in persona]))

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


def local_generation(hexaco_template, model, question, likert_scale, batching=False, persona_str=None, persona_base_text=None):
    
    if batching:
        pass
        # generator = Generator(model)
        # prompt = hexaco_template(text=list_to_str(question_batch), likert_scale = ", ".join(likert_scale), base_text = persona_base_text, attributes = persona_str)
        # answers = generator(prompt).strip().splitlines()
        # results = [re.sub(r"[^a-zA-Z]", "", answer).strip() for answer in answers]
    else:
                
        # results = []
        # for question in question_batch:
        prompt = hexaco_template(text=question, likert_scale = ", ".join(likert_scale), base_text = persona_base_text, attributes = persona_str)
        predicted = model(
            prompt,
            Literal[*likert_scale]
            )
            # prompt_list.append(prompt)
            # inputs = tokenizer(prompt, return_tensors="pt").to(device)
            # with t.no_grad():
            #     outputs = hf_model(**inputs)
            #     logits = outputs.logits[:, -1, :]  # Only look at the last token's logits
            
            #     option_ids = [
            #         tokenizer.encode(option, add_special_tokens=False)[0]
            #         for option in likert_scale
            #     ]
            #     option_logits = logits[:, option_ids].squeeze()
            
            #     best_idx = t.argmax(option_logits).item()
            #     predicted = likert_scale[best_idx]
                # print(f"Predicteda:{answer}")
            
            # results.append(predicted)
            # args_list = [(prompt, likert_scale) for prompt in prompt_list]
    
            # with ProcessPoolExecutor(max_workers=4) as executor:
            #     results = list(executor.map(answer, args_list))
        
        return [predicted]

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
    
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(answer, prompt_list))
        return results
    
def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_answers(generation_function, model, question_list, likert_scale, job_title=None, n_times = 30, batch_size = 5, batching = False, personas=personas):
    
    if job_title:
        answers = []
        hexaco_template = persona_hexaco_template
        base_text = personas[job_title]['base_text']
        persona_list = personas[job_title]['personas']
        for persona in tqdm(persona_list, desc="Personas", position=0):
            persona_str = ",\n".join([f"{key} : {persona[key]}" for key in persona])
            repeated_answers = []
            for i in tqdm(range(n_times), desc="Iterations", position=1):
                persona_answer = []
                # for question_batch in tqdm(batch_list(question_list, batch_size), desc="Batches", position=2):
                for question in tqdm(question_list, desc="Questions", position=2):
                    batch_answers = generation_function(hexaco_template = hexaco_template, 
                                                        model = model, 
                                                        question = question,
                                                        likert_scale = likert_scale,
                                                        persona_str = persona_str,
                                                        persona_base_text = base_text,
                                                        batching = batching)
                    persona_answer.extend(batch_answers)
                    del batch_answers
                    t.cuda.empty_cache()
                    t.cuda.synchronize()
                    # print(t.cuda.memory_allocated() / 1024**2, "MB")
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
                batch_answers = generation_function(hexaco_template = hexaco_template, 
                                                        question_batch = question_batch,
                                                        likert_scale = likert_scale,
                                                        batching = batching)
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

def run_experiment(data_dir, model_name,job_title=None):
    
    if "gpt" in model_name:
        print("Running Openai generation")
        model = openai_model
        generation_model = openai_generation
    else:
        print("Running Local generation")
        model = local_model
        generation_model = local_generation
    
    # print("################ Normal Questions, Normal Likert, Refusal Allowed ################")
    # normal_hexaco_answers = generate_answers(generation_model, model, question_list, likert_scale, job_title)
    # for persona_dict in normal_hexaco_answers:
    #     persona_dict['config']['likert_scale'] = "normal"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_answers, os.path.join(data_dir,f"normal_hexaco_answers_{model_name}.json"))
    
    # print("################ Normal Questions, Normal Likert, Refusal Not Allowed ################")
    # normal_hexaco_answers_without_no = generate_answers(generation_model, model, question_list, likert_scale_without_no, job_title)
    # for persona_dict in normal_hexaco_answers_without_no:
    #     persona_dict['config']['likert_scale'] = "normal"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "no_refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_answers_without_no, os.path.join(data_dir,f"normal_hexaco_answers_without_no_{model_name}.json"))
    
    # print("################ Normal Questions, Inverted Likert, Refusal Allowed ################")
    # normal_hexaco_inverted_likert_answers = generate_answers(generation_model, model, question_list, inverted_likert, job_title)
    # for persona_dict in normal_hexaco_inverted_likert_answers:
    #     persona_dict['config']['likert_scale'] = "inverted"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_inverted_likert_answers, os.path.join(data_dir,f"normal_hexaco_inverted_likert_answers_{model_name}.json"))
    
    # print("################ Normal Questions, Inverted Likert, Refusal Not Allowed ################")
    # normal_hexaco_inverted_likert_without_no_answers = generate_answers(generation_model, model, question_list, inverted_likert_without_no, job_title)
    # for persona_dict in normal_hexaco_inverted_likert_without_no_answers:
    #     persona_dict['config']['likert_scale'] = "inverted"
    #     persona_dict['config']['paraphrase'] = "normal"
    #     persona_dict['config']['refusal'] = "no_refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(normal_hexaco_inverted_likert_without_no_answers, os.path.join(data_dir,f"normal_hexaco_inverted_likert_without_no_answers_{model_name}.json"))
    
    # print("################ Paraphrase Questions, Normal Likert, Refusal Allowed ################")
    # paraphrase_hexaco_answers = generate_answers(generation_model, model, paraphrased_question_list, likert_scale, job_title)
    # for persona_dict in paraphrase_hexaco_answers:
    #     persona_dict['config']['likert_scale'] = "normal"
    #     persona_dict['config']['paraphrase'] = "paraphrase"
    #     persona_dict['config']['refusal'] = "refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(paraphrase_hexaco_answers, os.path.join(data_dir,f"paraphrase_hexaco_answers_{model_name}.json"))
    
    # print("################ Paraphrase Questions, Normal Likert, Refusal Not Allowed ################")
    # paraphrase_hexaco_answers_without_no = generate_answers(generation_model, model, paraphrased_question_list, likert_scale_without_no, job_title)
    # for persona_dict in paraphrase_hexaco_answers_without_no:
    #     persona_dict['config']['likert_scale'] = "normal"
    #     persona_dict['config']['paraphrase'] = "paraphrase"
    #     persona_dict['config']['refusal'] = "no_refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(paraphrase_hexaco_answers_without_no, os.path.join(data_dir,f"paraphrase_hexaco_answers_without_no_{model_name}.json"))
    
    # print("################ Paraphrase Questions, Inverted Likert, Refusal Allowed ################")
    # paraphrase_hexaco_inverted_likert_answers = generate_answers(generation_model, model, paraphrased_question_list, inverted_likert, job_title)
    # for persona_dict in paraphrase_hexaco_inverted_likert_answers:
    #     persona_dict['config']['likert_scale'] = "inverted"
    #     persona_dict['config']['paraphrase'] = "paraphrase"
    #     persona_dict['config']['refusal'] = "refusal"
    #     persona_dict['config']['model_name'] = model_name
    # write_to_json(paraphrase_hexaco_inverted_likert_answers, os.path.join(data_dir,f"paraphrase_hexaco_inverted_likert_answers_{model_name}.json"))
    
    print("################ Paraphrase Questions, Inverted Likert, Refusal Not Allowed ################")
    paraphrase_hexaco_inverted_likert_without_no_answers = generate_answers(generation_model, model, paraphrased_question_list, inverted_likert_without_no, job_title)
    for persona_dict in paraphrase_hexaco_inverted_likert_without_no_answers:
        persona_dict['config']['likert_scale'] = "inverted"
        persona_dict['config']['paraphrase'] = "paraphrase"
        persona_dict['config']['refusal'] = "no_refusal"
        persona_dict['config']['model_name'] = model_name
    write_to_json(paraphrase_hexaco_inverted_likert_without_no_answers, os.path.join(data_dir,f"paraphrase_hexaco_inverted_likert_without_no_answers_{model_name}.json"))

def main():
    os.makedirs("persona_experiment_results_v2", exist_ok = True)
    # generate_answers(local_generation,hf_model, question_list, likert_scale, "law_enforcement")
    run_experiment(data_dir="persona_experiment_results_v2", model_name = "llama32_1b_it",job_title = "law_enforcement")

# if __name__ == "__main__":

#     generate_answers(local_generation,hf_model, question_list, likert_scale, "law_enforcement")