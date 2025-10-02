from jinja2 import Template
import os
import sys
import json
from pydantic import BaseModel
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
sys.path.append("../")
from src.utils_v0 import list_to_str, openai_api_call
from src.prompt_templates.sjt_llm_judge_templates import SJT_LLM_JUDGE_EVALUATION_WITH_SEEDS_TEMPLATE_STR, SJT_LLM_JUDGE_EVALUATION_WITHOUT_SEEDS_TEMPLATE_STR
device = "cpu"


class WithSeedOutputFormat(BaseModel):
    score: float
    justification: str
    
class TraitAlignment(BaseModel):
    score: float
    justification: str
    overlaps: list
    
class HexacoTraits(BaseModel):
    honesty_humility: TraitAlignment
    emotionality: TraitAlignment
    extraversion: TraitAlignment
    agreeableness: TraitAlignment
    conscientiousness: TraitAlignment
    openness: TraitAlignment
    
class WithOutSeedOutputFormat(BaseModel):
    value: str
    confidence: float
    justification: str 
    
class HexacoWithOutSeedOutputFormat(BaseModel):
    values: dict
    confidence: float
    justification: str
    
class HexacoResponse(BaseModel):
    first_option: HexacoWithOutSeedOutputFormat
    second_option: HexacoWithOutSeedOutputFormat
    third_option: HexacoWithOutSeedOutputFormat
    fourth_option: HexacoWithOutSeedOutputFormat
    fifth_option: HexacoWithOutSeedOutputFormat
    sixt_option: HexacoWithOutSeedOutputFormat

class SjtLLMWithSeedsJudge(BaseModel):
    scenario_realism: WithSeedOutputFormat
    trait_alignment: HexacoTraits
    ethical_tension: WithSeedOutputFormat
    fairness: WithSeedOutputFormat

class SjtLLMWithoutSeedsJudge(BaseModel):
    urgency_level: WithOutSeedOutputFormat
    threat_level: WithOutSeedOutputFormat
    ambiguity_level: WithOutSeedOutputFormat
    individuals_involved: WithOutSeedOutputFormat
    authority_relationships: WithOutSeedOutputFormat
    situation_type: WithOutSeedOutputFormat
    time_of_day: WithOutSeedOutputFormat
    race: WithOutSeedOutputFormat
    gender: WithOutSeedOutputFormat
    age: WithOutSeedOutputFormat
    hexaco_traits: HexacoResponse
    rubric_quality: WithOutSeedOutputFormat
    
    
def write_to_json(file, file_path):
    with open(file_path, 'w') as f:
        json.dump(file, f)
        
def read_json(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file

def batch_list(lst, n, start_index=0):
    for idx, i in enumerate(range(0, len(lst), n)):
        if idx < start_index:
            continue  # skip batches before resume point
        yield idx, lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sjt-source", type=str, default="huggingface",
                        help="Source of Persona (huggingface | local)")
    parser.add_argument("--hf-sjt-path", type=str, default="thoughtworks/psychometric_SJTs",
                        help="HF Path for SJTs")
    parser.add_argument("--local-sjt-path", type=str, default="../data/sjt_data/synthetic_generated_sjt_list_v6.1_temp_point9_for_annotations.json",
                        help="HF Path for SJTs")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Huggingface token")
    parser.add_argument("--n-sjtsample", type=int, default=1,
                        help="Number of SJTs to be sampled for each template")
    parser.add_argument("--batch-size", type=int, default=40,
                        help="Batch Size to divide SJT processing")
    return parser.parse_args()

def load_sjts(args):
    
    if args.sjt_source == "huggingface":
        print("Using Huggingface SJTs")
        print(f"Loading SJTs from {args.hf_sjt_path}")
        hf_sjt_dataset = load_dataset(args.hf_sjt_path)
        sjt_datasets_total = hf_sjt_dataset['train']
        total_sjt_df = sjt_datasets_total.to_pandas()
        sampled_sjts = total_sjt_df.groupby("template_no").sample(n=args.n_sjtsample, random_state=42)
        sjt_datasets = sampled_sjts.to_dict("records")
        print(f"No of SJts: {len(sjt_datasets)}")
    elif args.sjt_source == "local":
        print("Loading SJTs from local storage")
        print(f"Loading SJTs from {args.local_sjt_path}")
        sjt_datasets = read_json(args.local_sjt_path)
        print(f"No of SJts: {len(sjt_datasets)}")
    else:
        raise NotImplementedError("Non Huggingface Synthetic SJTs is not implemented yet")
    
    return sjt_datasets
        

SJT_LLM_JUDGE_EVALUATION_WITH_SEEDS_TEMPLATE = Template(SJT_LLM_JUDGE_EVALUATION_WITH_SEEDS_TEMPLATE_STR)
SJT_LLM_JUDGE_EVALUATION_WITHOUT_SEEDS_TEMPLATE = Template(SJT_LLM_JUDGE_EVALUATION_WITHOUT_SEEDS_TEMPLATE_STR)

sjt_answer_default_order = ['honesty_humility_option', 'emotionality_option', 'extraversion_option', 'agreeableness_option', 'conscientiousness_option', 'openness_option']

def sjt_with_seeds_evaluation(sjt_dict):
    sjt = sjt_dict['corrected_sjt']
    config_dict = sjt_dict['config'].copy()
    config_dict['question'] = sjt['question']
    config_dict['answer_options'] = list_to_str([f"{key} : {sjt[key]}" for key in sjt_answer_default_order if  key in sjt and "_option" in key])

    sjt_evaluation_prompt = SJT_LLM_JUDGE_EVALUATION_WITH_SEEDS_TEMPLATE.render(config_dict)
    openai_sjt_response = openai_api_call(prompt=sjt_evaluation_prompt, response_format=SjtLLMWithSeedsJudge ,
                                          model="gpt-4.1", temperature=0, top_p=1, presence_penalty=0, frequency_penalty=0)

    response = openai_sjt_response.model_dump()
    # response['question_hash_id'] = sjt_dict['hash_id']
    return response

def sjt_without_seeds_evaluation(sjt_dict):

    sjt = sjt_dict['corrected_sjt']
    config_dict = {}
    config_dict['question'] = sjt['question']
    config_dict['answer_options'] = list_to_str([sjt[key] for key in sjt_answer_default_order if "_option" in key])

    sjt_evaluation_prompt = SJT_LLM_JUDGE_EVALUATION_WITHOUT_SEEDS_TEMPLATE.render(config_dict)
    openai_sjt_response = openai_api_call(prompt=sjt_evaluation_prompt, response_format=SjtLLMWithoutSeedsJudge ,
                                          model="gpt-4.1", temperature=0, top_p=1, presence_penalty=0, frequency_penalty=0)

    response = openai_sjt_response.model_dump()
    # response['question_hash_id'] = sjt_dict['hash_id']

    return response

def sjt_llm_judge_evaluation(synthetic_sjt_list):

    sjt_evaluation_result = {}

    for sjt_dict in tqdm(synthetic_sjt_list, desc="synthetic SJTs"):
        
        # sjt_with_seeds_evaluation_response = sjt_with_seeds_evaluation(sjt_dict)

        # sjt_without_seeds_evaluation_response = sjt_without_seeds_evaluation(sjt_dict)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(sjt_with_seeds_evaluation, sjt_dict)
            future2 = executor.submit(sjt_without_seeds_evaluation, sjt_dict)

            # get results (waits for them to finish)
            sjt_with_seeds_evaluation_response = future1.result()
            sjt_without_seeds_evaluation_response = future2.result()

        sjt_evaluation_dict = {
            "sjt_rubric_1_evaluation": sjt_with_seeds_evaluation_response,
            "sjt_rubric_2_evaluation": sjt_without_seeds_evaluation_response
        }
        sjt_evaluation_result[sjt_dict['hash_id']] = sjt_evaluation_dict
        
    return sjt_evaluation_result


if __name__ == "__main__":
    args = parse_args()
    
    if args.hf_token:
        login(args.hf_token)

    synthetic_sjt_dataset = load_sjts(args)

    start_index = 0

    for batch_idx, sjt_batch in tqdm(batch_list(synthetic_sjt_dataset, args.batch_size, start_index=start_index),
                                        desc="Total SJTs",
                                        position=0,
                                        initial=start_index,
                                        total=(len(synthetic_sjt_dataset) + args.batch_size - 1) // args.batch_size):
        
        print(f"Starting SJT evaluation for batch: {batch_idx}")

        sjt_evaluation_result = sjt_llm_judge_evaluation(sjt_batch)

        out_dir = "../data/sjt_llm_judge_evaluation"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"sjt_llmjudge_evaluation_result_v{batch_idx}_for_annotations.json")
        write_to_json(sjt_evaluation_result, out_file)
        print(f"Results saved to {out_file}")
        
