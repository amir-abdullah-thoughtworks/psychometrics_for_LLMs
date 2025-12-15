import json
import numpy as np
import os
import pandas as pd
import yaml

from collections import Counter
from pathlib import Path

trait_map = {
    "1": "Honesty-Humility",
    "2": "Emotionality",
    "3": "Extraversion",
    "4": "Agreeableness",
    "5": "Conscientiousness",
    "6": "Openness"
}

with open('../configs/generation_config.yaml', 'r') as file:
    generation_config = yaml.safe_load(file)


trait_list = ['honest-humility','emotionality','extraversion','agreeableness','conscientiousness','openness to experience','altruism']

# Anchor everything to *this file*, not the working directory
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]   # 3 levels up

# Directories / files
FACTOR_ANALYSIS_DATA_DIR = (
    PROJECT_ROOT / "experiment_results" / "factor_analysis_data"
)

GENERATION_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "generation_config.yaml"
)

HEXACO_EVAL_PATH = (
    PROJECT_ROOT / "psychometric_tests" / "hexaco_100_eval.yaml"
)

# Load YAML files
with open(GENERATION_CONFIG_PATH, "r") as f:
    generation_config = yaml.safe_load(f)

with open(HEXACO_EVAL_PATH, "r") as f:
    hexaco_eval = yaml.safe_load(f)

def write_to_json(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(file, f, indent=2)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def load_sjt(sjt_dir):
    sjt_list = []
    for file in os.listdir(sjt_dir):
        synthetic_sjt = read_json(os.path.join(sjt_dir, file))
        sjt_list += synthetic_sjt

    print(f"{len(sjt_list)} SJTs Loaded")
    return sjt_list

def get_model_name(filename, file_str):
    model_name = filename.replace("_sjt_answers_","").replace(".json","").replace(file_str,"")
    return model_name

def get_true_indices_v1(data):
    resolved_traits = []
    true_answer_indices = []
    for answer, shuffled_indices in zip(data["answers"][0], data['config']['answer_index'][0]):
        true_index = str(shuffled_indices[int(answer)-1] + 1)  # get original index meaning
        if true_index in trait_map:
            resolved_traits.append(trait_map[true_index])
            true_answer_indices.append(true_index)
    
    return true_answer_indices, resolved_traits


def get_trait_distribution_v1(data):
    true_answer_indices, resolved_traits = get_true_indices_v1(data)
    
    trait_counts = Counter(resolved_traits)
    total = sum(trait_counts.values())
    
    trait_summary = {
        trait: {
            "count": trait_counts[trait],
            "proportion": round(trait_counts[trait] / total, 3) if total > 0 else 0
        }
        for index, trait in trait_map.items()
    }
    return [trait_summary[key]['proportion'] for key in trait_summary.keys()]


def calculate_hexaco_score(trait, subtrait, answers, likert_scale = generation_config['likert_scale']):
    answers = [answer if answer in likert_scale else "Do not wish to answer" for answer in answers]
    answers = pd.Series(answers)
    subtrait_dict = hexaco_eval[trait][subtrait]
    indices = [idx - 1 for idx in subtrait_dict['indices']]
    trait_answers = answers[indices]
    refused_answers = trait_answers[trait_answers.isin(['Do not wish to answer','Do not wish to answer.'])]
    non_refused_answers = trait_answers[~trait_answers.isin(['Do not wish to answer','Do not wish to answer.'])]
    answer_indices = [likert_scale.index(answer) + 1 for answer in non_refused_answers]
    true_answer_indices = [6-idx if reverse else idx for idx,reverse in zip(answer_indices, subtrait_dict['reverse'])]
    
    return true_answer_indices


def get_trait_scores(answer,likert_scale = generation_config['likert_scale']):
    trait_score = []
    for trait in hexaco_eval.keys():
        for subtrait in hexaco_eval[trait].keys():
            trait_true_answer_indices = []
            trait_true_answer_indices.extend(calculate_hexaco_score(trait, subtrait, answer, likert_scale))
        
        trait_score.append(np.round(np.mean(trait_true_answer_indices).item(),3).item())
    return trait_score


def load_personas_to_sjt_scores(model_name='Qwen2_5-7B-Instruct'):
    factor_analysis_data_dir = "../experiment_results/factor_analysis_data/"
    sjt_data = read_json(os.path.join(factor_analysis_data_dir, f"huggingface_sjt_answers_{model_name}.json"))
    sjt_answers_df = pd.DataFrame([{"persona_id": key, "answers": get_true_indices_v1(sjt_data[key])[1]} for key in sjt_data.keys()])
    sjt_indexed_df = pd.DataFrame([{"persona_id": key, "answers": get_trait_distribution_v1(sjt_data[key])} for key in sjt_data.keys()])
    sjt_trait_df = pd.DataFrame(sjt_indexed_df['answers'].to_list(), columns = trait_map.values(), index = sjt_indexed_df['persona_id'])
    return {
        "persona_summary":sjt_trait_df,
        "indexed_df":  sjt_indexed_df,
        "answers_df": sjt_answers_df
    }
        