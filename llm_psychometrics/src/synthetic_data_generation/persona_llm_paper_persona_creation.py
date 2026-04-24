import yaml
import json
import os
import itertools
import uuid


def write_to_json(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(file, f, indent=2)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Load Persona seed values
with open('../configs/persona_llm_paper_seeds.yaml', 'r') as file:
    persona_llm_seeds = yaml.safe_load(file)


combinations = list(itertools.product(*persona_llm_seeds.values()))

persona_llm_paper_seed_combinations = [{"uuid": str(uuid.uuid4()), "persona_string":", ".join(attributes)} for attributes in combinations]

write_to_json(persona_llm_paper_seed_combinations,"../data/persona_llm_paper_seed_combinations.json")