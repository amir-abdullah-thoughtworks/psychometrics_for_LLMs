from jinja2 import Template
import pandas as pd
import yaml
import sys
import json
import random
import hashlib
from itertools import product
from pydantic import BaseModel
from tqdm import tqdm
sys.path.append("../")
from src.utils import list_to_str, openai_api_call
device = "cpu"


def write_to_json(file, file_path):
    with open(file_path, 'w') as f:
        json.dump(file, f)


def read_json(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file


def expand_dict_combinations(data):
    keys = list(data.keys())
    values = [data[k] for k in keys]

    combos = []
    for prod in product(*values):
        combos.append(dict(zip(keys, prod)))
    return combos


def generate_hash(prompt):
    hash_object = hashlib.sha256()
    hash_object.update(prompt.encode("utf-8"))
    return hash_object.hexdigest()


class SyntheticSJT(BaseModel):
    question: str
    honesty_humility_option: str
    emotionality_option: str
    extraversion_option: str
    agreeableness_option: str
    conscientiousness_option: str
    openness_option: str
    
    
with open('../configs/synthetic_sjt_seeds.yaml', 'r') as file:
    synthetic_sjt_seeds = yaml.safe_load(file)

handmade_sjt_template_df = pd.read_csv("sjt_data/sjt_jinja_template_v2.csv")


all_seed_combos = expand_dict_combinations(synthetic_sjt_seeds)

print(f"No of  Combos: {len(all_seed_combos)}")

random.seed(42)
n = 60
sampled_seed_combos = random.sample(all_seed_combos, n)

SJT_GENERATION_TEMPLATE_STR = """You are creating a new law enforcement Situational Judgment Test (SJT) scenario by modifying an existing scenario with new attribute values. Follow the template below to generate a realistic, professionally appropriate scenario that maintains the core decision-making structure while incorporating the specified attributes.

**Base Scenario to Modify:**
{{base_scenario}}


**Instructions**
* Keep the same personality trait dimensions being tested (Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience)
* Preserve the professional law enforcement context

**Attribute description**
* **Urgency Level:** Adjust time pressure and decision timeline accordingly
   * Low: Allow deliberation time, non-critical timing
   * Medium: Some time pressure, manageable deadlines
   * High: Immediate decisions required, critical timing
* **Threat Level:** Scale physical danger and safety concerns
   * Low: Administrative issues, minor policy matters
   * Medium: Potential for injury, safety protocols needed
   * High: Life-threatening situations, lethal force considerations
* **Ambiguity Level:** Modify clarity of protocols and guidance
   * Clear: Obvious procedures, straightforward application
   * Moderate: Some judgment required, minor gray areas
   * High: Conflicting guidance, novel situations, ethical dilemmas
* **Individuals Involved:** Adjust scenario complexity
   * Simple: Officer making individual decision
   * Moderate: 2-3 parties with different perspectives
   * Complex: Multiple stakeholders, witnesses, supervisors
* **Authority Relationships:** Frame interactions appropriately
   * Peer Level: Fellow officers, partners, colleagues
   * Subordinate: Supervisors, training officers, senior personnel
   * Authority: Suspects, witnesses, civilians, subordinates
* **Ethical Considerations:** Incorporate specified ethical tension
* **Situation Type:** Adapt setting and context to match type
* **Time of Day:** Include time context naturally in scenario
* **Demographics:** Integrate subject's race, gender, and age naturally without stereotyping if applicable.

**New Attribute Values:**
* **Urgency Level:** {{urgency_level}}
* **Threat Level:** {{threat_level}}
* **Ambiguity Level:** {{ambiguity_level}}
* **Individuals Involved:** {{individuals_involved}}
* **Authority Relationships:** {{authority_relationships}}
* **Ethical Considerations:** {{ethical_considerations}}
* **Situation Type:** {{situation_type}}
* **Time of Day:** {{time_of_day}}
* **Subject Race:** {{race}}
* **Subject Gender:** {{gender}}
* **Subject Age:** {{age}}


**Response Options Requirements**
Generate six response options (1-6) that:
* Clearly differentiate the six personality dimensions
* Remain professionally appropriate and realistic
* Reflect how each personality type would approach the modified scenario
* Maintain consistent quality and plausibility across all options
* The responses are qualitative descriptions of actions and do not contain specific speech suggestions.
* Integrate all specified attributes naturally into the scenario without explicitly naming them
* Do not use direct labels like "high-priority," "mental health crisis," "high threat," etc. in the responses.
* Make attribute levels apparent through context, actions, and circumstances rather than descriptive terms
* Let the urgency, threat level, and situation type emerge from the scenario details rather than being stated outright
* Ensure scenario is realistic and could occur in actual law enforcement
* The scenario should be 2-4 sentences
* Include specific, actionable decisions rather than vague choices
* Verify that all six personality responses are distinct and characteristic

**Output Format**
Provide the complete SJT scenario followed by the six response options as a JSON object with the following schema:

{
  'question': '<string>',
  'honesty_humility_option': '<string>',
  'emotionality_option': '<string>',
  'extraversion_option': '<string>',
  'agreeableness_option': '<string>',
  'conscientiousness_option': '<string>',
  'openness_option': '<string>'
}

Do not include any extra text, explanation, or formatting outside of the JSON object.
"""

SJT_TRAIT_BLEED_EVALUATION_TEMPLATE_STR = """ You are an expert SJT evaluator and corrector specializing in HEXACO-aligned scenarios.  You are given a situational judgment test (SJT) scenario with six answer options, each intended  to correspond to one HEXACO trait: Honesty-Humility, Emotionality, Extraversion, Agreeableness,  Conscientiousness, and Openness.

Your tasks:
1. **Trait Fit Evaluation**  
   - For each option, evaluate how strongly it aligns with its intended trait definition.  
   - Use a 1–5 scale:  
     5 = Very strong, clean representation, no leakage  
     4 = Strong but with minor overlap  
     3 = Moderate, noticeable blending  
     2 = Weak, trait unclear or diluted  
     1 = Poor, option does not represent the trait well  

2. **Separation Analysis**  
   - Highlight where options overlap or bleed into each other (e.g., Extraversion vs. Agreeableness).  
   - Explain why the overlap occurs.  

3. **Correction Suggestions**  
   - For any option rated below 5, propose a corrected rewrite that emphasizes the target trait more cleanly.  
   - Ensure each rewrite minimizes overlap with other traits.  

4. **Final Corrected SJT Object**  
   - Output an object with the exact same structure as the input SJT dictionary.  
   - Each option should contain the corrected version if a rewrite was needed, or the unchanged original if not.  

5. **Output Format**  
   Return results in structured JSON with this format:

{
  "scenario_summary": "<1-2 sentence summary of scenario>",
  "trait_evaluations": {
    "honesty_humility": {
      "score": <1-5>,
      "analysis": "<why it fits/doesn't fit>",
      "suggested_correction": "<if needed, otherwise null>"
    },
    "emotionality": { ... },
    "extraversion": { ... },
    "agreeableness": { ... },
    "conscientiousness": { ... },
    "openness": { ... }
  },
  "corrected_sjt": {
    "question": "<original or unchanged question>",
    "honesty_humility_option": "<corrected or unchanged option>",
    "emotionality_option": "<corrected or unchanged option>",
    "extraversion_option": "<corrected or unchanged option>",
    "agreeableness_option": "<corrected or unchanged option>",
    "conscientiousness_option": "<corrected or unchanged option>",
    "openness_option": "<corrected or unchanged option>"
  },
  "overall_notes": "<high-level summary of trait separation quality>"
}

---

### SJT Input
Here is the SJT you must evaluate:

{
  "question": {{ question }},
  "honesty_humility_option": {{ honesty_humility_option }},
  "emotionality_option": {{ emotionality_option }},
  "extraversion_option": {{ extraversion_option }},
  "agreeableness_option": {{ agreeableness_option }},
  "conscientiousness_option": {{ conscientiousness_option }},
  "openness_option": {{ openness_option }}
}

"""

sjt_example_template_str = """

Question: {{ question }}
Options:

{{ answer_options}}

"""

sjt_example_template = Template(sjt_example_template_str)
SJT_GENERATION_TEMPLATE = Template(SJT_GENERATION_TEMPLATE_STR)

option_cols = ['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5','Option 6']

synthetic_generated_sjt_list = []
handmade_sjt_sample = handmade_sjt_template_df.sample(7)
for index, row in tqdm(handmade_sjt_sample.iterrows(), desc="base_scenario",
                       position=0):
    question = row['Question']
    answer_options = list_to_str(row[option_cols])

    base_scenario = sjt_example_template.render(question=question,
                                                answer_options=answer_options)

    sampled_seeds = random.sample(sampled_seed_combos, 8)

    for seed_dict in tqdm(sampled_seeds, desc="seeds",
                          position=1):
        generated_sjt_dict = {}
        # generated_sjt_dict['base_scenario'] = base_scenario

        seed_copy = seed_dict.copy()
        seed_copy['base_scenario'] = base_scenario
        sjt_generation_prompt = SJT_GENERATION_TEMPLATE.render(seed_copy)
        generated_sjt_dict['config'] = seed_copy
        openai_sjt_response = openai_api_call(prompt=sjt_generation_prompt,
                                              response_format=SyntheticSJT,
                                              model="gpt-4.1", temperature=1.5,
                                              top_p=0.95)
        openai_response_dict = openai_sjt_response.model_dump()
        generated_sjt_dict['hash_id'] = generate_hash(json.dumps(openai_response_dict))
        generated_sjt_dict['sjt_generation_prompt'] = sjt_generation_prompt
        generated_sjt_dict.update(openai_response_dict)
        synthetic_generated_sjt_list.append(generated_sjt_dict)

write_to_json(synthetic_generated_sjt_list,
              "sjt_data/synthetic_generated_sjt_list_v8_temp_1point5.json")