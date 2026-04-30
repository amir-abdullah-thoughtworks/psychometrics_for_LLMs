"""
python src/synthetic_data_generation/synthetic_sjt_creation_v0.py \
  --source hf \
  --provider anthropic \
  --n-sample 100 \
  --push-to-hub
"""

import argparse
import functools
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
from jinja2 import Template
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils_v0 import anthropic_api_call, list_to_str, openai_api_call

device = "cpu"


# DEFAULT_SJT_GENERATION_MODEL = "gpt-4.1"
# DEFAULT_SJT_TRAIT_BLEED_EVALUATION_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 1
DEFAULT_TOP_P = 0.95


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["openai", "anthropic"], default="anthropic"
    )
    parser.add_argument(
        "--generation-model",
        default=None,
        help="Defaults: gpt-4.1 (openai) | claude-sonnet-4-6 (anthropic)",
    )
    parser.add_argument(
        "--evaluation-model",
        default=None,
        help="Defaults: gpt-4.1 (openai) | claude-sonnet-4-6 (anthropic)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Seeds per base template. Total SJTs = n_templates × n_seeds.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Row index in sjt_jinja_template_v2.csv to start from.",
    )
    parser.add_argument(
        "--output-path", default="data/sjt_data/synthetic_sjts_anthropic100.parquet"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the final dataset to HuggingFace Hub. Requires HF_TOKEN env var.",
    )
    parser.add_argument(
        "--hub-dataset-id",
        default="thoughtworks/psychometric_sjts_analysis",
        help="HuggingFace dataset repo id, e.g. 'thoughtworks/psychometric_SJTs'.",
    )
    parser.add_argument(
        "--hub-subset",
        default="comparison_anthropic",
        help="Dataset config/subset name (the 'name' param in push_to_hub).",
    )
    parser.add_argument(
        "--hub-split", default="train", help="Dataset split name, e.g. 'train', 'test'."
    )
    # HF-source mode
    parser.add_argument(
        "--source",
        choices=["local", "hf"],
        default="local",
        help="'local' = existing CSV/YAML behavior; 'hf' = sample from HF dataset.",
    )
    parser.add_argument(
        "--hf-source-path", default="thoughtworks/psychometric_sjts_analysis"
    )
    parser.add_argument("--hf-source-subset", default="analysis")
    parser.add_argument("--hf-source-split", default="train")
    parser.add_argument(
        "--n-sample",
        type=int,
        default=100,
        help="Number of SJTs to sample from HF source (hf mode only).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for reproducible HF sampling.",
    )
    return parser.parse_args()


def write_to_json(file, file_path):
    with open(file_path, "w") as f:
        json.dump(file, f)


def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


class TraitBleedEval(BaseModel):
    score: int
    analysis: str
    suggested_correction: Union[str, None]


class HexacoTrait(BaseModel):
    honesty_humility: TraitBleedEval
    emotionality: TraitBleedEval
    extraversion: TraitBleedEval
    agreeableness: TraitBleedEval
    conscientiousness: TraitBleedEval
    openness: TraitBleedEval


class SjtTraitBleedEval(BaseModel):
    scenario_summary: str
    trait_evaluations: HexacoTrait
    corrected_sjt: SyntheticSJT
    overall_notes: str


def qa_to_string(entry: dict) -> str:
    parts = []
    parts.append(f"Question: {entry.get('question', '')}")
    for key, value in entry.items():
        if key != "question":
            parts.append(f"{key.replace('_option', '').capitalize()}: {value}")
    return "\n".join(parts)


def embed_sjt(sjt, embed_model):
    corrected_sjt = sjt["corrected_sjt"]
    sjt_string = qa_to_string(corrected_sjt)
    return embed_model.encode([sjt_string], convert_to_numpy=True)[0].tolist()


def get_api_fn(provider, model, temperature, top_p):
    defaults = {"openai": "gpt-4.1", "anthropic": "claude-sonnet-4-6"}
    resolved_model = model or defaults[provider]
    if provider == "anthropic":
        return functools.partial(
            anthropic_api_call, model=resolved_model, temperature=temperature
        )
    return functools.partial(
        openai_api_call, model=resolved_model, temperature=temperature, top_p=top_p
    )


def sjt_generation_with_validation(seed_dict, generation_fn, evaluation_fn):
    generated_sjt_dict = {}
    sjt_generation_prompt = SJT_GENERATION_TEMPLATE.render(seed_dict)
    generated_sjt_dict["config"] = seed_dict

    original_sjt = generation_fn(
        prompt=sjt_generation_prompt,
        response_format=SyntheticSJT,
    )
    original_sjt_response_dict = original_sjt.model_dump()

    sjt_trait_bleed_evaluation_prompt = SJT_TRAIT_BLEED_EVALUATION_TEMPLATE.render(
        original_sjt_response_dict
    )
    evaluated_sjt = evaluation_fn(
        prompt=sjt_trait_bleed_evaluation_prompt,
        response_format=SjtTraitBleedEval,
    )
    corrected_sjt_response_dict = evaluated_sjt.model_dump()

    generated_sjt_dict["hash_id"] = generate_hash(
        json.dumps(corrected_sjt_response_dict["corrected_sjt"])
    )
    generated_sjt_dict["original_sjt"] = original_sjt_response_dict
    generated_sjt_dict["trait_bleed_evaluation"] = corrected_sjt_response_dict
    generated_sjt_dict["corrected_sjt"] = corrected_sjt_response_dict["corrected_sjt"]

    return generated_sjt_dict


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
   - Ensure each rewrite include specific, actionable decisions rather than vague choices.

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
    "emotionality": {
      "score": <1-5>,
      "analysis": "<why it fits/doesn't fit>",
      "suggested_correction": "<if needed, otherwise null>"
    },
    "extraversion": {
      "score": <1-5>,
      "analysis": "<why it fits/doesn't fit>",
      "suggested_correction": "<if needed, otherwise null>"
    },
    "agreeableness": {
      "score": <1-5>,
      "analysis": "<why it fits/doesn't fit>",
      "suggested_correction": "<if needed, otherwise null>"
    },
    "conscientiousness": {
      "score": <1-5>,
      "analysis": "<why it fits/doesn't fit>",
      "suggested_correction": "<if needed, otherwise null>"
    },
    "openness": {
      "score": <1-5>,
      "analysis": "<why it fits/doesn't fit>",
      "suggested_correction": "<if needed, otherwise null>"
    }
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
SJT_TRAIT_BLEED_EVALUATION_TEMPLATE = Template(SJT_TRAIT_BLEED_EVALUATION_TEMPLATE_STR)

option_cols = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5", "Option 6"]

SEED_ATTR_COLS = [
    "urgency_level",
    "threat_level",
    "ambiguity_level",
    "individuals_involved",
    "authority_relationships",
    "ethical_considerations",
    "situation_type",
    "time_of_day",
    "race",
    "gender",
    "age",
]


def load_and_sample_hf_source(hf_path, subset, split, n_sample, seed):
    from datasets import load_dataset

    ds = load_dataset(hf_path, name=subset)[split]
    n = min(n_sample, len(ds))
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    return [ds[i] for i in indices]


def build_seed_copy_from_hf_row(row):
    seed = {k: row[k] for k in SEED_ATTR_COLS if k in row}
    seed["base_scenario"] = row["base_scenario"]
    return seed


def main():
    args = parse_args()

    defaults = {"openai": "gpt-4.1", "anthropic": "claude-sonnet-4-6"}
    resolved_gen_model = args.generation_model or defaults[args.provider]
    resolved_eval_model = args.evaluation_model or defaults[args.provider]

    print(f"Source:            {args.source}")
    if args.source == "hf":
        print(
            f"  HF repo:         {args.hf_source_path} / {args.hf_source_subset} / {args.hf_source_split}"
        )
        print(f"  Sample size:     {args.n_sample} (seed={args.sample_seed})")
    print(f"Provider:          {args.provider}")
    print(f"Generation model:  {resolved_gen_model}")
    print(f"Evaluation model:  {resolved_eval_model}")
    print(f"Output path:       {args.output_path}")
    if args.push_to_hub:
        print("Push to Hub:       yes")
        print(f"  Dataset ID:      {args.hub_dataset_id}")
        print(f"  Subset:          {args.hub_subset}")
        print(f"  Split:           {args.hub_split}")
    else:
        print("Push to Hub:       no")
    print()

    generation_fn = get_api_fn(
        args.provider, args.generation_model, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
    )
    evaluation_fn = get_api_fn(
        args.provider, args.evaluation_model, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
    )
    validated_fn = functools.partial(
        sjt_generation_with_validation,
        generation_fn=generation_fn,
        evaluation_fn=evaluation_fn,
    )

    embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    all_rows = []
    sampled_rows = []

    if args.source == "hf":
        print(
            f"HF source:         {args.hf_source_path} / {args.hf_source_subset} / {args.hf_source_split}"
        )
        print(f"Sample size:       {args.n_sample} (seed={args.sample_seed})")
        print()

        sampled_rows = load_and_sample_hf_source(
            args.hf_source_path,
            args.hf_source_subset,
            args.hf_source_split,
            args.n_sample,
            args.sample_seed,
        )
        total_sjts = len(sampled_rows)
        print(f"Generating {total_sjts} SJTs (1 per sampled HF row)")

        input_batch = [build_seed_copy_from_hf_row(row) for row in sampled_rows]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                tqdm(
                    executor.map(validated_fn, input_batch),
                    total=total_sjts,
                    desc="sjts",
                )
            )

        for sjt, hf_row in zip(results, sampled_rows):
            config = sjt["config"]
            all_rows.append(
                {
                    "config": config,
                    "hash_id": sjt["hash_id"],
                    "original_sjt": sjt["original_sjt"],
                    "trait_bleed_evaluation": sjt["trait_bleed_evaluation"],
                    "corrected_sjt": sjt["corrected_sjt"],
                    "source_hash_id": hf_row.get("hash_id"),
                    "sjt_embedding": embed_sjt(sjt, embed_model),
                    **config,
                }
            )

    else:  # local — existing behavior
        with open("configs/synthetic_sjt_seeds.yaml", "r") as file:
            synthetic_sjt_seeds = yaml.safe_load(file)

        handmade_sjt_template_df = pd.read_csv(
            "data/sjt_data/sjt_jinja_template_v2.csv"
        )
        all_seed_combos = expand_dict_combinations(synthetic_sjt_seeds)

        handmade_sjt_sample = handmade_sjt_template_df.iloc[args.start_index :]
        n_templates = len(handmade_sjt_sample)
        total_sjts = n_templates * args.n_seeds
        print(
            f"Generating {total_sjts} SJTs ({n_templates} templates × {args.n_seeds} seeds each)"
        )

        for index, row in tqdm(
            handmade_sjt_sample.iterrows(),
            desc="base_scenario",
            position=0,
            total=n_templates,
        ):
            print(f"Index for base SJT dataframe: {index}")
            question = row["Question"]
            answer_options = list_to_str(row[option_cols])

            base_scenario = sjt_example_template.render(
                question=question, answer_options=answer_options
            )

            sampled_seeds = random.sample(all_seed_combos, args.n_seeds)

            for seed_batch in tqdm(
                batch_list(sampled_seeds, 5), desc="seeds", position=1
            ):
                input_seed_batch = []
                for seed_dict in seed_batch:
                    seed_copy = seed_dict.copy()
                    seed_copy["base_scenario"] = base_scenario
                    input_seed_batch.append(seed_copy)

                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(validated_fn, input_seed_batch))

                for sjt in results:
                    config = sjt["config"]
                    all_rows.append(
                        {
                            "config": config,
                            "hash_id": sjt["hash_id"],
                            "original_sjt": sjt["original_sjt"],
                            "trait_bleed_evaluation": sjt["trait_bleed_evaluation"],
                            "corrected_sjt": sjt["corrected_sjt"],
                            "template_no": index,
                            "sjt_embedding": embed_sjt(sjt, embed_model),
                            **config,
                        }
                    )

    df = pd.DataFrame(all_rows)
    try:
        df.to_parquet(args.output_path, index=False)
        print(f"Saved {len(df)} SJTs to {args.output_path}")
    except Exception as e:
        print(f"Warning: failed to save parquet to {args.output_path}: {e}")

    if args.push_to_hub:
        from datasets import Dataset
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset.push_to_hub(
            args.hub_dataset_id,
            config_name=args.hub_subset,
            split=args.hub_split,
        )
        print(f"Pushed to HuggingFace Hub: {args.hub_dataset_id}")

        if sampled_rows:
            comparison_ds = Dataset.from_list(sampled_rows)
            comparison_ds.push_to_hub(
                args.hub_dataset_id,
                config_name="comparison_openai",
                split=args.hub_split,
            )
            print(
                f"Pushed {len(comparison_ds)} source SJTs to subset 'comparison_openai'"
            )


if __name__ == "__main__":
    main()
