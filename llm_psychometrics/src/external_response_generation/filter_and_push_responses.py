"""
Filters a response dataset to only keep rows whose persona_uuid and question_hash
appear in designated persona and SJT HuggingFace datasets, then pushes the result
to a target HuggingFace repo under a specified config name.
"""

import os

from datasets import Dataset, load_dataset
from huggingface_hub import login

# ── Configuration ──────────────────────────────────────────────────────────────

# Source response dataset to filter
SOURCE_RESPONSE_HF_PATH = "thoughtworks/psychometric_personas_responses"
SOURCE_RESPONSE_HF_CONFIG = "police_sjt"
SOURCE_RESPONSE_HF_SPLIT = "train"

# Persona dataset — provides the set of persona UUIDs to keep
FILTER_PERSONA_HF_PATH = "thoughtworks/psychometric_personas"
FILTER_PERSONA_HF_CONFIG = "comparison_openai"
FILTER_PERSONA_HF_SPLIT = "train"
FILTER_PERSONA_ID_COL = "uuid"  # column in the persona dataset holding the IDs

# SJT dataset — provides the set of question hashes to keep
FILTER_SJT_HF_PATH = "thoughtworks/psychometric_sjts_analysis"
FILTER_SJT_HF_CONFIG = "comparison_openai"
FILTER_SJT_HF_SPLIT = "train"
FILTER_SJT_HASH_COL = "hash_id"  # column in the SJT dataset holding the hashes

# Target dataset to push the filtered responses into
TARGET_HF_PATH = "thoughtworks/psychometric_personas_responses"
TARGET_HF_CONFIG = (
    "cmp_openai_personas_cmp_openai_sjts"  # becomes the dataset config / subset name
)


# ── Main ───────────────────────────────────────────────────────────────────────
login(token=os.environ.get("HF_TOKEN", ""))


def load_hf(path, config, split):
    return load_dataset(path, config)[split].to_pandas()


print("Loading response dataset...")
responses = load_hf(
    SOURCE_RESPONSE_HF_PATH, SOURCE_RESPONSE_HF_CONFIG, SOURCE_RESPONSE_HF_SPLIT
)
print(f"  Loaded {len(responses):,} rows")

print("Loading persona filter dataset...")
personas = load_hf(
    FILTER_PERSONA_HF_PATH, FILTER_PERSONA_HF_CONFIG, FILTER_PERSONA_HF_SPLIT
)
persona_ids = set(personas[FILTER_PERSONA_ID_COL].dropna().unique())
print(f"  {len(persona_ids):,} unique persona IDs to keep")

print("Loading SJT filter dataset...")
sjts = load_hf(FILTER_SJT_HF_PATH, FILTER_SJT_HF_CONFIG, FILTER_SJT_HF_SPLIT)
question_hashes = set(sjts[FILTER_SJT_HASH_COL].dropna().unique())
print(f"  {len(question_hashes):,} unique question hashes to keep")

print("Filtering...")
mask = responses["persona_uuid"].isin(persona_ids) & responses["question_hash"].isin(
    question_hashes
)
filtered = responses[mask].reset_index(drop=True)
print(f"  {len(filtered):,} rows retained ({len(responses) - len(filtered):,} dropped)")

EXPECTED_ROWS = 50_000
print(f"\nFiltered dataset size: {len(filtered):,} rows")
if len(filtered) != EXPECTED_ROWS:
    print(
        f"  [ABORT] Expected {EXPECTED_ROWS:,} rows but got {len(filtered):,}. Not pushing."
    )
else:
    print(f"Pushing to {TARGET_HF_PATH} / config={TARGET_HF_CONFIG} ...")
    Dataset.from_pandas(filtered).push_to_hub(
        TARGET_HF_PATH, config_name=TARGET_HF_CONFIG
    )
    print("Done.")
