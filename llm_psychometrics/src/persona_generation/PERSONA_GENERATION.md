# Persona Generation

This directory contains scripts for generating structured Law Enforcement Officer (LEO) personas using LLMs with constrained/structured output.

## Files

| File | Description |
|------|-------------|
| `pydantic_persona_generation.py` | Original OpenAI version (GPT-4.x via Responses API) |
| `pydantic_persona_generation_anthropic.py` | Anthropic version (Claude Sonnet via tool_use) |
| `ablation_persona_generation.py` | Ablation variant — adds `ABLATION_CONFIGS` and `ablation_config` param; leave original untouched |
| `pydantic_parliamentarian_generation.py` | Variant for parliamentarian personas |

## How Persona Generation Works

Each persona is built from four seed inputs:
1. **Demographics row** — picked from `balanced_us_police_officers.csv` by index
2. **Archetype** — one of the archetypes in `populated_police_seeds.yaml` (cycled by index)
3. **Memoir** — one of the memoir titles/summaries in the YAML (cycled by index)
4. **Appearance/behavior categories** — sampled randomly per index

These seeds are baked into a Pydantic schema with `Literal[value]` types so the LLM cannot deviate from them. Generated prose fields (appearance, behavior, speech, cognition, etc.) are left as free-text `str`.

After generation, `build_concat_and_embedding()` concatenates all generated prose fields and embeds them with `Qwen/Qwen3-Embedding-0.6B` (1024-dim). The embedding is stored as `concat_embedding`.

## Running Locally

### OpenAI version
```bash
cd llm_psychometrics/src/persona_generation
python pydantic_persona_generation.py \
  --n 2 \
  --seeds-yaml ../../configs/populated_police_seeds.yaml \
  --officers-csv ../../data/demographics/balanced_us_police_officers.csv \
  --model gpt-4.1-mini \
  --version local_test \
  --api-key $OPENAI_API_KEY
```

### Anthropic version
```bash
python pydantic_persona_generation_anthropic.py \
  --n 2 \
  --seeds-yaml ../../configs/populated_police_seeds.yaml \
  --officers-csv ../../data/demographics/balanced_us_police_officers.csv \
  --model claude-sonnet-4-6 \
  --version local_test \
  --api-key $ANTHROPIC_API_KEY
```

### Re-generating from existing personas (`--from-personas`)

Use this flag to re-generate prose for personas whose seed fields (demographics, archetype, memoir, appearance/behavior category) are already fixed. The new run re-uses the exact seeds from the existing records — appearance and behavior prose from the original record are passed as style references, not re-sampled from the YAML.

```bash
# From a local JSONL file
python pydantic_persona_generation_anthropic.py \
  --from-personas existing_personas.jsonl \
  --api-key $ANTHROPIC_API_KEY

# From HuggingFace dataset (format: owner/repo:config:split)
python pydantic_persona_generation_anthropic.py \
  --from-personas thoughtworks/psychometric_personas:leo_v1:train \
  --api-key $ANTHROPIC_API_KEY
```

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | `claude-sonnet-4-6` (Anthropic) / `gpt-4.1-mini` (OpenAI) | |
| `temperature` | `1.0` (Anthropic, max allowed) / `2.0` (OpenAI) | Anthropic hard-caps at 1.0 |
| `top_p` | `0.98` | |
| `rng_seed` | `1337` | Controls archetype/memoir offsets and appearance/behavior sampling |
| `workers` | `10` | Thread workers for `run_batch()` / `run_batch_from_seeds()` |

## Output Schema (35 fields)

The generated record matches `thoughtworks/psychometric_personas` on HuggingFace:

**Pinned from seeds (Literals):** `uuid`, `version`, `name`, `age`, `sex`, `location`, `education_level`, `bachelors_field`, `ethnic_background`, `marital_status`, `archetype`, `memoir`, `appearance_category`, `behavior_category`

**Overwritten post-generation:** `archetype_description`, `memoir_summary` (exact values from YAML, regardless of model output)

**Generated prose:** `memoir_narrative`, `appearance`, `behavior`, `speech`, `mood_affect`, `educational_vocational_history`, `medical_developmental_history`, `family_history`, `presenting_problems` (List[str]), `thought_content`, `insight_judgment`, `cognition`, `emotional_behavioral_functioning`, `social_functioning`, `summary_of_psychological_profile`

**Computed post-generation:** `persona_string`, `persona_hash`, `concat_field`, `concat_embedding`

## Required Config Files

- `llm_psychometrics/configs/populated_police_seeds.yaml` — archetypes, memoir titles/summaries, appearance/behavior category seeds
- `llm_psychometrics/data/demographics/balanced_us_police_officers.csv` — demographics rows; required columns: `sex`, `age`, `city`, `state`, `first_name`, `last_name`, `education_level`, `marital_status`, `ethnic_background`, `bachelors_field`, `uuid`
