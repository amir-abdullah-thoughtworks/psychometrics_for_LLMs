# SJT Response Generator

`sjt_response_generator_merged.py` runs Situational Judgment Test (SJT) scenarios through an LLM, with or without persona injection, and records which of 6 HEXACO trait options the model selects.

## What it does

For each persona × SJT item × iteration:
1. Optionally shuffles the 6 answer options in a deterministic order (seeded by persona hash + question hash + iteration)
2. Renders a prompt via a Jinja template (persona-aware or base)
3. Calls vLLM with constrained decoding (`guided_choices=["1","2","3","4","5","6"]`) so the model outputs exactly one digit
4. Maps the digit back through the shuffle permutation to a canonical HEXACO trait name

Results are saved as a local JSON file and optionally pushed to HuggingFace.

## Prerequisites

- A running vLLM server (`python start_vllm.py --model-name <model>`)
- `HF_TOKEN` env var set (for loading HF datasets and pushing results)

## Basic usage

```bash
# Debug run (10 personas × 10 SJTs × 5 iterations, no hub push)
python sjt_response_generator_merged.py --debug

# Full run with all personas from the analysis config
python sjt_response_generator_merged.py \
  --no-debug \
  --model google/gemma-3-4b-it \
  --hf-persona-config analysis \
  --n-times 5 \
  --out-json /outputs/sjt_results.json

# Base model (no persona injection)
python sjt_response_generator_merged.py \
  --no-debug \
  --persona-source base_model

# Use the comparison_anthropic personas
python sjt_response_generator_merged.py \
  --no-debug \
  --hf-persona-path thoughtworks/psychometric_personas \
  --hf-persona-config comparison_anthropic \
  --target-hub-config comparison_anthropic_sjt
```

## Key arguments

| Argument | Default | Notes |
|---|---|---|
| `--model` | `google/gemma-3-4b-it` | vLLM model to use |
| `--persona-source` | `hf` | `hf` or `base_model` |
| `--hf-persona-path` | `thoughtworks/psychometric_personas` | |
| `--hf-persona-config` | `analysis` | Any config in that dataset |
| `--hf-persona-split` | `train` | |
| `--hf-sjt-path` | `thoughtworks/psychometric_sjts_analysis` | |
| `--hf-sjt-config` | `analysis` | |
| `--n-times` | `5` | Iterations per persona (for reliability measurement) |
| `--answer-shuffle` / `--no-answer-shuffle` | on | Deterministic per (persona, question, iter) |
| `--use-persona-template` / `--no-use-persona-template` | on | |
| `--template-key` | `gpt` | Template variant; see `sjt_persona_prompt_templates.py` |
| `--debug` / `--no-debug` | off | Debug: 10 personas × 10 SJTs × 5 iters, skips hub push |
| `--n-personasample` | `10` | Persona limit in debug mode |
| `--n-sjtsample` | `10` | SJT item limit in debug mode |
| `--out-json` | `/outputs/police_sjt_results.json` | Local output path |
| `--push-to-hub` / `--no-push-to-hub` | on | |
| `--target-hub-repo-id` | `thoughtworks/gemma_psychometrics_personas_responses` | |
| `--target-hub-config` | `analysis_base` | Config name on HF |

## Answer shuffling

Shuffling prevents the model from developing a position bias (always picking "1"). The shuffle is fully deterministic: each (persona_hash, question_hash, iteration) triple maps to the same permutation every run, so results are reproducible.

The raw answer (`"1"`–`"6"`) is stored alongside the `answer_index` permutation and the `normalized_answer` (one of the six HEXACO trait keys), so you can always reconstruct what happened.

```
"3" → permutation[2] → canonical_idx → "conscientiousness_option"
```

## Persona truncation

Persona strings are hard-capped at **1150 tokens** (using the model's tokenizer) before injection. The run logs how many personas were truncated.

## Output schema

### Local JSON (`--out-json`)

```
{
  "<persona_uuid>": {
    "persona": "<uuid>",
    "persona_hash": "<sha256>",
    "answers":             [iter][question] -> "1".."6" | null,
    "normalized_answers":  [iter][question] -> "honesty_humility_option" | ... | null,
    "question_hashes":     [question] -> hash string,
    "answer_index":        [iter][question] -> [int × 6]  (displayed→canonical permutation),
    "raw_prompts":         [iter][question] -> full prompt string,
    "guided_choices":      [iter][question] -> ["1","2","3","4","5","6"],
    "model_name": "...",
    "sjt_answer_options": "shuffle" | "normal"
  }
}
```

### HuggingFace dataset (flattened)

One row per (persona, iteration, question):

| Field | Description |
|---|---|
| `persona_uuid` | UUID of the persona |
| `persona_hash` | SHA256 of persona_string |
| `iter` | Iteration index (0-based) |
| `question_hash` | Hash of the SJT item |
| `answer` | Raw model output ("1"–"6") |
| `normalized_answer` | HEXACO trait key |
| `answer_index` | Displayed→canonical permutation |
| `raw_prompt` | Full prompt sent to vLLM |
| `model_name` | Model used |
| `run_timestamp_utc` | ISO timestamp |
| `answer_shuffle`, `n_times`, `template_key`, ... | Run metadata |

## HEXACO trait mapping

The 6 canonical answer options (in order) are:

```
1. honesty_humility_option
2. emotionality_option
3. extraversion_option
4. agreeableness_option
5. conscientiousness_option
6. openness_option
```

After shuffling, the displayed order changes but the mapping back to these keys is always preserved via `answer_index`.

## Dependencies

- `utils.vllm_utils.VLLMServerManager` — manages the vLLM server connection
- `prompt_templates.sjt_base_prompt_templates` / `sjt_persona_prompt_templates` — Jinja templates
- `utils_v0.list_to_str` — formats the 6 options into a numbered list for the prompt
