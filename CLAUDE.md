# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Dataset reference**: See [`DATASETS.md`](DATASETS.md) for the full inventory of HuggingFace datasets, their configs, subset relationships, and how they feed into response runs.

## Project Purpose

Research framework for running standardized psychometric tests (HEXACO personality inventory, Situational Judgment Tests) against LLMs. The core idea: generate synthetic personas with specific demographics and psychological profiles, inject those personas into prompts, and assess whether models respond consistently and validly across conditions. Used to measure validity, reliability, and diversity of LLM responses.

## Setup & Key Commands

**Python version**: 3.11.9 (see `.python-version`)

```bash
# Install dependencies
poetry install
# or
pip install -r requirements.txt

# Run local vLLM inference server (serves on port 9000)
python start_vllm.py --model-name <hf_model_id> --hf-token <token>

# Run distributed experiments on Modal.com (requires modal auth)
modal run llm_psychometrics/src/modal_trigger.py
```

There is no formal test suite or CI. Experimental validation happens through Jupyter notebooks in `llm_psychometrics/notebooks/`.

## Architecture

The pipeline flows through three stages:

### 1. Persona Generation (`src/persona_generation/`)

`pydantic_persona_generation.py` is the main entry point. It uses the OpenAI API (structured outputs via Pydantic `create_model`) to generate synthetic personas from seed YAML files in `configs/`. Each persona has demographics, psychological traits, memoir-style background, and a presenting problem.

`ABLATION_CONFIGS` controls five experiment modes: `full`, `no_attribute_injections`, `no_memoir_grounding`, `no_demographic_grounding`, `no_archetype_grounding`. These remove specific persona components to isolate their effect on model responses.

Personas are written to `.jsonl` files keyed by a stable SHA-256 hash to prevent duplicate generation across runs.

### 2. Prompt Construction (`src/prompt_templates/`)

Separate template modules for each test type:
- `hexaco_base_prompt_templates.py` / `hexaco_persona_prompt_templates.py` — HEXACO-100 Likert-scale questions
- `sjt_base_prompt_templates.py` / `sjt_persona_prompt_templates.py` — Situational Judgment Tests
- `sjt_llm_judge_templates.py` — LLM-as-judge prompts for scoring SJT free-text responses

Templates are model-aware and adapted for GPT (JSON format), Llama (chat tokens), and Qwen chat formats.

### 3. Inference & Evaluation

**Local inference**: `start_vllm.py` starts a vLLM OpenAI-compatible server; `src/utils/vllm_utils.py` provides `VLLMServerManager` for programmatic control.

**Cloud inference**: `src/modal_trigger.py` + `src/modal_scripts/` deploy jobs to Modal.com with A100 GPUs and a persistent output volume (`my-outputs`).

**OpenAI API calls**: `src/utils/openai_utils.py` wraps all API calls with retry logic and `diskcache` disk caching. Always use this wrapper—`CACHE_VERSION` in that file controls cache invalidation.

**Diversity metrics**: `src/evals/diversity_metrics.py` computes text diversity using Sentence-Transformers embeddings, spaCy lemmatization, silhouette scores, TTR, MTLD, vendi-score, and compression ratio.

### Supporting Structure

- `configs/` — YAML seeds for persona generation and generation hyperparameters (`generation_config.yaml`)
- `data/` — Census data, SJT response data, inter-rater agreement files, annotations
- `psychometric_tests/` — HEXACO question banks and evaluation YAML (`hexaco_100_eval.yaml`, `hexaco_100_questions.yaml`)
- `llm_psychometrics/notebooks/` — Experimental notebooks organized by stage: `response_generation/`, `evaluations/`, `post_processing/`, `analysis/`, `synthetic_data_generation/`
- `experiments/` — Legacy experiment scripts
- `eval_framework/` — Standalone evaluation framework

## Key Patterns

**Adding new psychometric tests**: Add question YAML under `psychometric_tests/`, create corresponding base and persona prompt templates in `src/prompt_templates/`, then wire up in `start_vllm.py` or a modal script.

**Caching API calls**: All OpenAI calls must go through `src/utils/openai_utils.py`. Bump `CACHE_VERSION` there when prompt logic changes to invalidate cached responses.

**Persona deduplication**: The generation script uses `stable_hash()` (SHA-256 of normalized seed text) and writes to `./{version}.jsonl`. Always pass a consistent `version` string per ablation run.

**Structured outputs**: Use `pydantic.create_model` to build dynamic models matching the response schema, then pass to `client.beta.parseds.parse()`. See `pydantic_persona_generation.py` for the established pattern.
