# LLM Psychometrics

A framework for running standardized psychometric tests against large language models to evaluate personality consistency, validity, and reliability. Models are tested both as a base (no persona) and while roleplaying synthetic personas drawn from a curated dataset.

## What it does

Two primary psychometric instruments are used:

- **HEXACO** — a 100-item personality questionnaire measuring six traits: Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, and Openness to Experience.
- **SJT (Situational Judgment Tests)** — scenario-based questions where a model chooses a response option, each option corresponding to a different personality trait.

Experiments test whether models respond consistently across repeated runs (reliability), whether persona-conditioned responses shift trait scores in the expected direction (validity), and whether different models exhibit systematically different personality profiles.

---

## Project Structure

```
llm_psychometrics/
├── src/                        # All production source code
├── notebooks/                  # Exploratory notebooks and converted scripts
├── configs/                    # YAML configuration files
├── psychometric_tests/         # Test question banks and scoring rubrics
├── data/                       # Processed data, annotations, and generated datasets
├── experiment_results/         # Raw and aggregated experiment outputs
└── modal_scripts/              # Cloud execution wrappers (Modal.com)
```

---

## `src/` — Source Code

### `src/analysis/`

Scripts that load experiment results and produce statistical analyses and visualizations. Each script corresponds to a specific analysis pipeline:

- `advbench_regression_analysis.py` — OLS and logistic regression on refusal rates across HEXACO traits and archetypes using AdvBench benchmark data.
- `benchmark_behaviour_analysis.py` — ICC (intraclass correlation) and JS divergence analysis across EmoBench and TruthfulQA benchmarks, broken down by persona and model.
- `sjt_correlation_analysis_v1.py` — Full SJT pipeline: JS stability, ICC per trait, train/test split, identity retrieval via JS similarity, and SJT–HEXACO trait correlation heatmap.
- `sjt_histograms.py` — Histogram visualizations of SJT trait score distributions.
- `zero_compute_analysis_hexaco.py` — Eigenvalue histograms, UMAP clustering, and trait correlation heatmaps over raw HEXACO answers, grouped by archetype, ethnicity, and age.
- `zero_compute_analysis_sjt.py` — Same clustering and correlation pipeline applied to SJT one-hot encoded answers.

### `src/evals/`

Evaluation utilities used to score and judge model outputs:

- `diversity_metrics.py` — Computes diversity and distributional spread metrics across persona responses.
- `sjt_evaluation_llm_judge_v0.py` — Prompts an LLM judge to evaluate SJT responses against per-trait rubrics.

### `src/external_response_generation/`

Scripts that run psychometric tests against external model APIs or local vLLM servers and save results:

- `hexaco_response_generator.py` — Class-based, vLLM-backed HEXACO runner. Supports configurable Likert shuffling, paraphrase mode, n-repetitions, per-answer audit logging (raw prompts, guided choices, Likert orderings), and push-to-HuggingFace-Hub.
- `sjt_response_generator_merged.py` — Unified SJT runner with the same vLLM backend, supporting persona and base-model modes.
- `adv_bench_generator.py` — Generates model responses to AdvBench adversarial prompts for refusal-rate measurement.
- `prompt_set_generator_base.py` — Abstract base class for prompt set generators, providing hashing and metadata tracking.
- `test_prompt_generation.py` — Validates that prompt templates render correctly before a run.

### `src/persona_generation/`

Scripts that synthetically generate persona datasets using LLMs:

- `pydantic_persona_generation.py` — Generic Pydantic-validated persona generation pipeline (traits, demographics, psychological profile).
- `pydantic_parliamentarian_generation.py` — Persona generation seeded from UK Parliament member profiles.
- `british_persona_seed_generator.py` — Generates HEXACO-anchored persona seeds for British parliamentary archetypes.
- `singapore_patients_seed_generator.py` — Generates patient personas seeded from Singapore demographic data.
- `instances/handmade/` — A small set of manually authored reference personas (Frank Ladd, John Singleton, Norman Spencer, Peter Moskos).

### `src/prompt_templates/`

Jinja2 prompt template definitions used by the response generators:

- `hexaco_base_prompt_templates.py` / `hexaco_persona_prompt_templates.py` — System and user message templates for base-model and persona-conditioned HEXACO runs.
- `sjt_base_prompt_templates.py` / `sjt_persona_prompt_templates.py` — Equivalent templates for SJT runs.
- `sjt_llm_judge_templates.py` — Detailed rubric-grounded templates for LLM-based SJT evaluation.

### `src/synthetic_data_generation/`

Scripts for generating synthetic SJT questions:

- `synthetic_sjt_creation_v0.py` — Generates HEXACO-aligned SJT scenarios from seed attributes (urgency, threat level, ambiguity, demographics, etc.) using GPT-4, with a two-pass trait-bleed evaluation and correction step.
- `persona_llm_paper_persona_creation.py` — Generates personas matching the PersonaLLM paper configuration.

### `src/utils/`

Shared infrastructure and I/O utilities:

- `vllm_utils.py` — `VLLMServerManager`: starts, monitors, and connects to a vLLM OpenAI-compatible server.
- `hf_utils.py` — `HFStreamingAppender`: incrementally appends rows to a HuggingFace dataset without loading the full dataset into memory.
- `openai_utils.py` — Wrappers around the OpenAI API for structured-output generation.
- `file_utils.py` — JSON read/write helpers with directory creation.
- `census_utils.py` — Utilities for loading and sampling from census demographic data.
- `check_vllm.py` / `start_vllm.py` — Server health-check and startup helpers.
- `data/` — Data processing utilities:
  - `sjt_responses_processing.py` — Cleans and reshapes raw SJT answer files.
  - `demographics_data_processing.py` — Processes demographic fields for persona datasets.
  - `adv_bench_processing.py` — Parses AdvBench output files.

### Root-level `src/` files

- `experiment.py` — Core experiment orchestration: loops over personas, runs the psychometric test, collects and saves results.
- `utils_v0.py` — Legacy shared utilities (Likert inversion, list formatting, OpenAI API call wrapper).
- `add_persona_str.py` — Converts structured persona objects into a single formatted string field for HuggingFace dataset embedding.
- `modal_trigger.py` — Modal.com entry point for running experiments on cloud GPU instances.
- `push_to_hub.py` — Uploads local results to a HuggingFace Hub dataset repo.
- `run.py` — Example script showing how to invoke an experiment locally.
- `run_blind_hexaco_eval.py` — Blind HEXACO trait detection evaluation: presents SJT response options to an LLM judge *without* trait labels and measures whether the judge can correctly identify the intended HEXACO trait for each option (top-predicted trait vs. ground truth). Runs in parallel across both GPT-4o-mini and Claude-3-5-Sonnet on 500 SJTs from `thoughtworks/psychometric_SJTs`. Used to validate that trait-option mappings are genuinely distinguishable without prior knowledge of the mapping.

---

## `notebooks/`

Jupyter notebooks organized by purpose. Most have a corresponding converted `.py` script:

- `analysis/` — Exploratory analysis notebooks (HEXACO profiles, SJT correlations, factor analysis, benchmark behavior, adversarial regression). Converted versions live in `src/analysis/`.
- `reliability/` — Test-retest and inter-rater reliability experiments for both HEXACO and SJT.
- `post_processing/` — Score computation and answer cleaning notebooks run after generation.
- `evaluations/` — LLM judge runs (Anthropic, OpenAI) for persona and SJT quality review.
- `generation_personas/` — Persona generation workflow notebooks.
- `response_generation/` — Earlier notebook-style response generation scripts (pre-`src/external_response_generation/`).
- `synthetic_data_generation/` — SJT question synthesis notebooks.
- `archive/` — Older experiment notebooks, kept for reference.

---

## `configs/`

YAML files controlling experiment behavior:

- `generation_config.yaml` — Likert scale definition, temperature, batch size, max tokens.
- `personas_v2.yaml` — Handcrafted local persona definitions.
- `synthetic_sjt_seeds.yaml` / `sjt_seeds.yaml` — Attribute seeds (urgency, threat level, demographics, etc.) used for SJT generation.
- `police_seeds.yaml` / `parliament_seeds_enriched.yaml` / `skeleton_singapore_patients.yaml` — Domain-specific persona seed files.

## `psychometric_tests/`

- `hexaco_100_questions.yaml` — The 100 HEXACO questionnaire items.
- `paraphrased_hexaco_100_questions.yaml` — Paraphrased variants for robustness testing.
- `hexaco_100_eval.yaml` — Scoring rubric: maps question indices to traits and marks which items require reverse-scoring.

---

## Setup

Dependencies are managed per-module. The vLLM-based generators require a running vLLM server; see `src/utils/vllm_utils.py` for the `VLLMServerManager` interface.

HuggingFace Hub access (for persona datasets and result uploads) requires either `huggingface-cli login` or an `HF_TOKEN` environment variable.

## Citation

```
@misc{yost2025measurematterspsychometricevaluation,
      title={Measure what Matters: Psychometric Evaluation of AI with Situational Judgment Tests},
      author={Alexandra Yost and Shreyans Jain and Shivam Raval and Grant Corser and Allen Roush and Nina Xu and Jacqueline Hammack and Ravid Shwartz-Ziv and Amirali Abdullah},
      year={2025},
      eprint={2510.22170},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.22170},
}
```

## Contact

For questions or collaboration contact: amir.abdullah@thoughtworks.com, jshrey8@gmail.com
