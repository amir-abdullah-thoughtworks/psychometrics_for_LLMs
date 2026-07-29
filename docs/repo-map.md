# Repository map

File-by-file reference. For dataset structure see [`DATASETS.md`](../DATASETS.md).

```
llm_psychometrics/
├── src/                        # Production source code
├── notebooks/                  # Exploratory notebooks and converted scripts
├── configs/                    # YAML seed and generation configs
├── psychometric_tests/         # Question banks and scoring keys
├── data/                       # Census data, SJT templates, annotations
├── experiment_results/         # Raw and aggregated outputs
└── modal_scripts/              # Cloud execution wrappers (Modal.com)
```

---

## `src/analysis/`

Scripts that load experiment results and produce statistical analyses and visualizations.

| File | Purpose |
|---|---|
| `advbench_regression_analysis.py` | OLS and logistic regression on refusal rates across HEXACO traits and archetypes, using AdvBench data |
| `benchmark_behaviour_analysis.py` | ICC and JS divergence across EmoBench and TruthfulQA, broken down by persona and model |
| `sjt_correlation_analysis_v1.py` | Full SJT pipeline: JS stability, per-trait ICC, train/test split, identity retrieval via JS similarity, SJT–HEXACO correlation heatmap |
| `sjt_combination_comparison.py` | Cross-generator comparison across the paired OpenAI/Anthropic splits |
| `sjt_histograms.py` | Histogram visualizations of SJT trait score distributions |
| `zero_compute_analysis_hexaco.py` | Eigenvalue histograms, UMAP clustering, and trait correlation heatmaps over raw HEXACO answers, grouped by archetype, ethnicity, and age |
| `zero_compute_analysis_sjt.py` | The same clustering and correlation pipeline over one-hot encoded SJT answers |

## `src/evals/`

| File | Purpose |
|---|---|
| `diversity_metrics.py` | Diversity and distributional spread: Sentence-Transformers embeddings, spaCy lemmatization, silhouette scores, TTR, MTLD, vendi-score, compression ratio |
| `sjt_evaluation_llm_judge_v0.py` | LLM judge scoring SJT responses against per-trait rubrics |

## `src/external_response_generation/`

Runs psychometric tests and benchmarks against local vLLM servers or external APIs.

| File | Purpose |
|---|---|
| `hexaco_response_generator.py` | Class-based vLLM HEXACO runner. Likert shuffling, paraphrase mode, n-repetitions, per-answer audit logging, push-to-Hub. Runs persona-conditioned and base conditions automatically |
| `sjt_response_generator_merged.py` | Unified SJT runner, same vLLM backend, persona and base-model modes, option shuffling with permutation tracking |
| `truthful_qa_mc.py` / `emo_bench_qa.py` | External behavioral benchmarks under persona conditioning |
| `adv_bench_generator.py` | AdvBench adversarial prompts, for refusal-rate measurement |
| `prompt_set_generator_base.py` | Abstract base for prompt set generators — hashing and metadata tracking |
| `create_comparison_splits.py` | Builds the paired cross-generator comparison splits |
| `filter_and_push_responses.py` | Result curation and Hub upload |
| `test_prompt_generation.py` | Validates that prompt templates render correctly before a run |

## `src/persona_generation/`

See [customizing.md](customizing.md) for the full guide.

| File | Purpose |
|---|---|
| `pydantic_persona_generation.py` | Core Pydantic-validated persona pipeline, exposed as `run_batch()` |
| `ablation_persona_generation.py` | Same pipeline with a CLI and `--ablation-config` grounding ablations |
| `pydantic_persona_generation_anthropic.py` | Claude-backed variant for cross-generator comparison |
| `pydantic_parliamentarian_generation.py` | UK Parliament domain port — 2,200 personas across 10 archetypes |
| `british_persona_seed_generator.py` | HEXACO-anchored seeds for British parliamentary archetypes |
| `singapore_patients_seed_generator.py` | Patient personas seeded from Singapore demographic data |
| `instances/handmade/` | Manually authored reference personas |

## `src/prompt_templates/`

Jinja2 prompt definitions, keyed by model family (`gpt`, `llama`, `qwen`, plus `gpt_cot` for ranked chain-of-thought).

| File | Purpose |
|---|---|
| `hexaco_base_prompt_templates.py` / `hexaco_persona_prompt_templates.py` | Base-model and persona-conditioned HEXACO messages |
| `sjt_base_prompt_templates.py` / `sjt_persona_prompt_templates.py` | The equivalents for SJT runs |
| `sjt_llm_judge_templates.py` | Rubric-grounded templates for LLM-based SJT evaluation |

## `src/synthetic_data_generation/`

| File | Purpose |
|---|---|
| `synthetic_sjt_creation_v0.py` | Generates HEXACO-aligned SJT scenarios from seed attributes, with the two-pass trait-bleed evaluation and correction step |
| `persona_llm_paper_persona_creation.py` | Personas matching the PersonaLLM paper configuration, for comparison |

## `src/utils/`

| File | Purpose |
|---|---|
| `vllm_utils.py` | `VLLMServerManager` — starts, monitors, and connects to a vLLM OpenAI-compatible server |
| `start_vllm.py` / `check_vllm.py` | Server startup and health check (defaults to port 9000) |
| `openai_utils.py` | Retry-wrapped, `diskcache`-backed OpenAI calls. **All OpenAI calls must go through this**; bump `CACHE_VERSION` when prompt logic changes |
| `hf_utils.py` | `HFStreamingAppender` — appends rows to a HF dataset without loading it fully into memory |
| `file_utils.py` | JSON read/write helpers with directory creation |
| `census_utils.py` | Loading and sampling census demographic data |
| `data/sjt_responses_processing.py` | Cleans and reshapes raw SJT answer files |
| `data/demographics_data_processing.py` | Processes demographic fields for persona datasets |
| `data/adv_bench_processing.py` | Parses AdvBench output files |

## Root-level `src/` files

| File | Purpose |
|---|---|
| `experiment.py` | Core orchestration: loops over personas, runs the test, collects and saves results |
| `modal_trigger.py` | Modal.com entry point for cloud GPU runs |
| `push_to_hub.py` | Uploads local results to a HuggingFace dataset repo |
| `add_persona_str.py` | Converts structured persona objects into the single `persona_string` field |
| `utils_v0.py` | Legacy shared utilities (Likert inversion, list formatting, OpenAI/Anthropic call wrappers) |
| `run.py` | Minimal example of invoking an experiment locally |

---

## `notebooks/`

Organized by stage; most have a corresponding converted `.py` script.

| Directory | Contents |
|---|---|
| `analysis/` | HEXACO profiles, SJT correlations, factor analysis, benchmark behavior, adversarial regression. Converted versions in `src/analysis/` |
| `reliability/` | Test-retest and inter-rater reliability for both instruments |
| `post_processing/` | Score computation and answer cleaning, run after generation |
| `evaluations/` | LLM judge runs (Anthropic, OpenAI) for persona and SJT quality review |
| `generation_personas/` | Persona generation workflows |
| `archive/` | Older experiments — including earlier notebook-style response generation and SJT synthesis, pre-`src/` |

---

## `configs/`

| File | Purpose |
|---|---|
| `generation_config.yaml` | Likert scale definition, temperature sweep, batch size, traits list, base prompt variants |
| `populated_police_seeds.yaml` | The complete police persona seed catalog — archetypes, memoirs **with summaries**, appearance and behavior categories. This is the one the generator wants |
| `police_seeds.yaml` | Earlier police seed file (no `MemoirSummaries`) |
| `synthetic_sjt_seeds.yaml` | Scenario variation axes for SJT generation |
| `parliament_seeds_enriched.yaml` / `skeleton_parliament_seeds.yaml` | UK Parliament domain seeds |
| `skeleton_singapore_patients.yaml` | Singapore patient domain seeds |
| `persona_llm_paper_seeds.yaml` | PersonaLLM comparison seeds |
| `personas.yaml` / `personas_v2.yaml` | Handcrafted local persona definitions |

## `psychometric_tests/`

| File | Purpose |
|---|---|
| `hexaco_100_questions.yaml` | The 100 HEXACO items, as a flat list |
| `paraphrased_hexaco_100_questions.yaml` | Paraphrased variants for wording-robustness testing (same order) |
| `hexaco_100_eval.yaml` | Scoring key: trait → facet → 1-based item indices and reverse-scoring flags |

## `data/`

| Directory | Contents |
|---|---|
| `demographics/` | `balanced_us_police_officers.csv` — 9,148 census-grounded rows |
| `census_data/` | Raw census inputs |
| `sjt_data/` | `sjt_jinja_template_v2.csv` (20 expert-authored base scenarios) and generated SJT parquet files |
| `persona_human_annotations/`, `persona_inter_rater_agreement/`, `persona_llm_judge/` | Persona validation data |
| `sjt_llm_judge_evaluation/`, `llm_judge_inter_rater_agreement/` | SJT validation data |
| `figures/` | Generated plots |
