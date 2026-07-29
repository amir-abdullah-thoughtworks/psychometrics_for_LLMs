# Customizing the framework

Nothing here is police-specific. There are two levers, and they are independent — you can swap either without touching the other:

| Lever | What it changes | Section |
|---|---|---|
| **The population** | Who the model is roleplaying — a seed YAML plus a demographics CSV | [Build your own persona set](#build-your-own-persona-set) |
| **The instrument** | What you are measuring — item banks, SJT scenarios, or a new response format | [Bring your own psychometric assessment](#bring-your-own-psychometric-assessment) |

Both start from working files in this repo, so you can diff your version against a known-good one. When you have something, [validate it](#validating-what-you-build) before trusting the numbers.

---

## Build your own persona set

Everything that makes personas differ from each other comes from two files — a **seed YAML** (the creative catalog) and a **demographics CSV** (the population). Swap both and you get nurses, teachers, or survey respondents instead of patrol officers, with no code changes.

The pipeline is designed to produce diverse but internally consistent personas at scale, to be reproducible (any persona can be regenerated from its seed index), and to support controlled ablation of individual grounding channels.

### 1 · The seed YAML

Four catalogs under a single top-level key.

- **Working reference:** [`llm_psychometrics/configs/populated_police_seeds.yaml`](../llm_psychometrics/configs/populated_police_seeds.yaml) — the file used for the paper's 8,500 personas.
- **Annotated starter template:** `persona_generation_overview/police_seeds_example.yaml` — same structure, heavily commented, trimmed down.

```yaml
PoliceOfficerPersonaSeeds:          # top-level key (the loader falls back to the file root)

  archetypes:                        # 6–10 psychological orientations — not job titles
    - name: The Problem Solver / Investigator
      core_trait: Analytical, detail-oriented, thrives on solving complex puzzles
      focus: Making sense of messy scenes and building airtight cases
      strengths:
        - Organizes chaos into coherent, fact-based reports
        - Builds strong cases that hold up in court
      challenges:
        - Can over-prioritize analysis at the expense of relationships

  MemoirSeeds:                       # narrative milieu — the biggest single diversity driver
    - "Blue Blood — Edward Conlon (2004)"
    - "Ghettoside — Jill Leovy (2015)"

  MemoirSummaries:                   # one entry per MemoirSeed, ~20–30 words, factual
    "Blue Blood — Edward Conlon (2004)": >
      NYPD officer blends personal narrative with reflections on urban policing, crime,
      and the humanity of both officers and civilians in New York City.

  AppearanceCategories:              # sensory style registers, ~20 phrases each
    Uniform/Official:
      - Crisp uniform, pressed and polished badge
      - Tactical vest strapped tightly across chest
    Plainclothes/Casual:
      - Faded windbreaker over a department polo

  BehaviorCategories:                # posture, gesture, interactional style
    Vigilant/Scanning:
      - Eyes tracking exits before sitting down
    Analytical/Focused:
      - Pauses mid-sentence to reorder the sequence of events
```

**What the loader enforces:**

- Every `MemoirSeeds` entry **must** have a matching `MemoirSummaries` key. Missing summaries raise with the list of offenders.
- `AppearanceCategories` and `BehaviorCategories` must be non-empty dicts. Individual category lists *may* be empty (`[]`) and filled in later without regenerating existing personas.
- `archetypes` must be a non-empty list. Entries may be plain strings, but dicts with `core_trait` / `focus` / `strengths` / `challenges` give the model far more to work with.

**Design guidance:**

*Archetypes* — aim for 6–10 genuinely distinct psychological orientations, grounded where possible in role-orientation or clinical typology literature. Include competent, conflicted, *and* struggling types; an all-negative catalog produces caricature. The model is instructed to *embody* the archetype, never to quote or paraphrase it, so texture should emerge in the prose rather than appear as a label.

*Memoirs* — invest here; it is the largest single driver of stylistic variety. Target 50–100 titles spanning geography, era, gender, and tone. For domains without a memoir literature, substitute blogs, oral histories, forum archives, or interview transcripts — anything that supplies a *specific narrative world* rather than a generic role description. Summaries need accuracy, not prose quality.

*Appearance / behavior phrases* — sensory and specific. *"Dusty boots caked from ranch backroads"* steers generation; *"looks tired"* does not.

**Fastest path to a new YAML.** Writing one from scratch is the highest-friction step. In practice:

1. Ask an LLM for the four catalogs for your target population — e.g. *"Generate 8 psychologically distinct archetypes for emergency-room nurses, each with a core trait, focus, strengths, and challenges."*
2. Use the police YAML as the formatting template; LLM output slots in with minimal editing.
3. Iterate on thin archetypes or short memoir lists before handing off for review.

That turns a multi-day desk-research task into roughly an hour of review-and-edit.

### 2 · The demographics CSV

One row per persona, treated as ground truth. These values are pinned into the generation schema as `Literal` types, so the model **cannot** drift from them.

**Validated at load:** `sex`, `age`, `city`, `state`, `first_name`, `last_name`, `education_level`, `marital_status`, `ethnic_background`.
**Also read:** `bachelors_field`, and `uuid` — the deduplication key.

Reference file: [`llm_psychometrics/data/demographics/balanced_us_police_officers.csv`](../llm_psychometrics/data/demographics/balanced_us_police_officers.csv) — 9,148 rows drawn from a probabilistic graphical model over census data. `src/utils/census_utils.py` and `src/utils/data/demographics_data_processing.py` cover loading and sampling if you are building one from census inputs.

Swapping this file is how you shift population — UK officers instead of US, nurses instead of soldiers, a target market segment instead of an occupation.

> ⚠️ The generator filters rows to ages **21–70**, an occupational constraint for serving officers. Change or drop that filter for other populations (`pydantic_persona_generation.py`, in `PersonaGenerator.__init__`).

### 3 · Generate

```bash
cd llm_psychometrics

python src/persona_generation/ablation_persona_generation.py \
  --populated-seeds-yaml   configs/populated_police_seeds.yaml \
  --balanced-officers-csv  data/demographics/balanced_us_police_officers.csv \
  --count 100 \
  --version my_personas_v1 \
  --out-jsonl my_personas_v1.jsonl \
  --ablation-config full \
  --model gpt-4.1-mini \
  --temperature 2.0 \
  --workers 10 \
  --no-push-to-hub
```

**What happens per persona:**

1. **Seeds are assigned by index, not randomly.** Demographics come from `df.iloc[idx]`; archetype and memoir each cycle independently (`(offset + idx) % N`); appearance and behavior categories draw from a per-index RNG seeded on `base_seed + c * idx`. Same `--base-seed` → identical seed assignment on every run. A persona at `idx=0` differs from `idx=1` even when they share an archetype.

2. **A Pydantic schema is built on the fly** with `create_model()`, declaring every seeded field as `Literal[value]`. Structured outputs then make it *impossible* for the model to return a different name, age, location, or archetype — non-conforming responses are rejected and retried. Free-text fields stay `str` with rich `Field(description=...)` prompts specifying word counts, tone, and constraints.

3. **The narrative comes first.** The model writes a 180–250 word `memoir_narrative` — a concrete, scene-level story in the selected memoir's milieu — and then fills roughly fifteen psychological-profile fields consistent with it. Demographics win any conflict with narrative tone.

4. **Ground truth is restored post-parse.** `archetype_description` and `memoir_summary` are overwritten with the exact YAML values regardless of model output, so downstream analysis groups by known archetype rather than a model paraphrase.

5. **Each row gets** a `persona_string` (prose joined into a single injectable block), a SHA-256 `persona_hash`, and a 1024-dim embedding from `Qwen/Qwen3-Embedding-0.6B` over the *generated* text only — seeds and demographics excluded — for diversity analysis.

Generation runs concurrently via `ThreadPoolExecutor` (default 10 workers). Re-running the same `--version` skips rows already present in the JSONL (deduped by `uuid`), so it is resumable.

Three prompt-level guards against stereotype collapse are worth knowing about, since you may want to adapt them:

- The archetype and memoir summary are both passed as *"guidance only — do not copy or paraphrase"*, and the model is told not to reuse five or more consecutive words from either.
- `presenting_problems` requires 3–6 items, **at least two of which must be unrelated to the occupation or the archetype** — otherwise every profile collapses into a role stereotype.
- Temperature is deliberately high (2.0 for OpenAI; cap at 1.0 for Anthropic) to maximize within-schema diversity.

**Output fields:**

| Group | Fields |
|---|---|
| **Pinned from seeds** | demographics, `archetype`, `memoir`, `appearance_category`, `behavior_category` — fixed at generation time, cannot drift |
| **Generated prose** | `memoir_narrative`, `appearance`, `behavior`, `speech`, `mood_affect`, `educational_vocational_history`, `medical_developmental_history`, `family_history`, `presenting_problems`, `thought_content`, `insight_judgment`, `cognition`, `emotional_behavioral_functioning`, `social_functioning`, `summary_of_psychological_profile` |
| **Computed post-generation** | `persona_string`, `persona_hash`, `concat_field`, `concat_embedding` |

### 4 · Grounding ablations

`--ablation-config` strips one grounding channel at a time, to isolate what each contributes to downstream response patterns:

| Config | Removes |
|---|---|
| `full` | nothing — the complete pipeline |
| `no_attribute_injections` | appearance/behavior style seeding |
| `no_memoir_grounding` | the narrative milieu |
| `no_demographic_grounding` | pinned demographics |
| `no_archetype_grounding` | the psychological archetype |

Each config pushes to its own HF dataset config, so ablation arms stay separable.

### 5 · Other entry points

| Script | Use |
|---|---|
| `src/persona_generation/pydantic_persona_generation.py` | The core pipeline, exposed as a library call (`run_batch()`) rather than a CLI |
| `src/persona_generation/ablation_persona_generation.py` | Same pipeline with a CLI and `--ablation-config` |
| `src/persona_generation/pydantic_persona_generation_anthropic.py` | Claude-backed variant, used for the `comparison_anthropic` cross-generator splits |
| `src/persona_generation/pydantic_parliamentarian_generation.py` | Worked example of porting the method to a new domain — 2,200 British parliamentarian personas across 10 archetypes, with domain-specific attributes (The Proceduralist, The Media Gladiator, The Policy Technocrat) and party context. The core generation methodology is unchanged |
| `src/persona_generation/british_persona_seed_generator.py` | Generates HEXACO-anchored seeds for the parliamentary domain |
| `src/persona_generation/singapore_patients_seed_generator.py` | Patient personas seeded from Singapore demographic data |
| `src/add_persona_str.py` | Converts structured persona objects into the single `persona_string` field used for prompt injection |
| `src/push_to_hub.py` | Uploads a local persona set to a HuggingFace dataset repo |

Once you have a set, check diversity before trusting anything downstream: `src/evals/diversity_metrics.py` computes silhouette scores, vendi-score, TTR, MTLD, and compression ratio over the generated fields.

---

## Bring your own psychometric assessment

Three levels of effort, depending on how far you are moving from HEXACO.

| Level | You want to | Effort |
|---|---|---|
| **1** | Keep the Likert self-report format, change the items | Edit two YAML files |
| **2** | Author a new SJT-style instrument for your own domain | Write base scenarios + variation axes, run the augmentation pipeline |
| **3** | Use a response format that is neither Likert nor six-way forced choice | New prompt templates, runner, and scoring |

### Level 1 — Swap the item bank of a self-report instrument

Replace [`llm_psychometrics/psychometric_tests/hexaco_100_questions.yaml`](../llm_psychometrics/psychometric_tests/hexaco_100_questions.yaml) — a flat YAML list of statements:

```yaml
- I would be quite bored by a visit to an art gallery.
- I clean my office or home quite frequently.
- I rarely hold a grudge, even against people who have badly wronged me.
```

Then update the scoring key, `hexaco_100_eval.yaml`, which maps trait → facet → **1-based** item indices plus per-item reverse-scoring flags:

```yaml
honest-humility:
  sincerity:
    indices: [6, 30, 54, 78]
    reverse: [True, False, True, False]
  fairness:
    indices: [12, 36, 60, 84]
    reverse: [True, True, False, True]
```

`indices` and `reverse` must be the same length, and every index must exist in your question list.

**Other knobs:**

- The response scale lives in `configs/generation_config.yaml` under `likert_scale`. Edit it there for a 7-point, bipolar, or differently-labeled scale. `--inverted-likert` reverses it; `--no-refusal` drops the appended *"Do not wish to answer"* option.
- `paraphrased_hexaco_100_questions.yaml` holds a parallel paraphrased bank used by `--paraphrase` for wording-robustness checks. Keep item order identical if you replace it, since the scoring key indexes both.
- The runner loads these three files **by name**, so either keep the filenames or edit `load_questions_and_likert()` in `hexaco_response_generator.py`.

**Run it:**

```bash
cd llm_psychometrics

python src/external_response_generation/hexaco_response_generator.py \
  --model-name google/gemma-3-4b-it \
  --persona-source huggingface \
  --hf-persona-path thoughtworks/psychometric_personas --hf-persona-config analysis \
  --n-personasample 500 --n-times 5 \
  --likert-shuffle \
  --vllm-port 9000 \
  --no-push-to-hub
```

The runner executes **two conditions automatically** — persona-conditioned and unconditioned base model — pushing to separate HF configs so you always have the baseline to difference against.

`--likert-shuffle` (on by default) randomizes scale order per answer, and every raw prompt, guided choice, and Likert ordering is logged per answer. Position and anchoring effects become auditable after the fact rather than silently baked into the trait scores. Add `--debug` to run and push only the first 10 personas.

### Level 2 — Author a new SJT-style instrument

An SJT item is a scenario plus exactly six response options, one per HEXACO trait. You supply two artifacts, and the pipeline grows a bank from them.

#### (a) Expert-authored base scenarios

[`llm_psychometrics/data/sjt_data/sjt_jinja_template_v2.csv`](../llm_psychometrics/data/sjt_data/sjt_jinja_template_v2.csv) — 20 rows in the released version.

| Column | Contents |
|---|---|
| `Question` | The scenario. Bracketed placeholders (e.g. `[time_of_day]`) mark the axes the augmentation step will vary |
| `Option 1` … `Option 6` | Six reasonable, feasible courses of action |
| `Option 1 Trait` … `Option 6 Trait` | The HEXACO trait each option keys to |

**This is where the domain expertise goes.** The paper's 20 base scenarios were written by industrial-organizational psychologists and a serving patrol officer, spanning interpersonal conflicts, high-stakes emergencies, and ethical dilemmas. Everything downstream inherits their quality — the augmentation pipeline varies scenarios, it does not invent good ones.

Each option must be genuinely plausible. If the "correct" answer is obvious, the item measures nothing.

#### (b) Variation axes

[`llm_psychometrics/configs/synthetic_sjt_seeds.yaml`](../llm_psychometrics/configs/synthetic_sjt_seeds.yaml) — a flat map of attribute → allowed values:

```yaml
urgency_level:   [low, medium, high]
threat_level:    [low, medium, high]
ambiguity_level: [clear, moderate, high]
individuals_involved:    [simple, moderate, complex]
authority_relationships: [peer_level, subordinate, authority]
situation_type:  [patrol_traffic_stop, crime_scene_investigation, emergency_response,
                  administrative_reporting, training_supervision, inter_agency_cooperation,
                  mental_health_crises]
ethical_considerations: [policy_compliance_vs_shortcuts, transparency_vs_self_protection,
                         individual_vs_team_loyalty, authority_vs_compassion,
                         procedure_vs_innovation]
time_of_day: [morning, afternoon, evening, night]
race:   [white, black_or_african_american, hispanic_latino, asian, ...]
gender: [male, female, non_binary, unknown]
age:    [juvenile, young_adult, adult, middle_aged, senior, unknown]
```

Every combination is enumerated, then sampled uniformly. That is what keeps the bank from collapsing onto the most stereotypical scenario: low-probability-but-plausible combinations (an *administrative reporting* situation at *high urgency* with *clear ambiguity*) get coverage too. Subject demographics are controlled explicitly for the same reason.

#### Run the augmentation pipeline

```bash
cd llm_psychometrics

python src/synthetic_data_generation/synthetic_sjt_creation_v0.py \
  --source local \
  --provider openai --generation-model gpt-4.1 --evaluation-model gpt-4.1 \
  --n-seeds 5 \
  --output-path data/sjt_data/my_sjts.parquet
```

Total items = `n_templates × n_seeds`. `--start-index` resumes partway through the base-scenario CSV.

**What happens per item:**

1. The generator rewrites the base scenario against the sampled attribute values — a controlled instruction-evolution step that varies one dimension at a time while preserving the decision structure.
2. A second LLM pass scores each option's **trait fit** from 1 (poor) to 5 (very strong) and flags **trait bleed**: options that read as two traits at once. This is the dominant noise source in downstream scoring, since the evaluation requires each response to map to a single trait.
3. Anything scoring below 5 is fed back to the model to sharpen its correspondence to the intended trait.
4. Both versions are retained — `original_sjt`, `corrected_sjt`, `trait_bleed_evaluation` — so the correction pass is fully auditable.

The paper used GPT-4.1 at `temperature=1.5`, `top_p=0.95`, `presence_penalty=0.4`, `frequency_penalty=0.3`, with [Outlines](https://github.com/dottxt-ai/outlines) for structured generation. `--provider anthropic` swaps in Claude.

#### The schema everything downstream expects

A `hash_id` plus a `corrected_sjt` dict with exactly these keys:

```python
{
  "question":                 ...,
  "honesty_humility_option":  ...,
  "emotionality_option":      ...,
  "extraversion_option":      ...,
  "agreeableness_option":     ...,
  "conscientiousness_option": ...,
  "openness_option":          ...,
}
```

Produce rows in that shape by any means — hand-written, another pipeline, a different trait model with six dimensions — and the SJT runner will consume them.

#### Run the instrument

```bash
python src/external_response_generation/sjt_response_generator_merged.py \
  --model google/gemma-3-4b-it \
  --hf-sjt-path     <your_sjt_dataset>     --hf-sjt-config     <config> \
  --persona-source  hf \
  --hf-persona-path <your_persona_dataset> --hf-persona-config <config> \
  --template-key gpt --use-persona-template \
  --n-times 5 --answer-shuffle \
  --out-json outputs/my_sjt_results.json \
  --no-push-to-hub
```

Option order is shuffled per item (`--answer-shuffle`, on by default) and the displayed→canonical permutation is recorded in `answer_index`, so position bias can be measured and corrected rather than absorbed into the trait scores. Answers are also stored in a canonical trait space (`normalized_answers`) alongside the raw ones.

Add `--debug` for a fast 10 personas × 10 items × 1 iteration smoke run. `--persona-source base_model` runs the unconditioned baseline. `--template-key gpt_cot` switches from single forced choice to a ranked-with-reasoning format.

### Level 3 — A genuinely different response format

If your instrument is neither Likert agreement nor six-way forced choice, you will add three pieces.

#### 1 · Prompt templates

`src/prompt_templates/` holds one module per instrument × condition. Each is a module-level dict keyed by model family, since chat formatting differs across GPT, Llama, and Qwen:

```python
sjt_persona_prompt_templates = {
    "gpt": [
        {"role": "system", "content": """You are a law enforcement officer with the following attributes:

{{ attributes }}

Task: Answer the below multiple choice question:"""},
        {"role": "user", "content": """Question: {{ question }}

Choices for this question are:

{{ answer_options }}
Give ONLY the option most consistent with your persona ..."""},
    ],
    "gpt_cot": [...],   # ranked, with reasoning, returned as JSON
    "llama": """<|begin_of_text|><|start_header_id|>system<|end_header_id|> ...""",
}
```

Placeholders are Jinja: `{{ attributes }}` (the persona string), `{{ question }}`, `{{ answer_options }}`. You need both a `*_persona_prompt_templates.py` and a `*_base_prompt_templates.py` so the unconditioned baseline stays comparable. Validate rendering with `src/external_response_generation/test_prompt_generation.py` before burning GPU hours.

#### 2 · A runner

Copy the closest of `hexaco_response_generator.py` (Likert, one question per prompt) or `sjt_response_generator_merged.py` (forced choice, option shuffling, permutation tracking). Both are vLLM-backed and follow the same shape: load personas → load items → build prompts → batch → normalize → save → optionally push to Hub.

`src/external_response_generation/prompt_set_generator_base.py` is the abstract base providing prompt hashing and metadata tracking, if you would rather start from the bottom.

#### 3 · Scoring

For deterministic scoring, a rubric YAML in `psychometric_tests/` following the `hexaco_100_eval.yaml` shape (trait → facet → indices + reverse flags).

For free-text responses, add an evaluator under `src/evals/`. `sjt_evaluation_llm_judge_v0.py` and `sjt_llm_judge_templates.py` demonstrate the rubric-grounded judge pattern used in the paper:

- **Two independent judges** (`gpt-4o-mini` and `claude-3-5-sonnet`) to limit self-preference bias.
- Ratings aggregated alongside human annotations on a hand-annotated slice.
- Agreement reported as Cohen's κ **and** mean absolute deviation — κ deflates badly when ratings concentrate in a narrow range, which happens constantly with high-quality generations, so both numbers are needed to tell "genuine disagreement" from "everyone said 5."

---

## Validating what you build

Whatever you build, reuse the paper's validation protocol:

1. **Item discrimination** under a MIRT nominal response model — are your items separating respondents at all?
2. **Test–retest ICC** across repeated sampling runs (the paper's threshold: > 0.86).
3. **JS divergence** between iterations, per condition.
4. **Human-vs-LLM-judge agreement** on a hand-annotated slice, reported as Cohen's κ *and* mean absolute deviation.
5. **Correlation against an external behavioral benchmark** — internal consistency alone proves nothing about validity.

Reference implementations: `src/analysis/sjt_correlation_analysis_v1.py` (JS stability, per-trait ICC, train/test split, identity retrieval, trait correlation heatmaps) and `src/analysis/benchmark_behaviour_analysis.py` (ICC and JS divergence against EmoBench and TruthfulQA). The external benchmark runners are already wired up: `truthful_qa_mc.py`, `emo_bench_qa.py`, and `adv_bench_generator.py` (refusal rates under persona conditioning).

### What "good" looked like on the released datasets

Useful as a yardstick for your own set.

**Personas** (n = 55 annotated) — rated by human annotators and two independent LLM judges on ten rubric axes: clarity, originality, coherence, diversity, realism, psychological depth, consistency, informativeness, ethical considerations, and demographic fidelity.

- Average scores exceed **4 / 5** on every axis (realism: 4.27 human, 4.78 GPT).
- Inter-rater **Cohen's κ ≈ 0.68** on average, ranging from near-perfect to near-random depending on the criterion.
- Overall mean absolute deviation: 0.47 (human vs. GPT), 0.24 (GPT vs. Claude).

**SJT items** (n = 30 annotated) — annotated by an industrial-organizational psychologist and a serving patrol officer, plus the same two LLM judges, on scenario realism, ethical tension, bias fairness, and per-trait alignment.

- Average rubric scores exceed **4 / 5** across all axes.
- Trait-alignment κ is 1.00 for most HEXACO traits. Agreeableness is weakest at 0.53 — the trait most prone to bleed.
- MAD is **0.00** for most trait-alignment axes: human experts and LLM judges converge.

**Diversity** — deterministic lexical and semantic metrics, confirming the corpus is not superficially repetitive:

| Metric | Value |
|---|---|
| MSTTR-100 | 0.802 |
| Compression ratio | 0.302 |
| Average cosine distance | 0.445 |

Full results, including the MIRT item analysis and the variance decomposition, are in [§7 of the paper](https://arxiv.org/abs/2510.22170).
