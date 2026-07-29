# LLM Psychometrics

[![arXiv](https://img.shields.io/badge/arXiv-2510.22170-b31b1b.svg)](https://arxiv.org/abs/2510.22170) &nbsp; [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset%20Collection-yellow)](https://huggingface.co/collections/thoughtworks/psychometrics-resources) &nbsp; [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](.python-version)

**Persona conditioning is everywhere. Does it give a model stable behavioral structure — or just a costume?**

This is the research framework behind [*Measure what Matters: Psychometric Evaluation of AI with Situational Judgment Tests*](https://arxiv.org/abs/2510.22170).

Most LLM personality work asks a model to rate its agreement with statements like *"I would never accept a bribe."* That is self-report: easy to game, likely memorized from pretraining, loosely connected to what the model does. We take the route organizational psychology uses instead: drop the model into a concrete, expert-authored scenario and force a choice between six plausible actions, where **each action maps to exactly one HEXACO trait**. Repeat across thousands of personas and scenarios, and behavior becomes measurable with the standard machinery: item discrimination, test–retest ICC, multidimensional item response theory, external validity.

| | |
|---|---|
| 🧪 **A new instrument** | 4,000 HEXACO-keyed law-enforcement SJT items, grown from 20 expert-authored base scenarios and screened for *trait bleed* |
| 🧍 **A persona pipeline** | 8,500 demographically grounded synthetic police officers + 2,200 British parliamentarians — swap one YAML to make your own |
| ⚙️ **Response runners** | vLLM and Modal backed generation for HEXACO, SJT, TruthfulQA, EmoBench, and AdvBench |
| 📊 **Analysis** | MIRT, ICC, Jensen–Shannon divergence, diversity metrics, LLM-as-judge against human-annotated ground truth |

```python
from datasets import load_dataset

sjts     = load_dataset("thoughtworks/psychometric_sjts_analysis", "analysis")["train"]
personas = load_dataset("thoughtworks/psychometric_personas",      "analysis")["train"]

item = sjts[0]["corrected_sjt"]
print(item["question"], "\n→", item["conscientiousness_option"])
print(personas[0]["persona_string"][:400])
```

---

## Three findings

From ~4M persona–scenario responses across the Gemma and Qwen families, over five sampling runs.

- **Behavior predicts better than self-report.** SJT-derived Agreeableness correlates with EmoBench at **0.70**; HEXACO self-report Agreeableness manages **0.51**. The two instruments correlate at only 0.25–0.35 with each other — convergent, not redundant.
- **Persona conditioning is a real intervention, not noise.** Against the base model, personas trade emotional calibration for epistemic accuracy (EmoBench **z = −1.06**, TruthfulQA **z = +1.13**), consistently across all eight archetypes.
- **It replicates.** **ICC > 0.86** on every HEXACO dimension; mean JS divergence of **0.003** between SJT response distributions across runs.

→ The other three findings — trait clustering, archetype discriminability, and the full MIRT psychometrics — along with the validation and human-annotation results, are in **[§7 of the paper](https://arxiv.org/abs/2510.22170)**.

> **Scope note.** These are stable *behavioral tendencies under persona and scenario conditioning* — not claims about model personality or internal states. See §9 and §10 of the paper, particularly on the risks of applying any of this in real hiring or law-enforcement settings.

---

## Setup

```bash
poetry install                # or: pip install -r requirements.txt

export OPENAI_API_KEY=...     # persona generation, SJT synthesis, LLM-as-judge
export HF_TOKEN=...           # dataset pulls and result pushes — or run `huggingface-cli login`
```

Python 3.11.9. Two things worth knowing before your first run:

- Several generation scripts resolve config paths relative to the working directory — **run them from `llm_psychometrics/`**.
- All OpenAI calls route through `src/utils/openai_utils.py`, which adds retries and disk caching. **Bump `CACHE_VERSION` there whenever prompt logic changes**, or you will silently score stale responses.

## Running the experiments

```bash
cd llm_psychometrics/src

# Serve a model (vLLM, OpenAI-compatible, port 9000)
python utils/start_vllm.py --model-name google/gemma-3-4b-it --hf-token $HF_TOKEN

# HEXACO — runs the persona-conditioned and unconditioned base conditions
python external_response_generation/hexaco_response_generator.py \
  --model-name google/gemma-3-4b-it --n-personasample 500 --n-times 5 --vllm-port 9000

# SJT
python external_response_generation/sjt_response_generator_merged.py --model google/gemma-3-4b-it

# External benchmarks: truthful_qa_mc.py | emo_bench_qa.py | adv_bench_generator.py
# Analysis:            analysis/sjt_correlation_analysis_v1.py | analysis/benchmark_behaviour_analysis.py
```

At scale, run on Modal (A100s, persistent `my-outputs` volume): `modal run llm_psychometrics/src/modal_trigger.py`.

> ⚠️ `start_vllm.py` defaults to port **9000** while the runners default to **8000** — pass `--vllm-port 9000` or set the base URL explicitly.

---

## Make it yours

The framework is not police-specific. **[docs/customizing.md](docs/customizing.md)** covers both levers, starting from working files in this repo:

| Section | What it covers |
|---|---|
| [Build your own persona set](docs/customizing.md#build-your-own-persona-set) | Custom personas from your own YAML — the four seed catalogs, the demographics CSV schema, deterministic seed assignment, grounding ablations, porting to a new population |
| [Bring your own assessment](docs/customizing.md#bring-your-own-psychometric-assessment) | Swapping the item bank of a self-report instrument, authoring and augmenting a new SJT-style instrument, or wiring a different response format |
| [Validating what you build](docs/customizing.md#validating-what-you-build) | The five-step protocol from the paper, plus what "good" looked like on the released datasets |

---

## Datasets

| Dataset | Contents |
|---|---|
| [`thoughtworks/psychometric_personas`](https://huggingface.co/datasets/thoughtworks/psychometric_personas) | 8,500 synthetic law-enforcement personas across 8 configs |
| [`thoughtworks/psychometric_SJTs`](https://huggingface.co/datasets/thoughtworks/psychometric_SJTs) | The raw 4,000-item HEXACO-keyed SJT bank, pre-curation |
| [`thoughtworks/psychometric_sjts_analysis`](https://huggingface.co/datasets/thoughtworks/psychometric_sjts_analysis) | Curated SJT subsets with embeddings, scenario metadata, and inter-rater reliability scores |

Subset hierarchies, row counts, and how each config feeds into response runs: **[DATASETS.md](DATASETS.md)**.

## Repository layout

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

File-by-file reference: **[docs/repo-map.md](docs/repo-map.md)**. There is no formal test suite; experimental validation lives in `llm_psychometrics/notebooks/`.

---

## Citation

```bibtex
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

Questions or collaboration: amir.abdullah@thoughtworks.com, jshrey8@gmail.com
