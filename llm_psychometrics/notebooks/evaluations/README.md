# Evaluations

LLM-judge and human rubric scores for persona quality assessment.

## Annotation files

### `open_ai_persona_reviews.json`
GPT-judge scores for 55 personas (1–5 scale). Fields: `clarity`, `originality`, `coherence`, `diversity`, `realism`, `psychological_depth`, `consistency`, `informativeness`, `ethical_considerations`, `demographic_fidelity`, `overall_score`. Keyed by persona UUID.

Means match Table 3 (GPT column) in the paper exactly.

### `anthropic_persona_reviews.json`
Claude-judge scores for the same 55 personas (1–5 scale), same schema. Keyed by the same UUIDs as `open_ai_persona_reviews.json`.

Means match Table 3 (Claude column) in the paper exactly.

### `persona_reviews.json`
100 personas, same rubric schema (1–5 scale). Likely an earlier or broader GPT-judge run covering more personas.

### `final_list_of_reviews.json`
55 personas on a **1–10 scale** — a different scoring run, not directly comparable to the 1–5 files above.

## Notebooks

- `personas_llm_judge_openai.ipynb` / `personas_llm_judge_anthropic.ipynb` — scripts that produced the review JSONs
- `paper_metrics.ipynb` — computes Table 3 and other paper metrics
- `diversity_judge.ipynb` — diversity scoring
