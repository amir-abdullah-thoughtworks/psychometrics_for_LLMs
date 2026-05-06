# external_response_generation

Scripts for running psychometric test scenarios through a local vLLM server and recording model responses. All scripts expect a running vLLM OpenAI-compatible server (default port 8000, except where noted).

## Scripts

### `sjt_response_generator_merged.py` — Situational Judgment Tests

Runs SJT scenarios through an LLM with optional persona injection. For each persona × item × iteration the model picks one of 6 HEXACO trait options via constrained decoding (`guided_choices=["1","2","3","4","5","6"]`). Answer options are deterministically shuffled per (persona, question, iteration) to prevent position bias.

See [`SJT_RESPONSE_GENERATOR.md`](SJT_RESPONSE_GENERATOR.md) for full documentation.

```bash
# Debug run (5 personas × 10 SJTs × 3 iters, no hub push)
python sjt_response_generator_merged.py --debug --no-push-to-hub

# Full run
python sjt_response_generator_merged.py \
  --no-debug \
  --model google/gemma-3-4b-it \
  --hf-persona-config analysis \
  --n-times 5 \
  --out-json /outputs/sjt_results.json
```

---

### `hexaco_response_generator.py` — HEXACO-100 Personality Inventory

Runs HEXACO-100 Likert-scale questions through an LLM with optional persona injection. Responses are one of five Likert options (Strongly Disagree → Strongly Agree).

```bash
python hexaco_response_generator.py \
  --model-name google/gemma-3-4b-it \
  --hf-persona-config analysis \
  --n-personasample 10 \
  --n-times 5 \
  --out-dir /outputs/hexaco
```

Key flags: `--paraphrase` (use paraphrased questions), `--inverted-likert`, `--no-refusal`, `--batching`.

---

### `adv_bench_generator.py` — AdvBench Safety Scenarios

Runs AdvBench harmful-behavior prompts through personas to assess refusal/compliance patterns. Results are saved to a JSONL directory and optionally pushed to HuggingFace.

```bash
python adv_bench_generator.py
```

Output goes to `outputs/gemma_advbench_persona_responses/` by default.

---

### `prompt_set_generator_base.py` — Base class

`PromptSetGeneratorBase` is the shared base for prompt-set generators. Handles deduplication via a seen-pairs JSONL log and run-ID generation. Not run directly.

---

## Prerequisites

- A running vLLM server: `python start_vllm.py --model-name <model>` (from repo root)
- `HF_TOKEN` env var set for loading personas/SJTs from HuggingFace and pushing results
