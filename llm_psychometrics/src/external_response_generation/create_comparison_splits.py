"""
create_comparison_splits.py
===========================
Creates five comparison response splits in:
  thoughtworks/gemma_psychometrics_personas_responses

These splits form a 2x2 + 1 design for comparing OpenAI vs. Anthropic authorship
of both personas and SJT items, holding the Gemma response model constant.

SPLITS
------
1. cmp_openai_personas_cmp_openai_sjts   [COPY from analysis_sjt]
   Personas : thoughtworks/psychometric_personas           | comparison_openai (100)
   SJTs     : thoughtworks/psychometric_sjts_analysis      | comparison_openai (100)
   How      : Filtered directly from analysis_sjt (no re-run needed).
              comparison_openai personas and SJTs are both subsets of analysis,
              so all 100x100x5 = 50,000 rows already exist.

2. cmp_openai_personas_cmp_anthropic_sjts   [COPY from gpt_persona_claude_sjt]
   Personas : thoughtworks/psychometric_personas           | comparison_openai (100)
   SJTs     : thoughtworks/psychometric_sjts_analysis      | comparison_anthropic (100)
   How      : Already exists as config 'gpt_persona_claude_sjt' (50,000 rows, verified).
              hf_persona_config='comparison_openai', hf_sjt_config='comparison_anthropic'.
              Copy by filtering that config to the 100 comparison_openai persona UUIDs
              and 100 comparison_anthropic SJT hashes.

3. cmp_anthropic_personas_cmp_openai_sjts   [FRESH RUN]
   Personas : thoughtworks/psychometric_personas           | comparison_anthropic (100)
   SJTs     : thoughtworks/psychometric_sjts_analysis      | comparison_openai (100)
   How      : comparison_anthropic personas share UUIDs with comparison_openai but have
              different persona strings (Claude-generated prose), so analysis_sjt rows
              cannot be reused — fresh run required.

4. cmp_anthropic_personas_cmp_anthropic_sjts   [FRESH RUN]
   Personas : thoughtworks/psychometric_personas           | comparison_anthropic (100)
   SJTs     : thoughtworks/psychometric_sjts_analysis      | comparison_anthropic (100)
   How      : Both sides differ from analysis_sjt. Fresh run required.

5. cmp_anthropic_personas_analysis_openai_sjts   [FRESH RUN]
   Personas : thoughtworks/psychometric_personas           | comparison_anthropic (100)
   SJTs     : thoughtworks/psychometric_sjts_analysis      | analysis (300)
   How      : Gives full SJT coverage for Anthropic personas (100x300x5 = 150,000 rows).
              SJTs are the same as analysis_sjt but persona strings differ.

PROVENANCE NOTES
----------------
- comparison_openai personas: first 100 rows of the analysis persona set (OpenAI-generated).
- comparison_anthropic personas: Claude Sonnet re-generations of the same 100 seeds.
  Same UUIDs and seed fields; all prose fields differ.
- comparison_openai SJTs: 100-item subset of the analysis SJT set (OpenAI-generated).
- comparison_anthropic SJTs: Claude-generated from the same template+seed combinations
  as comparison_openai SJTs. source_hash_id links each item to its paired comparison_openai
  counterpart. Independently generated — not a rewrite of the OpenAI text.
- analysis SJTs: 300-item curated set used for the main 500-persona run.

USAGE
-----
# Steps 1-2: copy splits (no vLLM needed)
python create_comparison_splits.py --split cmp_openai_personas_cmp_openai_sjts
python create_comparison_splits.py --split cmp_openai_personas_cmp_anthropic_sjts

# Steps 3-5: fresh runs (requires running vLLM server on port 9000)
#   python start_vllm.py --model-name google/gemma-3-4b-it
python create_comparison_splits.py --split cmp_anthropic_personas_cmp_openai_sjts
python create_comparison_splits.py --split cmp_anthropic_personas_cmp_anthropic_sjts
python create_comparison_splits.py --split cmp_anthropic_personas_analysis_openai_sjts

# Run all (copy first, then fresh runs sequentially)
python create_comparison_splits.py --all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("HF_HOME", "/workspace/HF/cache")

from datasets import load_dataset, DatasetDict
from huggingface_hub import login

HF_REPO_ID = "thoughtworks/gemma_psychometrics_personas_responses"
HF_PERSONA_REPO = "thoughtworks/psychometric_personas"
HF_SJT_REPO = "thoughtworks/psychometric_sjts_analysis"
MODEL = "google/gemma-3-4b-it"
VLLM_PORT = 9000

SPLITS = {
    "cmp_openai_personas_cmp_openai_sjts": {
        "persona_config": "comparison_openai",
        "sjt_config": "comparison_openai",
        "method": "copy",
        "source_config": "analysis_sjt",
        "expected_rows": 50_000,
    },
    "cmp_openai_personas_cmp_anthropic_sjts": {
        "persona_config": "comparison_openai",
        "sjt_config": "comparison_anthropic",
        "method": "copy",
        "source_config": "gpt_persona_claude_sjt",
        "expected_rows": 50_000,
    },
    "cmp_anthropic_personas_cmp_openai_sjts": {
        "persona_config": "comparison_anthropic",
        "sjt_config": "comparison_openai",
        "method": "run",
        "expected_rows": 50_000,
    },
    "cmp_anthropic_personas_cmp_anthropic_sjts": {
        "persona_config": "comparison_anthropic",
        "sjt_config": "comparison_anthropic",
        "method": "run",
        "expected_rows": 50_000,
    },
    "cmp_anthropic_personas_analysis_openai_sjts": {
        "persona_config": "comparison_anthropic",
        "sjt_config": "analysis",
        "method": "run",
        "expected_rows": 150_000,
    },
}


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = os.path.join(os.environ.get("HF_HOME", ""), "token")
        if os.path.exists(token_path):
            token = open(token_path).read().strip()
    if token:
        login(token=token)
    else:
        raise RuntimeError("HF_TOKEN not set and ~/.cache/huggingface/token not found.")


def run_copy_split(split_name: str, cfg: dict):
    """Filter analysis_sjt to the comparison_openai persona x SJT subset."""
    print(f"\n[{split_name}] Method: copy from {cfg['source_config']}")

    persona_uuids = set(
        r["uuid"]
        for r in load_dataset(HF_PERSONA_REPO, name=cfg["persona_config"])["train"]
    )
    sjt_hashes = set(
        r["hash_id"]
        for r in load_dataset(HF_SJT_REPO, name=cfg["sjt_config"])["train"]
    )

    print(f"  Filtering to {len(persona_uuids)} personas x {len(sjt_hashes)} SJTs...")
    source = load_dataset(HF_REPO_ID, name=cfg["source_config"])["train"]
    filtered = source.filter(
        lambda r: r["persona_uuid"] in persona_uuids and r["question_hash"] in sjt_hashes,
        num_proc=4,
    )

    print(f"  Filtered rows: {len(filtered)} (expected {cfg['expected_rows']})")
    assert len(filtered) == cfg["expected_rows"], (
        f"Row count mismatch: got {len(filtered)}, expected {cfg['expected_rows']}"
    )

    print(f"  Pushing to {HF_REPO_ID} / {split_name} ...")
    filtered.push_to_hub(HF_REPO_ID, config_name=split_name, split="train")
    print(f"  Done.")


def run_inference_split(split_name: str, cfg: dict, out_dir: str = "/tmp"):
    """Run SJT inference for a persona x SJT config and push to hub."""
    print(f"\n[{split_name}] Method: fresh inference run")
    print(f"  Personas: {HF_PERSONA_REPO}/{cfg['persona_config']}")
    print(f"  SJTs:     {HF_SJT_REPO}/{cfg['sjt_config']}")

    from external_response_generation.sjt_response_generator_merged import (
        SJTResponseRunner,
        VLLMServerManager,
        push_results_to_hub,
    )

    mgr = VLLMServerManager(model=MODEL, kill_existing=False, port=VLLM_PORT)
    if not mgr._is_up():
        raise RuntimeError(
            f"vLLM server not running on port {VLLM_PORT}. "
            f"Start it with: python start_vllm.py --model-name {MODEL}"
        )

    out_json = os.path.join(out_dir, f"{split_name}.json")

    args = argparse.Namespace(
        model=MODEL,
        persona_source="hf",
        hf_persona_path=HF_PERSONA_REPO,
        hf_persona_config=cfg["persona_config"],
        hf_persona_split="train",
        hf_sjt_path=HF_SJT_REPO,
        hf_sjt_config=cfg["sjt_config"],
        hf_sjt_split="train",
        n_times=5,
        answer_shuffle=True,
        use_persona_template=True,
        template_key="gpt",
        debug=False,
        n_personasample=None,
        n_sjtsample=None,
        out_json=out_json,
        push_to_hub=True,
        target_hub_repo_id=HF_REPO_ID,
        target_hub_config=split_name,
        target_hub_split="train",
        max_tokens=1,
        temperature=0.3,
        top_p=0.9,
        batch_size=500,
        num_workers=6,
    )

    runner = SJTResponseRunner(args=args, mgr=mgr)
    results = runner.run()

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results.to_jsonable(), f, ensure_ascii=False)
    print(f"  Saved local results -> {out_json}")

    push_results_to_hub(results, args)
    print(f"  Pushed to {HF_REPO_ID} / {split_name}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--split",
        choices=list(SPLITS.keys()),
        help="Run a single split.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all splits (copy first, then fresh runs in order).",
    )
    p.add_argument(
        "--out-dir",
        default="/tmp",
        help="Directory for local JSON output from inference runs.",
    )
    return p.parse_args()


def run_split(split_name: str, out_dir: str):
    cfg = SPLITS[split_name]
    if cfg["method"] == "copy":
        run_copy_split(split_name, cfg)
    else:
        run_inference_split(split_name, cfg, out_dir=out_dir)


def main():
    args = parse_args()
    hf_login()

    if args.all:
        for split_name in SPLITS:
            run_split(split_name, args.out_dir)
    else:
        run_split(args.split, args.out_dir)


if __name__ == "__main__":
    main()
