"""
vLLM-only SJT response generator (patched).

Key behaviors:
- Default is DEBUG mode: runs 10 personas x 10 SJTs x 1 iteration, with answer shuffling ON.
- NO-DEBUG mode: uses ALL provided personas and ALL provided SJTs (no dropping) and can push to HF Hub:
    repo_id: thoughtworks/psychometric_personas_responses
    config:  police_sjt

Outputs:
- Per persona: PersonaRunConfig with:
  - answers: List[List[str]] (iterations x questions)
  - normalized_answers: List[List[Optional[str]]] (iterations x questions), trait keys in canonical space
  - answer_index: List[List[List[int]]] (iterations x questions x 6), permutation displayed->canonical
  - raw_prompts, guided_choices, question_hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from jinja2 import Template
from pydantic import BaseModel, RootModel
from tqdm import tqdm

# --- Custom imports (keep as-is in your repo) ---
from utils_v0 import list_to_str
from prompt_templates.sjt_base_prompt_templates import sjt_base_prompt_templates
from prompt_templates.sjt_persona_prompt_templates import sjt_persona_prompt_templates
from utils.vllm_utils import VLLMServerManager


# =========================
# Constants
# =========================

SJT_ANSWER_CHOICES: List[str] = ["1", "2", "3", "4", "5", "6"]

DEFAULT_ANSWER_OPTION_ORDERING: List[str] = [
    "honesty_humility_option",
    "emotionality_option",
    "extraversion_option",
    "agreeableness_option",
    "conscientiousness_option",
    "openness_option",
]

# =========================
# Models
# =========================

class PersonaRunConfig(BaseModel):
    persona: str
    persona_hash: Optional[str]  # SHA256(persona_string), None for base_model

    # Per iteration -> per question -> chosen option (ideally "1".."6")
    answers: List[List[str]]

    # Per iteration -> per question -> normalized trait option key in canonical space
    # One of DEFAULT_ANSWER_OPTION_ORDERING values.
    normalized_answers: List[List[Optional[str]]]

    # Per question (same across iterations)
    question_hashes: List[str]

    # Per iteration -> per question -> permutation used to shuffle options (length 6).
    # idx[j] tells which canonical option index appears at displayed position j.
    answer_index: List[List[List[int]]]

    # Per iteration -> per question -> exact prompt passed to vLLM
    raw_prompts: List[List[str]]

    # Per iteration -> per question -> allowed outputs for guided decoding
    guided_choices: List[List[List[str]]]

    model_name: str

    # Optional provenance
    hf_persona_path: Optional[str] = None
    hf_persona_config: Optional[str] = None
    hf_sjt_path: Optional[str] = None
    hf_sjt_config: Optional[str] = None
    hf_sjt_split: Optional[str] = None
    sjt_answer_options: Optional[Literal["normal", "shuffle"]] = None


class ExperimentResults(RootModel[Dict[str, PersonaRunConfig]]):
    """
    persona_uuid (or 'base_model') -> PersonaRunConfig
    """
    def to_jsonable(self) -> Dict[str, Any]:
        return {k: v.model_dump() for k, v in self.root.items()}


# =========================
# Prompt helpers
# =========================

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def prompt_seed_int(*parts: str, salt: str = "sjt_shuffle_v1") -> int:
    """
    Deterministic 32-bit seed from stable SHA256 over concatenated parts.
    Use parts that uniquely define the prompt instance (persona, question, etc).
    """
    s = salt + "||" + "||".join(parts)
    # take first 8 hex chars = 32 bits (enough for random.Random seed)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def compile_message_templates(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return [{"role": m["role"], "content": Template(m["content"])} for m in messages]

def render_messages(compiled_messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, str]]:
    return [{"role": m["role"], "content": m["content"].render(**kwargs)} for m in compiled_messages]

def messages_to_prompt_text(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        lines.append(f"{role}:\n{content}".strip())
    lines.append("ASSISTANT:\n")
    return "\n\n".join(lines)

def validate_prompt_text(prompt: str, where: str) -> None:
    if "{{" in prompt or "}}" in prompt:
        raise ValueError(f"[{where}] Prompt still contains Jinja tokens.")
    if not prompt.strip():
        raise ValueError(f"[{where}] Prompt is empty after rendering.")

def normalize_answer_to_trait(
    answer: str,
    permutation_displayed_to_canonical: List[int],
) -> Optional[str]:
    """
    answer: model output, ideally "1".."6" (displayed index +1)
    permutation_displayed_to_canonical: length-6 list mapping displayed position -> canonical option index
    """
    a = str((answer or "").strip())[0]
    try:
        if a not in SJT_ANSWER_CHOICES:
            raise ValueError(f"Invalid answer choice: {a}, {a} not {SJT_ANSWER_CHOICES}")
        displayed_idx = int(a) - 1
        if not (0 <= displayed_idx < 6):
            raise ValueError(f"Invalid answer displayed index: {displayed_idx}")
        canonical_idx = permutation_displayed_to_canonical[displayed_idx]
        if not (0 <= canonical_idx < 6):
            raise ValueError(f"Invalid permutation displayed index: {canonical_idx}")
        return DEFAULT_ANSWER_OPTION_ORDERING[canonical_idx]
    except ValueError:
        return None


# =========================
# SJT row normalization
# =========================

@dataclass(frozen=True)
class SJTSpec:
    idx: int
    hash_id: str
    question: str
    options_canonical: List[str]  # length 6, in DEFAULT_ANSWER_OPTION_ORDERING order
    corrected_sjt: Dict[str, Any]

def normalize_sjt_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sjt = row.get("corrected_sjt") or row.get("sjt") or row
    if not isinstance(sjt, dict):
        raise ValueError(f"SJT row missing dict payload. Keys: {list(row.keys())}")
    return sjt

def extract_sjt_spec(row: Dict[str, Any], idx: int) -> SJTSpec:
    sjt = normalize_sjt_row(row)
    question = sjt.get("question")
    if not question:
        raise ValueError(f"SJT payload missing 'question'. Keys: {list(sjt.keys())}")

    options: List[str] = []
    for k in DEFAULT_ANSWER_OPTION_ORDERING:
        if k not in sjt:
            raise ValueError(f"SJT payload missing option key '{k}'. Keys: {list(sjt.keys())}")
        options.append(sjt[k])

    hash_id = row["hash_id"]

    return SJTSpec(
        idx=idx,
        hash_id=str(hash_id),
        question=str(question),
        options_canonical=[str(x) for x in options],
        corrected_sjt=sjt,
    )


# =========================
# Hub push helpers
# =========================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_source_meta(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "debug": bool(args.debug),
        "persona_source": getattr(args, "persona_source", None),

        "hf_persona_path": getattr(args, "hf_persona_path", None),
        "hf_persona_config": getattr(args, "hf_persona_config", None),
        "hf_persona_split": getattr(args, "hf_persona_split", None),

        "hf_sjt_path": getattr(args, "hf_sjt_path", None),
        "hf_sjt_config": getattr(args, "hf_sjt_config", None),
        "hf_sjt_split": getattr(args, "hf_sjt_split", None),

        "template_key": getattr(args, "template_key", None),
        "use_persona_template": bool(getattr(args, "use_persona_template", False)),
        "answer_shuffle": bool(getattr(args, "answer_shuffle", False)),
        "n_times": int(getattr(args, "n_times", 1)),
        "model": getattr(args, "model", None),
        "max_tokens": int(getattr(args, "max_tokens", 1)),
        "temperature": float(getattr(args, "temperature", 0.0)),
    }

def flatten_results_for_hub(exp: ExperimentResults, source_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for persona_uuid, cfg in exp.root.items():
        d = cfg.model_dump()
        answers = d["answers"]
        normalized = d["normalized_answers"]
        q_hashes = d["question_hashes"]
        ans_idx = d["answer_index"]
        raw_prompts = d["raw_prompts"]
        guided_choices = d["guided_choices"]

        for t in range(len(answers)):
            for qi in range(len(q_hashes)):
                rows.append({
                    "persona_uuid": persona_uuid,
                    "persona_hash": d["persona_hash"],
                    "iter": t,
                    "question_hash": q_hashes[qi],
                    "answer": answers[t][qi],
                    "normalized_answer": normalized[t][qi],
                    "answer_index": ans_idx[t][qi],
                    "raw_prompt": raw_prompts[t][qi],
                    "guided_choices": guided_choices[t][qi],
                    "model_name": d["model_name"],
                    "run_timestamp_utc": _utc_now_iso(),
                    **source_meta,
                })
    return rows

def push_results_to_hub(exp: ExperimentResults, args: argparse.Namespace) -> None:
    source_meta = build_source_meta(args)
    rows = flatten_results_for_hub(exp, source_meta)
    ds = Dataset.from_list(rows)
    dsd = DatasetDict({args.target_hub_split: ds})

    desc = {
        "created_utc": _utc_now_iso(),
        "repo_id": args.target_hub_repo_id,
        "config": args.target_hub_config,
        "split": args.target_hub_split,
        "source_meta": source_meta,
        "counts": {
            "rows": len(ds),
            "unique_personas": len(set(ds["persona_uuid"])) if len(ds) else 0,
            "unique_questions": len(set(ds["question_hash"])) if len(ds) else 0,
        },
    }
    dsd[args.target_hub_split].info.description = json.dumps(desc, indent=2, sort_keys=True)

    dsd.push_to_hub(
        repo_id=args.target_hub_repo_id,
        config_name=args.target_hub_config,
    )


# =========================
# Runner
# =========================

class SJTResponseRunner:
    def __init__(self, args: argparse.Namespace, mgr: VLLMServerManager):
        self.args = args
        self.mgr = mgr

        transformers.logging.set_verbosity_error()

        # Templates
        base_msgs = sjt_base_prompt_templates[self.args.template_key]
        persona_msgs = sjt_persona_prompt_templates[self.args.template_key]
        chosen = persona_msgs if self.args.use_persona_template else base_msgs
        self.compiled_templates = compile_message_templates(chosen)

    def _make_shuffled_options(
            self, canonical_options: List[str],
            *, seed: Optional[int] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Returns:
          displayed_options: List[str] length 6 (possibly shuffled)
          permutation: List[int] length 6 mapping displayed position -> canonical index
        """
        if not self.args.answer_shuffle:
            return canonical_options[:], list(range(6))

        perm = list(range(6))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(perm)
        displayed = [canonical_options[i] for i in perm]
        return displayed, perm

    def _build_prompt(
        self,
        *,
        persona_str: Optional[str],
        sjt: SJTSpec,
        displayed_options: List[str],
    ) -> str:
        rendered = render_messages(
            self.compiled_templates,
            attributes=persona_str,
            question=sjt.question,
            answer_options=list_to_str(displayed_options),
            sjt=sjt.corrected_sjt,
        )
        prompt = messages_to_prompt_text(rendered)
        validate_prompt_text(prompt, where=f"persona={persona_str is not None} sjt={sjt.idx}")
        return prompt

    def _run_one_persona(
        self,
        persona_uuid: str,
        persona_hash: Optional[str],
        persona_str: Optional[str],
        sjts: List[SJTSpec],
    ) -> PersonaRunConfig:
        answers_all: List[List[str]] = []
        norm_all: List[List[Optional[str]]] = []
        q_hashes: List[str] = [s.hash_id for s in sjts]
        idx_all: List[List[List[int]]] = []
        prompts_all: List[List[str]] = []
        guided_all: List[List[List[str]]] = []

        invalid_rows = 0

        for t in range(self.args.n_times):
            prompts: List[str] = []
            perms: List[List[int]] = []
            displayed_options_list: List[List[str]] = []

            for sjt in sjts:
                seed = prompt_seed_int(
                    str(persona_hash or persona_uuid or "base_model"),
                    str(sjt.hash_id)
                )
                displayed, perm = self._make_shuffled_options(sjt.options_canonical, seed=seed)
                displayed_options_list.append(displayed)
                perms.append(perm)
                prompts.append(self._build_prompt(persona_str=persona_str, sjt=sjt, displayed_options=displayed))

            outputs = self.mgr.vllm_chat_batched(
                prompts=prompts,
                model=self.args.model,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                guided_choices=SJT_ANSWER_CHOICES,
            )

            # Normalize / record
            iter_answers: List[str] = []
            iter_norm: List[Optional[str]] = []
            iter_idx: List[List[int]] = []
            iter_guided: List[List[str]] = []

            for out, perm in zip(outputs, perms):
                a = str((out or "").strip())
                # Keep only first token-ish to be safe
                a = str(a.split()[0] if a else "")
                if a not in SJT_ANSWER_CHOICES:
                    invalid_rows += 1
                    a = None
                    iter_answers.append(a)
                    iter_norm.append(a)
                    iter_idx.append(perm)
                    iter_guided.append(SJT_ANSWER_CHOICES)

                else:
                    iter_answers.append(a)
                    iter_norm.append(normalize_answer_to_trait(a, perm))
                    iter_idx.append(perm)
                    iter_guided.append(SJT_ANSWER_CHOICES)

            answers_all.append(iter_answers)
            norm_all.append(iter_norm)
            idx_all.append(iter_idx)
            prompts_all.append(prompts)
            guided_all.append(iter_guided)

        print(f"Total invalid rows: {invalid_rows}")

        return PersonaRunConfig(
            persona=persona_uuid,
            persona_hash=persona_hash,
            answers=answers_all,
            normalized_answers=norm_all,
            question_hashes=q_hashes,
            answer_index=idx_all,
            raw_prompts=prompts_all,
            guided_choices=guided_all,
            model_name=self.args.model,
            hf_persona_path=getattr(self.args, "hf_persona_path", None),
            hf_persona_config=getattr(self.args, "hf_persona_config", None),
            hf_sjt_path=getattr(self.args, "hf_sjt_path", None),
            hf_sjt_config=getattr(self.args, "hf_sjt_config", None),
            hf_sjt_split=getattr(self.args, "hf_sjt_split", None),
            sjt_answer_options="shuffle" if self.args.answer_shuffle else "normal",
        )

    def run(self) -> ExperimentResults:
        results: Dict[str, PersonaRunConfig] = {}

        # --- Load SJT dataset
        if self.args.hf_sjt_config:
            sjt_ds = load_dataset(self.args.hf_sjt_path, name=self.args.hf_sjt_config)[self.args.hf_sjt_split]
        else:
            sjt_ds = load_dataset(self.args.hf_sjt_path)[self.args.hf_sjt_split]

        # Select SJTs: debug => sample; no-debug => ALL
        if self.args.debug:
            sjt_count = min(self.args.n_sjtsample, len(sjt_ds))
        else:
            sjt_count = len(sjt_ds)

        sjts: List[SJTSpec] = [extract_sjt_spec(sjt_ds[i], i) for i in range(sjt_count)]

        # Base-model mode (no personas)
        if self.args.persona_source == "base_model":
            print("Running SJTs on Base Model without Personas")
            results["base_model"] = self._run_one_persona("base_model", None, None, sjts)
            return ExperimentResults(results)

        # Persona mode
        if not self.args.hf_persona_path:
            raise ValueError("persona_source=hf requires --hf-persona-path")

        if self.args.hf_persona_config:
            persona_ds = load_dataset(self.args.hf_persona_path, name=self.args.hf_persona_config)[self.args.hf_persona_split]
        else:
            persona_ds = load_dataset(self.args.hf_persona_path)[self.args.hf_persona_split]

        if self.args.debug:
            persona_count = min(self.args.n_personasample, len(persona_ds))
        else:
            persona_count = len(persona_ds)

        for i in tqdm(range(persona_count), desc="Personas"):
            persona_row = persona_ds[i]
            persona_uuid = (
                persona_row.get("uuid")
                or persona_row.get("persona_uuid")
                or persona_row.get("id")
                or f"persona_{i}"
            )
            persona_hash = persona_row.get("persona_hash")
            persona_str = persona_row.get("persona_string") or persona_row.get("attributes") or persona_row.get("persona")

            if persona_str is None:
                raise ValueError(f"Persona row missing persona_string/attributes/persona. Keys={list(persona_row.keys())}")

            results[str(persona_uuid)] = self._run_one_persona(str(persona_uuid), persona_hash, str(persona_str), sjts)

        return ExperimentResults(results)


# =========================
# CLI / main
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # vLLM settings
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=70)

    # Templates / SJT behavior
    p.add_argument("--template-key", type=str, default="gpt")
    p.add_argument("--use-persona-template", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--answer-shuffle", action=argparse.BooleanOptionalAction, default=True)

    # Experiment sizing
    p.add_argument("--n-times", type=int, default=1)
    p.add_argument("--n-personasample", type=int, default=10)
    p.add_argument("--n-sjtsample", type=int, default=10)

    # Debug default ON (your preference)
    p.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

    # Persona source
    p.add_argument("--persona-source", type=str, choices=["hf", "base_model"], default="hf")

    p.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
    p.add_argument("--hf-persona-config", type=str, default="expanded")
    p.add_argument("--hf-persona-split", type=str, default="train")

    # SJT dataset
    p.add_argument("--hf-sjt-path", type=str, default="thoughtworks/psychometric_sjts_analysis")
    p.add_argument("--hf-sjt-config", type=str, default="expanded")
    p.add_argument("--hf-sjt-split", type=str, default="train")

    # Output
    p.add_argument("--out-json", type=str, default="outputs/police_sjt_results.json")

    # Hub push (only in no-debug)
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--target-hub-repo-id", type=str, default="thoughtworks/psychometric_personas_responses")
    p.add_argument("--target-hub-config", type=str, default="police_sjt")
    p.add_argument("--target-hub-split", type=str, default="train")

    args = p.parse_args()

    # Debug-mode overrides (exactly as you described)
    if args.debug:
        args.n_times = 1
        args.n_personasample = 10
        args.n_sjtsample = 10
        args.answer_shuffle = True

    return args


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    # NOTE: assumes you already have a running vLLM OpenAI-compatible server,
    # and VLLMServerManager is configured to talk to it.
    mgr = VLLMServerManager()

    runner = SJTResponseRunner(args=args, mgr=mgr)
    results = runner.run()

    # Write local JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results.to_jsonable(), f, ensure_ascii=False)

    print(f"Wrote local results -> {args.out_json}")

    # Push to hub only in no-debug mode
    if args.push_to_hub:
        push_results_to_hub(results, args)
        print(f"Pushed to hub -> {args.target_hub_repo_id} (config={args.target_hub_config})")
    else:
        print("Skipping hub push (debug mode or push disabled).")

if __name__ == "__main__":
    main()
