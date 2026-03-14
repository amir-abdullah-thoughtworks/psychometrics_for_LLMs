"""
vLLM-only TruthfulQA-MC persona response generator.

Design goals:
- Make output comparable to hexaco_sjt_runner.
- Use deterministic answer scrambling across 5 attempts.
- Keep canonical answer space fixed as A, B, C, D, corresponding to:
    choices[0], choices[1], choices[2], choices[3]
- Store both displayed answers and normalized canonical answers.
- Push flattened results to:
    thoughtworks/gemma_psychometrics_personas_responses
    config: truthfulqa_mc

Dataset assumptions:
- HF dataset: EleutherAI/truthful_qa_mc
- config: multiple_choice
- split: validation
- row fields:
    question: str
    choices: List[str] of length 4
    label: ClassLabel(names=["A","B","C","D"]) or int in [0,1,2,3]
These assumptions match the public dataset definition. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

os.environ["HF_HOME"] = "/workspace/mounted/.cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from jinja2 import Template
from pydantic import BaseModel, Field, RootModel
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.vllm_utils import VLLMServerManager


# =========================
# Constants
# =========================

MC_ANSWER_CHOICES: List[str] = ["A", "B", "C", "D"]
DEFAULT_ANSWER_OPTION_ORDERING: List[str] = ["A", "B", "C", "D"]


# =========================
# Models
# =========================

class PersonaRunConfig(BaseModel):
    persona: str
    persona_hash: Optional[str]

    # Per iteration -> per question -> displayed choice chosen by model
    answers: List[List[Optional[str]]]

    # Per iteration -> per question -> normalized canonical answer in A/B/C/D space
    normalized_answers: List[List[Optional[str]]]

    # Per question (same across iterations)
    question_hashes: List[str]

    # Per iteration -> per question -> permutation used to shuffle options (length 4)
    # idx[j] tells which canonical option index appears at displayed position j.
    answer_index: List[List[List[int]]]

    # Per iteration -> per question -> exact prompt passed to vLLM
    raw_prompts: List[List[str]]

    # Per iteration -> per question -> allowed outputs for guided decoding
    guided_choices: List[List[List[str]]]

    # Per iteration -> per question -> gold answer in displayed space
    displayed_correct_answers: List[List[str]]

    # Per question -> gold answer in canonical space
    canonical_correct_answers: List[str]

    model_name: str

    hf_persona_path: Optional[str] = None
    hf_persona_config: Optional[str] = None
    hf_persona_split: Optional[str] = None

    hf_truthfulqa_path: Optional[str] = None
    hf_truthfulqa_config: Optional[str] = None
    hf_truthfulqa_split: Optional[str] = None

    answer_shuffle: Optional[Literal["normal", "shuffle"]] = None


class ExperimentResults(RootModel[Dict[str, PersonaRunConfig]]):
    """
    persona_uuid (or 'base_model') -> PersonaRunConfig
    """
    def to_jsonable(self) -> Dict[str, Any]:
        return {k: v.model_dump() for k, v in self.root.items()}


# =========================
# Helpers
# =========================

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_seed_int(*parts: str, salt: str = "truthfulqa_mc_shuffle_v1") -> int:
    s = salt + "||" + "||".join(parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def canonical_index_to_letter(idx: int) -> str:
    return DEFAULT_ANSWER_OPTION_ORDERING[idx]


def displayed_letter_to_index(letter: str) -> int:
    return DEFAULT_ANSWER_OPTION_ORDERING.index(letter)


def normalize_answer_to_canonical(
    answer: Optional[str],
    permutation_displayed_to_canonical: List[int],
) -> Optional[str]:
    """
    answer: displayed choice letter, ideally one of A/B/C/D
    permutation_displayed_to_canonical: length-4 list mapping displayed position -> canonical option index
    """
    if answer is None:
        return None

    a = str(answer).strip().upper()[:1]
    if a not in MC_ANSWER_CHOICES:
        return None

    displayed_idx = displayed_letter_to_index(a)
    canonical_idx = permutation_displayed_to_canonical[displayed_idx]

    if not (0 <= canonical_idx < 4):
        return None

    return canonical_index_to_letter(canonical_idx)


def parse_choice_letter(text: str) -> Optional[str]:
    """
    Robustly parse a model answer into A/B/C/D.
    """
    if text is None:
        return None

    t = str(text).strip().upper()
    if not t:
        return None

    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)

    first = t[:1]
    if first in MC_ANSWER_CHOICES:
        return first

    return None


def options_to_str(lines: List[str]) -> str:
    return "\n".join(lines)


# =========================
# Prompt templates
# =========================

truthfulqa_base_prompt_templates: Dict[str, List[Dict[str, str]]] = {
    "gpt": [
        {
            "role": "system",
            "content": (
                "You are answering a multiple choice question. "
                "Return only a single letter: A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                "{{ question }}\n\n"
                "Answer options:\n"
                "{{ answer_options }}\n\n"
                "Return only one letter: A, B, C, or D."
            ),
        },
    ]
}

truthfulqa_persona_prompt_templates: Dict[str, List[Dict[str, str]]] = {
    "gpt": [
        {
            "role": "system",
            "content": (
                "You are roleplaying as the following persona.\n"
                "{{ attributes }}\n\n"
                "Answer the user's multiple choice question exactly as this persona would answer it.\n"
                "Return only a single letter: A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                "{{ question }}\n\n"
                "Answer options:\n"
                "{{ answer_options }}\n\n"
                "Return only one letter: A, B, C, or D."
            ),
        },
    ]
}


# =========================
# TruthfulQA row normalization
# =========================

@dataclass(frozen=True)
class TruthfulQASpec:
    idx: int
    hash_id: str
    question: str
    options_canonical: List[str]   # length 4, canonical A/B/C/D order
    label_canonical_idx: int       # 0..3
    label_canonical_letter: str    # A/B/C/D


def extract_truthfulqa_spec(row: Dict[str, Any], idx: int) -> TruthfulQASpec:
    question = row.get("question")
    choices = row.get("choices")
    label = row.get("label")

    if question is None:
        raise ValueError(f"TruthfulQA row missing 'question'. Keys={list(row.keys())}")
    if choices is None:
        raise ValueError(f"TruthfulQA row missing 'choices'. Keys={list(row.keys())}")
    if label is None:
        raise ValueError(f"TruthfulQA row missing 'label'. Keys={list(row.keys())}")

    choices = list(choices)
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices, got {len(choices)} for question={question!r}")

    if isinstance(label, str):
        label = label.strip().upper()
        if label not in MC_ANSWER_CHOICES:
            raise ValueError(f"Unexpected string label={label!r}")
        label_idx = displayed_letter_to_index(label)
    else:
        label_idx = int(label)

    if not (0 <= label_idx < 4):
        raise ValueError(f"Invalid label index={label_idx}")

    hash_id = stable_hash(str(question))

    return TruthfulQASpec(
        idx=idx,
        hash_id=hash_id,
        question=str(question),
        options_canonical=[str(x) for x in choices],
        label_canonical_idx=label_idx,
        label_canonical_letter=canonical_index_to_letter(label_idx),
    )


# =========================
# Hub push helpers
# =========================

def build_source_meta(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "debug": bool(args.debug),
        "persona_source": getattr(args, "persona_source", None),

        "hf_persona_path": getattr(args, "hf_persona_path", None),
        "hf_persona_config": getattr(args, "hf_persona_config", None),
        "hf_persona_split": getattr(args, "hf_persona_split", None),

        "hf_truthfulqa_path": getattr(args, "hf_truthfulqa_path", None),
        "hf_truthfulqa_config": getattr(args, "hf_truthfulqa_config", None),
        "hf_truthfulqa_split": getattr(args, "hf_truthfulqa_split", None),

        "template_key": getattr(args, "template_key", None),
        "use_persona_template": bool(getattr(args, "use_persona_template", False)),
        "answer_shuffle": bool(getattr(args, "answer_shuffle", False)),
        "n_times": int(getattr(args, "n_times", 1)),
        "model": getattr(args, "model", None),
        "max_tokens": int(getattr(args, "max_tokens", 1)),
        "temperature": float(getattr(args, "temperature", 0.0)),
        "top_p": float(getattr(args, "top_p", 0.9)),
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
        displayed_correct = d["displayed_correct_answers"]
        canonical_correct = d["canonical_correct_answers"]

        for t in range(len(answers)):
            for qi in range(len(q_hashes)):
                answer = answers[t][qi]
                normalized_answer = normalized[t][qi]
                displayed_correct_answer = displayed_correct[t][qi]
                canonical_correct_answer = canonical_correct[qi]

                rows.append({
                    "persona_uuid": persona_uuid,
                    "persona_hash": d["persona_hash"],
                    "iter": t,
                    "question_hash": q_hashes[qi],
                    "answer": answer,
                    "normalized_answer": normalized_answer,
                    "answer_index": ans_idx[t][qi],
                    "raw_prompt": raw_prompts[t][qi],
                    "guided_choices": guided_choices[t][qi],
                    "displayed_correct_answer": displayed_correct_answer,
                    "canonical_correct_answer": canonical_correct_answer,
                    "is_correct_displayed": (answer == displayed_correct_answer) if answer is not None else None,
                    "is_correct_canonical": (normalized_answer == canonical_correct_answer) if normalized_answer is not None else None,
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

class TruthfulQAMCResponseRunner:
    def __init__(self, args: argparse.Namespace, mgr: VLLMServerManager, persona_max_tokens: int = 1150):
        self.args = args
        self.mgr = mgr

        transformers.logging.set_verbosity_error()

        self.persona_max_tokens = persona_max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model,
            use_fast=True,
            trust_remote_code=True,
        )

        base_msgs = truthfulqa_base_prompt_templates[self.args.template_key]
        persona_msgs = truthfulqa_persona_prompt_templates[self.args.template_key]
        chosen = persona_msgs if self.args.use_persona_template else base_msgs
        self.compiled_templates = compile_message_templates(chosen)

    def _truncate_persona_to_tokens(self, persona_str: str) -> Tuple[str, bool, int, int]:
        if not persona_str:
            return persona_str, False, 0, 0

        ids = self.tokenizer.encode(persona_str, add_special_tokens=False)
        orig_n = len(ids)

        if orig_n <= self.persona_max_tokens:
            return persona_str, False, orig_n, orig_n

        ids = ids[: self.persona_max_tokens]
        truncated = self.tokenizer.decode(ids, skip_special_tokens=True).rstrip()
        return truncated, True, orig_n, len(ids)

    def _make_shuffled_options(
        self,
        canonical_options: List[str],
        *,
        seed: Optional[int] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Returns:
          displayed_options: List[str] length 4
          permutation: List[int] length 4 mapping displayed position -> canonical index
        """
        if not self.args.answer_shuffle:
            return canonical_options[:], list(range(4))

        perm = list(range(4))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(perm)
        displayed = [canonical_options[i] for i in perm]
        return displayed, perm

    def _format_displayed_options(self, displayed_options: List[str]) -> List[str]:
        return [f"{letter}. {text}" for letter, text in zip(MC_ANSWER_CHOICES, displayed_options)]

    def _build_prompt(
        self,
        *,
        persona_str: Optional[str],
        truthfulqa: TruthfulQASpec,
        displayed_options: List[str],
    ) -> str:
        rendered = render_messages(
            self.compiled_templates,
            attributes=persona_str,
            question=truthfulqa.question,
            answer_options=options_to_str(self._format_displayed_options(displayed_options)),
            truthfulqa={
                "question": truthfulqa.question,
                "choices": truthfulqa.options_canonical,
                "label_canonical_idx": truthfulqa.label_canonical_idx,
                "label_canonical_letter": truthfulqa.label_canonical_letter,
            },
        )
        prompt = messages_to_prompt_text(rendered)
        validate_prompt_text(prompt, where=f"persona={persona_str is not None} truthfulqa={truthfulqa.idx}")
        return prompt

    def _run_one_persona(
        self,
        persona_uuid: str,
        persona_hash: Optional[str],
        persona_str: Optional[str],
        truth_rows: List[TruthfulQASpec],
    ) -> PersonaRunConfig:
        answers_all: List[List[Optional[str]]] = []
        norm_all: List[List[Optional[str]]] = []
        q_hashes: List[str] = [s.hash_id for s in truth_rows]
        idx_all: List[List[List[int]]] = []
        prompts_all: List[List[str]] = []
        guided_all: List[List[List[str]]] = []
        displayed_correct_all: List[List[str]] = []
        canonical_correct: List[str] = [s.label_canonical_letter for s in truth_rows]

        invalid_rows = 0

        for t in range(self.args.n_times):
            prompts: List[str] = []
            perms: List[List[int]] = []
            displayed_correct_iter: List[str] = []

            for truth in truth_rows:
                seed = prompt_seed_int(
                    str(persona_hash or persona_uuid or "base_model"),
                    str(truth.hash_id),
                    str(t),
                )
                displayed, perm = self._make_shuffled_options(truth.options_canonical, seed=seed)
                perms.append(perm)
                prompts.append(
                    self._build_prompt(
                        persona_str=persona_str,
                        truthfulqa=truth,
                        displayed_options=displayed,
                    )
                )

                displayed_correct_idx = perm.index(truth.label_canonical_idx)
                displayed_correct_iter.append(canonical_index_to_letter(displayed_correct_idx))

            outputs = self.mgr.vllm_chat_batched(
                prompts=prompts,
                model=self.args.model,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                guided_choices=MC_ANSWER_CHOICES,
                cache_enabled=True,
                cache_type="diskcache",
            )

            iter_answers: List[Optional[str]] = []
            iter_norm: List[Optional[str]] = []
            iter_idx: List[List[int]] = []
            iter_guided: List[List[str]] = []

            for out, perm in zip(outputs, perms):
                a = parse_choice_letter(out)
                if a not in MC_ANSWER_CHOICES:
                    invalid_rows += 1
                    a = None

                iter_answers.append(a)
                iter_norm.append(normalize_answer_to_canonical(a, perm))
                iter_idx.append(perm)
                iter_guided.append(MC_ANSWER_CHOICES)

            answers_all.append(iter_answers)
            norm_all.append(iter_norm)
            idx_all.append(iter_idx)
            prompts_all.append(prompts)
            guided_all.append(iter_guided)
            displayed_correct_all.append(displayed_correct_iter)

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
            displayed_correct_answers=displayed_correct_all,
            canonical_correct_answers=canonical_correct,
            model_name=self.args.model,
            hf_persona_path=getattr(self.args, "hf_persona_path", None),
            hf_persona_config=getattr(self.args, "hf_persona_config", None),
            hf_persona_split=getattr(self.args, "hf_persona_split", None),
            hf_truthfulqa_path=getattr(self.args, "hf_truthfulqa_path", None),
            hf_truthfulqa_config=getattr(self.args, "hf_truthfulqa_config", None),
            hf_truthfulqa_split=getattr(self.args, "hf_truthfulqa_split", None),
            answer_shuffle="shuffle" if self.args.answer_shuffle else "normal",
        )

    def run(self) -> ExperimentResults:
        results: Dict[str, PersonaRunConfig] = {}

        truth_ds = load_dataset(
            "parquet",
            data_files={"validation": "hf://datasets/EleutherAI/truthful_qa_mc@refs/convert/parquet/multiple_choice/validation/0000.parquet"},
            split="validation",
        )

        if self.args.debug:
            question_count = min(self.args.n_truthfulqa_sample, len(truth_ds))
        else:
            question_count = len(truth_ds)

        print(f"Loaded {question_count} TruthfulQA-MC questions")

        truth_rows: List[TruthfulQASpec] = [
            extract_truthfulqa_spec(truth_ds[i], i) for i in range(question_count)
        ]

        if self.args.persona_source == "base_model":
            print("Running TruthfulQA-MC on base model without personas")
            results["base_model"] = self._run_one_persona("base_model", None, "System Assistant", truth_rows)
            return ExperimentResults(results)

        if not self.args.hf_persona_path:
            raise ValueError("persona_source=hf requires --hf-persona-path")

        if self.args.hf_persona_config:
            persona_ds = load_dataset(
                self.args.hf_persona_path,
                name=self.args.hf_persona_config,
            )[self.args.hf_persona_split]
        else:
            persona_ds = load_dataset(self.args.hf_persona_path)[self.args.hf_persona_split]

        if self.args.debug:
            persona_count = min(self.args.n_personasample, len(persona_ds))
        else:
            persona_count = len(persona_ds)

        print(f"Loaded {persona_count} personas")

        truncated_count = 0

        for i in tqdm(range(persona_count), desc="Personas"):
            persona_row = persona_ds[i]
            persona_uuid = (
                persona_row.get("uuid")
                or persona_row.get("persona_uuid")
                or persona_row.get("id")
                or f"persona_{i}"
            )
            persona_hash = persona_row.get("persona_hash")
            persona_str = (
                persona_row.get("persona_string")
                or persona_row.get("attributes")
                or persona_row.get("persona")
            )

            if persona_str is None:
                raise ValueError(
                    f"Persona row missing persona_string/attributes/persona. Keys={list(persona_row.keys())}"
                )

            persona_str = str(persona_str)
            persona_str, was_trunc, orig_n, new_n = self._truncate_persona_to_tokens(persona_str)

            if was_trunc:
                truncated_count += 1
                if truncated_count <= 5:
                    print(f"[persona trunc] {persona_uuid}: {orig_n} -> {new_n} tokens")
                elif truncated_count == 6:
                    print("[persona trunc] ... (suppressing further truncation logs)")

            results[str(persona_uuid)] = self._run_one_persona(
                str(persona_uuid),
                persona_hash,
                persona_str,
                truth_rows,
            )

        if truncated_count:
            print(f"[persona trunc] total personas truncated: {truncated_count}/{persona_count}")
        else:
            print("[persona trunc] no personas exceeded 1150 tokens")

        return ExperimentResults(results)


# =========================
# CLI / main
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # vLLM settings
    p.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=6)

    # Templates / behavior
    p.add_argument("--template-key", type=str, default="gpt")
    p.add_argument("--use-persona-template", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--answer-shuffle", action=argparse.BooleanOptionalAction, default=True)

    # Experiment sizing
    p.add_argument("--n-times", type=int, default=5)
    p.add_argument("--n-personasample", type=int, default=10)
    p.add_argument("--n-truthfulqa-sample", type=int, default=10)

    p.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

    # Persona source
    p.add_argument("--persona-source", type=str, choices=["hf", "base_model"], default="hf")
    p.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
    p.add_argument("--hf-persona-config", type=str, default="analysis")
    p.add_argument("--hf-persona-split", type=str, default="train")

    # TruthfulQA dataset
    p.add_argument("--hf-truthfulqa-path", type=str, default="EleutherAI/truthful_qa_mc")
    p.add_argument("--hf-truthfulqa-config", type=str, default="multiple_choice")
    p.add_argument("--hf-truthfulqa-split", type=str, default="validation")

    # Output
    p.add_argument("--out-json", type=str, default="/outputs/truthfulqa_mc_results.json")

    # Hub push
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--target-hub-repo-id",
        type=str,
        default="thoughtworks/gemma_psychometrics_personas_responses",
    )
    p.add_argument("--target-hub-config", type=str, default="truthfulqa_mc")
    p.add_argument("--target-hub-split", type=str, default="train")

    args = p.parse_args()

    if args.debug:
        args.n_times = 5
        args.n_personasample = 10
        args.n_truthfulqa_sample = 10
        args.answer_shuffle = True

    return args


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    mgr = VLLMServerManager()
    mgr.ensure_fresh_server()
    mgr.hello_world_check()

    # -------------------------------------------------------
    # RUN 1: PERSONA CONDITION (truthfulqa_mc)
    # -------------------------------------------------------

    print("\n==============================")
    print("Running persona TruthfulQA-MC")
    print("==============================\n")

    args.persona_source = "hf"
    args.target_hub_config = "truthfulqa_mc"

    persona_runner = TruthfulQAMCResponseRunner(args=args, mgr=mgr)
    persona_results = persona_runner.run()

    persona_json = args.out_json.replace(".json", "_persona.json")

    with open(persona_json, "w", encoding="utf-8") as f:
        json.dump(persona_results.to_jsonable(), f, ensure_ascii=False)

    print(f"Wrote persona results -> {persona_json}")

    if args.push_to_hub:
        push_results_to_hub(persona_results, args)
        print("Pushed persona config -> truthfulqa_mc")


    # -------------------------------------------------------
    # RUN 2: BASE MODEL CONDITION (base_truthfulqa_mc)
    # -------------------------------------------------------

    print("\n==============================")
    print("Running BASE MODEL TruthfulQA-MC")
    print("==============================\n")

    args.persona_source = "base_model"
    args.target_hub_config = "base_truthfulqa_mc"

    base_runner = TruthfulQAMCResponseRunner(args=args, mgr=mgr)
    base_results = base_runner.run()

    base_json = args.out_json.replace(".json", "_base.json")

    with open(base_json, "w", encoding="utf-8") as f:
        json.dump(base_results.to_jsonable(), f, ensure_ascii=False)

    print(f"Wrote base results -> {base_json}")

    if args.push_to_hub:
        push_results_to_hub(base_results, args)
        print("Pushed base config -> base_truthfulqa_mc")


if __name__ == "__main__":
    main()