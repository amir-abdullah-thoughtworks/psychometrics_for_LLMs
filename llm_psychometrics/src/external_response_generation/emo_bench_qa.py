"""
vLLM-only EmoBench persona response generator.

Design goals:
- Match the same flattened-output pattern as the TruthfulQA-MC runner.
- Evaluate both EmoBench subsets together:
    * emotional_application (4 choices)
    * emotional_understanding (emotion_choices only, 6 choices)
- Use deterministic answer scrambling across 5 attempts.
- Keep canonical answer space fixed by displayed letter positions.
- Support variable answer counts across subsets:
    * emotional_application -> 4 choices
    * emotional_understanding -> 6 choices
- Store both displayed answers and normalized canonical answers.
- Push flattened results to:
    thoughtworks/gemma_psychometrics_personas_responses
    configs:
        - analysis_emo_bench   (persona-conditioned)
        - base_emo_bench       (base model)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

os.environ["HF_HOME"] = "/workspace/mounted/.cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from jinja2 import Template
from pydantic import BaseModel, RootModel
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.vllm_utils import VLLMServerManager


# =========================
# Constants
# =========================

DISPLAY_LETTERS: List[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_EMOBENCH_CONFIGS: List[str] = ["emotional_application", "emotional_understanding"]


# =========================
# Models
# =========================

class PersonaRunConfig(BaseModel):
    persona: str
    persona_hash: Optional[str]

    # Per iteration -> per question -> displayed choice chosen by model
    answers: List[List[Optional[str]]]

    # Per iteration -> per question -> normalized canonical answer in letter space
    normalized_answers: List[List[Optional[str]]]

    # Per question (same across iterations)
    question_hashes: List[str]

    # Per iteration -> per question -> permutation used to shuffle options
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

    hf_emo_bench_path: Optional[str] = None
    hf_emo_bench_configs: Optional[List[str]] = None
    hf_emo_bench_split: Optional[str] = None

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


def prompt_seed_int(*parts: str, salt: str = "emo_bench_shuffle_v2") -> int:
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


def letter_for_index(idx: int) -> str:
    if not (0 <= idx < len(DISPLAY_LETTERS)):
        raise ValueError(f"Index out of range for display letters: {idx}")
    return DISPLAY_LETTERS[idx]


def displayed_letter_to_index(letter: str) -> int:
    return DISPLAY_LETTERS.index(letter)


def normalize_answer_to_canonical(
    answer: Optional[str],
    permutation_displayed_to_canonical: List[int],
) -> Optional[str]:
    """
    answer: displayed choice letter
    permutation_displayed_to_canonical: list mapping displayed position -> canonical option index
    """
    if answer is None:
        return None

    a = str(answer).strip().upper()[:1]
    allowed_letters = DISPLAY_LETTERS[: len(permutation_displayed_to_canonical)]
    if a not in allowed_letters:
        return None

    displayed_idx = displayed_letter_to_index(a)
    canonical_idx = permutation_displayed_to_canonical[displayed_idx]

    if not (0 <= canonical_idx < len(permutation_displayed_to_canonical)):
        return None

    return letter_for_index(canonical_idx)


def parse_choice_letter(text: str, n_choices: int) -> Optional[str]:
    """
    Robustly parse a model answer into one of the first n choice letters.
    """
    if text is None:
        return None

    allowed_letters = DISPLAY_LETTERS[:n_choices]
    t = str(text).strip().upper()
    if not t:
        return None

    pattern = r"\b([" + "".join(allowed_letters) + r"])\b"
    m = re.search(pattern, t)
    if m:
        return m.group(1)

    first = t[:1]
    if first in allowed_letters:
        return first

    return None


def options_to_str(lines: List[str]) -> str:
    return "\n".join(lines)


def first_present(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


# =========================
# Prompt templates
# =========================

emo_bench_base_prompt_templates: Dict[str, List[Dict[str, str]]] = {
    "gpt": [
        {
            "role": "system",
            "content": (
                "You are answering a multiple choice emotional reasoning question. "
                "Read the scenario carefully and choose the best answer option. "
                "Return only a single letter corresponding to one answer option."
            ),
        },
        {
            "role": "user",
            "content": (
                "Dataset subset: {{ subset_name }}\n"
                "Language: {{ language }}\n"
                "Category: {{ category }}\n"
                "Question type: {{ question_type }}\n"
                "Subject: {{ subject }}\n\n"
                "Scenario:\n"
                "{{ scenario }}\n\n"
                "Answer options:\n"
                "{{ answer_options }}\n\n"
                "Return only one letter."
            ),
        },
    ]
}

emo_bench_persona_prompt_templates: Dict[str, List[Dict[str, str]]] = {
    "gpt": [
        {
            "role": "system",
            "content": (
                "You are roleplaying as the following persona.\n"
                "{{ attributes }}\n\n"
                "Answer the user's multiple choice emotional reasoning question exactly as this persona would answer it.\n"
                "Read the scenario carefully and select the option this persona would most likely choose.\n"
                "Return only a single letter corresponding to one answer option."
            ),
        },
        {
            "role": "user",
            "content": (
                "Dataset subset: {{ subset_name }}\n"
                "Language: {{ language }}\n"
                "Category: {{ category }}\n"
                "Question type: {{ question_type }}\n"
                "Subject: {{ subject }}\n\n"
                "Scenario:\n"
                "{{ scenario }}\n\n"
                "Answer options:\n"
                "{{ answer_options }}\n\n"
                "Return only one letter."
            ),
        },
    ]
}


# =========================
# EmoBench row normalization
# =========================

@dataclass(frozen=True)
class EmoBenchSpec:
    idx: int
    subset_name: str
    question_source: str
    qid: str
    hash_id: str
    language: str
    category: str
    question_type: str
    scenario: str
    subject: str
    options_canonical: List[str]
    label_canonical_idx: int
    label_canonical_letter: str
    label_text: str
    n_choices: int


def extract_emo_bench_specs(row: Dict[str, Any], idx: int, subset_name: str) -> List[EmoBenchSpec]:
    qid = first_present(row, ["qid", "id"], default=str(idx))
    language = str(first_present(row, ["language"], default="unknown"))
    scenario = first_present(row, ["scenario"])
    subject = str(first_present(row, ["subject"], default=""))

    if scenario is None:
        raise ValueError(f"EmoBench row missing 'scenario'. Keys={list(row.keys())}")

    def _build_spec(
        *,
        local_idx: int,
        question_source: str,
        category: str,
        question_type: str,
        choices_raw: Any,
        label_raw: Any,
        expected_n_choices: int,
    ) -> EmoBenchSpec:
        if choices_raw is None:
            raise ValueError(
                f"EmoBench row missing choices for question_source={question_source}. Keys={list(row.keys())}"
            )
        if label_raw is None:
            raise ValueError(
                f"EmoBench row missing label for question_source={question_source}. Keys={list(row.keys())}"
            )

        choices = [str(x) for x in list(choices_raw)]
        if len(choices) != expected_n_choices:
            raise ValueError(
                f"Expected {expected_n_choices} choices, got {len(choices)} for "
                f"qid={qid!r}, question_source={question_source!r}"
            )

        label_text = str(label_raw).strip()

        label_idx: Optional[int] = None
        for i, choice in enumerate(choices):
            if choice.strip() == label_text:
                label_idx = i
                break

        if label_idx is None:
            raise ValueError(
                f"Could not match label text to one of the {expected_n_choices} choices for "
                f"qid={qid!r}, question_source={question_source!r}. "
                f"label={label_text!r} choices={choices!r}"
            )

        hash_basis = json.dumps(
            {
                "subset_name": subset_name,
                "question_source": question_source,
                "qid": str(qid),
                "language": language,
                "category": category,
                "question_type": question_type,
                "subject": subject,
                "scenario": str(scenario),
                "choices": choices,
                "label": label_text,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        hash_id = stable_hash(hash_basis)

        return EmoBenchSpec(
            idx=local_idx,
            subset_name=subset_name,
            question_source=question_source,
            qid=str(qid),
            hash_id=hash_id,
            language=language,
            category=category,
            question_type=question_type,
            scenario=str(scenario),
            subject=subject,
            options_canonical=choices,
            label_canonical_idx=label_idx,
            label_canonical_letter=letter_for_index(label_idx),
            label_text=label_text,
            n_choices=expected_n_choices,
        )

    if subset_name == "emotional_application":
        category = str(first_present(row, ["category", "coarse_category"], default="unknown"))
        question_type = str(
            first_present(row, ["question type", "question_type", "finegrained_category"], default="unknown")
        )

        return [
            _build_spec(
                local_idx=idx,
                question_source="emotional_application",
                category=category,
                question_type=question_type,
                choices_raw=first_present(row, ["choices"]),
                label_raw=first_present(row, ["label"]),
                expected_n_choices=4,
            )
        ]

    if subset_name == "emotional_understanding":
        coarse = str(first_present(row, ["coarse_category"], default="unknown"))
        fine = str(first_present(row, ["finegrained_category"], default="unknown"))
        category = f"{coarse} | {fine}"

        emotion_choices = first_present(row, ["emotion_choices"])
        emotion_label = first_present(row, ["emotion_label"])

        return [
            _build_spec(
                local_idx=idx,
                question_source="emotional_understanding_emotion",
                category=category,
                question_type="emotion",
                choices_raw=emotion_choices,
                label_raw=emotion_label,
                expected_n_choices=6,
            )
        ]

    raise ValueError(f"Unknown EmoBench subset_name={subset_name!r}")


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

        "hf_emo_bench_path": getattr(args, "hf_emo_bench_path", None),
        "hf_emo_bench_configs": list(getattr(args, "hf_emo_bench_configs", []) or []),
        "hf_emo_bench_split": getattr(args, "hf_emo_bench_split", None),
        "hf_emo_bench_language_filter": getattr(args, "hf_emo_bench_language_filter", None),

        "template_key": getattr(args, "template_key", None),
        "use_persona_template": bool(getattr(args, "use_persona_template", False)),
        "answer_shuffle": bool(getattr(args, "answer_shuffle", False)),
        "n_times": int(getattr(args, "n_times", 1)),
        "model": getattr(args, "model", None),
        "max_tokens": int(getattr(args, "max_tokens", 1)),
        "temperature": float(getattr(args, "temperature", 0.0)),
        "top_p": float(getattr(args, "top_p", 0.9)),
    }


def flatten_results_for_hub(
    exp: ExperimentResults,
    source_meta: Dict[str, Any],
    emo_rows: List[EmoBenchSpec],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    question_meta = [
        {
            "subset_name": s.subset_name,
            "question_source": s.question_source,
            "qid": s.qid,
            "language": s.language,
            "category": s.category,
            "question_type": s.question_type,
            "subject": s.subject,
            "scenario": s.scenario,
            "canonical_choices": s.options_canonical,
            "label_text": s.label_text,
            "n_choices": s.n_choices,
        }
        for s in emo_rows
    ]

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
                meta = question_meta[qi]
                answer = answers[t][qi]
                normalized_answer = normalized[t][qi]
                displayed_correct_answer = displayed_correct[t][qi]
                canonical_correct_answer = canonical_correct[qi]

                rows.append({
                    "persona_uuid": persona_uuid,
                    "persona_hash": d["persona_hash"],
                    "iter": t,

                    "question_hash": q_hashes[qi],
                    "subset_name": meta["subset_name"],
                    "question_source": meta["question_source"],
                    "qid": meta["qid"],
                    "language": meta["language"],
                    "category": meta["category"],
                    "question_type": meta["question_type"],
                    "subject": meta["subject"],
                    "scenario": meta["scenario"],
                    "canonical_choices": meta["canonical_choices"],
                    "label_text": meta["label_text"],
                    "n_choices": meta["n_choices"],

                    "answer": answer,
                    "normalized_answer": normalized_answer,
                    "answer_index": ans_idx[t][qi],
                    "raw_prompt": raw_prompts[t][qi],
                    "guided_choices": guided_choices[t][qi],
                    "displayed_correct_answer": displayed_correct_answer,
                    "canonical_correct_answer": canonical_correct_answer,

                    "is_correct": (normalized_answer == canonical_correct_answer) if normalized_answer is not None else None,
                    "is_correct_displayed": (answer == displayed_correct_answer) if answer is not None else None,
                    "is_correct_canonical": (normalized_answer == canonical_correct_answer) if normalized_answer is not None else None,

                    "model_name": d["model_name"],
                    "run_timestamp_utc": _utc_now_iso(),
                    **source_meta,
                })

    return rows


def push_results_to_hub(
    exp: ExperimentResults,
    args: argparse.Namespace,
    emo_rows: List[EmoBenchSpec],
) -> None:
    source_meta = build_source_meta(args)
    rows = flatten_results_for_hub(exp, source_meta, emo_rows)

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
            "unique_subsets": len(set(ds["subset_name"])) if len(ds) else 0,
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

class EmoBenchQAResponseRunner:
    def __init__(self, args: argparse.Namespace, mgr: VLLMServerManager, persona_max_tokens: int = 1150):
        self.args = args
        self.mgr = mgr
        self.persona_max_tokens = persona_max_tokens

        transformers.logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model,
            use_fast=True,
            trust_remote_code=True,
        )

        base_msgs = emo_bench_base_prompt_templates[self.args.template_key]
        persona_msgs = emo_bench_persona_prompt_templates[self.args.template_key]
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
          displayed_options: length N
          permutation: length N mapping displayed position -> canonical index
        """
        n = len(canonical_options)
        if not self.args.answer_shuffle:
            return canonical_options[:], list(range(n))

        perm = list(range(n))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(perm)
        displayed = [canonical_options[i] for i in perm]
        return displayed, perm

    def _format_displayed_options(self, displayed_options: List[str]) -> List[str]:
        n = len(displayed_options)
        letters = DISPLAY_LETTERS[:n]
        return [f"{letter}. {text}" for letter, text in zip(letters, displayed_options)]

    def _build_prompt(
        self,
        *,
        persona_str: Optional[str],
        emo: EmoBenchSpec,
        displayed_options: List[str],
    ) -> str:
        rendered = render_messages(
            self.compiled_templates,
            attributes=persona_str,
            subset_name=emo.subset_name,
            language=emo.language,
            category=emo.category,
            question_type=emo.question_type,
            subject=emo.subject,
            scenario=emo.scenario,
            answer_options=options_to_str(self._format_displayed_options(displayed_options)),
            emo_bench={
                "subset_name": emo.subset_name,
                "qid": emo.qid,
                "language": emo.language,
                "category": emo.category,
                "question_type": emo.question_type,
                "subject": emo.subject,
                "scenario": emo.scenario,
                "choices": emo.options_canonical,
                "label_canonical_idx": emo.label_canonical_idx,
                "label_canonical_letter": emo.label_canonical_letter,
                "label_text": emo.label_text,
                "n_choices": emo.n_choices,
            },
        )
        prompt = messages_to_prompt_text(rendered)
        validate_prompt_text(prompt, where=f"persona={persona_str is not None} emo_qid={emo.qid}")
        return prompt

    def _run_one_persona(
        self,
        persona_uuid: str,
        persona_hash: Optional[str],
        persona_str: Optional[str],
        emo_rows: List[EmoBenchSpec],
    ) -> PersonaRunConfig:
        answers_all: List[List[Optional[str]]] = []
        norm_all: List[List[Optional[str]]] = []
        q_hashes: List[str] = [s.hash_id for s in emo_rows]
        idx_all: List[List[List[int]]] = []
        prompts_all: List[List[str]] = []
        guided_all: List[List[List[str]]] = []
        displayed_correct_all: List[List[str]] = []
        canonical_correct: List[str] = [s.label_canonical_letter for s in emo_rows]

        invalid_rows = 0

        for t in range(self.args.n_times):
            prompts: List[str] = []
            perms: List[List[int]] = []
            displayed_correct_iter: List[str] = []
            guided_choices_iter: List[List[str]] = []

            for emo in emo_rows:
                seed = prompt_seed_int(
                    str(persona_hash or persona_uuid or "base_model"),
                    str(emo.hash_id),
                    str(t),
                )
                displayed, perm = self._make_shuffled_options(emo.options_canonical, seed=seed)
                perms.append(perm)
                prompts.append(
                    self._build_prompt(
                        persona_str=persona_str,
                        emo=emo,
                        displayed_options=displayed,
                    )
                )

                n = emo.n_choices
                guided_letters = DISPLAY_LETTERS[:n]
                guided_choices_iter.append(guided_letters)

                displayed_correct_idx = perm.index(emo.label_canonical_idx)
                displayed_correct_iter.append(letter_for_index(displayed_correct_idx))

            outputs = self.mgr.vllm_chat_batched(
                prompts=prompts,
                model=self.args.model,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                guided_choices=guided_choices_iter,
                cache_enabled=False,
                cache_type="diskcache",
                default_guided_choice="A"
            )

            iter_answers: List[Optional[str]] = []
            iter_norm: List[Optional[str]] = []
            iter_idx: List[List[int]] = []
            iter_guided: List[List[str]] = []

            for out, perm, emo, guided in zip(outputs, perms, emo_rows, guided_choices_iter):
                a = parse_choice_letter(out, emo.n_choices)
                if a not in guided:
                    invalid_rows += 1
                    a = None

                iter_answers.append(a)
                iter_norm.append(normalize_answer_to_canonical(a, perm))
                iter_idx.append(perm)
                iter_guided.append(guided)

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
            hf_emo_bench_path=getattr(self.args, "hf_emo_bench_path", None),
            hf_emo_bench_configs=list(getattr(self.args, "hf_emo_bench_configs", []) or []),
            hf_emo_bench_split=getattr(self.args, "hf_emo_bench_split", None),
            answer_shuffle="shuffle" if self.args.answer_shuffle else "normal",
        )

    def load_emo_rows(self) -> List[EmoBenchSpec]:
        all_specs: List[EmoBenchSpec] = []
        ds_parts = []

        for cfg_name in self.args.hf_emo_bench_configs:
            ds = load_dataset(
                self.args.hf_emo_bench_path,
                name=cfg_name,
                split=self.args.hf_emo_bench_split,
            )

            if self.args.hf_emo_bench_language_filter:
                target_lang = self.args.hf_emo_bench_language_filter
                ds = ds.filter(lambda x: str(x.get("language", "")).lower() == target_lang.lower())

            ds_parts.append((cfg_name, ds))

        total_rows = sum(len(ds) for _, ds in ds_parts)
        print(f"Loaded {total_rows} EmoBench rows across configs={self.args.hf_emo_bench_configs}")

        idx = 0
        for cfg_name, ds in ds_parts:
            limit = len(ds)
            if self.args.debug:
                limit = min(limit, self.args.n_emo_bench_sample_per_config)

            print(f"Using {limit} rows from EmoBench config={cfg_name}")

            for i in range(limit):
                specs = extract_emo_bench_specs(ds[i], idx=idx, subset_name=cfg_name)
                all_specs.extend(specs)
                idx += len(specs)

        print(f"Expanded to {len(all_specs)} total QA items after schema normalization")
        counts = Counter(s.question_source for s in all_specs)
        print(f"Question source counts: {dict(counts)}")

        return all_specs

    def run(self) -> Tuple[ExperimentResults, List[EmoBenchSpec]]:
        results: Dict[str, PersonaRunConfig] = {}
        emo_rows = self.load_emo_rows()

        if self.args.persona_source == "base_model":
            print("Running EmoBench-QA on base model without personas")
            results["base_model"] = self._run_one_persona("base_model", None, "System Assistant", emo_rows)
            return ExperimentResults(results), emo_rows

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
                emo_rows,
            )

        if truncated_count:
            print(f"[persona trunc] total personas truncated: {truncated_count}/{persona_count}")
        else:
            print("[persona trunc] no personas exceeded 1150 tokens")

        return ExperimentResults(results), emo_rows


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
    p.add_argument("--num-workers", type=int, default=50)

    # Templates / behavior
    p.add_argument("--template-key", type=str, default="gpt")
    p.add_argument("--use-persona-template", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--answer-shuffle", action=argparse.BooleanOptionalAction, default=True)

    # Experiment sizing
    p.add_argument("--n-times", type=int, default=5)
    p.add_argument("--n-personasample", type=int, default=1000)
    p.add_argument("--n-emo-bench-sample-per-config", type=int, default=5000)

    p.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

    # Persona source
    p.add_argument("--persona-source", type=str, choices=["hf", "base_model"], default="hf")
    p.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
    p.add_argument("--hf-persona-config", type=str, default="analysis")
    p.add_argument("--hf-persona-split", type=str, default="train")

    # EmoBench dataset
    p.add_argument("--hf-emo-bench-path", type=str, default="SahandSab/EmoBench")
    p.add_argument(
        "--hf-emo-bench-configs",
        type=str,
        nargs="+",
        default=DEFAULT_EMOBENCH_CONFIGS,
    )
    p.add_argument("--hf-emo-bench-split", type=str, default="train")
    p.add_argument("--hf-emo-bench-language-filter", type=str, default="en")

    # Output
    p.add_argument("--out-json", type=str, default="/outputs/emo_bench_results.json")

    # Hub push
    p.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--target-hub-repo-id",
        type=str,
        default="thoughtworks/gemma_psychometrics_personas_responses",
    )
    p.add_argument("--target-hub-config", type=str, default="analysis_emo_bench")
    p.add_argument("--target-hub-split", type=str, default="train")

    args = p.parse_args()

    if args.debug:
        args.n_times = 5
        args.n_personasample = 10
        args.n_emo_bench_sample_per_config = 10
        args.answer_shuffle = True

    return args


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)


    mgr = VLLMServerManager(port=9000)
    # mgr.ensure_fresh_server()

    mgr.hello_world_check()

    # -------------------------------------------------------
    # RUN 1: PERSONA CONDITION (analysis_emo_bench)
    # -------------------------------------------------------

    print("\n==============================")
    print("Running persona EmoBench-QA")
    print("==============================\n")

    args.persona_source = "hf"
    args.target_hub_config = "analysis_emo_bench"

    persona_runner = EmoBenchQAResponseRunner(args=args, mgr=mgr)
    persona_results, emo_rows = persona_runner.run()

    persona_json = args.out_json.replace(".json", "_persona.json")

    with open(persona_json, "w", encoding="utf-8") as f:
        json.dump(persona_results.to_jsonable(), f, ensure_ascii=False)

    print(f"Wrote persona results -> {persona_json}")

    if args.push_to_hub:
        push_results_to_hub(persona_results, args, emo_rows)
        print("Pushed persona config -> analysis_emo_bench")

    # -------------------------------------------------------
    # RUN 2: BASE MODEL CONDITION (base_emo_bench)
    # -------------------------------------------------------

    print("\n==============================")
    print("Running BASE MODEL EmoBench-QA")
    print("==============================\n")

    args.persona_source = "base_model"
    args.target_hub_config = "base_emo_bench"

    base_runner = EmoBenchQAResponseRunner(args=args, mgr=mgr)
    base_results, emo_rows = base_runner.run()

    base_json = args.out_json.replace(".json", "_base.json")

    with open(base_json, "w", encoding="utf-8") as f:
        json.dump(base_results.to_jsonable(), f, ensure_ascii=False)

    print(f"Wrote base results -> {base_json}")

    if args.push_to_hub:
        push_results_to_hub(base_results, args, emo_rows)
        print("Pushed base config -> base_emo_bench")


if __name__ == "__main__":
    main()
