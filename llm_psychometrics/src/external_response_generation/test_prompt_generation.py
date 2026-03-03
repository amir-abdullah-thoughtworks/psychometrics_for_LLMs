#!/usr/bin/env python3
"""
SJT prompt smoke test (refactored):
- Load personas from HF (same as HEXACO runner pattern)
- Load SJT items from HF
- Render prompts using your existing Jinja templates
- Validate: no missing keys, no leftover {{ }} tokens
- Write rendered prompts to JSONL

Run:
  python scripts/sjt_prompt_smoke_test.py \
    --hf-persona-path thoughtworks/psychometric_personas \
    --hf-persona-config expanded \
    --n-personas 2 \
    --hf-sjt-path <YOUR_SJT_DATASET_ID> \
    --n-sjts 10 \
    --template-key gpt \
    --use-persona-template
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, load_dataset
from jinja2 import Template

# Your repo imports (match your project layout)
from utils_v0 import list_to_str
from prompt_templates.sjt_base_prompt_templates import sjt_base_prompt_templates
from prompt_templates.sjt_persona_prompt_templates import sjt_persona_prompt_templates


SJT_ANSWER_CHOICES: List[str] = ["1", "2", "3", "4", "5", "6"]

DEFAULT_ANSWER_OPTION_ORDERING: List[str] = [
    "honesty_humility_option",
    "emotionality_option",
    "extraversion_option",
    "agreeableness_option",
    "conscientiousness_option",
    "openness_option",
]


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class PromptTemplates:
    key: str
    use_persona_template: bool
    compiled_messages: List[Dict[str, Any]]  # [{"role":..., "content": Template(...)}, ...]


@dataclass(frozen=True)
class PersonaSpec:
    idx: int
    attributes: Optional[str]


@dataclass(frozen=True)
class SJTSpec:
    idx: int
    hash_id: Optional[str]
    question: str
    options: List[str]
    corrected_sjt: Dict[str, Any]
    raw_row: Dict[str, Any]


@dataclass(frozen=True)
class PromptRecord:
    persona_index: int
    persona_present: bool
    sjt_index: int
    hash_id: Optional[str]
    prompt: str
    guided_choices: List[str]
    question: str
    options: List[str]


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Personas (same pattern as HEXACO)
    p.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
    p.add_argument("--hf-persona-config", type=str, default="restricted")
    p.add_argument("--n-personas", type=int, default=1)
    p.add_argument("--attributes-field", type=str, default="persona_string",
                   help="Field in persona dataset that contains persona text")

    # SJTs
    p.add_argument("--hf-sjt-path", type=str, default="thoughtworks/psychometric_sjts_analysis")
    p.add_argument("--hf-sjt-config", type=str, default="debug")
    p.add_argument("--n-sjts", type=int, default=10)

    # Templates
    p.add_argument("--template-key", type=str, default="gpt",
                   help="Key inside sjt_*_prompt_templates dicts")
    p.add_argument("--use-persona-template", action="store_true",
                   help="Use persona SJT template instead of base")

    # Output
    p.add_argument("--out", type=str, default="outputs/sjt_prompts_preview.jsonl")

    return p


# ----------------------------
# Template loading / rendering
# ----------------------------

def compile_message_templates(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert [{"role":..., "content": "...jinja..."}] to compiled templates.
    """
    compiled: List[Dict[str, Any]] = []
    for m in messages:
        compiled.append({"role": m["role"], "content": Template(m["content"])})
    return compiled


def load_prompt_templates(template_key: str, use_persona_template: bool) -> PromptTemplates:
    base_msgs = sjt_base_prompt_templates[template_key]
    persona_msgs = sjt_persona_prompt_templates[template_key]

    chosen = persona_msgs if use_persona_template else base_msgs
    compiled = compile_message_templates(chosen)

    return PromptTemplates(
        key=template_key,
        use_persona_template=use_persona_template,
        compiled_messages=compiled,
    )


def render_messages(compiled_messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, str]]:
    """
    Render compiled Jinja templates into messages with content strings.
    """
    rendered: List[Dict[str, str]] = []
    for m in compiled_messages:
        rendered.append(
            {"role": m["role"], "content": m["content"].render(**kwargs)}
        )
    return rendered


def messages_to_prompt_text(messages: List[Dict[str, str]]) -> str:
    """
    Match your existing linearization convention.
    """
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        lines.append(f"{role}:\n{content}".strip())
    lines.append("ASSISTANT:\n")
    return "\n\n".join(lines)


def validate_prompt_text(prompt: str, where: str) -> None:
    """
    Fail fast if prompt is clearly malformed.
    """
    if "{{" in prompt or "}}" in prompt:
        raise ValueError(f"[{where}] Prompt still contains Jinja tokens '{{{{' / '}}}}'.")
    if not prompt.strip():
        raise ValueError(f"[{where}] Prompt is empty after rendering.")


# ----------------------------
# Dataset loading
# ----------------------------

def load_persona_dataset(hf_path: str, hf_config: str) -> Dataset:
    return load_dataset(hf_path, name=hf_config)["train"]


def load_sjt_dataset(hf_path: str, hf_sjt_config: str) -> Dataset:
    return load_dataset(hf_path, name=hf_sjt_config)["train"]


def iter_personas(ds: Dataset, n: int, attributes_field: str) -> Iterable[PersonaSpec]:
    """
    Yields up to n personas. If n <= 0, yields a single None persona.
    """
    if n <= 0:
        yield PersonaSpec(idx=0, attributes=None)
        return

    n = min(n, len(ds))
    if n == 0:
        yield PersonaSpec(idx=0, attributes=None)
        return

    for i in range(n):
        row = ds[i]
        yield PersonaSpec(idx=i, attributes=row.get(attributes_field))


# ----------------------------
# SJT normalization
# ----------------------------

def normalize_sjt_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the SJT dict from common fields. Adjust if your dataset differs.
    """
    sjt = row.get("corrected_sjt") or row.get("sjt") or row
    if not isinstance(sjt, dict):
        raise ValueError(f"SJT row missing dict payload. Keys: {list(row.keys())}")
    return sjt


def extract_sjt_spec(row: Dict[str, Any], idx: int) -> SJTSpec:
    """
    Convert a raw HF dataset row into the canonical SJTSpec the prompt builder expects.
    """
    sjt = normalize_sjt_row(row)

    question = sjt.get("question")
    if not question:
        raise ValueError(f"SJT payload missing 'question'. Keys: {list(sjt.keys())}")

    options: List[str] = []
    for k in DEFAULT_ANSWER_OPTION_ORDERING:
        if k not in sjt:
            raise ValueError(f"SJT payload missing option key '{k}'. Keys: {list(sjt.keys())}")
        options.append(sjt[k])

    hash_id = row.get("hash_id") or sjt.get("hash_id") or row.get("id")

    return SJTSpec(
        idx=idx,
        hash_id=hash_id,
        question=question,
        options=options,
        corrected_sjt=sjt,
        raw_row=row,
    )


def iter_sjts(ds: Dataset, n: int) -> Iterable[SJTSpec]:
    n = min(max(n, 0), len(ds))
    for i in range(n):
        yield extract_sjt_spec(ds[i], idx=i)


# ----------------------------
# Prompt building
# ----------------------------

def build_prompt_kwargs(persona: PersonaSpec, sjt: SJTSpec) -> Dict[str, Any]:
    """
    Central place to map your template variables.
    If your templates expect different names, change *only this function*.
    """
    return dict(
        attributes=persona.attributes,
        question=sjt.question,
        answer_options=list_to_str(sjt.options),
        sjt=sjt.corrected_sjt,
    )


def build_prompt_record(templates: PromptTemplates, persona: PersonaSpec, sjt: SJTSpec) -> PromptRecord:
    """
    Build a PromptRecord for one (persona, sjt) pair.
    """
    kwargs = build_prompt_kwargs(persona, sjt)
    rendered_msgs = render_messages(templates.compiled_messages, **kwargs)
    prompt_text = messages_to_prompt_text(rendered_msgs)

    validate_prompt_text(prompt_text, where=f"persona[{persona.idx}] sjt[{sjt.idx}]")

    return PromptRecord(
        persona_index=persona.idx,
        persona_present=persona.attributes is not None,
        sjt_index=sjt.idx,
        hash_id=sjt.hash_id,
        prompt=prompt_text,
        guided_choices=SJT_ANSWER_CHOICES,
        question=sjt.question,
        options=sjt.options,
    )


def generate_prompt_records(
    templates: PromptTemplates,
    personas: Iterable[PersonaSpec],
    sjts: Iterable[SJTSpec],
) -> Iterable[PromptRecord]:
    """
    Cross product of personas x sjts.
    """
    sjt_list = list(sjts)  # small for smoke tests; later you can stream
    for p in personas:
        for s in sjt_list:
            yield build_prompt_record(templates, p, s)


# ----------------------------
# Output
# ----------------------------

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: Iterable[PromptRecord], out_path: Path) -> int:
    ensure_parent_dir(out_path)
    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
            wrote += 1
    return wrote


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = build_arg_parser().parse_args()

    templates = load_prompt_templates(args.template_key, args.use_persona_template)

    persona_ds = load_persona_dataset(args.hf_persona_path, args.hf_persona_config)
    sjt_ds = load_sjt_dataset(args.hf_sjt_path, args.hf_sjt_config)

    personas = iter_personas(persona_ds, args.n_personas, args.attributes_field)
    sjts = iter_sjts(sjt_ds, args.n_sjts)

    records = generate_prompt_records(templates, personas, sjts)

    out_path = Path(args.out)
    wrote = write_jsonl(records, out_path)

    print(f"Wrote {wrote} prompts -> {out_path}")


if __name__ == "__main__":
    main()
