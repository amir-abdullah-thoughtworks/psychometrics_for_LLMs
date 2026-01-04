"""
vLLM-only, class-based SJT response generator (single consolidated file).

This file is intentionally patterned after `hexaco_response_generator.py` in the same repo.

Key behavior:
- vLLM only (no transformers/OpenAI client usage).
- Does NOT start/boot/kill any vLLM server.
- Uses an externally created VLLMServerManager passed from main().
- Uses mgr.vllm_chat_batched(prompts=..., guided_choices=...) with per-prompt constraints.
- Optional per-answer shuffling of answer-option ordering (iid) when --answer-shuffle is enabled.
- Pulls Personas from the *same* HF persona dataset/config used by the HEXACO runner.
- Pushes results to the *same* HF responses repo_id but under a different HF dataset *config* (default: "sjt").

Expected repo layout (mirrors HEXACO runner assumptions):
root/
  configs/
  data/
  src/
    external_response_generation/
      sjt_response_generator/   <-- this file lives here (or similar)
  psychometric_tests/

Notes:
- Paths are resolved relative to this file, not CWD.
- Requires the repo's `root/src` to be importable (we insert it into sys.path).

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from jinja2 import Template  # type: ignore
from pydantic import BaseModel  # type: ignore
from tqdm import tqdm  # type: ignore

from prompt_templates.sjt_persona_prompt_templates import sjt_persona_prompt_templates

# =========================
# Paths (match HEXACO runner style)
# =========================

THIS_FILE = Path(__file__).resolve()
# sjt_response_generator -> external_response_generation -> src -> root
ROOT_DIR = THIS_FILE.parents[3]

DATA_DIR = ROOT_DIR / "data"
PSYCHOMETRIC_DIR = ROOT_DIR / "psychometric_tests"
SRC_DIR = ROOT_DIR / "src"
CONFIGS_DIR = ROOT_DIR / "configs"

# Ensure imports work no matter where script is run from
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Custom imports (these live under root/src) ---
from utils_v0 import list_to_str  # type: ignore
from utils.vllm_utils import VLLMServerManager  # type: ignore
from prompt_templates.sjt_base_prompt_templates import sjt_base_prompt_templates  # type: ignore
from prompt_templates.sjt_persona_prompt_templates import sjt_persona_prompt_templates  # type: ignore


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

def render_llm_messages(template_messages, **kwargs):
    rendered =  [
        {"role": msg["role"], "content": msg["content"].render(**kwargs)}
        for msg in template_messages
    ]
    return rendered


def build_prompts(sjt_template, question_batch, answer_index, batching=False,
                        persona_str=None, answer_shuffle=False):
    """Generate a batch of questions."""
    if batching:
        raise NotImplementedError("Batching not implemented")

    # Non-batched
    prompt_list = []
    hash_list = []
    answer_index_list = []
    for sjt_dict in question_batch:

        answer_index_copy = answer_index.copy()
        if answer_shuffle:

            random.shuffle(answer_index_copy)
        answer_index_list.append(list(answer_index_copy))

        sjt = sjt_dict['corrected_sjt']
        # print("raw sjt", sjt)
        answer_options = [sjt[key] for key in DEFAULT_ANSWER_OPTION_ORDERING]
        # print("answer options after default ordering ", answer_options)
        answer_options = [answer_options[idx] for idx in answer_index_copy]
        # print("final answer options", answer_options)
        question = sjt['question']

        prompt = render_llm_messages(
            sjt_template,
            attributes=persona_str,
            question=question,
            answer_options=list_to_str(answer_options)
        )
        # print("final prompt",prompt)
        prompt_list.append(prompt)
        hash_list.append(sjt_dict['hash_id'])

def load_prompt_templates(model_name):
    """
    Use chat-style (OpenAI/vLLM) templates for all models to avoid Outlines/HF.
    Signature unchanged.
    """
    # We ignore non-GPT branches to keep everything in chat format.
    print("Loading Prompt Templates for Chat-style (OpenAI/vLLM) models")
    base_templates = sjt_base_prompt_templates["gpt"]
    persona_templates = sjt_persona_prompt_templates["gpt"]

    # compile GPT messages into Jinja templates
    def compile_message_templates(messages):
        return [
            {"role": msg["role"], "content": Template(msg["content"])}
            for msg in messages
        ]

    base_sjt_template = compile_message_templates(base_templates)
    persona_sjt_template = compile_message_templates(persona_templates)
    return base_sjt_template, persona_sjt_template


# =========================
# Result schemas (row-wise push mirrors HEXACO runner)
# =========================

class PersonaRunConfig(BaseModel):
    persona: str
    persona_hash: str

    # Per iteration -> per question -> answers (normalized to "1".."6" when possible)
    answers: List[List[str]]

    # Per iteration -> per question -> hashes of the SJT items (stable across runs)
    question_hashes: List[List[str]]

    # Per iteration -> per question -> list[int] permutation of DEFAULT_ANSWER_OPTION_ORDERING
    answer_index: List[List[List[int]]]

    # Per iteration -> per question -> exact prompt passed to vLLM
    raw_prompts: List[List[str]]

    # Per iteration -> per question -> allowed outputs for guided decoding
    guided_choices: List[List[List[str]]]

    model_name: str

    # Audit fields
    hf_persona_path: Optional[str] = None
    hf_persona_config: Optional[str] = None
    hf_sjt_path: Optional[str] = None
    hf_sjt_config: Optional[str] = None
    hf_sjt_split: Optional[str] = None


class PersonaRunResult(BaseModel):
    config: PersonaRunConfig
    # Raw model outputs (before normalization), per iteration -> per question
    raw_texts: List[List[str]]


class SJTExperimentResults(BaseModel):
    # persona_id -> result
    root: Dict[str, PersonaRunResult]


# =========================
# Runner
# =========================

class SJTResponseRunner:
    def __init__(self, args: argparse.Namespace, mgr: VLLMServerManager):
        self.args = args
        self.mgr = mgr

        self.model: str = args.model_name
        self.max_tokens: int = args.max_tokens
        self.temperature: float = args.temperature

        # batching inside mgr.vllm_chat_batched
        self.mp_batch_size: int = args.mp_batch_size
        self.mp_workers: int = args.mp_workers

        # retries passed to mgr.vllm_chat_batched
        self.max_retries: int = args.max_retries
        self.retry_backoff_s: float = args.retry_backoff_s

        self.base_sjt_template, self.persona_sjt_template = load_prompt_templates(self.model)

    # ----------------------------
    # Hashing / helpers
    # ----------------------------

    @staticmethod
    def stable_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _batch_list(lst: List[Any], n: int):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    @staticmethod
    def _normalize_choice(text: str, allowed: List[str]) -> str:
        """Map raw model output to one of allowed choices if possible."""
        if text is None:
            return ""
        t = str(text).strip()
        # common patterns
        if t in allowed:
            return t
        # allow prefixes like "1." or "1)" or "Answer: 1"
        for a in allowed:
            if t.startswith(a):
                return a
        # fallback: first digit 1-6 found
        for ch in t:
            if ch in allowed:
                return ch
        return t

    @staticmethod
    def _messages_to_text(messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages list to a single prompt string."""
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}:\n{content}".strip())
        lines.append("ASSISTANT:\n")
        return "\n\n".join(lines)

    # ----------------------------
    # Data loaders
    # ----------------------------

    def load_sjts(self) -> List[Dict[str, Any]]:
        """Load SJT items from HF dataset."""
        print("Using Huggingface SJTs")
        print(f"Loading SJTs from {self.args.hf_sjt_path}")
        name = getattr(self.args, "hf_sjt_config", None) or "debug"
        split = getattr(self.args, "hf_sjt_split", None) or "train"

        hf_sjt_dataset = load_dataset(self.args.hf_sjt_path, name=name)
        if split not in hf_sjt_dataset:
            raise ValueError(
                f"Requested SJT split '{split}' not found in {self.args.hf_sjt_path}. " 
                f"Available splits: {list(hf_sjt_dataset.keys())}"
            )

        sjt_ds = hf_sjt_dataset[split]
        print(f"Loaded {len(sjt_ds)} SJTs from {self.args.hf_sjt_path} name={name!r} split={split!r}")

        # Keep as list of dicts for easy templating
        return [dict(x) for x in sjt_ds]

    def load_personas(self) -> Dataset:
        """Load personas (mirrors HEXACO runner: dataset path + config name)."""
        if self.args.persona_source != "huggingface":
            raise ValueError("This runner currently supports persona_source='huggingface' only (to match HEXACO workflow).")

        print("Using Huggingface Personas")
        print(f"Loading Personas from {self.args.hf_persona_path}")
        hf_config = getattr(self.args, "hf_persona_config", None) or None

        hf_persona_dataset = load_dataset(self.args.hf_persona_path, name=hf_config)
        persona_ds = hf_persona_dataset["train"]
        print(f"Loaded {len(persona_ds)} Personas from {self.args.hf_persona_path} name={hf_config!r}")

        if self.args.debug:
            persona_ds = persona_ds.select(range(min(10, len(persona_ds))))
            print(f"Debug mode enabled; using {len(persona_ds)} personas.")

        return persona_ds

    # ----------------------------
    # Prompt building
    # ----------------------------

    def build_prompt_for_sjt(
        self,
        persona_str: str,
        sjt_item: Dict[str, Any],
        answer_shuffle: bool,
    ) -> Tuple[str, List[str], List[int], str]:
        """
        Returns:
          prompt_text, guided_choices, answer_index, question_hash
        """
        # Determine answer-option order
        if answer_shuffle:
            idxs = list(range(len(DEFAULT_ANSWER_OPTION_ORDERING)))
            random.shuffle(idxs)
        else:
            idxs = list(range(len(DEFAULT_ANSWER_OPTION_ORDERING)))

        # Render the prompt using the same template pattern as existing SJT runner
        # Expect sjt_item to contain the scenario text + the six options keyed by DEFAULT_ANSWER_OPTION_ORDERING
        option_order = [DEFAULT_ANSWER_OPTION_ORDERING[i] for i in idxs]
        ordered_options = [sjt_item.get(k, "") for k in option_order]


        scenario = sjt_item.get("scenario", sjt_item.get("prompt", ""))
        if not scenario:
            raise ValueError("SJT item missing 'scenario' (or 'prompt') field.")

        # Format options nicely (1..6 correspond to the *ordered* options)
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(ordered_options)])

        prompt_text = self.persona_sjt_template.render(
            persona=persona_str,
            scenario=scenario,
            options=options_text,
        )

        # Guided decoding choices are always 1..6
        guided_choices = SJT_ANSWER_CHOICES

        # Hash should be stable per SJT item (use full JSON canonical form)
        question_hash = self.stable_hash(json.dumps(sjt_item, sort_keys=True, ensure_ascii=False))

        return prompt_text, guided_choices, idxs, question_hash

    # ----------------------------
    # vLLM generation (batched)
    # ----------------------------

    def vllm_generate_batched(self, prompts: List[str], guided_choices: List[List[str]]) -> List[str]:
        return self.mgr.vllm_chat_batched(
            prompts=prompts,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            batch_size=self.mp_batch_size,
            num_workers=self.mp_workers,
            guided_choices=guided_choices,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
        )

    # ----------------------------
    # Push to Hub (mirrors HEXACO runner)
    # ----------------------------

    def push_results_to_hub(
        self,
        results: SJTExperimentResults,
        repo_id: str,
        split_name: str,
        config_name: str = "sjt",
    ) -> None:
        """Push row-wise results to HF under the given dataset config and split."""
        rows: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for persona_id, run_result in results.root.items():
            cfg = run_result.config
            rows.append(
                {
                    "persona_id": persona_id,
                    "persona_hash": cfg.persona_hash,
                    "model_name": cfg.model_name,
                    "n_times": len(cfg.answers),
                    "n_questions": (len(cfg.answers[0]) if cfg.answers else 0),
                    "answers": cfg.answers,
                    "raw_texts": run_result.raw_texts,
                    "question_hashes": cfg.question_hashes,
                    "answer_index": cfg.answer_index,
                    "raw_prompts": cfg.raw_prompts,
                    "guided_choices": cfg.guided_choices,
                    "hf_persona_path": cfg.hf_persona_path,
                    "hf_persona_config": cfg.hf_persona_config,
                    "hf_sjt_path": cfg.hf_sjt_path,
                    "hf_sjt_config": cfg.hf_sjt_config,
                    "hf_sjt_split": cfg.hf_sjt_split,
                    "created_at": now_iso,
                    "generator": "sjt_response_generator.py",
                }
            )

        ds = Dataset.from_list(rows)
        dsd = DatasetDict({split_name: ds})

        # datasets supports configs via config_name in push_to_hub in recent versions.
        # If user's environment has an older datasets, they can remove config_name and instead use split_name isolation.
        try:
            dsd.push_to_hub(repo_id, config_name=config_name)
        except TypeError:
            # Back-compat: no config support
            dsd.push_to_hub(repo_id)

        print(f"Pushed {len(rows)} rows to HF dataset {repo_id} config '{config_name}' split '{split_name}'")

    # ----------------------------
    # Main run loop
    # ----------------------------

    def run(self) -> str:
        print(f"Model Used: {self.model}")

        personas = self.load_personas()
        sjt_items = self.load_sjts()

        # Sample SJTs per persona if requested
        if self.args.n_sjtsample is not None and self.args.n_sjtsample > 0:
            if self.args.n_sjtsample < len(sjt_items):
                sjt_items_sampled = random.sample(sjt_items, k=self.args.n_sjtsample)
            else:
                sjt_items_sampled = sjt_items
        else:
            sjt_items_sampled = sjt_items

        results: Dict[str, PersonaRunResult] = {}

        for row in tqdm(personas, desc="Personas"):
            persona_uuid = str(row.get("uuid", row.get("persona_id", row.get("id", ""))))
            persona_str = str(row.get("persona_string", row.get("persona", row.get("text", ""))))
            if not persona_uuid:
                persona_uuid = self.stable_hash(persona_str)[:16]

            persona_hash = self.stable_hash(persona_str)

            repeated_answers: List[List[str]] = []
            repeated_raw_texts: List[List[str]] = []
            repeated_question_hashes: List[List[str]] = []
            repeated_answer_indexes: List[List[List[int]]] = []
            repeated_raw_prompts: List[List[str]] = []
            repeated_guided_choices: List[List[List[str]]] = []

            for _ in range(self.args.n_times):
                all_prompts: List[str] = []
                all_guided: List[List[str]] = []
                question_hashes: List[str] = []
                answer_indexes: List[List[int]] = []

                for sjt_batch in self._batch_list(sjt_items_sampled, self.args.batch_size):
                    prompts_text: List[str] = []
                    batch_guided: List[List[str]] = []
                    batch_hashes: List[str] = []
                    batch_answer_idx: List[List[int]] = []

                    for sjt_item in sjt_batch:
                        ptxt, guided, aidx, qhash = self.build_prompt_for_sjt(
                            persona_str=persona_str,
                            sjt_item=sjt_item,
                            answer_shuffle=bool(self.args.answer_shuffle),
                        )
                        prompts_text.append(ptxt)
                        batch_guided.append(guided)
                        batch_hashes.append(qhash)
                        batch_answer_idx.append(aidx)

                    all_prompts.extend(prompts_text)
                    all_guided.extend(batch_guided)
                    question_hashes.extend(batch_hashes)
                    answer_indexes.extend(batch_answer_idx)

                raw_texts = self.vllm_generate_batched(all_prompts, guided_choices=all_guided)
                persona_answers = [self._normalize_choice(t, SJT_ANSWER_CHOICES) for t in raw_texts]

                repeated_answers.append(persona_answers)
                repeated_raw_texts.append(raw_texts)
                repeated_question_hashes.append(question_hashes)
                repeated_answer_indexes.append(answer_indexes)
                repeated_raw_prompts.append(all_prompts)
                repeated_guided_choices.append(all_guided)

            cfg = PersonaRunConfig(
                persona=persona_uuid,
                persona_hash=persona_hash,
                answers=repeated_answers,
                question_hashes=repeated_question_hashes,
                answer_index=repeated_answer_indexes,
                raw_prompts=repeated_raw_prompts,
                guided_choices=repeated_guided_choices,
                model_name=self.model,
                hf_persona_path=getattr(self.args, "hf_persona_path", None),
                hf_persona_config=getattr(self.args, "hf_persona_config", None),
                hf_sjt_path=getattr(self.args, "hf_sjt_path", None),
                hf_sjt_config=getattr(self.args, "hf_sjt_config", None),
                hf_sjt_split=getattr(self.args, "hf_sjt_split", None),
            )

            results[persona_uuid] = PersonaRunResult(config=cfg, raw_texts=repeated_raw_texts)

        final = SJTExperimentResults(root=results)

        # Always write local output
        out_dir = Path(self.args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sjt_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(final.model_dump(), f, indent=2)
        print(f"Wrote local results to {out_path}")

        if bool(self.args.push_to_hub):
            self.push_results_to_hub(
                results=final,
                repo_id=self.args.hub_repo,
                config_name=self.args.hub_config,
                split_name=self.args.hub_split,
            )

        return str(out_path)


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    # Personas (match HEXACO runner: dataset path + config name)
    parser.add_argument(
        "--persona-source",
        type=str,
        default="huggingface",
        help="Only 'huggingface' is supported (matches HEXACO workflow).",
    )
    parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
    parser.add_argument("--hf-persona-config", type=str, default="expanded")
    parser.add_argument("--n-personasample", type=int, default=1)

    # SJTs
    parser.add_argument("--hf-sjt-path", type=str, default="thoughtworks/psychometric_sjts_analysis")
    parser.add_argument("--hf-sjt-config", type=str, default="debug")
    parser.add_argument("--hf-sjt-split", type=str, default="train")
    parser.add_argument("--n-sjtsample", type=int, default=1)

    # Prompt/answer behavior
    parser.add_argument("--answer-shuffle", action="store_true")
    parser.add_argument("--batching", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-times", type=int, default=1)

    # Output
    parser.add_argument("--out-dir", type=str, default=str(ROOT_DIR / "outputs"))

    # vLLM generation params
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)

    # multiprocessing / batching inside mgr.vllm_chat_batched
    parser.add_argument("--mp-batch-size", type=int, default=256)
    parser.add_argument("--mp-workers", type=int, default=8)

    # retry behavior passed to mgr.vllm_chat_batched
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=0.5)

    # vLLM server connection (we DO NOT start it; mgr only uses these for URL)
    parser.add_argument("--vllm-host", type=str, default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-timeout-s", type=int, default=180)

    # Debug + push (match HEXACO runner flags)
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction,
        default=True, help="Debug mode (default True): only run first 10 personas.",
    )
    parser.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction,
        default=True, help="Push results to Hugging Face Hub (default True).",
    )
    parser.add_argument(
        "--hub-repo", type=str,
        default="thoughtworks/psychometric_test_responses",
        help="HF dataset repo to push to (same as HEXACO runner).",
    )
    parser.add_argument(
        "--hub-config", type=str, default="sjt",
        help="HF dataset config name for SJT outputs.",
    )
    parser.add_argument(
        "--hub-split", type=str, default="train",
        help="HF dataset split name inside the SJT config.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    mgr = VLLMServerManager(
        model=args.model_name,
        host=args.vllm_host,
        port=args.vllm_port,
        timeout_s=args.vllm_timeout_s,
        kill_existing=False,     # do not kill anything
        server_extra_args=[],    # do not start anything
    )

    runner = SJTResponseRunner(args=args, mgr=mgr)
    runner.run()


if __name__ == "__main__":
    main()
