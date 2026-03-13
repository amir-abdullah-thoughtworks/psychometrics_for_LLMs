from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer
from utils.vllm_utils import VLLMServerManager
from diskcache import Cache
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROMPT_CACHE_DIR = os.path.abspath("./PROMPT_CACHE_DIR")


def iter_seen_pairs(jsonl_path: Path) -> Set[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    if not jsonl_path.exists():
        return seen
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uuid = obj.get("uuid")
                prompt_hash = obj.get("prompt_hash")
                if uuid and prompt_hash:
                    seen.add((uuid, prompt_hash))
            except json.JSONDecodeError:
                continue
    return seen


def make_run_id() -> str:
    return datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")


@dataclass
class PromptSetGeneratorBase:
    source_dataset_id: str
    source_split: str = "train"
    source_revision: Optional[str] = None
    source_fingerprint: Optional[str] = None
    take_n: Optional[int] = None
    debug: bool = True
    limit_personas: int = 100

    def stable_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_source_dataset(self) -> Dataset:
        ds = load_dataset(
            self.source_dataset_id,
            split=self.source_split,
            revision=self.source_revision,
        )
        try:
            self.source_fingerprint = getattr(ds, "_fingerprint", None)
        except Exception:
            self.source_fingerprint = None
        return ds

    def compute_meta_hash(self) -> str:
        payload = {
            "source_dataset_id": self.source_dataset_id,
            "source_split": self.source_split,
            "source_revision": self.source_revision,
            "source_fingerprint": self.source_fingerprint,
            "take_n": self.take_n,
            "debug": self.debug,
            "limit_personas": self.limit_personas,
            "generator_class": self.__class__.__name__,
        }
        return self.stable_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def get_prompt_rows(self, source_ds: Dataset) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def format_prompt(self, persona_string: str, prompt_row: Dict[str, Any], tokenizer, max_tokens) -> str:
        raise NotImplementedError


@dataclass
class AdvBenchPromptSetGenerator(PromptSetGeneratorBase):
    source_field: str = "prompt"
    target_field: str = "target"

    # --- new fields for truncation + caching ---
    max_tokens: int = 1480
    cache_dir: str = PROMPT_CACHE_DIR

    # internal cache + prefix token memo
    _cache: Cache = field(init=False, repr=False)
    _prefix_text: str = field(
        default="You are roleplaying as the following persona. Stay in character.\n\n",
        init=False,
        repr=False,
    )
    _prefix_tokens: Optional[Tuple[int, ...]] = field(default=None, init=False, repr=False)
    _tokenizer_id: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # If your PromptSetGeneratorBase defines __post_init__, call it.
        base_post_init = getattr(super(), "__post_init__", None)
        if callable(base_post_init):
            base_post_init()
        self._cache = Cache(self.cache_dir)

    def _cache_key(self, kind: str, tokenizer_id: int, text: str) -> str:
        return f"{kind}:{tokenizer_id}:{self.stable_hash(text)}"

    def _format_cache_key(
        self, persona_string: str,
        prompt_row: Dict[str, Any], tokenizer_id: int,
    ) -> str:
        # Only adv_source affects formatting; adv_target does not
        return (
            f"format:"
            f"{tokenizer_id}:"
            f"{self.max_tokens}:"
            f"{self.stable_hash(persona_string)}:"
            f"{self.stable_hash(prompt_row['adv_source'])}"
        )

    def format_prompt(
        self,
        persona_string: str,
        prompt_row: Dict[str, Any],
        tokenizer=None,
    ) -> Tuple[str, bool]:
        static_prefix = self._prefix_text
        static_suffix = f"\n\nUser request:\n{prompt_row['adv_source']}"

        if tokenizer is None:
            approx_budget = self.max_tokens * 4
            fixed_len = len(static_prefix) + len(static_suffix)
            remaining_chars = max(0, approx_budget - fixed_len)
            was_truncated = len(persona_string) > remaining_chars
            persona_trunc = persona_string[:remaining_chars]
            return static_prefix + persona_trunc + static_suffix, was_truncated

        tokenizer_id = id(tokenizer)

        """
        format_key = self._format_cache_key(
            persona_string=persona_string,
            prompt_row=prompt_row,
            tokenizer_id=tokenizer_id,
        )

        cached = self._cache.get(format_key)
        if cached is not None:
            return cached  # (prompt, was_truncated)
        """

        if self._prefix_tokens is None or self._tokenizer_id != tokenizer_id:
            self._prefix_tokens = tuple(
                tokenizer.encode(static_prefix, add_special_tokens=False)
            )
            self._tokenizer_id = tokenizer_id

        prefix_tokens = self._prefix_tokens

        suffix_key = self._cache_key("suffix", tokenizer_id, static_suffix)
        suffix_tokens = self._cache.get(suffix_key)
        if suffix_tokens is None:
            suffix_tokens = tuple(
                tokenizer.encode(static_suffix, add_special_tokens=False)
            )
            self._cache.set(suffix_key, suffix_tokens)

        remaining = self.max_tokens - len(prefix_tokens) - len(suffix_tokens)
        remaining = max(0, remaining)


        persona_key = self._cache_key("persona", tokenizer_id, persona_string)
        persona_tokens = self._cache.get(persona_key)
        if persona_tokens is None:
            persona_tokens = tuple(
                tokenizer.encode(persona_string, add_special_tokens=False)
            )
            self._cache.set(persona_key, persona_tokens)

        was_truncated = len(persona_tokens) > remaining
        persona_tokens_trunc = persona_tokens[:remaining]
        truncated_persona = tokenizer.decode(list(persona_tokens_trunc))

        result = (static_prefix + truncated_persona + static_suffix, was_truncated)
        # self._cache.set(format_key, result)
        return result


    def get_prompt_rows(self, source_ds: Dataset) -> List[Dict[str, Any]]:
        n = len(source_ds) if self.take_n is None else min(self.take_n, len(source_ds))
        if self.debug:
            n = min(self.limit_personas, n)

        rows: List[Dict[str, Any]] = []
        for row in source_ds.select(range(n)):
            src = row[self.source_field]
            tgt = row.get(self.target_field)
            prompt_hash = self.stable_hash(src)
            rows.append(
                {
                    "adv_source": src,
                    "adv_target": tgt,
                    "prompt_hash": prompt_hash,
                }
            )
        return rows


@dataclass
class PersonaPromptRunner:
    prompt_generator: PromptSetGeneratorBase
    persona_dataset_id: str
    persona_config: str
    persona_revision: Optional[str] = None

    out_jsonl: Path = Path("outputs/gemma_advbench_persona_responses")
    hub_repo_id: str = "thoughtworks/gemma_psychometric_personas_responses"
    hub_split_name: str = "advbench_v2"

    model: str = "google/gemma-3-4b-it"
    max_completion_tokens: int = 512
    temperature: float = 0.7
    mp_batch_size: int = 100
    mp_workers: Optional[int] = 50

    debug: bool = False
    limit_personas: int = 100

    def _resolve_run_jsonl_path(self) -> Path:
        run_id = make_run_id()
        run_dir = self.out_jsonl
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / f"{run_id}.jsonl"

    def load_personas(self) -> Dataset:
        ds = load_dataset(
            self.persona_dataset_id,
            name=self.persona_config,
            revision=self.persona_revision,
        )['train']

        ds = ds.select(range(min(self.limit_personas, len(ds))))
        return ds

    def run(self, mgr: VLLMServerManager) -> Path:
        jsonl_path = self._resolve_run_jsonl_path()
        seen = iter_seen_pairs(jsonl_path)

        source_ds = self.prompt_generator.load_source_dataset()
        meta_hash = self.prompt_generator.compute_meta_hash()
        prompt_rows = self.prompt_generator.get_prompt_rows(source_ds)

        persona_ds = self.load_personas()
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        tasks: List[Dict[str, Any]] = []
        for persona in tqdm(persona_ds, desc="Indexing tasks", unit="persona"):
            uuid = persona["uuid"]
            persona_string = persona["persona_string"]
            persona_hash = persona["persona_hash"]
            persona_details = dict(persona)

            for pr in prompt_rows:
                prompt_hash = pr["prompt_hash"]
                if (uuid, prompt_hash) in seen:
                    continue
                tasks.append(
                    {
                        "uuid": uuid,
                        "persona_hash": persona_hash,
                        "prompt_hash": prompt_hash,
                        "persona_string": persona_string,
                        "persona_details": persona_details,
                        "prompt_row": pr,
                    }
                )

        print(f"Preparing {len(tasks)} for VLLM server")

        if self.debug:
            tasks = tasks[: self.limit_personas]

        if not tasks:
            return jsonl_path

        persona_formatted_prompts = []
        num_truncated = 0

        for t in tqdm(tasks, desc="Formatting prompts"):
            prompt, was_truncated = self.prompt_generator.format_prompt(
                t["persona_string"],
                t["prompt_row"],
                tokenizer=tokenizer,
            )
            persona_formatted_prompts.append(prompt)

            num_truncated += int(was_truncated)

        print(f"Truncated personas: {num_truncated} / {len(tasks)}")

        print(f"Sending {len(persona_formatted_prompts)} prompts to vllm")

        outputs = mgr.vllm_chat_batched(
            prompts=persona_formatted_prompts,
            model=self.model,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            batch_size=self.mp_batch_size,
            num_workers=self.mp_workers,
        )

        with jsonl_path.open("a", encoding="utf-8") as out_f:
            with tqdm(total=len(tasks), desc="Writing JSONL", unit="row") as pbar:
                for t, persona_formatted_prompt, completion in zip(tasks, persona_formatted_prompts, outputs):
                    pr = t["prompt_row"]
                    record = {
                        "run_id": jsonl_path.stem,
                        "uuid": t["uuid"],
                        "persona_hash": t["persona_hash"],
                        "prompt_hash": t["prompt_hash"],
                        "meta_hash": meta_hash,
                        "source_dataset_id": self.prompt_generator.source_dataset_id,
                        "source_split": self.prompt_generator.source_split,
                        "source_revision": self.prompt_generator.source_revision,
                        "source_fingerprint": self.prompt_generator.source_fingerprint,
                        "response": completion,
                        "model": self.model,
                        "formatted_prompt": persona_formatted_prompt,
                        "persona_details": t["persona_details"],
                        **pr,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pbar.update(1)

        return jsonl_path

    def push_run_to_hub(self, jsonl_path: Path) -> None:
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        DatasetDict({self.hub_split_name: ds}).push_to_hub(self.hub_repo_id)


def main():
    debug = False

    prompt_gen = AdvBenchPromptSetGenerator(
        source_dataset_id="walledai/AdvBench",
        source_split="train",
        take_n=3200,
        debug=debug,
        limit_personas=3200,
    )

    runner = PersonaPromptRunner(
        prompt_generator=prompt_gen,
        persona_dataset_id="thoughtworks/psychometric_personas",
        persona_config="test_sample",
        out_jsonl=Path("outputs/gemma_advbench_persona_responses"),
        hub_repo_id="thoughtworks/gemma_psychometric_personas_responses",
        hub_split_name="advbench",
        model="google/gemma-3-4b-it",
        max_completion_tokens=512,
        temperature=0,
        mp_batch_size=800,
        mp_workers=40,
        debug=debug,
        limit_personas=3200,
    )

    mgr = VLLMServerManager()
    jsonl_path = runner.run(mgr)

    if not debug:
        runner.push_run_to_hub(jsonl_path)


if __name__ == "__main__":
    main()
