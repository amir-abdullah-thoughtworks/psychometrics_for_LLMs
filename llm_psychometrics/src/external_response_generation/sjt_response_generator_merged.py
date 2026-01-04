"""
vLLM-only, class-based SJT response generator (consolidated single file).

Features:
- vLLM only (no OpenAI support).
- Does NOT start/boot/kill any vLLM server.
- Uses an externally created VLLMServerManager passed from main().
- Uses mgr.vllm_chat_batched(prompts=..., guided_choices=...) with per-prompt constraints.
- Fresh shuffle per answer (iid) when --answer-shuffle is enabled.
- Stores:
  - persona_hash = SHA256(persona_string)
  - raw_prompts (exact strings sent to vLLM)
  - guided_choices (exact decoding constraints per prompt)
  - answer_index (per prompt permutation of options)
  - question_hashes
  - answers (normalized to "1".."6" when possible)

Output JSON schema:
{
  "<persona_uuid or base_model>": {
    "config": {...},
    "answers": [[...], [...], ...]   # n_times iterations
  },
  ...
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import transformers
from datasets import Dataset, load_dataset
from jinja2 import Template
from pydantic import BaseModel
from tqdm import tqdm

# --- Custom imports (keep as-is in your repo) ---
from utils_v0 import list_to_str
from prompt_templates.sjt_base_prompt_templates import sjt_base_prompt_templates
from prompt_templates.sjt_persona_prompt_templates import sjt_persona_prompt_templates
from utils.vllm_utils import VLLMServerManager

from pydantic import RootModel


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
# Pydantic output schema
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


class ExperimentResults(RootModel[Dict[str, "PersonaRunConfig"]]):
    """
    persona_uuid (or 'base_model') -> PersonaRunConfig
    """
    def to_jsonable(self) -> Dict[str, Any]:
        # In pydantic v2: RootModel stores value in `.root`
        return {k: v.model_dump() for k, v in self.root.items()}
# =========================
# Runner
# =========================

class SJTResponseRunner:
    def __init__(self, args: argparse.Namespace, mgr: VLLMServerManager):
        self.args = args
        self.mgr = mgr

        transformers.logging.set_verbosity_error()

        # vLLM generation settings
        self.model: str = args.model_name
        self.max_tokens: int = args.max_tokens
        self.temperature: float = args.temperature
        self.mp_batch_size: int = args.mp_batch_size
        self.mp_workers: int = args.mp_workers
        self.max_retries: int = args.max_retries
        self.retry_backoff_s: float = args.retry_backoff_s

        # Templates: list of dicts with role + compiled Jinja template for content
        self.base_sjt_template: List[Dict[str, Any]] = []
        self.persona_sjt_template: List[Dict[str, Any]] = []

    # ----------------------------
    # Hashing
    # ----------------------------
    def stable_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ----------------------------
    # CLI
    # ----------------------------
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")

        # By default we run a small smoke test (10 personas x 10 SJTs) unless --no-debug is set.
        parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

        parser.add_argument(
            "--persona-source",
            type=str,
            default="huggingface",
            help="(base_model | huggingface | personallm_paper | local)",
        )

        # Personas (same HF dataset as HEXACO runner)
        parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
        parser.add_argument("--hf-persona-config", type=str, default="expanded")
        parser.add_argument("--hf-persona-split", type=str, default="train")
        parser.add_argument("--n-personasample", type=int, default=10)

        # SJTs
        parser.add_argument("--hf-sjt-path", type=str, default="thoughtworks/psychometric_SJTs")
        parser.add_argument("--hf-sjt-config", type=str, default=None)
        parser.add_argument("--hf-sjt-split", type=str, default="restricted")
        parser.add_argument("--n-sjtsample", type=int, default=10)

        # Prompt / generation
        parser.add_argument("--batching", action="store_true")
        parser.add_argument("--batch-size", type=int, default=64)

        parser.add_argument("--n-times", type=int, default=1)

        # Shuffle answer options (default: ON)
        parser.add_argument("--answer-shuffle", action=argparse.BooleanOptionalAction, default=True)

        parser.add_argument("--out-dir", type=str, default=".")

        # vLLM generation params
        parser.add_argument("--max-tokens", type=int, default=16)
        parser.add_argument("--temperature", type=float, default=0.0)

        # multiprocessing / batching inside mgr.vllm_chat_batched
        parser.add_argument("--mp-batch-size", type=int, default=256)
        parser.add_argument("--mp-workers", type=int, default=8)

        # retry behavior passed to mgr.vllm_chat_batched
        parser.add_argument("--max-retries", type=int, default=3)
        parser.add_argument("--retry-backoff-s", type=float, default=0.1)

        # manager connection details (we do NOT start any server here)
        parser.add_argument("--vllm-host", type=str, default="127.0.0.1")
        parser.add_argument("--vllm-port", type=int, default=8000)
        parser.add_argument("--vllm-timeout-s", type=int, default=180)

        args = parser.parse_args()

        # Debug mode pins a small, predictable run unless explicitly overridden.
        if args.debug:
            args.n_times = 1
            args.n_personasample = 10
            args.n_sjtsample = 10
            args.answer_shuffle = True

        return args

# ----------------------------
    # Small utils
    # ----------------------------
    @staticmethod
    def _write_json(obj: Dict[str, Any], file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def _read_json(file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _batch_list(lst: List[Any], n: int):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # ----------------------------
    # Templates
    # ----------------------------
    @staticmethod
    def _compile_message_templates(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        return [{"role": msg["role"], "content": Template(msg["content"])} for msg in messages]

    def load_prompt_templates(self) -> None:
        # Keep using "gpt" chat-style templates, but we linearize to plain text prompts.
        base_templates = sjt_base_prompt_templates["gpt"]
        persona_templates = sjt_persona_prompt_templates["gpt"]
        self.base_sjt_template = self._compile_message_templates(base_templates)
        self.persona_sjt_template = self._compile_message_templates(persona_templates)

    @staticmethod
    def render_messages(template_messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"].render(**kwargs)} for msg in template_messages]

    @staticmethod
    def messages_to_prompt_text(messages: List[Dict[str, str]]) -> str:
        """
        Converts chat-style messages into a single plain text prompt.
        We store exactly this prompt for audit/replay.
        """
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
        print("Using Huggingface SJTs")
        print(f"Loading SJTs from {self.args.hf_sjt_path} (config={self.args.hf_sjt_config}, split={self.args.hf_sjt_split})")

        if self.args.hf_sjt_config:
            ds = load_dataset(self.args.hf_sjt_path, name=self.args.hf_sjt_config)[self.args.hf_sjt_split]
        else:
            ds = load_dataset(self.args.hf_sjt_path)[self.args.hf_sjt_split]

        n = min(self.args.n_sjtsample, len(ds))
        if n <= 0:
            return []

        # Sample TOTAL n items (not per-template)
        ds = ds.shuffle(seed=42).select(range(n))

        sjt_records = ds.to_pandas().to_dict("records")
        print(f"No of SJTs: {len(sjt_records)}")
        return sjt_records
    def load_personas(self) -> Optional[Union[Dataset, List[Dict[str, Any]]]]:
        if self.args.persona_source == "huggingface":
            print("Using Huggingface Personas")
            print(f"Loading Personas from {self.args.hf_persona_path} (config={self.args.hf_persona_config}, split={self.args.hf_persona_split})")

            ds = load_dataset(self.args.hf_persona_path, name=self.args.hf_persona_config)[self.args.hf_persona_split]
            n = min(self.args.n_personasample, len(ds))
            if n <= 0:
                print("No personas requested; returning empty dataset.")
                return Dataset.from_list([])

            ds = ds.shuffle(seed=42).select(range(n))
            print(f"No of Personas: {len(ds)}")
            return ds

        if self.args.persona_source == "personallm_paper":
            print("Using Persona LLM Paper Personas")
            persona_datasets_total = load_dataset("proj-persona/Personas_paper", split="train")
            total_persona_df = persona_datasets_total.to_pandas()
            persona_datasets = Dataset.from_pandas(total_persona_df.sample(n=self.args.n_personasample, random_state=42))
            print(f"No of Personas: {len(persona_datasets)}")
            return persona_datasets

        if self.args.persona_source == "local":
            raise ValueError("persona_source=local not wired in this script yet")

        # base_model: no personas
        return None

    def _normalize_choice(text: str, choices: List[str]) -> str:
        """
        Map outputs back to canonical choice if possible.
        - exact match "3"
        - or first occurrence of any digit in choices in the output string
        """
        t = (text or "").strip()
        if t in choices:
            return t
        for ch in t:
            if ch in choices:
                return ch
        return t

    @staticmethod
    def _choice_to_trait(choice: str, perm: List[int]) -> Optional[str]:
        """
        Given a model choice in displayed space ("1".."6") and the permutation used
        to shuffle options, return the canonical trait option key.
        perm[j] = canonical option index shown at displayed position j.
        """
        c = (choice or "").strip()
        if c not in SJT_ANSWER_CHOICES:
            raise ValueError(f"Invalid choice: {c}")
        j = int(c) - 1
        if j < 0 or j >= len(perm):
            raise ValueError(f"Invalid perm: {j}")
        canonical_idx = perm[j]
        if canonical_idx < 0 or canonical_idx >= len(DEFAULT_ANSWER_OPTION_ORDERING):
            raise ValueError(f"Invalid canonical_idx {canonical_idx}")
        return DEFAULT_ANSWER_OPTION_ORDERING[canonical_idx]


    @staticmethod
    def _extract_texts(outputs: Any) -> List[str]:
        """
        vllm_chat_batched may return:
        - list[str]
        - list[dict] with {"text": "..."} (common pattern)
        - other: stringified
        """
        if outputs is None:
            return []
        if isinstance(outputs, list) and (len(outputs) == 0 or isinstance(outputs[0], str)):
            return outputs
        if isinstance(outputs, list) and isinstance(outputs[0], dict) and "text" in outputs[0]:
            return [o.get("text", "") for o in outputs]
        return [str(o) for o in outputs]

    def vllm_generate_batched(self, prompts_text: List[str], guided_choices: List[List[str]]) -> List[str]:
        if len(prompts_text) != len(guided_choices):
            raise ValueError(
                f"len(prompts_text) ({len(prompts_text)}) != len(guided_choices) ({len(guided_choices)})"
            )

        outputs = self.mgr.vllm_chat_batched(
            prompts=prompts_text,
            guided_choices=guided_choices,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            batch_size=self.mp_batch_size,
            num_workers=self.mp_workers,
            max_retries=self.max_retries,
            retry_backoff_s=self.retry_backoff_s,
        )
        return self._extract_texts(outputs)

    # ----------------------------
    # Prompt building (fresh shuffle per answer)
    # ----------------------------
    def build_prompts_for_questions(
        self,
        sjt_template: List[Dict[str, Any]],
        question_batch: List[Dict[str, Any]],
        persona_str: Optional[str],
        answer_shuffle: bool,
    ) -> Tuple[List[str], List[str], List[List[int]], List[List[str]]]:
        prompt_texts: List[str] = []
        hash_list: List[str] = []
        answer_index_list: List[List[int]] = []
        guided_choices_list: List[List[str]] = []

        base_index = [0, 1, 2, 3, 4, 5]

        for sjt_dict in question_batch:
            # Fresh iid shuffle PER ANSWER
            idx = base_index[:]  # fresh copy each question
            if answer_shuffle:
                random.shuffle(idx)

            answer_index_list.append(idx)

            sjt = sjt_dict["corrected_sjt"]
            answer_options_text = [sjt[k] for k in DEFAULT_ANSWER_OPTION_ORDERING if "_option" in k]
            answer_options_text = [answer_options_text[i] for i in idx]

            question = sjt["question"]
            messages = self.render_messages(
                sjt_template,
                attributes=persona_str,
                question=question,
                answer_options=list_to_str(answer_options_text),
            )

            prompt_texts.append(self.messages_to_prompt_text(messages))
            hash_list.append(sjt_dict["hash_id"])

            # Constrain each prompt to outputs "1".."6"
            guided_choices_list.append(SJT_ANSWER_CHOICES)

        return prompt_texts, hash_list, answer_index_list, guided_choices_list

    # ----------------------------
    # Experiment loop
    # ----------------------------
    def generate_answers(
        self,
        synthetic_sjts: List[Dict[str, Any]],
        persona_datasets: Optional[Union[Dataset, List[Dict[str, Any]]]],
        answer_shuffle: bool,
    ) -> ExperimentResults:
        # NOTE: We keep the existing vLLM batched call path (mgr.vllm_chat_batched).
        if self.args.batching:
            raise NotImplementedError("Batching flag is deprecated here; use --batch-size / --mp-* knobs instead.")

        sjt_answer_options_mode: Literal["normal", "shuffle"] = "shuffle" if answer_shuffle else "normal"
        print("Shuffling answer options for SJTs" if answer_shuffle else "Default Ordering of answer options for SJTs")
        print(f"No of SJTs: {len(synthetic_sjts)}")

        if persona_datasets:
            print(f"No of personas: {len(persona_datasets)}")
        else:
            print("Answering the SJTs using the base model, without any personas")

        results: Dict[str, PersonaRunConfig] = {}

        def _run_one_persona(persona_uuid: str, persona_hash: Optional[str], persona_str: Optional[str]) -> PersonaRunConfig:
            sjt_template = self.persona_sjt_template if persona_str else self.base_sjt_template

            repeated_answers: List[List[str]] = []
            repeated_normalized: List[List[Optional[str]]] = []
            repeated_answer_indexes: List[List[List[int]]] = []
            repeated_raw_prompts: List[List[str]] = []
            repeated_guided_choices: List[List[List[str]]] = []

            question_hashes: List[str] = []

            for _ in tqdm(range(self.args.n_times), desc=f"Iterations ({persona_uuid})"):
                all_prompts: List[str] = []
                all_guided: List[List[str]] = []
                iter_hashes: List[str] = []
                answer_indexes: List[List[int]] = []

                for q_batch in tqdm(self._batch_list(synthetic_sjts, self.args.batch_size), desc="SJT Batches"):
                    prompts_text, batch_hashes, batch_answer_idx, batch_guided = self.build_prompts_for_questions(
                        sjt_template=sjt_template,
                        question_batch=q_batch,
                        persona_str=persona_str,
                        answer_shuffle=answer_shuffle,
                    )
                    all_prompts.extend(prompts_text)
                    all_guided.extend(batch_guided)
                    iter_hashes.extend(batch_hashes)
                    answer_indexes.extend(batch_answer_idx)

                # keep hashes from first iteration (same order each time)
                if not question_hashes:
                    question_hashes = iter_hashes

                raw_texts = self.vllm_generate_batched(all_prompts, guided_choices=all_guided)
                persona_answer = [self._normalize_choice(t, SJT_ANSWER_CHOICES) for t in raw_texts]
                persona_norm = [self._choice_to_trait(a, perm) for a, perm in zip(persona_answer, answer_indexes)]

                repeated_answers.append(persona_answer)
                repeated_normalized.append(persona_norm)
                repeated_answer_indexes.append(answer_indexes)
                repeated_raw_prompts.append(all_prompts)
                repeated_guided_choices.append(all_guided)

            cfg = PersonaRunConfig(
                persona=persona_uuid,
                persona_hash=persona_hash,
                answers=repeated_answers,
                normalized_answers=repeated_normalized,
                question_hashes=question_hashes,
                answer_index=repeated_answer_indexes,
                raw_prompts=repeated_raw_prompts,
                guided_choices=repeated_guided_choices,
                model_name=self.model,
                hf_persona_path=getattr(self.args, "hf_persona_path", None),
                hf_persona_config=getattr(self.args, "hf_persona_config", None),
                hf_sjt_path=getattr(self.args, "hf_sjt_path", None),
                hf_sjt_config=getattr(self.args, "hf_sjt_config", None),
                hf_sjt_split=getattr(self.args, "hf_sjt_split", None),
                sjt_answer_options=sjt_answer_options_mode,
            )
            return cfg

        # --- Base model mode (no personas)
        if self.args.persona_source == "base_model":
            print("Running SJTs on Base Model without Personas")
            results["base_model"] = _run_one_persona("base_model", None, None)
            return ExperimentResults(results)

        # --- Persona mode
        if persona_datasets is None:
            raise ValueError("persona_source implies personas, but persona_datasets is None")

        for persona_row in tqdm(persona_datasets, desc="Personas"):
            persona_uuid = persona_row.get("uuid") or persona_row.get("persona_uuid") or persona_row.get("id") or "unknown_persona"
            persona_hash = persona_row.get("persona_hash")
            persona_str = persona_row.get("persona_string") or persona_row.get("attributes") or persona_row.get("persona")

            if persona_str is None:
                raise ValueError(f"Persona row missing persona_string/attributes/persona. Keys={list(persona_row.keys())}")

            results[persona_uuid] = _run_one_persona(persona_uuid, persona_hash, persona_str)

        return ExperimentResults(results)


    def run(self) -> str:
        print(f"Model Used: {self.model}")
        print(f"Writing Output in: {self.args.out_dir}")

        self.load_prompt_templates()
        synthetic_sjts = self.load_sjts()
        persona_datasets = self.load_personas()

        results = self.generate_answers(
            synthetic_sjts=synthetic_sjts,
            persona_datasets=persona_datasets,
            answer_shuffle=self.args.answer_shuffle,
        )

        model_name_safe = self.model.replace(".", "_").split("/")[-1]
        out_file = os.path.join(
            self.args.out_dir,
            f"{self.args.persona_source}_sjt_answers_{model_name_safe}.json",
        )
        self._write_json(results.to_jsonable(), out_file)
        print(f"Results saved to {out_file}")
        return out_file


# =========================
# Main (manager created here; does NOT start server)
# =========================

def main():
    args = SJTResponseRunner.parse_args()

    # Create manager and pass it in. This assumes a server is already reachable.
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
