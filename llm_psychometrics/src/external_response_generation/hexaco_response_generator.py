"""
vLLM-only, class-based HEXACO response generator (single consolidated file).

What this refactor does:
- Removes OpenAI + transformers usage (vLLM only).
- Does NOT start/boot/kill any vLLM server.
- Uses an externally created VLLMServerManager passed from main().
- Uses mgr.vllm_chat_batched(prompts=..., guided_choices=...) with per-prompt constraints.
- Fresh Likert shuffle per answer when --likert-shuffle is enabled (iid per question).
- Stores:
  - persona_hash = SHA256(persona_string)
  - raw_prompts (exact strings sent to vLLM)
  - guided_choices (exact constraints per prompt)
  - likert_orders (the exact likert option ordering used per prompt)
  - answers
  - metadata: paraphrase flag, likert mode, refusal allowed, model name

Assumptions:
- VLLMServerManager.vllm_chat_batched signature is:
    def vllm_chat_batched(
        self,
        prompts: List[str],
        guided_choices=None,
        model: str = "...",
        max_tokens: int = 128,
        temperature: float = 0.0,
        batch_size: int = 100,
        num_workers: int = 100,
        max_retries: int = 3,
        retry_backoff_s: float = 0.1,
    )

- It returns list[str] OR list[dict] with "text" fields (handled).

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from datasets import Dataset, load_dataset
from jinja2 import Template
from pydantic import BaseModel
from tqdm import tqdm

# Custom imports (keep relative paths consistent with your repo structure)
from utils_v0 import inverse_likert
from utils.vllm_utils import VLLMServerManager
from prompt_templates.hexaco_base_prompt_templates import hexaco_base_prompt_templates
from prompt_templates.hexaco_persona_prompt_templates import hexaco_persona_prompt_templates


NO_ANSWER = "Do not wish to answer"


# =========================
# Pydantic output schema
# =========================

class HexacoRunConfig(BaseModel):
    persona: str
    persona_hash: Optional[str]  # SHA256(persona_string), None for base_model
    paraphrase: Literal["paraphrased", "normal"]
    likert_scale_mode: Literal["inverted", "shuffle", "normal"]
    refusal_allowed: Literal["refusal", "no refusal"]
    model_name: str

    # Per iteration -> per question -> exact prompt passed to vLLM
    raw_prompts: List[List[str]]

    # Per iteration -> per question -> allowed outputs for guided decoding
    guided_choices: List[List[List[str]]]

    # Per iteration -> per question -> the EXACT likert option ordering shown in the prompt
    likert_orders: List[List[List[str]]]


class HexacoRunResult(BaseModel):
    config: HexacoRunConfig
    # Per iteration -> per question -> chosen likert string
    answers: List[List[str]]


class HexacoExperimentResults(BaseModel):
    """
    persona_uuid (or 'base_model') -> HexacoRunResult
    """
    __root__: Dict[str, HexacoRunResult]

    def to_jsonable(self) -> Dict[str, Any]:
        return {k: v.model_dump() for k, v in self.__root__.items()}


# =========================
# Runner
# =========================

class HexacoResponseRunner:
    def __init__(self, args: argparse.Namespace, mgr: VLLMServerManager):
        self.args = args
        self.mgr = mgr

        self.model: str = args.model_name
        self.max_tokens: int = args.max_tokens
        self.temperature: float = args.temperature
        self.mp_batch_size: int = args.mp_batch_size
        self.mp_workers: int = args.mp_workers
        self.max_retries: int = args.max_retries
        self.retry_backoff_s: float = args.retry_backoff_s

        # templates: list[{"role": str, "content": Template}]
        self.base_hexaco_template: List[Dict[str, Any]] = []
        self.persona_hexaco_template: List[Dict[str, Any]] = []

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

        parser.add_argument(
            "--persona-source",
            type=str,
            default="huggingface",
            help="Source of Persona (huggingface | base_model | personallm_paper | local)",
        )
        parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
        parser.add_argument("--hf-persona-config", type=str, default="expanded")
        parser.add_argument("--n-personasample", type=int, default=1)

        # question generation options
        parser.add_argument("--paraphrase", action="store_true", help="Use paraphrased HEXACO questions")
        parser.add_argument("--inverted-likert", action="store_true", help="Invert Likert scale")
        parser.add_argument("--no-refusal", action="store_true", help="Disallow refusal option")
        parser.add_argument("--likert-shuffle", action="store_true", help="Shuffle likert scale per answer (iid)")

        # batching (kept to match old interface; we still do one question per prompt)
        parser.add_argument("--batching", action="store_true")
        parser.add_argument("--batch-size", type=int, default=5)

        parser.add_argument("--n-times", type=int, default=1)

        parser.add_argument("--out-dir", type=str, default=".")

        # vLLM generation params
        parser.add_argument("--max-tokens", type=int, default=16)
        parser.add_argument("--temperature", type=float, default=0.0)

        # batching inside vllm_chat_batched
        parser.add_argument("--mp-batch-size", type=int, default=256)
        parser.add_argument("--mp-workers", type=int, default=8)

        # retry behavior
        parser.add_argument("--max-retries", type=int, default=3)
        parser.add_argument("--retry-backoff-s", type=float, default=0.1)

        # manager connection details (we do NOT start any server here)
        parser.add_argument("--vllm-host", type=str, default="127.0.0.1")
        parser.add_argument("--vllm-port", type=int, default=8000)
        parser.add_argument("--vllm-timeout-s", type=int, default=180)

        return parser.parse_args()

    # ----------------------------
    # Utils
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
        base_templates = hexaco_base_prompt_templates["gpt"]
        persona_templates = hexaco_persona_prompt_templates["gpt"]
        self.base_hexaco_template = self._compile_message_templates(base_templates)
        self.persona_hexaco_template = self._compile_message_templates(persona_templates)

    @staticmethod
    def render_messages(template_messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"].render(**kwargs)} for msg in template_messages]

    @staticmethod
    def messages_to_prompt_text(messages: List[Dict[str, str]]) -> str:
        """
        Linearize messages to a single plain text prompt (stored for audit).
        """
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}:\n{content}".strip())
        lines.append("ASSISTANT:\n")
        return "\n\n".join(lines)

    # ----------------------------
    # Data loading
    # ----------------------------
    def load_questions_and_likert(self) -> Tuple[List[str], List[str]]:
        with open("../configs/generation_config.yaml", "r", encoding="utf-8") as f:
            generation_config = yaml.safe_load(f)

        with open("../psychometric_tests/hexaco_100_questions.yaml", "r", encoding="utf-8") as f:
            question_list = yaml.safe_load(f)

        with open("../psychometric_tests/paraphrased_hexaco_100_questions.yaml", "r", encoding="utf-8") as f:
            paraphrased_question_list = yaml.safe_load(f)

        # Likert scale
        if self.args.inverted_likert:
            print("Inverting Likert")
            likert_scale = inverse_likert(generation_config["likert_scale"].copy())
        else:
            print("Normal Likert")
            likert_scale = generation_config["likert_scale"].copy()

        if self.args.no_refusal:
            print("Refusal Not Allowed")
        else:
            print("Refusal Allowed")
            likert_scale.append(NO_ANSWER)

        if self.args.paraphrase:
            print("Using Paraphrased Questions")
            question_list = paraphrased_question_list
        else:
            print("Using Normal Questions")

        # Ensure it's a list[str]; YAML might load as list already, but be safe.
        if not isinstance(question_list, list):
            raise ValueError("Expected question_list to be a list loaded from YAML.")
        if not isinstance(likert_scale, list):
            raise ValueError("Expected likert_scale to be a list loaded from YAML.")

        return question_list, likert_scale

    def load_personas(self) -> Optional[Union[Dataset, List[Dict[str, Any]]]]:
        if self.args.persona_source == "huggingface":
            print("Using Huggingface Personas")
            print(f"Loading Personas from {self.args.hf_persona_path}")
            hf_config = self.args.hf_persona_config
            hf_persona_dataset = load_dataset(self.args.hf_persona_path, name=hf_config)

            persona_datasets_total = hf_persona_dataset["train"]
            print(f"Loaded {len(persona_datasets_total)} Personas from {self.args.hf_persona_path} and config {hf_config}")
            return persona_datasets_total

        if self.args.persona_source == "personallm_paper":
            print("Using Persona LLM Paper Personas")
            persona_datasets_total = self._read_json("../data/persona_llm_paper_seed_combinations.json")
            random.seed(42)
            persona_datasets = random.sample(persona_datasets_total, self.args.n_personasample)
            print(f"No of Personas: {len(persona_datasets)}")
            return persona_datasets

        if self.args.persona_source == "base_model":
            return None

        # local YAML personas
        print("Using Local Personas")
        with open("../configs/personas_v2.yaml", "r", encoding="utf-8") as f:
            local_personas = yaml.safe_load(f)

        job_title = "law_enforcement"
        persona_datasets = local_personas[job_title]["personas"]
        print(f"No of Personas: {len(persona_datasets)}")
        return persona_datasets

    # ----------------------------
    # vLLM generation
    # ----------------------------
    @staticmethod
    def _extract_texts(outputs: Any) -> List[str]:
        if outputs is None:
            return []
        if isinstance(outputs, list) and (len(outputs) == 0 or isinstance(outputs[0], str)):
            return outputs
        if isinstance(outputs, list) and isinstance(outputs[0], dict) and "text" in outputs[0]:
            return [o.get("text", "") for o in outputs]
        return [str(o) for o in outputs]

    @staticmethod
    def _normalize_answer(text: str, options: List[str]) -> str:
        """
        Guided decoding should already constrain outputs, but be defensive:
        - exact match
        - case-insensitive match
        - otherwise return stripped text
        """
        t = (text or "").strip()
        if t in options:
            return t
        low = t.lower()
        norm_map = {opt.strip().lower(): opt for opt in options}
        return norm_map.get(low, t)

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
    # Prompt building (fresh Likert shuffle per answer)
    # ----------------------------
    def build_prompts_for_questions(
        self,
        hexaco_template: List[Dict[str, Any]],
        question_batch: List[str],
        base_likert_scale: List[str],
        persona_str: Optional[str],
        likert_shuffle: bool,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Returns:
          prompts_text: list[str]
          guided_choices_list: list[list[str]]  (one set of allowed outputs per prompt)
          likert_orders_list: list[list[str]]   (the exact likert ordering shown per prompt)
        """
        prompts_text: List[str] = []
        guided_choices_list: List[List[str]] = []
        likert_orders_list: List[List[str]] = []

        for q in question_batch:
            # Fresh iid shuffle PER ANSWER (on a copy)
            likert = base_likert_scale[:]  # copy
            if likert_shuffle:
                random.shuffle(likert)

            messages = self.render_messages(
                hexaco_template,
                attributes=persona_str,
                text=q,
                likert_scale=", ".join(likert),
            )
            prompt = self.messages_to_prompt_text(messages)

            prompts_text.append(prompt)
            guided_choices_list.append(likert)     # constrain decoding to what's shown
            likert_orders_list.append(likert)      # store ordering for audit

        return prompts_text, guided_choices_list, likert_orders_list

    # ----------------------------
    # Experiment loop
    # ----------------------------
    def generate_answers(
        self,
        persona_datasets: Optional[Union[Dataset, List[Dict[str, Any]]]],
        question_list: List[str],
        base_likert_scale: List[str],
    ) -> HexacoExperimentResults:
        if self.args.batching:
            raise NotImplementedError("Batching not implemented (kept consistent with your script).")

        results: Dict[str, HexacoRunResult] = {}

        # Select template + persona list
        if persona_datasets is None:
            print("Running HEXACO on Base Model without Personas")
            personas_iter = [{"uuid": "base_model", "persona_string": ""}]
            hexaco_template = self.base_hexaco_template
        else:
            print("Running HEXACO with Personas")
            personas_iter = persona_datasets
            hexaco_template = self.persona_hexaco_template

        if self.args.likert_shuffle:
            print("Shuffling likert scale for every question (iid per answer)")
        else:
            print("Using the default likert scale order without shuffling")

        paraphrase_mode: Literal["paraphrased", "normal"] = "paraphrased" if self.args.paraphrase else "normal"
        refusal_mode: Literal["refusal", "no refusal"] = "no refusal" if self.args.no_refusal else "refusal"

        if self.args.inverted_likert:
            likert_mode: Literal["inverted", "shuffle", "normal"] = "inverted"
        elif self.args.likert_shuffle:
            likert_mode = "shuffle"
        else:
            likert_mode = "normal"

        for persona in tqdm(personas_iter, desc="Personas"):
            persona_str = persona["persona_string"]
            persona_id = persona.get("uuid", "base_model")
            persona_hash = None if persona_id == "base_model" else persona["persona_hash"]

            repeated_answers: List[List[str]] = []
            repeated_raw_prompts: List[List[str]] = []
            repeated_guided: List[List[List[str]]] = []
            repeated_likert_orders: List[List[List[str]]] = []

            for _ in tqdm(range(self.args.n_times), desc="Iterations"):
                all_prompts: List[str] = []
                all_guided: List[List[str]] = []
                all_likert_orders: List[List[str]] = []

                for q_batch in tqdm(self._batch_list(question_list, self.args.batch_size), desc="Batches"):
                    prompts_text, guided_choices_list_of_lists, likert_orders_list_of_lists = self.build_prompts_for_questions(
                        hexaco_template=hexaco_template,
                        question_batch=q_batch,
                        base_likert_scale=base_likert_scale,
                        persona_str=persona_str,
                        likert_shuffle=self.args.likert_shuffle,
                    )
                    all_prompts.extend(prompts_text)
                    all_guided.extend(guided_choices_list_of_lists)
                    all_likert_orders.extend(likert_orders_list_of_lists)

                print(f"Passing {len(all_prompts)} prompts and {len(all_guided)} guided choices.")
                raw_texts = self.vllm_generate_batched(all_prompts, guided_choices=all_guided)
                answers = [self._normalize_answer(t, opts) for t, opts in zip(raw_texts, all_guided)]

                repeated_answers.append(answers)
                repeated_raw_prompts.append(all_prompts)
                repeated_guided.append(all_guided)
                repeated_likert_orders.append(all_likert_orders)

            cfg = HexacoRunConfig(
                persona=persona_id,
                persona_hash=persona_hash,
                paraphrase=paraphrase_mode,
                likert_scale_mode=likert_mode,
                refusal_allowed=refusal_mode,
                model_name=self.model,
                raw_prompts=repeated_raw_prompts,
                guided_choices=repeated_guided,
                likert_orders=repeated_likert_orders,
            )
            results[persona_id] = HexacoRunResult(config=cfg, answers=repeated_answers)

        return HexacoExperimentResults(__root__=results)

    # ----------------------------
    # Orchestrate
    # ----------------------------
    def run(self) -> str:
        print(f"Model Used: {self.model}")
        print(f"Writing Output in: {self.args.out_dir}")

        self.load_prompt_templates()
        question_list, likert_scale = self.load_questions_and_likert()
        persona_datasets = self.load_personas()

        results = self.generate_answers(
            persona_datasets=persona_datasets,
            question_list=question_list,
            base_likert_scale=likert_scale,
        )

        model_name_safe = self.model.replace(".", "_").split("/")[-1]
        out_file = os.path.join(self.args.out_dir, f"{self.args.persona_source}_hexaco_answers_{model_name_safe}.json")
        self._write_json(results.to_jsonable(), out_file)
        print(f"Results saved to {out_file}")
        return out_file


# =========================
# Main (manager created here; does NOT start server)
# =========================

def main():
    args = HexacoResponseRunner.parse_args()

    mgr = VLLMServerManager(
        model=args.model_name,
        host=args.vllm_host,
        port=args.vllm_port,
        timeout_s=args.vllm_timeout_s,
        kill_existing=False,     # do not kill anything
        server_extra_args=[],    # do not start anything
    )

    runner = HexacoResponseRunner(args=args, mgr=mgr)
    runner.run()


if __name__ == "__main__":
    main()
