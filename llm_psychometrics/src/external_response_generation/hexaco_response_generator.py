"""
vLLM-only, class-based HEXACO response generator (single consolidated file).

Repo layout (given):
root/
  configs/
  data/
  src/
    external_response_generation/
      hexaco_response_generator/   <-- this file lives here
  psychometric_tests/

This version:
- Resolves paths relative to *this file* (not CWD).
- Uses ROOT_DIR = 3 levels up from this file (hexaco_response_generator -> external_response_generation -> src -> root).
- Loads from:
    ROOT_DIR/configs/...
    ROOT_DIR/data/...
    ROOT_DIR/psychometric_tests/...
- Keeps imports robust by inserting ROOT_DIR/src into sys.path.
- Runs two conditions automatically:
    1) persona-conditioned -> pushed to HF config: hexaco_analysis
    2) base model only     -> pushed to HF config: hexaco_base
- Defaults to 5 iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from datetime import datetime, timezone
from jinja2 import Template
from pydantic import BaseModel, RootModel
from tqdm import tqdm

# =========================
# Path resolution (relative to this file)
# =========================

THIS_FILE = Path(__file__).resolve()
# this file is at: root/src/external_response_generation/hexaco_response_generator.py
ROOT_DIR = THIS_FILE.parents[2]  # external_response_generation -> src -> root

DATA_DIR = ROOT_DIR / "data"
PSYCHOMETRIC_DIR = ROOT_DIR / "psychometric_tests"
SRC_DIR = ROOT_DIR / "src"
CONFIGS_DIR = ROOT_DIR / "configs"

# Ensure imports work no matter where script is run from
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Custom imports (these live under root/src)
from utils_v0 import inverse_likert  # type: ignore
from utils.vllm_utils import VLLMServerManager  # type: ignore
from prompt_templates.hexaco_base_prompt_templates import hexaco_base_prompt_templates  # type: ignore
from prompt_templates.hexaco_persona_prompt_templates import hexaco_persona_prompt_templates  # type: ignore

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


class HexacoExperimentResults(RootModel[Dict[str, HexacoRunResult]]):
    """
    persona_uuid (or 'base_model') -> HexacoRunResult
    """
    def to_jsonable(self) -> Dict[str, Any]:
        return {k: v.model_dump() for k, v in self.root.items()}


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
    @staticmethod
    def stable_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ----------------------------
    # CLI
    # ----------------------------
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        parser.add_argument("--model-name", type=str, default="google/gemma-3-4b-it")

        parser.add_argument(
            "--persona-source",
            type=str,
            default="huggingface",
            choices=["huggingface", "base_model", "personallm_paper", "local"],
            help="Source of Persona (huggingface | base_model | personallm_paper | local)",
        )
        parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas")
        parser.add_argument("--hf-persona-config", type=str, default="test_sample")
        parser.add_argument("--hf-persona-split", type=str, default="train")
        parser.add_argument("--n-personasample", type=int, default=500)

        # question generation options
        parser.add_argument("--paraphrase", action="store_true", help="Use paraphrased HEXACO questions")
        parser.add_argument("--inverted-likert", action="store_true", help="Invert Likert scale")
        parser.add_argument("--no-refusal", action="store_true", help="Disallow refusal option")
        parser.add_argument(
            "--likert-shuffle",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Shuffle likert scale per answer (default True)",
        )

        # batching (kept to match old interface; we still do one question per prompt)
        parser.add_argument("--batching", action="store_true")
        parser.add_argument("--batch-size", type=int, default=5)

        parser.add_argument("--n-times", type=int, default=5)

        # Outdir default: repo-root/outputs
        parser.add_argument("--out-dir", type=str, default=str(ROOT_DIR / "outputs"))

        # vLLM generation params
        parser.add_argument("--max-tokens", type=int, default=16)
        parser.add_argument("--temperature", type=float, default=0.0)

        # batching inside vllm_chat_batched
        parser.add_argument("--mp-batch-size", type=int, default=256)
        parser.add_argument("--mp-workers", type=int, default=8)

        # retry behavior
        parser.add_argument("--max-retries", type=int, default=3)
        parser.add_argument("--retry-backoff-s", type=float, default=0.1)

        # manager connection details
        parser.add_argument("--vllm-host", type=str, default="127.0.0.1")
        parser.add_argument("--vllm-port", type=int, default=8000)
        parser.add_argument("--vllm-timeout-s", type=int, default=180)

        parser.add_argument(
            "--debug",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Debug mode (default True): only run/push first 10 personas.",
        )

        parser.add_argument(
            "--push-to-hub",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Push results to Hugging Face Hub (default True).",
        )

        parser.add_argument(
            "--hub-repo",
            type=str,
            default="thoughtworks/gemma_psychometrics_personas_responses",
            help="HF dataset repo to push to.",
        )

        parser.add_argument(
            "--analysis-config",
            type=str,
            default="analysis_hexaco",
            help="HF config name for persona-conditioned runs.",
        )

        parser.add_argument(
            "--base-config",
            type=str,
            default="base_hexaco",
            help="HF config name for base-model runs.",
        )

        parser.add_argument(
            "--hub-split",
            type=str,
            default="train",
            help="HF split name inside each config.",
        )

        return parser.parse_args()

    # ----------------------------
    # Utils
    # ----------------------------
    @staticmethod
    def _write_json(obj: Dict[str, Any], file_path: str) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _read_json(file_path: Union[str, Path]) -> Any:
        p = Path(file_path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _batch_list(lst: List[Any], n: int):
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

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
        gen_cfg_path = CONFIGS_DIR / "generation_config.yaml"
        q_path = PSYCHOMETRIC_DIR / "hexaco_100_questions.yaml"
        q_para_path = PSYCHOMETRIC_DIR / "paraphrased_hexaco_100_questions.yaml"

        with gen_cfg_path.open("r", encoding="utf-8") as f:
            generation_config = yaml.safe_load(f)

        with q_path.open("r", encoding="utf-8") as f:
            question_list = yaml.safe_load(f)

        with q_para_path.open("r", encoding="utf-8") as f:
            paraphrased_question_list = yaml.safe_load(f)

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

            split = getattr(self.args, "hf_persona_split", "train")
            persona_datasets_total = hf_persona_dataset[split]
            print(
                f"Loaded {len(persona_datasets_total)} Personas from "
                f"{self.args.hf_persona_path} config={hf_config} split={split}"
            )
            return persona_datasets_total

        if self.args.persona_source == "personallm_paper":
            print("Using Persona LLM Paper Personas")
            paper_json = DATA_DIR / "persona_llm_paper_seed_combinations.json"
            persona_datasets_total = self._read_json(paper_json)

            random.seed(42)
            persona_datasets = random.sample(persona_datasets_total, self.args.n_personasample)
            print(f"No of Personas: {len(persona_datasets)}")
            return persona_datasets

        if self.args.persona_source == "base_model":
            return None

        print("Using Local Personas")
        local_yaml = CONFIGS_DIR / "personas_v2.yaml"
        with local_yaml.open("r", encoding="utf-8") as f:
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
            cache_enabled=False,
            cache_type="diskcache",
            retry_backoff_s=self.retry_backoff_s,
            default_guided_choice=NO_ANSWER

        )
        return self._extract_texts(outputs)

    def push_results_to_hub(
        self,
        results: HexacoExperimentResults,
        repo_id: str,
        config_name: str,
        split_name: str = "train",
    ) -> None:
        """
        Push to HF datasets hub under the provided config name.
        One row per persona, storing config + answers + audit fields.
        """
        rows: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for persona_id, run_result in results.root.items():
            cfg = run_result.config

            rows.append(
                {
                    "persona_id": persona_id,
                    "persona_hash": cfg.persona_hash,
                    "model_name": cfg.model_name,
                    "paraphrase": cfg.paraphrase,
                    "likert_scale_mode": cfg.likert_scale_mode,
                    "refusal_allowed": cfg.refusal_allowed,
                    "n_times": len(run_result.answers),
                    "n_questions": (len(run_result.answers[0]) if run_result.answers else 0),
                    "answers": run_result.answers,
                    "raw_prompts": cfg.raw_prompts,
                    "guided_choices": cfg.guided_choices,
                    "likert_orders": cfg.likert_orders,
                    "created_at_utc": now_iso,
                    "persona_source": self.args.persona_source,
                    "hf_persona_path": getattr(self.args, "hf_persona_path", None),
                    "hf_persona_config": getattr(self.args, "hf_persona_config", None),
                    "hf_persona_split": getattr(self.args, "hf_persona_split", None),
                    "debug": bool(self.args.debug),
                }
            )

        ds = Dataset.from_list(rows)
        dsd = DatasetDict({split_name: ds})
        dsd.push_to_hub(repo_id, config_name=config_name)
        print(f"Pushed {len(rows)} rows to HF dataset {repo_id} config '{config_name}' split '{split_name}'")

    # ----------------------------
    # Prompt building
    # ----------------------------
    def build_prompts_for_questions(
        self,
        hexaco_template: List[Dict[str, Any]],
        question_batch: List[str],
        base_likert_scale: List[str],
        persona_str: Optional[str],
        likert_shuffle: bool,
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        prompts_text: List[str] = []
        guided_choices_list: List[List[str]] = []
        likert_orders_list: List[List[str]] = []

        for q in question_batch:
            likert = base_likert_scale[:]
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
            guided_choices_list.append(likert)
            likert_orders_list.append(likert)

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

        if persona_datasets is None:
            print("Running HEXACO on Base Model without Personas")
            personas_iter = [{"uuid": "base_model", "persona_string": "", "persona_hash": None}]
            hexaco_template = self.base_hexaco_template
        else:
            print("Running HEXACO with Personas")
            personas_iter = persona_datasets
            hexaco_template = self.persona_hexaco_template

        if self.args.debug:
            print("[DEBUG] Limiting to first 10 personas for generation + push.")
            try:
                if hasattr(personas_iter, "select"):
                    personas_iter = personas_iter.select(range(min(10, len(personas_iter))))
                else:
                    personas_iter = list(personas_iter)[:10]
            except Exception:
                personas_iter = list(personas_iter)[:10]

        print(f"Processing {len(personas_iter)} personas")

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
            persona_id = persona.get("uuid", "base_model")
            persona_str = persona.get("persona_string", "")
            persona_hash = None if persona_id == "base_model" else persona.get("persona_hash")

            repeated_answers: List[List[str]] = []
            repeated_raw_prompts: List[List[str]] = []
            repeated_guided: List[List[List[str]]] = []
            repeated_likert_orders: List[List[List[str]]] = []

            for _ in tqdm(range(self.args.n_times), desc="Iterations", leave=False):
                all_prompts: List[str] = []
                all_guided: List[List[str]] = []
                all_likert_orders: List[List[str]] = []

                for q_batch in tqdm(self._batch_list(question_list, self.args.batch_size), desc="Batches", leave=False):
                    prompts_text, guided_choices_list, likert_orders_list = self.build_prompts_for_questions(
                        hexaco_template=hexaco_template,
                        question_batch=q_batch,
                        base_likert_scale=base_likert_scale,
                        persona_str=persona_str,
                        likert_shuffle=self.args.likert_shuffle,
                    )
                    all_prompts.extend(prompts_text)
                    all_guided.extend(guided_choices_list)
                    all_likert_orders.extend(likert_orders_list)

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
            results[str(persona_id)] = HexacoRunResult(config=cfg, answers=repeated_answers)

        return HexacoExperimentResults(results)

    # ----------------------------
    # Single-condition run
    # ----------------------------
    def run_single_condition(
        self,
        *,
        persona_source: str,
        out_suffix: str,
        hub_config: str,
    ) -> str:
        original_persona_source = self.args.persona_source
        self.args.persona_source = persona_source

        print(f"\n==============================")
        print(f"Running condition: {hub_config}")
        print(f"persona_source={persona_source}")
        print(f"n_times={self.args.n_times}")
        print(f"==============================\n")

        self.load_prompt_templates()
        question_list, likert_scale = self.load_questions_and_likert()
        persona_datasets = self.load_personas()

        results = self.generate_answers(
            persona_datasets=persona_datasets,
            question_list=question_list,
            base_likert_scale=likert_scale,
        )

        model_name_safe = self.model.replace(".", "_").replace("/", "__")
        out_file = Path(self.args.out_dir) / f"{out_suffix}_{model_name_safe}.json"
        self._write_json(results.to_jsonable(), str(out_file))
        print(f"Results saved to {out_file}")

        if getattr(self.args, "push_to_hub", True):
            self.push_results_to_hub(
                results=results,
                repo_id=self.args.hub_repo,
                config_name=hub_config,
                split_name=self.args.hub_split,
            )

        self.args.persona_source = original_persona_source
        return str(out_file)

    # ----------------------------
    # Orchestrate both runs
    # ----------------------------
    def run(self) -> Dict[str, str]:
        print(f"Repo root: {ROOT_DIR}")
        print(f"Model Used: {self.model}")
        print(f"Writing Output in: {self.args.out_dir}")

        analysis_out = self.run_single_condition(
            persona_source="huggingface",
            out_suffix="analysis_hexaco",
            hub_config=self.args.analysis_config,
        )

        base_out = self.run_single_condition(
            persona_source="base_model",
            out_suffix="base_hexaco",
            hub_config=self.args.base_config,
        )

        return {
            "hexaco_analysis": analysis_out,
            "hexaco_base": base_out,
        }


# =========================
# Main
# =========================
def main():
    args = HexacoResponseRunner.parse_args()

    mgr = VLLMServerManager(
        model=args.model_name,
        host=args.vllm_host,
        port=args.vllm_port,
        timeout_s=args.vllm_timeout_s,
        kill_existing=False,
        server_extra_args=[],
    )

    runner = HexacoResponseRunner(args=args, mgr=mgr)
    outputs = runner.run()

    print("\nDone.")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
