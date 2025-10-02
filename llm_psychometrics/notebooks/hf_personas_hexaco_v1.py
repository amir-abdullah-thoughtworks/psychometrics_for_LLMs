import argparse
import json
import os
import re
import sys
from typing import List, Literal
import torch as t
import transformers
from outlines import Generator
from jinja2 import Template
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from huggingface_hub import login
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor
import random

# Custom imports
sys.path.append("../")
from src.utils_v0 import list_to_str, inverse_likert
from src.prompt_templates.hexaco_base_prompt_templates import hexaco_base_prompt_templates
from src.prompt_templates.hexaco_persona_prompt_templates import hexaco_persona_prompt_templates

import psutil
import subprocess
import time
import requests
import socket

vllm_client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="-",
)

class VLLMServerManager:
    """
    Minimal manager for a local vLLM OpenAI-compatible server.
    - Kills any existing vLLM api_server processes (optional).
    - Starts a fresh server on host:port for the requested model.
    - Waits until /v1/models responds.
    """
    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 host: str = "127.0.0.1", port: int = 8000,
                 python_executable: str = sys.executable,
                 server_extra_args=None, env=None,
                 log_file: str = "vllm_server.log",
                 timeout_s: int = 180, kill_existing: bool = True):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.python_executable = python_executable
        self.server_extra_args = server_extra_args or []
        self.env = {**os.environ, **(env or {})}
        self.log_file = log_file
        self.timeout_s = timeout_s
        self.kill_existing = kill_existing
        self._proc = None

    def ensure_fresh_server(self, run_benchmark: bool = False):
        if self.kill_existing:
            print("Killing existing vLLM server")
            self._kill_existing_servers()
        if not self._is_up():
            print("Starting vLLM server")
            self._start()
            print("Waiting for vLLM server to start")
            self._wait_ready()

        print("Running hello world check")
        self.hello_world_check()

        if run_benchmark:
            print("Sanity checking tokens per second")
            benchmark = self.benchmark_tps()
            with open("benchmark_tps.json", "w") as f:
                json.dump(benchmark, f)
            print(f"Finished benchmark with results \n {benchmark}")

    def benchmark_tps(
            self,
            delay_s: int = 5,
            max_tokens: int = 1024,
            trials: int = 30,
            model_override: str | None = None,
    ) -> dict:
        """
        Benchmark generation throughput after a post-startup delay.

        - Waits `delay_s` seconds (default 30) before benchmarking.
        - Runs `trials` chat.completions with `max_tokens` tokens each.
        - Uses /v1/chat/completions and reads usage.completion_tokens.
        - Returns a dict with per-trial stats and averaged TPS.

        NOTE: Assumes the server is already up (call ensure_fresh_server first).
        """
        import time
        import requests

        model_name = model_override or self.model
        url = f"{self.base_url}/v1/chat/completions"

        # Simple, high-entropy prompt to avoid early stopping
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Write a long, continuous stream of varied text about GPUs, kernels, "
                    "and parallelism without concluding, avoiding numbered lists. "
                    "Do not stop, ever."
                ),
            },
        ]

        # One-time post-startup delay (e.g., to let CUDA warm up)
        if delay_s > 0:
            time.sleep(delay_s)

        results = []
        for _ in tqdm(range(max(1, trials))):
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.0,
            }
            t0 = time.perf_counter()
            r = requests.post(url, json=payload, timeout=600)
            t1 = time.perf_counter()
            r.raise_for_status()
            data = r.json()

            # vLLM and OpenAI-compatible servers return usage.{completion_tokens,prompt_tokens,total_tokens}
            usage = data.get("usage", {}) or {}
            completion_tokens = int(usage.get("completion_tokens", 0))
            elapsed = max(t1 - t0, 1e-9)
            tps = completion_tokens / elapsed if completion_tokens else 0.0

            results.append(
                {
                    "completion_tokens": completion_tokens,
                    "elapsed_s": elapsed,
                    "tps": tps,
                }
            )

        avg_tps = sum(item["tps"] for item in results) / len(results)
        total_tokens = sum(item["completion_tokens"] for item in results)
        total_time = sum(item["elapsed_s"] for item in results)

        return {
            "model": model_name,
            "delay_s": delay_s,
            "trials": trials,
            "per_trial": results,
            "avg_tps": avg_tps,
            "total_tokens": total_tokens,
            "total_elapsed_s": total_time,
        }

    # ---------- internals ----------
    def _is_up(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=1.5)
            is_up = (r.status_code == 200)
            print(f"Server is up.")
        except Exception:
            print(f"Server is not up.")
            return False

    def _kill_existing_servers(self):
        pids = []
        for p in psutil.process_iter(["pid","cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                if "vllm" in cmd and "entrypoints" in cmd and "openai" in cmd and "api_server" in cmd:
                    print(f"Found existing vLLM server on {p.info['pid']}")
                    pids.append(p.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        procs = []
        for pid in pids:
            try:
                print(f"Attempting to terminate vLLM server on {pid}")
                pr = psutil.Process(pid)
                pr.terminate()
                procs.append(pr)
            except Exception:
                print(f"Failed to terminate vLLM server on {pid}")
                pass
        if procs:
            psutil.wait_procs(procs, timeout=5)
        # nudge port (best-effort)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((self.host, self.port))
            except Exception:
                pass  # not listening or already free

    def _start(self):
        print(f"Starting vLLM server on {self.host}:{self.port}")
        cmd = [
            self.python_executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--gpu-memory-utilization", "0.9",
            "--max-num-batched-tokens", "70000",
            "--max-num-seqs", "100",
            "--enforce-eager", # Needed to prevent erroring out on cluster.
            "--disable-log-requests",
            "--max-model-len", "2048",
            "--disable-log-stats",
            "--enable-chunked-prefill",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", "bfloat16",
        ] + self.server_extra_args
        stdout = open(self.log_file, "a", buffering=1, encoding="utf-8")
        self._proc = subprocess.Popen(cmd, stdout=stdout, stderr=stdout, env=self.env, start_new_session=True)

    def hello_world_check(self, model_override: str | None = None) -> str:
        """
        Run a quick smoke test once the server is up:
        Ask 'What is 2+2?' and return the model's raw output string.
        Caller can check `'4' in output` to validate.
        """
        import requests

        model_name = model_override or self.model
        url = f"{self.base_url}/v1/chat/completions"

        messages = [
            {"role": "system", "content": "You are a math assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 16,
        }

        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        output = (data["choices"][0]["message"]["content"] or "").strip()
        if "4" in output:
            print(f"Hello world check successful returned 2+2='{output}'")
        else:
            print(f"Hello world check unexpected 2+2={output}")

        return output

    def _wait_ready(self):
        start = time.time()
        last_err = None
        while time.time() - start < self.timeout_s:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {self._proc.returncode}. Check {self.log_file}.")
            try:
                r = requests.get(f"{self.base_url}/v1/models", timeout=2)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            time.sleep(1.0)
        raise TimeoutError(f"Timed out waiting for vLLM at {self.base_url} ({last_err})")


# ----------------------------
# Setup
# ----------------------------
transformers.logging.set_verbosity_error()

NO_ANSWER = "Do not wish to answer"


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model Name")
    parser.add_argument("--persona-source", type=str, default="huggingface",
                        help="Source of Persona (huggingface | base_model | personallm_paper)")
    parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas_temp",
                        help="HF Path for Personas")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Huggingface token")
    parser.add_argument("--batching", action="store_true",
                        help="Enable batching mode (default: False)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of questions per batch")
    parser.add_argument("--n-times", type=int, default=1,
                        help="Number of repetitions per persona")
    parser.add_argument("--paraphrase", action="store_true",
                        help="Use paraphrased versions of HEXACO (default: False)")
    parser.add_argument("--n-personasample", type=int, default=1,
                        help="Number of Personas to be sampled for each archetype")
    parser.add_argument("--inverted-likert", action="store_true",
                        help="Whether Likert Scale needs to be inverted or not")
    parser.add_argument("--no-refusal", action="store_true",
                        help="Whether Refusal is allowed or not")
    parser.add_argument("--likert-shuffle", action="store_true",
                        help="Enabling Shuffling of likert scale for hexaco (default: False)")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "vllm"],
                        help="Backend provider: openai (API), vllm (OpenAI-compatible server), or hf (local Transformers).")
    parser.add_argument("--vllm-base-url", type=str, default="http://127.0.0.1:8000/v1",
                        help="Base URL for vLLM OpenAI-compatible server, e.g., http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key", type=str, default="-",
                        help="API key to send to vLLM (usually ignored but required by the OpenAI client).")
    return parser.parse_args()


# ----------------------------
# Utility Functions
# ----------------------------
def write_to_json(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(file, f, indent=2)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def batch_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ----------------------------
# Model Setup
# ----------------------------
class OpenaiResponse(BaseModel):
    response: str


def load_model(model_name: str, hf_token: str = None):
    """Load OpenAI or vLLM (OpenAI-compatible) client. Signature unchanged."""
    # Do not use HF; keep hf_token param for compatibility.
    if args.provider == "openai":
        client = OpenAI()  # uses OPENAI_API_KEY from environment
    else:

        client = OpenAI(base_url=args.vllm_base_url.rstrip("/"), api_key=args.vllm_api_key)
    # NOTE: add provider so local_answer can branch to structured outputs for vLLM
    return {"client": client, "model_name": model_name, "provider": args.provider}


# ----------------------------
# Data Setup
# ----------------------------
def load_data(args):
    with open('../configs/generation_config.yaml', 'r') as file:
        generation_config = yaml.safe_load(file)

    with open('../psychometric_tests/hexaco_100_questions.yaml', 'r') as file:
        question_list = yaml.safe_load(file)

    with open('../psychometric_tests/paraphrased_hexaco_100_questions.yaml', 'r') as file:
        paraphrased_question_list = yaml.safe_load(file)

    # with open('../psychometric_tests/hexaco_100_eval.yaml', 'r') as file:
    #     hexaco_eval = yaml.safe_load(file)

    # with open('../configs/personas_v2.yaml', 'r') as file:
    #     personas = yaml.safe_load(file)

    # Scale
    if args.inverted_likert:
        print("Inverting Likert")
        likert_scale = inverse_likert(generation_config['likert_scale'].copy())
    else:
        print("Normal Likert")
        likert_scale = generation_config['likert_scale'].copy()

    if args.no_refusal:
        print("Refusal Not Allowed")
    else:
        print("Refusal Allowed")
        likert_scale.append(NO_ANSWER)

    if args.paraphrase:
        print("Using Paraphrased Questions")
        question_list = paraphrased_question_list
    else:
        print("Using Normal Questions")

    return question_list, likert_scale

def load_prompt_templates(model_name):
    """
    Use chat-style (OpenAI/vLLM) templates for all models to avoid Outlines/HF.
    Signature unchanged.
    """
    # We ignore non-GPT branches to keep everything in chat format.
    print("Loading Prompt Templates for Chat-style (OpenAI/vLLM) models")
    base_templates = hexaco_base_prompt_templates["gpt"]
    persona_templates = hexaco_persona_prompt_templates["gpt"]

    # compile GPT messages into Jinja templates
    def compile_message_templates(messages):
        return [
            {"role": msg["role"], "content": Template(msg["content"])}
            for msg in messages
        ]

    base_hexaco_template = compile_message_templates(base_templates)
    persona_hexaco_template = compile_message_templates(persona_templates)
    return base_hexaco_template, persona_hexaco_template


def load_personas(args):
    if args.persona_source == "huggingface":
        print("Using Huggingface Personas")
        print(f"Loading Personas from {args.hf_persona_path}")
        hf_persona_dataset = load_dataset(args.hf_persona_path)
        persona_datasets_total = hf_persona_dataset['train']
        total_persona_df = persona_datasets_total.to_pandas()
        sampled_personas = total_persona_df.groupby("archetype").sample(n=args.n_personasample, random_state=42)
        persona_datasets = Dataset.from_pandas(sampled_personas)
        print(f"No of Personas: {len(persona_datasets)}")
    elif args.persona_source == "personallm_paper":
        print("Using Persona LLM Paper Personas")
        persona_datasets = read_json("../data/persona_llm_paper_seed_combinations.json")
    elif args.persona_source == "base_model":
        return None
    else:
        print("Using Local Personas")
        with open('../configs/personas_v2.yaml', 'r') as file:
            local_personas = yaml.safe_load(file)

        job_title = 'law_enforcement'
        persona_datasets = local_personas[job_title]['personas']
        print(f"No of Personas: {len(persona_datasets)}")

    return persona_datasets


def openai_answer(model, prompt: List[dict], answer_options: List):
    """
    Minimal change: still called openai_answer(model, prompt) but now
    `prompt` is a list of OpenAI chat messages, and `model` is a bundle from load_model.
    Returns the assistant text directly.
    """
    client = model["client"]
    model_name = model["model_name"]
    # Dynamically constrain to one of the Likert strings
    AnswerLiteral = Literal[tuple(answer_options)]

    class LikertAnswer(BaseModel):
        answer: AnswerLiteral
    resp = client.responses.parse(
        model=model_name,
        input=prompt,               # list of messages, unchanged
        text_format=LikertAnswer
    )

    parsed = resp.output_parsed    # LikertAnswer instance
    return parsed.answer

    content = (resp.choices[0].message.content or "").strip()
    return content

def local_answer(model, prompt, answer_options: List[str]):
    """
    Use vLLM guided_choice to constrain the output to one of answer_options.
    Works with an OpenAI-compatible client pointed at your vLLM server.
    """

    model_name = model["model_name"].strip()

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]

    resp = vllm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=16,  # small cap; response is a single choice
        extra_body={"guided_choice": answer_options},
    )

    text = resp.choices[0].message.content.strip()

    # Safety: normalize and map back to the exact canonical option
    norm_map = {opt.strip().lower(): opt for opt in answer_options}
    return norm_map.get(text.lower(), text)

def render_openai_messages(template_messages, **kwargs):
    rendered =  [
        {"role": msg["role"], "content": msg["content"].render(**kwargs)}
        for msg in template_messages
    ]
    return rendered

def generation_function(model, hexaco_template, question_batch, likert_scale,
                        batching=False, persona_str=None, persona_base_text=None, likert_shuffle=False):

    if batching:
        raise NotImplementedError("Batching not implemented")

    # Non-batched
    prompt_list = []
    for question in question_batch:
        if likert_shuffle:
            random.shuffle(likert_scale)


        prompt = render_openai_messages(
                            hexaco_template,
                            attributes=persona_str,
                            text=question,
                            likert_scale=", ".join(likert_scale),
                )
        prompt_list.append(prompt)

    if "gpt" in args.model_name:
        with ThreadPoolExecutor(max_workers=12) as executor:
            return list(executor.map(lambda p: openai_answer(model, p, likert_scale), prompt_list))
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(lambda p: local_answer(model, p, likert_scale), prompt_list))

# ----------------------------
# Experiment Runner
# ----------------------------
def generate_answers(model, args, persona_datasets, question_list,
                     likert_scale):
    answers = {}
    hexaco_template = persona_hexaco_template
    base_text = "You are a law enforcement officer."

    if persona_datasets is None:
        print("Running HEXACO on Base Model without Personas")
        persona_datasets = [{"uuid": "base_model", "persona_text": ""}]
        hexaco_template = base_hexaco_template

    if args.batching:
        print(f"Batching {args.batch_size} questions together")
    else:
        print("Passing one question per prompt")

    if args.likert_shuffle:
        print("Shuffling likert scale for every persona and question combination")
    else:
        print("Using the default likert scale order without shuffling")

    for persona_dataset in tqdm(persona_datasets, desc="Personas"):
        persona_str = persona_dataset.get('persona_string', "")
        persona_id = persona_dataset.get('uuid', "base_model")

        repeated_answers = []
        for _ in tqdm(range(args.n_times), desc="Iterations"):
            persona_answer = []
            for question_batch in tqdm(batch_list(question_list, args.batch_size), desc="Batches"):
                batch_answers = generation_function(model, hexaco_template,
                                                    question_batch, likert_scale,
                                                    persona_str=persona_str,
                                                    persona_base_text=base_text,
                                                    batching=args.batching,
                                                    likert_shuffle=args.likert_shuffle)
                persona_answer.extend(batch_answers)
            repeated_answers.append(persona_answer)

        if args.inverted_likert:
            likert = "inverted"
        elif args.likert_shuffle:
            likert = "shuffle"
        else:
            likert = "normal"
        
        answers[persona_id] = {
            'config': {
                'persona': persona_id,
                'paraphrase': "paraphrased" if args.paraphrase else "normal",
                "likert_scale": likert,
                "refusal_allowed": "no refusal" if args.no_refusal else "refusal",
                'model_name': args.model_name
            },
            'answers': repeated_answers
        }

    return answers


# ----------------------------
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":

    args = parse_args()
    print(f"Model Used: {args.model_name}")

    # --- NEW: auto-boot vLLM server when provider==vllm ---
    if args.provider == "vllm":
        from urllib.parse import urlparse
        u = urlparse(args.vllm_base_url)
        host = u.hostname or "127.0.0.1"
        port = u.port or 8000

        # Optional: pass extra CLI flags via env var VLLM_SERVER_ARGS="--gpu-memory-utilization 0.9 --dtype bfloat16"
        server_extra_args = os.environ.get("VLLM_SERVER_ARGS", "").strip().split()
        vllm_mgr = VLLMServerManager(
            model=args.model_name,
            host=host,
            port=port,
            timeout_s=180,
            kill_existing=True,
            server_extra_args=server_extra_args,
        )
        vllm_mgr.ensure_fresh_server()
    
    base_hexaco_template, persona_hexaco_template = load_prompt_templates(args.model_name)

    # Load model
    model = load_model(args.model_name, args.hf_token)

    # Load data
    question_list, likert_scale = load_data(args)

    # Load persona
    persona_datasets = load_personas(args)

    print(f"model name is {args.model_name}.")
    # Run
    results = generate_answers(model, args, persona_datasets, question_list, likert_scale)

    # Save results
    model_name = args.model_name.replace(".", "_").split("/")[-1]
    out_dir = "../experiment_results/reliability_experiments/vllm_experiment_1"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.persona_source}_hexaco_answers_{model_name}.json")
    write_to_json(results, out_file)
    print(f"Results saved to {out_file}")
