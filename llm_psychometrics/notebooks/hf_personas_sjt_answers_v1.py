import argparse
import json
import os
import re
import sys
from typing import List, Literal
import random
import torch as t
import transformers
from jinja2 import Template
from outlines import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from huggingface_hub import login
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import OpenAI
import yaml
from concurrent.futures import ThreadPoolExecutor


# Custom imports
sys.path.append("../")
from src.utils_v0 import list_to_str
from src.prompt_templates.sjt_base_prompt_templates import sjt_base_prompt_templates
from src.prompt_templates.sjt_persona_prompt_templates import sjt_persona_prompt_templates

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
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct",
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
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Device: {device}")

sjt_answer_options = ["1", "2", "3", "4", "5", "6"]
default_answer_option_ordering = ['honesty_humility_option',
                                  'emotionality_option',
                                  'extraversion_option',
                                  'agreeableness_option',
                                  'conscientiousness_option',
                                  'openness_option']


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model Name")
    parser.add_argument("--persona-source", type=str, default="huggingface",
                        help="Source of Persona Being Used (base_model | huggingface)")
    parser.add_argument("--hf-persona-path", type=str, default="thoughtworks/psychometric_personas_temp",
                        help="HF Path for Personas")
    parser.add_argument("--hf-sjt-path", type=str, default="thoughtworks/psychometric_SJTs",
                        help="HF Path for SJTs")
    parser.add_argument("--sjt-dir", type=str, default=None,
                        help="Source directory for Synthetic SJTs")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Huggingface token")
    parser.add_argument("--batching", action="store_true",
                        help="Enable batching mode (default: False)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of questions per batch when batching is enabled")
    parser.add_argument("--n-times", type=int, default=1,
                        help="Number of repetitions per persona")
    parser.add_argument("--n-sjtsample", type=int, default=1,
                        help="Number of SJTs to be sampled")
    parser.add_argument("--n-personasample", type=int, default=1,
                        help="Number of Personas to be sampled for each archetype")
    parser.add_argument("--answer-shuffle", action="store_true",
                        help="Enabling Shuffling of answer index for SJTs (default: False)")
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


# class LocalResponse(BaseModel):
#     response: str


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


def load_sjt(args):
    print("Using Huggingface SJTs")
    print(f"Loading SJTs from {args.hf_sjt_path}")
    hf_sjt_dataset = load_dataset(args.hf_sjt_path)
    sjt_datasets_total = hf_sjt_dataset['train']
    total_sjt_df = sjt_datasets_total.to_pandas()
    sampled_sjt = total_sjt_df.groupby("template_no").sample(n=args.n_sjtsample, random_state=42)
    sjt_datasets = sampled_sjt.to_dict("records")
    print(f"No of SJTs: {len(sjt_datasets)}")
    # sjt_list = []
    # for file in os.listdir(args.hf_sjt_path):
    #     synthetic_sjt = read_json(os.path.join(args.sjt_dir, file))
    #     sampled_sjts = random.sample(synthetic_sjt, args.n_sjtsample)
    #     sjt_list += sampled_sjts

    # print(f"{len(sjt_list)} SJTs Loaded")
    # return sjt_list
    return sjt_datasets

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
        persona_datasets_total = read_json("../data/persona_llm_paper_seed_combinations.json")
        random.seed(42)
        persona_datasets = random.sample(persona_datasets_total, args.n_personasample)
        print(f"No of Personas: {len(persona_datasets)}")
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


# ----------------------------
# Answer Generation
# ----------------------------
def openai_answer(model, prompt: List[dict], answer_options: List):
    """
    Minimal change: still called openai_answer(model, prompt) but now
    `prompt` is a list of OpenAI chat messages, and `model` is a bundle from load_model.
    Returns the assistant text directly.
    """
    client = model["client"]
    model_name = model["model_name"]
    # Dynamically constrain to one of the SJT Answer Order
    AnswerLiteral = Literal[tuple(answer_options)]

    class SJTAnswer(BaseModel):
        answer: AnswerLiteral
    resp = client.responses.parse(
        model=model_name,
        input=prompt,               # list of messages, unchanged
        text_format=SJTAnswer
    )

    parsed = resp.output_parsed    # SJTAnswer instance
    return parsed.answer

def local_answer(model, prompt, answer_options: List[str]):
    """
    Use vLLM guided_choice to constrain the output to one of answer_options.
    Works with an OpenAI-compatible client pointed at your vLLM server.
    """

    model_name = model["model_name"].strip()

    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]

    # print(f"Working with {model_name}")
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

def generation_function(model, sjt_template, question_batch,answer_index ,batching=False,
                        persona_str=None, answer_shuffle=False):
    """Generate answers for a batch of questions."""
    if batching:
        raise NotImplementedError("Batching not implemented")

    # Non-batched
    prompt_list = []
    hash_list = []
    answer_index_list = []
    for sjt_dict in question_batch:
        
        if answer_shuffle:
            random.shuffle(answer_index)
        answer_index_list.append(list(answer_index))

        sjt = sjt_dict['corrected_sjt']

        answer_options = [sjt[key] for key in default_answer_option_ordering if "_option" in key]
        answer_options = [answer_options[idx] for idx in answer_index]
        question = sjt['question']
        
        prompt = render_openai_messages(
                            sjt_template,
                            attributes=persona_str,
                            question=question,
                            answer_options=list_to_str(answer_options)
                )
        prompt_list.append(prompt)
        hash_list.append(sjt_dict['hash_id'])


    if "gpt" in args.model_name:
        with ThreadPoolExecutor(max_workers=12) as executor:
            return list(executor.map(lambda p: openai_answer(model, p, sjt_answer_options), prompt_list)), hash_list, answer_index_list
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(lambda p: local_answer(model, p, sjt_answer_options), prompt_list)), hash_list, answer_index_list


# ----------------------------
# Experiment Runner
# ----------------------------
def generate_answers(model, args, synthetic_sjts, persona_datasets=None, answer_shuffle=False):

    answer_index = [0, 1, 2, 3, 4, 5]
    
    if answer_shuffle:
        sjt_answer_options = "shuffle"
        print("Shuffling answer options for SJTs")
    else:
        sjt_answer_options = "normal"
        print("Default Ordering of answer options for SJTs")

    if persona_datasets:
        print(f"No of personas: {len(persona_datasets)}")
    else:
        print("Answering the SJTs using the base model, without any personas")
    print(f"No of SJTs: {len(synthetic_sjts)}")
    
    """Run experiments for base model or persona-conditioned runs."""
    if args.batching:
        print(f"Batching {args.batch_size} questions together")
    else:
        print("Passing one question per prompt")

    answers = {}

    if args.persona_source == "base_model":
        print("Running SJTs on Base Model without Personas")
        sjt_template = base_sjt_template
        repeated_answers = []

        for _ in tqdm(range(args.n_times), desc="Iterations"):
            persona_answer = []
            question_hashes = []
            answer_indexes = []
            for q_batch in tqdm(batch_list(synthetic_sjts, args.batch_size), desc="SJT Batches"):
                batch_answers, batch_hash_list, batch_answer_index_list = generation_function(model, sjt_template, q_batch,answer_index=answer_index,
                                        batching=args.batching,
                                        answer_shuffle=answer_shuffle)
                persona_answer.extend(batch_answers)
                question_hashes.extend(batch_hash_list)
                answer_indexes.extend(batch_answer_index_list)
            repeated_answers.append(persona_answer)

        answers['base_model'] = {
            'config': {
                'persona': "base_model",
                'question_hashes': question_hashes,
                'sjt_answer_options': sjt_answer_options,
                'answer_index': answer_indexes,
                'model_name': args.model_name
            },
            'answers': repeated_answers
        }

    else:
        print("Running SJTs with Personas")
        sjt_template = persona_sjt_template

        for persona_dataset in tqdm(persona_datasets, desc="Personas"):
            persona_str = persona_dataset['persona_string']
            
            repeated_answers = []

            for _ in tqdm(range(args.n_times), desc="Iterations"):
                persona_answer = []
                question_hashes = []
                answer_indexes = []
                for q_batch in tqdm(batch_list(synthetic_sjts, args.batch_size), desc="Batches"):
                    batch_answers, batch_hash_list, batch_answer_index_list = generation_function(model, sjt_template, q_batch,
                                            answer_index=answer_index,
                                            persona_str=persona_str,
                                            batching=args.batching,
                                            answer_shuffle=answer_shuffle)
                    persona_answer.extend(batch_answers)
                    question_hashes.extend(batch_hash_list)
                    answer_indexes.extend(batch_answer_index_list)
                repeated_answers.append(persona_answer)

            answers[persona_dataset['uuid']] = {
                'config': {
                    'persona': persona_dataset['uuid'],
                    'question_hashes': question_hashes,
                    'sjt_answer_options': sjt_answer_options,
                    'answer_index': answer_indexes,
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
        
    base_sjt_template, persona_sjt_template = load_prompt_templates(args.model_name)

    # Load model
    model = load_model(args.model_name, args.hf_token)

    # Load SJTs
    synthetic_sjts = load_sjt(args)

    # Load Personas
    persona_datasets = load_personas(args)


    # Run
    results = generate_answers(model, args, synthetic_sjts, persona_datasets, args.answer_shuffle)

    # Save results
    model_name = args.model_name.replace(".", "_").split("/")[-1]
    out_dir = "../experiment_results/reliability_experiments/vllm_experiment_6"
    out_file = os.path.join(out_dir,
                            f"{args.persona_source}_sjt_answers_{model_name}.json")
    write_to_json(results, out_file)
    print(f"Results saved to {out_file}")
