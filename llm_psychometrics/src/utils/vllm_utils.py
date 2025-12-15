import json
import os
import psutil
import socket
import subprocess
import sys
import time

import requests
from tqdm import tqdm


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

    def list_vllm_models(self) -> list[str]:
        """
        Return model IDs served by the vLLM server.
        """
        resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

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

    def vllm_chat(
        self, prompt: str, model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        """
        Send a prompt via the OpenAI-compatible chat endpoint.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def benchmark_tps(
            self,
            delay_s: int = 5,
            max_tokens: int = 1024,
            trials: int = 30,
            model_override: str | None = None,
    ) -> dict:
        """
        Benchmark dsdsa throughput after a post-startup delay.

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
        """
        Check whether the vLLM server is reachable.
        """
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass

        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
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
            "--gpu-memory-utilization", "0.7",
            "--max-num-batched-tokens", "70000",
            "--max-num-seqs", "100",
            "--enforce-eager", # Needed to prevent erroring out on cluster.
            "--disable-log-requests",
            "--max-model-len", "2048",
            "--disable-log-stats",
            "--enable-chunked-prefill",
            "--tensor-parallel-size", "1",
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


    def stable_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def vllm_chat(
        self,
        prompt: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _chat_one(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
        retry_backoff_s: float,
    ) -> str:
        last_err = None
        for attempt in range(max_retries):
            try:
                return self.vllm_chat(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                last_err = e
                time.sleep(retry_backoff_s * (2 ** attempt))
        raise RuntimeError("vllm_chat failed") from last_err

    def _mp_chat_chunk(
        self,
        prompts: List[str],
        model: str,
        max_tokens: int,
        temperature: float,
        num_workers: int,
        max_retries: int,
        retry_backoff_s: float,
    ) -> List[str]:
        results: List[Optional[str]] = [None] * len(prompts)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(
                    _mp_chat_one_worker,
                    self.base_url,
                    p,
                    model,
                    max_tokens,
                    temperature,
                    max_retries,
                    retry_backoff_s,
                ): idx
                for idx, p in enumerate(prompts)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return [r for r in results if r is not None]

    def vllm_chat_batched(
        self,
        prompts: List[str],
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 128,
        temperature: float = 0.0,
        batch_size: int = 100,
        num_workers: int = 50,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
    ) -> List[str]:
        outputs: List[str] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            outputs.extend(
                self._mp_chat_chunk(
                    prompts=chunk,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_workers=num_workers,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                )
            )
        return outputs


def _mp_chat_one_worker(
    base_url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_backoff_s: float,
) -> str:
    mgr = VLLMServerManager(base_url=base_url)
    last_err = None
    for attempt in range(max_retries):
        try:
            return mgr.vllm_chat(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            last_err = e
            time.sleep(retry_backoff_s * (2 ** attempt))
    raise RuntimeError("vllm_chat failed") from last_err


def make_math_prompts(n: int) -> List[str]:
    return [f"What is {i} + {i + 1}? Show your reasoning briefly." for i in range(n)]


def main():
    NUM_PROMPTS = 2000
    MP_WORKERS = 50
    BATCH_SIZE = 100

    out_jsonl = Path("outputs/debug_math_vllm_batching.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    mgr = VLLMServerManager(base_url="http://localhost:8000")
    prompts = make_math_prompts(NUM_PROMPTS)

    outputs = mgr.vllm_chat_batched(
        prompts=prompts,
        model="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=128,
        temperature=0.0,
        batch_size=BATCH_SIZE,
        num_workers=MP_WORKERS,
    )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, outputs):
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "prompt_hash": mgr.stable_hash(prompt),
                        "response": response,
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )