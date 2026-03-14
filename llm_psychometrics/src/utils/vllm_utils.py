import json
import os
import psutil
import socket
import subprocess
import hashlib
import json
import requests
import sys
import time
from dataclasses import dataclass
from collections import Counter
from pydantic import BaseModel, Field
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Union
import sqlite3
from contextlib import contextmanager
from diskcache import Cache
from datetime import datetime, UTC


# default_model = ""Qwen/Qwen2.5-7B-Instruct"
default_model = "google/gemma-3-4b-it"

from diskcache import Cache, FanoutCache

class VLLMServerManager:
    """
    Minimal manager for a local vLLM OpenAI-compatible server.
    - Kills any existing vLLM api_server processes (optional).
    - Starts a fresh server on host:port for the requested model.
    - Waits until /v1/models responds.
    """

    def __init__(self, model: str = default_model,
                 host: str = "127.0.0.1", port: int = 8000,
                 python_executable: str = sys.executable,
                 server_extra_args=None, env=None,
                 log_file: str = "outputs/vllm_server.log",
                 timeout_s: int = 500, kill_existing: bool = True):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.backup_url = f"http://{host}:8001"
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

    def _cache_key(
            self,
            prompt: str,
            model: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            guided_choices: Optional[List[str]] = None,
    ) -> str:
        guided_choices = guided_choices or []
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "guided_choices": guided_choices,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

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
            "--gpu-memory-utilization", "0.85",
            "--max-num-batched-tokens", "16384",
            "--max-num-seqs", "64",
            "--disable-log-requests",
            "--max-model-len", "4096",
            "--disable-log-stats",
            "--enable-chunked-prefill",
            "--tensor-parallel-size", "1",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "> /outputs/vllm_server.log",
        ] + self.server_extra_args

        stdout = open(self.log_file, "a", buffering=1, encoding="utf-8")
        self._proc = subprocess.Popen(cmd, stdout=stdout, stderr=stdout, env=self.env, start_new_session=True)
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        stdout = log_path.open("a", buffering=1, encoding="utf-8")

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


    def _normalize_guided_choices(
            self,
            prompts: List[str],
            guided_choices: Optional[Union[List[str], List[Optional[List[str]]]]],
    ) -> List[List[str]]:
        """
        Normalize guided choices into a per-prompt List[List[str]].
        Supports:
          - None -> [[]] * len(prompts)
          - List[str] -> same choices for all prompts
          - List[Optional[List[str]]] with len == len(prompts) -> per-prompt choices
        """
        n = len(prompts)
        if guided_choices is None:
            print(f"No guided choices provided")
            return [[] for _ in range(n)]

        # Case A: shared list[str] for all prompts
        if guided_choices and isinstance(guided_choices[0], str):  # type: ignore[index]
            shared = list(guided_choices)  # type: ignore[assignment]
            return [shared for _ in range(n)]

        # Case B: per-prompt list
        per = guided_choices  # type: ignore[assignment]
        if len(per) != n:
            raise ValueError(
                f"guided_choices must be List[str] (shared) OR List[Optional[List[str]]] with len==len(prompts). "
                f"Got len(guided_choices)={len(per)} vs len(prompts)={n}."
            )
        return [gc or [] for gc in per]

    
    def vllm_chat(
            self,
            prompt: str,
            model: str = default_model,
            max_tokens: int = 128,
            temperature: float = 0.0,
            top_p: float = 1.0,
            guided_choices: Optional[List[str]] = None,
            response_format: Optional[BaseModel] = None
    ) -> str:
        guided_choices = guided_choices or []
        choice_set = {c.strip() for c in guided_choices}

        def _normalize(resp: str) -> str:
            # strict + simple: first token only
            return resp.strip().split(None, 1)[0]

        last_err: Optional[Exception] = None
        had_mismatch = False

        for attempt in range(3):
            text = None
            payload = None
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }

                if guided_choices and guided_choices[0]:
                    payload["extra_body"] = {"guided_choice": guided_choices}
                    
                if response_format:
                    payload['response_format'] = {
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": response_format.__name__,
                                                        "schema": response_format.model_json_schema()
                                                    }
                                                }

                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload
                )
                if not resp.ok:
                    raise RuntimeError(
                        f"status={resp.status_code} body={resp.text}\n"
                        f"payload_preview={json.dumps(payload, ensure_ascii=False)[:3000]}"
                    )
                    resp.raise_for_status()

                text = resp.json()["choices"][0]["message"]["content"]
                
                self._log(
                    f"RAW OUTPUT "
                    f"attempt={attempt + 1} "
                    f"raw={text!r} "
                )

                # No constraint → return raw
                if not choice_set:
                    return text

                token = _normalize(text)
                
                # text, token = self.call_llm_mode(payload=payload, N=5, choice_set=choice_set)
                # token = str(np.argmax(json.loads(text)['answer']).item())
                # token = str(json.loads(text)['answer'][0])
                # token = str(json.loads(text)['answer'])
                self._log(f"Token: {token}"
                          f"Choice Set: {choice_set}")
                if token in choice_set:
                    if had_mismatch:
                        self._log(
                            f"GUIDED_CHOICES_RECOVERED "
                            f"attempt={attempt + 1} "
                            f"token={token!r}"
                        )
                    return token

                # Mismatch → retry
                had_mismatch = True
                last_err = ValueError(
                    f"guided_choices mismatch: raw={text!r}, token={token!r}"
                )
                self._log(
                    f"GUIDED_CHOICES_MISMATCH "
                    f"attempt={attempt + 1} "
                    f"raw={text!r} "
                    f"token={token!r}"
                )
                time.sleep(0.01 * (2 ** attempt))

            except Exception as e:
                last_err = e
                self._log(
                    f"VLLM_EXCEPTION "
                    f"attempt={attempt + 1} "
                    f"err={text or None} as output "
                    f"err={repr(e)} on payload "
                    f"{json.dumps(payload, indent=4)}"
                )
                time.sleep(0.01 * (2 ** attempt))

        self._log(
            "GUIDED_CHOICES_FINAL_FAILURE "
            f"retries=3 "
            f"last_err={repr(last_err)}"
        )
        raise RuntimeError(f"vllm_chat failed after 3 attempts: {last_err}") from last_err

    def _log(self, msg: str):
        ts = datetime.now(UTC).isoformat()

        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")


    def _open_cache(
        self,
        cache_dir: str,
        cache_type: str = "fanout",   # "fanout" (recommended for multiproc) or "cache"
        shards: int = 64,
        timeout: float = 1.0,
        **kwargs,
    ):
        """
        Returns a diskcache cache instance.

        FanoutCache is strongly preferred for multi-process writers/readers.
        - cache_type="fanout": FanoutCache(cache_dir, shards=..., timeout=...)
        - cache_type="cache":  Cache(cache_dir)
        """
        cache_type = (cache_type or "fanout").lower().strip()

        if cache_type in {"fanout", "fanoutcache", "fan_out", "fan-out"}:
            # FanoutCache expects a directory; it will create per-shard subdirs.
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            return FanoutCache(cache_dir, shards=int(shards), timeout=float(timeout), **kwargs)

        if cache_type in {"cache", "diskcache"}:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            return Cache(cache_dir, **kwargs)

        raise ValueError(f"Unknown cache_type={cache_type!r}. Use 'fanout' or 'cache'.")

    @staticmethod
    def _cache_get_many(cache, keys: List[str]) -> dict[str, str]:
        """
        diskcache Cache supports __contains__/__getitem__.
        FanoutCache supports .get similarly; safest is using .get with default.
        """
        out: dict[str, str] = {}
        for k in keys:
            v = cache.get(k, default=None)
            if v is not None:
                out[k] = v
        return out

    @staticmethod
    def _cache_set_many(cache, items: List[tuple[str, str]]):
        for k, v in items:
            cache.set(k, v)

    # ...

    def _mp_chat_chunk(
        self,
        prompts: List[str],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_workers: int,
        max_retries: int,
        retry_backoff_s: float,
        pbar: Optional[tqdm] = None,
        guided_choices: Optional[List[List[str]]] = None,  # per-prompt
        response_format: Optional[BaseModel] = None,
        cache_dir: Optional[str] = None,
        cache_keys: Optional[List[str]] = None,
        # --- NEW ---
        cache_type: str = "fanout",
        cache_shards: int = 64,
        cache_timeout: float = 1.0,
    ) -> List[str]:
        """
        Returns outputs in the exact same order as `prompts`.

        guided_choices: per-prompt List[List[str]] aligned with prompts.
        """
        n = len(prompts)
        results: List[Optional[str]] = [None] * n

        guided_choices = guided_choices or [[] for _ in range(n)]
        if len(guided_choices) != n:
            raise ValueError("guided_choices must be the same length as prompts (per-prompt).")

        args_list = [
            (
                idx, self.base_url, # if idx % 2 == 0 else self.backup_url,
                prompt, model, max_tokens, temperature,top_p,
                max_retries, retry_backoff_s, guided_choices[idx], response_format
            )
            for idx, prompt in enumerate(prompts)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_mp_chat_one_worker_args, args) for args in args_list]

            failures: List[str] = []
            for fut in as_completed(futures):
                try:
                    idx, text = fut.result()
                    results[idx] = text
                except Exception as e:
                    failures.append(repr(e))
                finally:
                    if pbar is not None:
                        pbar.update(1)

        if failures:
            raise RuntimeError(f"One or more vLLM worker calls failed (showing up to 3): {failures[:3]}")

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"Missing outputs for indices: {missing[:10]} (total missing={len(missing)})")

        out: List[str] = results  # type: ignore[assignment]

        # write-through cache (parent process only)
        if cache_dir is not None:
            if cache_keys is None or len(cache_keys) != len(out):
                raise ValueError("cache_keys must be provided and match prompts length when cache is enabled.")

            with self._open_cache(
                cache_dir=cache_dir,
                cache_type=cache_type,
                shards=cache_shards,
                timeout=cache_timeout,
            ) as cache:
                self._cache_set_many(cache, list(zip(cache_keys, out)))

        return out

    def vllm_chat_batched(
        self,
        prompts: List[str],
        guided_choices: Optional[Union[List[str], List[Optional[List[str]]]]] = None,
        response_format: Optional[BaseModel] = None,
        model: str = default_model,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 100,
        num_workers: int = 100,
        max_retries: int = 5,
        retry_backoff_s: float = 0.1,
        chunk_max_retries: int = 2,
        chunk_retry_backoff_s: float = 0.05,
        cache_dir: Optional[str] = ".vllm_cache/diskcache",
        cache_enabled: bool = True,
        # --- NEW ---
        cache_type: str = "fanout",   # "fanout" recommended for multiproc
        cache_shards: int = 64,
        cache_timeout: float = 1.0,
    ) -> List[str]:
        outputs: List[str] = []

        total = len(prompts)
        cache_hits_total = 0

        per_prompt_guidance: List[List[str]] = self._normalize_guided_choices(prompts, guided_choices)

        with tqdm(total=total, desc=f"vLLM completions for guided choices: {guided_choices}", unit="req") as pbar:
            for i in range(0, total, batch_size):
                chunk = prompts[i: i + batch_size]
                chunk_guidance = per_prompt_guidance[i: i + batch_size]

                chunk_keys = [
                    self._cache_key(
                        prompt=p,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        guided_choices=gc,
                    )
                    for p, gc in zip(chunk, chunk_guidance)
                ]

                # Read cache first
                cached: dict[str, str] = {}
                if cache_enabled and cache_dir:
                    with self._open_cache(
                        cache_dir=cache_dir,
                        cache_type=cache_type,
                        shards=cache_shards,
                        timeout=cache_timeout,
                    ) as cache:
                        cached = self._cache_get_many(cache, chunk_keys)

                chunk_out: List[Optional[str]] = [None] * len(chunk)

                miss_prompts: List[str] = []
                miss_keys: List[str] = []
                miss_positions: List[int] = []
                miss_guidance: List[List[str]] = []

                for j, (k, p, gc) in enumerate(zip(chunk_keys, chunk, chunk_guidance)):
                    if k in cached:
                        chunk_out[j] = cached[k]
                        cache_hits_total += 1
                    else:
                        miss_positions.append(j)
                        miss_prompts.append(p)
                        miss_keys.append(k)
                        miss_guidance.append(gc)

                # Fetch misses
                if miss_prompts:
                    last_err: Optional[Exception] = None
                    for attempt in range(chunk_max_retries + 1):
                        try:
                            miss_out = self._mp_chat_chunk(
                                prompts=miss_prompts,
                                model=model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                num_workers=num_workers,
                                max_retries=max_retries,
                                retry_backoff_s=retry_backoff_s,
                                pbar=None,
                                guided_choices=miss_guidance,
                                response_format=response_format,
                                cache_dir=(cache_dir if (cache_enabled and cache_dir) else None),
                                cache_keys=miss_keys,
                                # --- pass through cache config ---
                                cache_type=cache_type,
                                cache_shards=cache_shards,
                                cache_timeout=cache_timeout,
                            )

                            for pos, text in zip(miss_positions, miss_out):
                                chunk_out[pos] = text

                            pbar.update(len(chunk))
                            last_err = None
                            break
                        except RuntimeError as e:
                            last_err = e
                            if attempt >= chunk_max_retries:
                                raise
                            time.sleep(chunk_retry_backoff_s * (2 ** attempt))

                    if last_err is not None:
                        raise last_err
                else:
                    pbar.update(len(chunk))

                missing = [idx for idx, r in enumerate(chunk_out) if r is None]
                if missing:
                    raise RuntimeError(f"Internal error: missing chunk outputs at positions {missing[:10]}")

                outputs.extend([r for r in chunk_out if r is not None])

        if cache_enabled and cache_dir:
            hits = cache_hits_total
            self._log(f"[vllm_chat_batched] Cache hits: {hits}/{total} ({(100.0 * hits / max(1, total)):.1f}%)")

        return outputs


def _mp_chat_one_worker(
    idx: int,
    base_url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    retry_backoff_s: float,
    guided_choices: Optional[List[str]],
    response_format: Optional[BaseModel]
) -> tuple[int, str]:
    mgr = VLLMServerManager(
        host=base_url.split("://", 1)[1].split(":", 1)[0],
        port=int(base_url.rsplit(":", 1)[1]),
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            text = mgr.vllm_chat(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                guided_choices=guided_choices,
                response_format=response_format
            )
            return idx, text
        except Exception as e:
            last_err = e
            time.sleep(retry_backoff_s * (2 ** attempt))
    raise RuntimeError(f"vllm_chat failed for idx={idx}, error={last_err}") from last_err


def _mp_chat_one_worker_args(args):
    return _mp_chat_one_worker(*args)


def make_math_prompts(n: int) -> list[str]:
    return [f"""What is {i} + {i+1}? Give answer only.""" for i in range(n)]

def main():
    
    NUM_PROMPTS = 20
    MP_WORKERS = 100
    BATCH_SIZE = 100

    out_jsonl = Path("outputs/debug_math_vllm_batching.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    mgr = VLLMServerManager()
    # mgr.ensure_fresh_server()
    prompts = make_math_prompts(NUM_PROMPTS)

    guided_choices = [str(i) for i in range(1, 2*NUM_PROMPTS+1)]

    outputs = mgr.vllm_chat_batched(
        prompts=prompts,
        model=default_model,
        max_tokens=128,
        temperature=0.0,
        batch_size=BATCH_SIZE,
        num_workers=MP_WORKERS,
        guided_choices=guided_choices
    )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, outputs):
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "prompt_hash": mgr.stable_hash(prompt),
                        "response": response,
                        "model": default_model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

if __name__ == "__main__":
    main()
