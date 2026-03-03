from __future__ import annotations

import hashlib
import json

import math
import sys
import threading
import time
from typing import Callable, Any, Optional, List, Dict

from diskcache import Cache
from openai import OpenAI
from tqdm import tqdm

CACHE_VERSION = "v3"  # bump whenever prompts/schemas change materially

def _call_with_retries(
    call_fn: Callable[[bool], Any],
    *,
    tries: int = 5,
    base_sleep_s: float = 0.8,
    always_force_fresh_on_retry: bool = True,
) -> Any:
    """
    call_fn(force_fresh) -> resp

    Key behavior:
    - First attempt uses force_fresh=False (cache allowed).
    - Any retry uses force_fresh=True if always_force_fresh_on_retry=True,
      preventing infinite loops on a bad cached result.
    """
    last_exc: Optional[Exception] = None
    for i in range(tries):
        force_fresh = (i > 0 and always_force_fresh_on_retry)
        try:
            return call_fn(force_fresh)
        except Exception as e:
            last_exc = e
            time.sleep(base_sleep_s * (i + 1))
    assert last_exc is not None
    raise last_exc


def responses_parse_with_progress(
    client: OpenAI,
    *,
    model: str,
    input: List[Dict[str, str]],
    text_format: Any,
    temperature: float,
    top_p: float,
    desc: str,
    spinner: bool = True,
    calls_pbar: Optional[tqdm] = None,
) -> Any:
    sp = Spinner(desc=desc, enabled=spinner)
    sp.start()
    t0 = time.time()
    try:
        resp = client.responses.parse(
            model=model,
            input=input,
            temperature=temperature,
            top_p=top_p,
            text_format=text_format,
        )
    finally:
        sp.stop()

    dt = time.time() - t0
    usage = _extract_usage(resp)
    if usage:
        it = usage.get("input_tokens", "?")
        ot = usage.get("output_tokens", "?")
        tt = usage.get("total_tokens", "?")
        tqdm.write(f"[DONE] {desc} | {dt:0.1f}s | tokens in={it} out={ot} total={tt}")
    else:
        tqdm.write(f"[DONE] {desc} | {dt:0.1f}s | token usage unavailable")

    if calls_pbar is not None:
        calls_pbar.update(1)

    return resp


def cache_key_for_parse(
    *,
    model: str,
    temperature: float,
    top_p: float,
    text_format: Any,
    input_messages: List[Dict[str, str]],
) -> str:
    fmt_name = getattr(text_format, "__name__", str(text_format))
    payload = {
        "cache_version": CACHE_VERSION,
        "kind": "responses.parse",
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "text_format": fmt_name,
        "messages": input_messages,
    }
    return stable_hash(payload)


class CacheLoopGuard:
    """
    Prevents infinite retry loops due to repeatedly reusing the same bad cached object.

    Rule:
    - If we hit the same cache key twice in a row (consecutive calls), force fresh on the next attempt.
    - If cached payload fails validation, evict it and force fresh.
    """
    def __init__(self):
        self._last_cache_key: Optional[str] = None
        self._last_was_cache_hit: bool = False

    def should_force_fresh_due_to_repeat_hit(self, key: str, cache_hit: bool) -> bool:
        if cache_hit and self._last_was_cache_hit and (self._last_cache_key == key):
            return True
        return False

    def update(self, key: str, cache_hit: bool):
        self._last_cache_key = key
        self._last_was_cache_hit = cache_hit


def responses_parse_cached_safe(
    client: OpenAI,
    cache: Cache,
    guard: CacheLoopGuard,
    *,
    model: str,
    input: List[Dict[str, str]],
    text_format: Any,
    temperature: float,
    top_p: float,
    desc: str,
    spinner: bool = True,
    calls_pbar: Optional[tqdm] = None,
    force_fresh: bool = False,
) -> Any:
    """
    Cached parse that is safe against "bad value stuck in cache" infinite loops.

    - If force_fresh=True: bypass cache read (but still write on success).
    - If cache hit twice in a row for same key: bypass cache read this time.
    - If cached payload fails validation: evict + bypass cache.
    Stores ONLY parsed model_dump() in cache.
    """
    key = cache_key_for_parse(
        model=model,
        temperature=temperature,
        top_p=top_p,
        text_format=text_format,
        input_messages=input,
    )

    cached = None
    cache_hit = False

    if not force_fresh:
        cached = cache.get(key)
        cache_hit = cached is not None

        if guard.should_force_fresh_due_to_repeat_hit(key, cache_hit):
            tqdm.write(f"[CACHE GUARD] repeat hit for {desc} → forcing fresh call")
            cached = None
            cache_hit = False

    if cache_hit:
        try:
            parsed_obj = text_format.model_validate(cached)
            guard.update(key, True)
            tqdm.write(f"[CACHE HIT] {desc}")
            # create a tiny dummy response
            class _DummyResp:
                def __init__(self, parsed_obj):
                    self.output_parsed = parsed_obj
                    self.usage = None
            if calls_pbar is not None:
                calls_pbar.update(1)
            return _DummyResp(parsed_obj)
        except Exception as e:
            tqdm.write(f"[CACHE EVICT] {desc} cached payload failed validation: {e}")
            try:
                cache.delete(key)
            except Exception:
                pass
            # fall through to fresh call

    resp = responses_parse_with_progress(
        client,
        model=model,
        input=input,
        text_format=text_format,
        temperature=temperature,
        top_p=top_p,
        desc=desc,
        spinner=spinner,
        calls_pbar=calls_pbar,
    )

    # cache only on success
    try:
        cache.set(key, resp.output_parsed.model_dump(), expire=None)
    except Exception as e:
        tqdm.write(f"[WARN] Failed to cache {desc}: {e}")

    guard.update(key, False)
    return resp


def embed_texts(client: OpenAI, cache: Cache, embedding_model: str, texts: List[str]) -> List[List[float]]:
    key = cache_key_for_embeddings(embedding_model, texts)
    hit = cache.get(key)
    if hit is not None:
        return hit
    resp = client.embeddings.create(model=embedding_model, input=texts)
    vecs = [d.embedding for d in resp.data]
    cache.set(key, vecs, expire=None)
    return vecs


def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def dedup_options_semantic(
    client: Optional[OpenAI],
    cache: Optional[Cache],
    embedding_model: str,
    items: List[Any],
    sim_threshold: float = 0.88,
) -> List[Any]:
    seen = set()
    exact_kept: List[Any] = []
    for it in items:
        k = _norm_key(it.value)
        if k not in seen:
            seen.add(k)
            exact_kept.append(it)

    if client is None or cache is None or len(exact_kept) <= 1:
        return exact_kept

    texts = [f"{it.value} :: {it.explanation}" for it in exact_kept]
    vecs = embed_texts(client, cache, embedding_model, texts)

    kept: List[Any] = []
    kept_vecs: List[List[float]] = []
    for it, v in zip(exact_kept, vecs):
        if any(cosine(v, kv) >= sim_threshold for kv in kept_vecs):
            continue
        kept.append(it)
        kept_vecs.append(v)
    return kept


class Spinner:
    """Lightweight spinner that runs in a background thread while a blocking call executes."""
    def __init__(self, desc: str, enabled: bool = True, interval_s: float = 0.25):
        self.desc = desc
        self.enabled = enabled
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self):
        if not self.enabled:
            return
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        sys.stderr.write("\r" + " " * 140 + "\r")
        sys.stderr.flush()

    def _run(self):
        glyphs = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            sys.stderr.write(f"\r{glyphs[i % len(glyphs)]} {self.desc}  (elapsed {elapsed:0.1f}s)")
            sys.stderr.flush()
            i += 1
            time.sleep(self.interval_s)


def cache_key_for_embeddings(model: str, texts: List[str]) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "kind": "embeddings.create",
        "model": model,
        "texts": texts,
    }
    return stable_hash(payload)

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _extract_usage(resp: Any) -> Optional[Dict[str, Any]]:
    """
    Try to pull usage from the OpenAI SDK response object.
    SDK shapes can vary; be defensive.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        usage = getattr(resp, "response", None)
        usage = getattr(usage, "usage", None) if usage is not None else None
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    out = {}
    for k in ["input_tokens", "output_tokens", "total_tokens"]:
        if hasattr(usage, k):
            out[k] = getattr(usage, k)
    return out or None


def _norm_key(s: str) -> str:
    return " ".join(s.strip().casefold().split())
