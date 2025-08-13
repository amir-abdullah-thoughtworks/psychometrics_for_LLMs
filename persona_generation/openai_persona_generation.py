#!/usr/bin/env python3
"""
Generate, embed, dedupe, and push personas to HF Hub (private) with **one column per persona field**.
Dataset is ALWAYS loaded from remote (force_redownload).

- Generation: OpenAI Responses.parse (gpt-4o)
- Embeddings: Qwen/Qwen3-Embedding-0.6B (dim auto-detected)
- Dedup: cosine similarity > 0.98
- Storage: local output.json + HF dataset (thoughtworks/psychometric_personas)
- HF schema is dynamic from schema.json and includes a column for each (section, field_name),
  normalized to snake-case "section__fieldname". Age is int64; all other fields are string.
- Metadata columns: uuid, persona_text, embedding, embedding_dim, embedding_model, script_version, archetype,
  display_name (mirror of Name), age (mirror of Age)
- Remote dataset load: download_mode="force_redownload"
"""

import argparse, json, os, sys, time, uuid, re
from typing import List, Union, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from openai._exceptions import RateLimitError, APIStatusError, APIConnectionError, APIError

from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
from sentence_transformers import SentenceTransformer

# ---------------- Config (test-friendly defaults) ----------------
SCRIPT_VERSION_DEFAULT = "0.1"
ARCHETYPE_DEFAULT = "none"

DEFAULT_HF_REPO = "thoughtworks/psychometric_personas"
DEFAULT_EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_SIM_THRESHOLD = 0.98

DEFAULT_MODEL = "gpt-4o"
DEFAULT_SCHEMA_PATH = "schema.json"
DEFAULT_OUT_PATH = "output.json"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TOTAL = 300
DEFAULT_PRIVATE = False  # always push private unless overridden

# ---------------- Pydantic for parse() ----------------
class FieldItem(BaseModel):
    section: str
    field_name: str
    value: Union[str, int]

class Persona(BaseModel):
    fields: List[FieldItem]
    script_version: str = SCRIPT_VERSION_DEFAULT
    archetype: str = ARCHETYPE_DEFAULT

class PersonaBatch(BaseModel):
    items: List[Persona]

# ---------------- IO helpers ----------------
def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---------------- Schema helpers ----------------
def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def column_key(section: str, field_name: str) -> str:
    return f"{to_snake(section)}__{to_snake(field_name)}"

def build_schema_maps(schema: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Returns:
      field_map: key -> {"section","field_name","type"} with normalized key "section__field"
      ordered_keys: list of keys in schema order
    """
    field_map: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []
    for f in schema.get("fields", []):
        sec = f.get("section", "")
        name = f.get("field_name", "")
        ftype = (f.get("field_type") or "string").lower()
        key = column_key(sec, name)
        field_map[key] = {"section": sec, "field_name": name, "type": ftype}
        ordered_keys.append(key)
    return field_map, ordered_keys

def build_hf_features(field_map: Dict[str, Dict[str, Any]]) -> Features:
    """
    Dynamic features: each persona field becomes a column.
    Age is int64; all others string. Plus stable metadata columns (no persona_json).
    """
    feats = {
        "uuid": Value("string"),
        "persona_text": Value("string"),
        "embedding": Sequence(Value("float32")),
        "embedding_dim": Value("int64"),
        "embedding_model": Value("string"),
        "script_version": Value("string"),
        "archetype": Value("string"),
        "display_name": Value("string"),
        "age": Value("int64"),
    }
    for key, _meta in field_map.items():
        if key.endswith("__age"):
            feats[key] = Value("int64")
        else:
            feats[key] = Value("string")
    return Features(feats)

def flatten_persona_columns(persona_obj: Dict[str, Any], field_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return dict mapping each schema column -> value (typed); blanks for missing."""
    out = {k: (0 if k.endswith("__age") else "") for k in field_map.keys()}
    display_name, age = "", -1
    for it in persona_obj.get("fields", []):
        key = column_key(it.get("section", ""), it.get("field_name", ""))
        if key in field_map:
            val = it.get("value")
            if key.endswith("__age"):
                try:
                    out[key] = int(val)
                except Exception:
                    out[key] = -1
                age = out[key]
            else:
                out[key] = str(val) if not isinstance(val, str) else val
            if key.endswith("__name") and isinstance(val, str):
                display_name = val
    # mirrors
    if display_name == "":
        display_name = str(next((it.get("value") for it in persona_obj.get("fields", [])
                                 if it.get("field_name") == "Name"), "")) or ""
    if age == -1:
        try:
            age = int(next((it.get("value") for it in persona_obj.get("fields", [])
                            if it.get("field_name") == "Age"), -1))
        except Exception:
            age = -1
    out["display_name"] = display_name
    out["age"] = int(age if isinstance(age, int) else -1)
    return out

# ---------------- Prompt construction ----------------
def summarize_schema(schema: Dict[str, Any]) -> str:
    lines = []
    for f in schema.get("fields", []):
        lines.append(
            f"- [{f.get('section')}] {f.get('field_name')} ({f.get('field_type','string')}); "
            f"field tokens {f.get('tokens_min')}-{f.get('tokens_max')}; "
            f"section tokens {f.get('section_tokens_min')}-{f.get('section_tokens_max')}; "
            f"style: {f.get('content_style')}; "
            f"restrictions: {', '.join(f.get('restrictions',[])) or 'none'}"
        )
    return "\n".join(lines)

def instance_instructions(n: int, script_version: str, archetype: str) -> str:
    return f"""
Create N={n} distinct, diverse police personas that strictly follow the flattened schema below.

Return ONLY a JSON object with this shape:
{{
  "items": [
    {{
      "fields": [
        {{"section": "Demographic Fields", "field_name": "Name", "value": "Full Name (Preferred Name: …)" }},
        {{"section": "Demographic Fields", "field_name": "Age", "value": 42 }},
        {{"section": "Demographic Fields", "field_name": "Location", "value": "City, ST" }},
        …
      ],
      "script_version": "{script_version}",
      "archetype": "{archetype}"
    }}
  ]
}}

Rules:
- Include every (section, field_name) from the schema exactly once per persona.
- Narrative fields are strings; Age is an integer.
- Respect token ranges approximately (±10%).
- Do NOT include Date of Birth anywhere.
- Keep content clinically realistic, nuanced, and internally consistent.
- Output ONLY the JSON object (no prose, no markdown).
"""

def build_messages(schema: Dict[str, Any], n: int, script_version: str, archetype: str):
    sys_msg = (
        "You generate realistic police personas for psychometric LLM testing. "
        "Follow the provided field schema, token budgets, and style notes closely."
    )
    schema_bullets = summarize_schema(schema)
    user_msg = f"""Flattened schema fields:
{schema_bullets}

Additional guidance:
- Vary age, career stage, city, and background.
- Avoid PII.

{instance_instructions(n, script_version, archetype)}
"""
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

# ---------------- Embeddings ----------------
def build_embedder(model_id: str) -> SentenceTransformer:
    print(f"[embeddings] Loading: {model_id}")
    return SentenceTransformer(model_id, trust_remote_code=True)

def get_embed_dim(embedder: SentenceTransformer) -> int:
    try:
        dim = embedder.get_sentence_embedding_dimension()
    except Exception:
        dim = int(embedder.encode(["dim_probe"], normalize_embeddings=True).shape[-1])
    print(f"[embeddings] dimension={dim}")
    return dim

def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 16) -> np.ndarray:
    vecs = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)

# ---------------- Pool + migration ----------------
def build_persona_text_from_columns(row: Dict[str, Any], field_map: Dict[str, Dict[str, Any]], script_version: str, archetype: str) -> str:
    parts = []
    for key in field_map.keys():
        sec = field_map[key]["section"]
        fld = field_map[key]["field_name"]
        val = row.get(key, "" if not key.endswith("__age") else -1)
        parts.append(f"{sec}: {fld}: {val}")
    parts.append(f"script_version: {row.get('script_version', script_version)}")
    parts.append(f"archetype: {row.get('archetype', archetype)}")
    return "\n".join(parts)

def ensure_embeddings_on_rows(rows: List[Dict[str, Any]], embedder: SentenceTransformer, dim: int, embed_model_id: str,
                              field_map: Dict[str, Dict[str, Any]], script_version: str, archetype: str) -> np.ndarray:
    print(f"[migrate] Ensuring embeddings on {len(rows)} rows…")
    need_idx, need_txt = [], []
    for i, r in enumerate(rows):
        r.setdefault("uuid", str(uuid.uuid4()))
        r.setdefault("persona_text", r.get("persona_text") or "")
        r.setdefault("script_version", r.get("script_version") or SCRIPT_VERSION_DEFAULT)
        r.setdefault("archetype", r.get("archetype") or ARCHETYPE_DEFAULT)
        r.setdefault("display_name", r.get("display_name") or "")
        r.setdefault("age", int(r.get("age") or -1))
        r["embedding_dim"] = int(dim)
        r["embedding_model"] = embed_model_id

        ok = isinstance(r.get("embedding"), list) and len(r["embedding"]) == dim and r.get("embedding_model") == embed_model_id
        if not ok:
            if not r["persona_text"]:
                r["persona_text"] = build_persona_text_from_columns(r, field_map, script_version, archetype)
            need_idx.append(i)
            need_txt.append(r["persona_text"])

    if need_txt:
        print(f"[migrate] Re-embedding {len(need_txt)}…")
        new_embs = embed_texts(embedder, need_txt, batch_size=16)
        for j, i in enumerate(need_idx):
            rows[i]["embedding"] = new_embs[j].astype(np.float32).tolist()

    embs = []
    for r in rows:
        v = np.array(r["embedding"], dtype=np.float32) if isinstance(r.get("embedding"), list) else np.zeros(dim, np.float32)
        embs.append(v)
    pool = np.vstack(embs) if embs else np.zeros((0, dim), dtype=np.float32)
    pool /= (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
    print(f"[migrate] pool={pool.shape}")
    return pool

def filter_near_duplicates(new_rows: List[Dict[str, Any]], existing_embs: np.ndarray, embedder: SentenceTransformer, threshold: float, dim: int):
    if not new_rows:
        return [], existing_embs
    texts = [r["persona_text"] for r in new_rows]
    new_embs = embed_texts(embedder, texts, batch_size=16)
    accepted, pool = [], existing_embs.copy() if existing_embs.size else np.zeros((0, dim), dtype=np.float32)
    print(f"[dedupe] compare {len(new_rows)} vs pool={pool.shape[0]}")
    for i, row in enumerate(new_rows):
        emb = new_embs[i]
        if pool.size:
            sims = pool @ emb
            mx = float(np.max(sims))
            print(f"[dedupe] uuid={row['uuid']} max_sim={mx:.5f}")
            if mx > threshold:
                print("[dedupe]   reject")
                continue
        row["embedding"] = emb.astype(np.float32).tolist()
        accepted.append(row)
        pool = np.vstack([pool, emb.reshape(1, -1)]) if pool.size else emb.reshape(1, -1)
        print("[dedupe]   accept")
    return accepted, pool

# ---------------- HF helpers (private; no persona_json) ----------------
def hf_features(field_map: Dict[str, Dict[str, Any]]) -> Features:
    return build_hf_features(field_map)

def load_hf_rows(repo_id: str) -> List[Dict[str, Any]]:
    """Always load the latest dataset from the Hub (ignore local cache)."""
    print(f"[hf] Loading REMOTE dataset (force_redownload) → {repo_id}")
    try:
        ds = load_dataset(
            repo_id,
            split="train",
            download_mode="force_redownload"  # <--- ensures fresh remote load
        )
        rows = ds.to_list()
        print(f"[hf] Loaded {len(rows)} rows from remote")
        return rows
    except Exception as e:
        print(f"[hf] load failed or missing: {e}")
        return []

def parse_persona_json_if_present(r: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy helper (if a very old row still has persona_json); parse to dict if present."""
    pj = r.get("persona_json")
    if isinstance(pj, str) and pj.strip():
        try:
            return json.loads(pj)
        except Exception:
            return {}
    if isinstance(r.get("persona"), dict):
        return r["persona"]
    return {}

def normalize_rows_for_features(rows: List[Dict[str, Any]], field_map: Dict[str, Dict[str, Any]],
                                script_version: str, archetype: str) -> List[Dict[str, Any]]:
    """
    Ensure rows have ONLY our feature keys. For legacy rows:
      - If persona_json exists, parse and fill field columns.
      - Else keep whatever matching columns exist; fill defaults for missing.
    """
    norm = []
    # base keys
    base = {
        "uuid": "", "persona_text": "",
        "embedding": [], "embedding_dim": 0, "embedding_model": "",
        "script_version": SCRIPT_VERSION_DEFAULT, "archetype": ARCHETYPE_DEFAULT,
        "display_name": "", "age": -1,
    }
    for k in field_map.keys():
        base[k] = (0 if k.endswith("__age") else "")
    keys = list(base.keys())

    for r in rows:
        # if legacy: attempt to parse and expand
        persona_obj = parse_persona_json_if_present(r)
        expanded = {}
        if persona_obj:
            expanded = flatten_persona_columns(persona_obj, field_map)

        clean = {}
        for k in keys:
            if k in expanded:
                v = expanded[k]
            else:
                v = r.get(k, base[k])
            if k.endswith("__age") or k == "age":
                try:
                    v = int(v)
                except Exception:
                    v = -1
            clean[k] = v

        # persona_text
        pt = r.get("persona_text", "")
        if not pt:
            pt = build_persona_text_from_columns(clean, field_map, script_version, archetype)

        # metadata + embedding
        clean["uuid"] = str(r.get("uuid") or uuid.uuid4())
        clean["persona_text"] = pt
        clean["embedding"] = r.get("embedding") or []
        clean["embedding_dim"] = int(r.get("embedding_dim") or 0)
        clean["embedding_model"] = r.get("embedding_model") or ""
        clean["script_version"] = r.get("script_version") or script_version
        clean["archetype"] = r.get("archetype") or archetype

        norm.append(clean)
    return norm

def push_hf_rows(repo_id: str, rows: List[Dict[str, Any]], features: Features, private: bool = DEFAULT_PRIVATE):
    print(f"[hf] Pushing {len(rows)} rows to {repo_id} (private={private}) …")
    ds = Dataset.from_list(rows, features=features)
    ds.push_to_hub(repo_id, private=private)
    print("[hf] push done")

# ---------------- Validation / retries ----------------
def backoff(attempt: int, base: float = 1.5, cap: float = 30.0) -> float:
    import random
    d = min(cap, (base ** attempt)) + random.random()
    print(f"[retry] sleep {d:.2f}s")
    return d

def validate_batch_schema_pairs(batch_personas: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[str]:
    req = {(f["section"], f["field_name"]) for f in schema["fields"]}
    issues = []
    for i, p in enumerate(batch_personas):
        seen = {(it.get("section"), it.get("field_name")) for it in p.get("fields", [])}
        miss = req - seen
        extra = seen - req
        if miss:
            issues.append(f"persona[{i}] missing: {sorted(list(miss))[:6]}")
        if extra:
            issues.append(f"persona[{i}] extra: {sorted(list(extra))[:6]}")
    return issues

# ---------------- OpenAI generation ----------------
def generate_persona_batch(client: OpenAI, model: str, schema: Dict[str, Any], n: int, script_version: str, archetype: str):
    msgs = build_messages(schema, n, script_version, archetype)
    for attempt in range(7):
        try:
            print(f"[gen] asking {model} for {n} personas …")
            resp = client.responses.parse(
                model=model,
                input=msgs,
                temperature=1.2,
                top_p=0.99,
                max_output_tokens=30000,
                text_format=PersonaBatch,
            )
            batch = resp.output_parsed
            items = [p.model_dump() for p in batch.items]
            print(f"[gen] got {len(items)}")
            return items
        except (RateLimitError, APIStatusError, APIConnectionError, APIError, Exception) as e:
            print(f"[gen] error: {e}")
            time.sleep(backoff(attempt))
    return []

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default=DEFAULT_SCHEMA_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    ap.add_argument("--private", action="store_true", default=DEFAULT_PRIVATE)
    ap.add_argument("--embed-model-id", default=DEFAULT_EMBED_MODEL_ID)
    ap.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    ap.add_argument("--script-version", default=SCRIPT_VERSION_DEFAULT)
    ap.add_argument("--archetype", default=ARCHETYPE_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Load schema and build dynamic maps/features
    print(f"[init] schema={args.schema}")
    schema = load_json(args.schema, default=None)
    if not schema or "fields" not in schema:
        sys.exit("Invalid schema (missing 'fields').")
    field_map, ordered_keys = build_schema_maps(schema)
    features = hf_features(field_map)
    print(f"[init] dynamic columns: {len(field_map)} (plus metadata)")

    # Local rows
    print(f"[init] loading local: {args.out}")
    local_rows_raw = load_json(args.out, default=[])
    if not isinstance(local_rows_raw, list):
        sys.exit("output.json must be a JSON array")
    print(f"[init] local rows: {len(local_rows_raw)}")

    # HF rows — FORCE remote load
    remote_rows_raw = load_hf_rows(args.hf_repo)
    print(f"[init] remote rows (fresh): {len(remote_rows_raw)}")

    # Normalize to current feature set (legacy persona_json gets expanded & dropped)
    normalized_remote = normalize_rows_for_features(remote_rows_raw, field_map, args.script_version, args.archetype)
    normalized_local  = normalize_rows_for_features(local_rows_raw,  field_map, args.script_version, args.archetype)
    existing_rows = normalized_remote + [r for r in normalized_local if r not in normalized_remote]
    print(f"[init] pool size (HF+local): {len(existing_rows)}")

    # Embedder
    embedder = build_embedder(args.embed_model_id)
    EMBED_DIM = get_embed_dim(embedder)

    # Ensure embeddings for pool
    existing_embs = ensure_embeddings_on_rows(existing_rows, embedder, EMBED_DIM, args.embed_model_id, field_map, args.script_version, args.archetype)

    # Save normalized local back to disk
    save_json(args.out, normalized_local)

    # Generation loop
    client = OpenAI()
    current_total = len(existing_rows)
    target_total = args.total
    print(f"[run] start current={current_total} target={target_total} batch={args.batch_size}")

    with tqdm(total=target_total, initial=current_total, desc="Personas (HF + Local)") as pbar:
        while current_total < target_total:
            to_go = target_total - current_total
            n = min(args.batch_size, to_go)

            if args.dry_run:
                msgs = build_messages(schema, n, args.script_version, args.archetype)
                print("---- SYSTEM ----\n", msgs[0]["content"])
                print("---- USER ----\n", msgs[1]["content"])
                return

            personas = generate_persona_batch(client, args.model, schema, n, args.script_version, args.archetype)
            if not personas:
                print("[run] empty batch; continue")
                time.sleep(0.5)
                continue

            # Validate content against schema pairs (non-fatal)
            issues = validate_batch_schema_pairs(personas, schema)
            if issues:
                print("[validate] Warnings:")
                for it in issues:
                    print("  -", it)

            # Build flat rows with one column per field (NO persona_json)
            new_rows = []
            for p in personas:
                p.setdefault("script_version", args.script_version)
                p.setdefault("archetype", args.archetype)
                cols = flatten_persona_columns(p, field_map)
                row = {
                    "uuid": str(uuid.uuid4()),
                    "persona_text": build_persona_text_from_columns(cols | {"script_version": args.script_version, "archetype": args.archetype}, field_map, args.script_version, args.archetype),
                    "embedding": [],
                    "embedding_dim": EMBED_DIM,
                    "embedding_model": args.embed_model_id,
                    "script_version": args.script_version,
                    "archetype": args.archetype,
                    "display_name": cols.get("display_name", ""),
                    "age": int(cols.get("age", -1)),
                }
                # add all persona field columns
                for key in field_map.keys():
                    row[key] = cols.get(key, 0 if key.endswith("__age") else "")
                new_rows.append(row)

            # Dedupe + embed
            accepted_rows, existing_embs = filter_near_duplicates(new_rows, existing_embs, embedder, args.sim_threshold, EMBED_DIM)
            if not accepted_rows:
                print("[run] no unique rows; continue")
                time.sleep(0.5)
                continue

            # Append to local + save
            normalized_local.extend(accepted_rows)
            save_json(args.out, normalized_local)
            print(f"[local] +{len(accepted_rows)} rows; local total={len(normalized_local)}")

            # Push merged to Hub (private)
            existing_rows.extend(accepted_rows)
            push_hf_rows(args.hf_repo, existing_rows, features, private=args.private)

            # Progress
            current_total = len(existing_rows)
            pbar.n = current_total
            pbar.refresh()
            time.sleep(0.4)

    print(f"[done] local saved: {len(normalized_local)} to {args.out}")
    print(f"[done] hub updated: {args.hf_repo} (private={args.private})")

if __name__ == "__main__":
    main()
