#!/usr/bin/env python3
"""
British Parliament persona seed generator (Westminster) — YAML-driven + title-by-title enrichment.

Repo structure:
  root/
    configs/
      skeleton_parliament_seeds.yaml
    src/
      persona_generation/
        british_persona_seed_generator.py   <-- this file

What this does:
- Loads configs/skeleton_parliament_seeds.yaml (relative to this file)
- Enriches CanonicalTexts.titles ONE BY ONE into CanonicalTexts.items (CanonItem schema)
- Expands PersonaCore + AdditionalTraits + PolicyDebates (optional; controlled by YAML + CLI flags)
- Uses diskcache for re-use of previously executed prompts + embeddings
- IMPORTANT FIX: retries do NOT get stuck on a bad cached value:
  - If a cached result fails validation, it is evicted and a fresh call is forced.
  - If two consecutive attempts return a cache hit for the same key, the second is forced fresh.
  - Every retry uses force_fresh=True by default.

Requires:
  pip install openai pydantic pyyaml tqdm diskcache

Run:
  python -m src.persona_generation.british_persona_seed_generator --out outputs/parliament_seeds_enriched.yaml
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import yaml
from diskcache import Cache
from openai import OpenAI
from pydantic import BaseModel, Field, conlist, model_validator
from tqdm import tqdm

from utils.openai_utils import _call_with_retries, responses_parse_cached_safe, CacheLoopGuard, dedup_options_semantic

# ---------------------------
# Paths (relative to this file)
# ---------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # .../root/src/persona_generation/british_persona_seed_generator.py
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "skeleton_parliament_seeds.yaml"

DEFAULT_OUT = str(REPO_ROOT / "outputs" / "parliament_seeds_enriched.yaml")
DEFAULT_CACHE_DIR = str(REPO_ROOT / ".cache_british_parliament_seeds")

# ---------------------------
# Defaults
# ---------------------------

DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 2.0
DEFAULT_TOP_P = 0.98
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------
# Schema / enums
# ---------------------------

Party = Literal[
    "Conservative",
    "Labour",
    "LiberalDemocrats",
    "SNP",
    "Green",
    "PlaidCymru",
    "DUP",
    "SinnFein",
    "SDLP",
    "Alliance",
    "Independent",
]

PARTIES: List[str] = [
    "Conservative", "Labour", "LiberalDemocrats", "SNP", "Green",
    "PlaidCymru", "DUP", "SinnFein", "SDLP", "Alliance", "Independent"
]


# ---------------------------
# Pydantic output models
# ---------------------------

class OptionItem(BaseModel):
    value: str = Field(..., description="Categorical label.")
    explanation: str = Field(..., description="1–2 sentences explaining what it signals or how it manifests.")
    weight: Optional[float] = Field(None, description="Optional sampling weight/probability.")


class OptionsOut(BaseModel):
    items: List[OptionItem]


class CanonItem(BaseModel):
    title: str
    author: str
    year: Optional[int] = None
    category: Literal[
        "ParliamentaryHistory",
        "PoliticalMemoir",
        "PoliticalPhilosophy",
        "PoliticalBiography",
        "PoliticalJournalism",
        "ConstitutionalLaw",
        "DevolutionAndUnion",
        "BrexitAndEurope",
        "CampaigningAndSpin",
        "PoliticalFiction",
    ]
    summary: str = Field(..., description="1–2 sentence persona-forming summary.")
    ideology_tags: List[str] = Field(..., description="Economic + constitutional + Brexit/Europe tags.")

# ---------------------------
# Policy debate models (Structured Outputs SAFE)
# ---------------------------

class PartyPrevalence(BaseModel):
    Conservative: float = 0.0
    Labour: float = 0.0
    LiberalDemocrats: float = 0.0
    SNP: float = 0.0
    Green: float = 0.0
    PlaidCymru: float = 0.0
    DUP: float = 0.0
    SinnFein: float = 0.0
    SDLP: float = 0.0
    Alliance: float = 0.0
    Independent: float = 0.0

class DebateBulletStance(BaseModel):
    bullet: str = Field(..., description="The specific bullet point the stance addresses (repeat verbatim across stances).")
    id: str = Field(..., description="snake_case id for the stance.")
    description: str = Field(..., description="1–2 sentences describing the stance.")
    party_distribution: PartyPrevalence
    rationale: str = Field(..., description="1–3 sentences explaining party_distribution intuition.")


class PolicyDebateExpanded(BaseModel):
    name: str
    description: str = Field(..., description="One-paragraph overall debate description.")
    stances: List[DebateBulletStance]


class PolicyDebatesOut(BaseModel):
    debates: conlist(PolicyDebateExpanded, min_length=20, max_length=20)

# ---------------------------
# Progress helpers
# ---------------------------


# ---------------------------
# Diskcache helpers
# ---------------------------


# ---------------------------
# Embeddings + semantic dedup
# ---------------------------


# ---------------------------
# Prompt templates
# ---------------------------

def prompt_canon_enrich_one(title: str) -> str:
    return f"""
Fill out metadata for this UK politics canon title.

Title:
{title}

Return a CanonItem with:
- title (same as provided, or minimally corrected for punctuation)
- author
- year (or null if uncertain)
- category (one of: ParliamentaryHistory, PoliticalMemoir, PoliticalPhilosophy, PoliticalBiography,
  PoliticalJournalism, ConstitutionalLaw, DevolutionAndUnion, BrexitAndEurope, CampaigningAndSpin, PoliticalFiction)
- summary: 1–2 sentences, persona-forming (max ~35 words)
- ideology_tags: a short list of tags covering:
  * economic dimension (e.g., social_democratic, free_market, communitarian, etc.)
  * constitutional/devolution/union dimension (e.g., unionist, devolutionist, federalist, secessionist_sympathetic, etc.)
  * Brexit/Europe dimension where relevant (e.g., brexit_hard, brexit_soft, remain, rejoin, regulatory_alignment)
If unsure on year, set year=null. Do not hallucinate obscure works; be conservative.
""".strip()


def prompt_expand_category(category_name: str, description: str, n: int, avoid: List[str]) -> str:
    return f"""
Expand the persona seed category: {category_name}

Category purpose:
{description}

Return at least {n} distinct items.
Each item must include:
- value: short label
- explanation: 1–2 sentences describing what it signals and how it manifests

Constraints:
- Avoid duplicating these existing values: {avoid}
- Keep items UK/Westminster-plausible where relevant
- No profanity, no slurs, no real-person accusations
""".strip()


def prompt_expand_additional_trait(subcategory: str, description: str, n: int, avoid: List[str]) -> str:
    return f"""
Expand the AdditionalTraits subcategory: {subcategory}

Purpose:
{description}

Return at least {n} distinct items with:
- value
- explanation (1–2 sentences)

Constraints:
- Avoid duplicating these existing values: {avoid}
- Make items distinctive and useful for synthetic persona generation (not vague synonyms)
""".strip()


def prompt_expand_category_fill(category_name: str, description: str, need_n: int, existing_values: List[str]) -> str:
    existing_values = "\n".join([f"- {v}" for v in existing_values])
    return f"""
We are expanding the persona seed category: {category_name}

Category purpose:
{description}

We already have these items (DO NOT repeat, do not paraphrase into near-synonyms):
{existing_values}

Task:
- Produce {need_n} NEW items that are clearly distinct from the above.
- Each item must include:
  - value: short label
  - explanation: 1–2 sentences describing what it signals and how it manifests

Novelty constraints:
- Avoid near-duplicates in wording or meaning.
- Prefer concrete behavioural/presentational signals over abstract adjectives.
- Keep UK/Westminster-plausible where relevant.
""".strip()


def prompt_expand_trait_fill(subcategory: str, description: str, need_n: int, existing_values: List[str]) -> str:
    existing_values = "\n".join([f"- {v}" for v in existing_values])
    return f"""
We are expanding the AdditionalTraits subcategory: {subcategory}

Purpose:
{description}

We already have these items (DO NOT repeat, do not paraphrase into near-synonyms):
{existing_values}

Task:
- Produce {need_n} NEW items that are clearly distinct from the above.
- Each item must include:
  - value
  - explanation (1–2 sentences)

Novelty constraints:
- Make items meaningfully different (not just rewordings).
- Prefer traits that create different downstream persona behaviour.
""".strip()

# ---------------------------
# Misc utils
# ---------------------------
def prompt_expand_single_policy_debate(debate: Dict[str, Any]) -> str:
    return f"""
You are expanding a UK policy debate for synthetic MP persona generation.

Debate name:
{debate["name"]}

Tasks:
- Provide a 1-paragraph description of the debate.
- Generate stances grouped by bullet points (sub-issues).
- Do NOT output a separate bullet list.
- Each stance must include a `bullet` field.
- Reuse the EXACT same bullet text across all stances that address it.

Requirements:
- Create 4–12 distinct bullet points (expressed via stance.bullet).
- For EACH bullet point, generate AT LEAST 4 discrete stances.

Each stance must include:
- bullet: exact bullet text it addresses
- id: snake_case
- description: 1–2 sentences
- party_distribution: INDEPENDENT PER-PARTY PREVALENCE values in [0, 1]
  * Interpret as: likelihood that a typical MP of that party holds this stance.
  * IMPORTANT: values DO NOT need to sum to 1 across parties.
  * It is valid for multiple parties to be high (e.g., 0.8 for both Conservative and Labour),
    or low for all parties.

Parties (include all of them with numeric values):
  {PARTIES}

- rationale: 1–3 sentences explaining why the per-party prevalences look like that.

Notes:
- These are plausible priors, not polling.
- Make stances meaningfully different, not rewordings.
""".strip()

def openai_expand_policy_debates_looped(
    client: OpenAI,
    cache: Cache,
    guard: CacheLoopGuard,   # ✅ add guard here
    model: str,
    seed_debates: List[Dict[str, Any]],
    *,
    temperature: float = 2.0,
    top_p: float = 0.98,
    calls_pbar: Optional[tqdm] = None,
    spinner: bool = True,
) -> List[Dict[str, Any]]:
    """
    Expands policy debates ONE-BY-ONE with tqdm progress.
    Robust: each debate is its own Structured Outputs call.
    Safe caching: retries won't get stuck on a bad cached value.
    """
    debates_out: List[Dict[str, Any]] = []

    pbar = tqdm(seed_debates, desc="Policy debates", unit="debate")

    for debate in pbar:
        name = str(debate.get("name", "")).strip()
        if not name:
            continue
        pbar.set_postfix_str(name)

        def call(force_fresh: bool):
            return responses_parse_cached_safe(
                client,
                cache,
                guard,  # ✅ pass guard here
                model=model,
                input=[
                    {"role": "system", "content": (
                        "You expand a UK policy debate into bullet-level stances "
                        "with party-conditioned prevalence priors."
                    )},
                    {"role": "user", "content": prompt_expand_single_policy_debate(debate)},
                ],
                text_format=PolicyDebateExpanded,
                temperature=temperature,
                top_p=top_p,
                desc=f"PolicyDebate {name}",
                spinner=spinner,
                calls_pbar=calls_pbar,
                force_fresh=force_fresh,
            )

        try:
            resp = _call_with_retries(
                call,
                tries=4,
                base_sleep_s=0.2,
                always_force_fresh_on_retry=True,
            )
            dd = resp.output_parsed.model_dump()

            # Normalize party distributions
            for s in dd.get("stances", []):
                pd_obj = PartyPrevalence(**(s.get("party_distribution") or {}))
                s["party_distribution"] = clamp_party_prevalence(pd_obj).model_dump()

            debates_out.append(dd)

        except Exception as e:
            tqdm.write(f"[WARN] Failed debate '{name}': {e}")
            continue

        time.sleep(0.1)  # gentle pacing

    return debates_out

# --- 3) Clamp-only (no renormalization) for PartyPrevalence ---

def clamp_party_prevalence(pd: "PartyPrevalence", lo: float = 0.0, hi: float = 1.0) -> "PartyPrevalence":
    """
    Clamp each party's prevalence independently to [0, 1].
    No renormalization (values do NOT have to sum to 1).
    """
    vals = pd.model_dump()
    clamped = {k: min(hi, max(lo, float(v))) for k, v in vals.items()}
    # optional rounding for nicer YAML
    clamped = {k: round(v, 4) for k, v in clamped.items()}
    return PartyPrevalence(**clamped)


def to_plain_options(items: List[OptionItem]) -> List[Dict[str, Any]]:
    return [it.model_dump() for it in items]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------
# YAML loading / writing
# ---------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


# ---------------------------
# OpenAI expanders
# ---------------------------

def openai_expand_options_fill(
    client: OpenAI,
    cache: Cache,
    guard: CacheLoopGuard,
    *,
    model: str,
    temperature: float,
    top_p: float,
    embedding_model: str,
    base_prompt: str,
    min_items: int,
    sim_threshold: float = 0.88,
    max_rounds: int = 4,
    oversample_factor: float = 2.0,
    fill_prompt_fn: Callable[[str, str, int, List[str]], str],
    category_name: str,
    category_description: str,
    calls_pbar: Optional[tqdm],
    spinner: bool = True,
) -> List[OptionItem]:
    pool: List[OptionItem] = []

    def do_parse(prompt: str, target: int, desc: str) -> List[OptionItem]:
        want = max(target, int(math.ceil(target * oversample_factor)))
        sys_msg = "You generate structured persona seed options. Return diverse, concrete, distinct items."
        user_msg = prompt + f"\n\nReturn at least {want} items."

        def call(force_fresh: bool):
            return responses_parse_cached_safe(
                client,
                cache,
                guard,
                model=model,
                input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                text_format=OptionsOut,
                temperature=temperature,
                top_p=top_p,
                desc=desc,
                spinner=spinner,
                calls_pbar=calls_pbar,
                force_fresh=force_fresh,
            )

        resp = _call_with_retries(call, tries=5, base_sleep_s=0.2, always_force_fresh_on_retry=True)
        out: OptionsOut = resp.output_parsed
        return out.items

    pool.extend(do_parse(base_prompt, min_items, f"{category_name}: round0"))
    pool = dedup_options_semantic(client, cache, embedding_model, pool, sim_threshold=sim_threshold)
    tqdm.write(f"[{category_name}] after dedup: {len(pool)}/{min_items}")

    round_i = 1
    while len(pool) < min_items and round_i <= max_rounds:
        need = min_items - len(pool)
        existing_vals = [it.value for it in pool]
        fill_prompt = fill_prompt_fn(category_name, category_description, need, existing_vals)

        pool.extend(do_parse(fill_prompt, need, f"{category_name}: round{round_i}"))
        pool = dedup_options_semantic(client, cache, embedding_model, pool, sim_threshold=sim_threshold)
        tqdm.write(f"[{category_name}] after dedup: {len(pool)}/{min_items}")

        round_i += 1
        time.sleep(0.1)

    return pool[:min_items]


def enrich_titles_one_by_one(
    client: OpenAI,
    cache: Cache,
    guard: CacheLoopGuard,
    *,
    model: str,
    temperature: float,
    top_p: float,
    titles: List[str],
    calls_pbar: Optional[tqdm],
    spinner: bool = True,
    max_tries_per_title: int = 6,
) -> List[Dict[str, Any]]:
    """
    Enrich every title into a CanonItem. Uses cache, but never allows retries to loop on the same cached bad payload.
    """
    out_items: List[Dict[str, Any]] = []

    for idx, title in enumerate(titles, start=1):
        title_clean = (title or "").strip()
        if not title_clean:
            continue

        desc = f"CanonEnrich {idx}/{len(titles)}"

        def call(force_fresh: bool):
            return responses_parse_cached_safe(
                client,
                cache,
                guard,
                model=model,
                input=[
                    {"role": "system", "content": "You fill in structured metadata for a UK politics canon item."},
                    {"role": "user", "content": prompt_canon_enrich_one(title_clean)},
                ],
                text_format=CanonItem,
                temperature=temperature,
                top_p=top_p,
                desc=desc,
                spinner=spinner,
                calls_pbar=calls_pbar,
                force_fresh=force_fresh,
            )

        # If cached result is bad, safe wrapper evicts. All retries force fresh.
        resp = _call_with_retries(
            call,
            tries=max_tries_per_title,
            base_sleep_s=0.2,
            always_force_fresh_on_retry=True,
        )
        out_items.append(resp.output_parsed.model_dump())

    return out_items


# ---------------------------
# Builders driven by YAML config
# ---------------------------

def build_persona_core_from_yaml(
    client: Optional[OpenAI],
    cache: Cache,
    guard: CacheLoopGuard,
    *,
    model: str,
    temperature: float,
    top_p: float,
    embedding_model: str,
    persona_core_cfg: Dict[str, Any],
    calls_pbar: Optional[tqdm],
    spinner: bool = True,
    use_openai: bool = True,
) -> Dict[str, Any]:
    """
    persona_core_cfg format expected (from YAML):
      PersonaCore:
        Appearance:
          explanation: ...
          seed_items: [{value, explanation}, ...]
          target_count: 25
        ...
    """
    out: Dict[str, Any] = {}
    for cat, spec in tqdm(persona_core_cfg.items(), desc="PersonaCore categories", unit="cat"):
        if not isinstance(spec, dict):
            continue
        explanation = spec.get("explanation", "")
        seed_items_raw = spec.get("seed_items", []) or []
        target_count = int(spec.get("target_count", 0) or 0)

        seed_items = [OptionItem(**x) for x in seed_items_raw]

        if not use_openai or client is None or target_count <= len(seed_items):
            out[cat] = {"items": to_plain_options(seed_items[:max(target_count, len(seed_items))]), "explanation": explanation}
            continue

        avoid = [s.value for s in seed_items]
        base_prompt = prompt_expand_category(cat, explanation, n=target_count, avoid=avoid)

        generated = openai_expand_options_fill(
            client=client,
            cache=cache,
            guard=guard,
            model=model,
            temperature=temperature,
            top_p=top_p,
            embedding_model=embedding_model,
            base_prompt=base_prompt,
            min_items=target_count,
            sim_threshold=0.88,
            max_rounds=5,
            oversample_factor=1.8,
            fill_prompt_fn=prompt_expand_category_fill,
            category_name=f"PersonaCore/{cat}",
            category_description=explanation,
            calls_pbar=calls_pbar,
            spinner=spinner,
        )

        pool = seed_items + generated
        pool = dedup_options_semantic(client, cache, embedding_model, pool, sim_threshold=0.88)
        out[cat] = {"items": to_plain_options(pool[:target_count]), "explanation": explanation}

    return out


def build_additional_traits_from_yaml(
    client: Optional[OpenAI],
    cache: Cache,
    guard: CacheLoopGuard,
    *,
    model: str,
    temperature: float,
    top_p: float,
    embedding_model: str,
    traits_cfg: Dict[str, Any],
    calls_pbar: Optional[tqdm],
    spinner: bool = True,
    use_openai: bool = True,
) -> Dict[str, Any]:
    """
    traits_cfg format expected:
      AdditionalTraits:
        subcategories:
          RhetoricalRegister:
            purpose: ...
            seed_items: [{value, explanation}, ...]
            target_count: 10
    """
    out: Dict[str, Any] = {}
    subcats = (traits_cfg or {}).get("subcategories", {}) or {}

    for subcat, spec in tqdm(subcats.items(), desc="AdditionalTraits categories", unit="cat"):
        if not isinstance(spec, dict):
            continue
        purpose = spec.get("purpose", "")
        seed_items_raw = spec.get("seed_items", []) or []
        target_count = int(spec.get("target_count", 0) or 0)

        seed_items = [OptionItem(**x) for x in seed_items_raw]

        if not use_openai or client is None or target_count <= len(seed_items):
            out[subcat] = {"items": to_plain_options(seed_items[:max(target_count, len(seed_items))]), "explanation": purpose}
            continue

        avoid = [s.value for s in seed_items]
        base_prompt = prompt_expand_additional_trait(subcat, purpose, n=target_count, avoid=avoid)

        generated = openai_expand_options_fill(
            client=client,
            cache=cache,
            guard=guard,
            model=model,
            temperature=temperature,
            top_p=top_p,
            embedding_model=embedding_model,
            base_prompt=base_prompt,
            min_items=target_count,
            sim_threshold=0.88,
            max_rounds=5,
            oversample_factor=1.8,
            fill_prompt_fn=prompt_expand_trait_fill,
            category_name=f"AdditionalTraits/{subcat}",
            category_description=purpose,
            calls_pbar=calls_pbar,
            spinner=spinner,
        )

        pool = seed_items + generated
        pool = dedup_options_semantic(client, cache, embedding_model, pool, sim_threshold=0.88)
        out[subcat] = {"items": to_plain_options(pool[:target_count]), "explanation": purpose}

    return out


# ---------------------------
# Main build pipeline
# ---------------------------
def load_seed_policy_debates(skeleton: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract policy debate seeds from a pre-loaded skeleton YAML.
    """
    return skeleton["BritishParliamentPersonaSeeds"]["PolicyDebates"]["debates"]

from typing import Any, Dict, Optional
from tqdm import tqdm
from openai import OpenAI
from diskcache import Cache


def build_from_skeleton(
    skeleton: Dict[str, Any],
    *,
    client: Optional[OpenAI],
    cache: Cache,
    guard: "CacheLoopGuard",
    model: str,
    temperature: float,
    top_p: float,
    embedding_model: str,
    use_openai: bool,
    expand_traits: bool,
    expand_persona_core: bool,
    expand_policy_debates: bool,
) -> Dict[str, Any]:
    """
    Loads everything from skeleton YAML and enriches/expands where requested.

    Assumptions:
    - skeleton is properly formatted (we do not coerce/normalize).
    - canonical titles live at: BritishParliamentPersonaSeeds -> CanonicalTexts -> titles
    - policy debate seeds live at: BritishParliamentPersonaSeeds -> PolicyDebates -> items
    """

    root_key = "BritishParliamentPersonaSeeds"
    seeds = (skeleton or {}).get(root_key, {}) or {}

    # GeneratorConfig: allow YAML to override embedding_model (CLI already applied upstream if you do that).
    gen_cfg = seeds.get("GeneratorConfig", {}) or {}
    embedding_model = str(gen_cfg.get("embedding_model", embedding_model) or embedding_model)

    can_call_openai = bool(use_openai and client is not None)
    calls_pbar = tqdm(desc="OpenAI calls", unit="call") if can_call_openai else None

    # -------------------------
    # Canon: titles -> items one by one
    # -------------------------
    canon = seeds.get("CanonicalTexts", {}) or {}
    titles = canon.get("titles", []) or []

    if can_call_openai:
        tqdm.write("==> Enriching canon titles one-by-one")
        canon_items = enrich_titles_one_by_one(
            client=client,
            cache=cache,
            guard=guard,
            model=model,
            temperature=temperature,
            top_p=top_p,
            titles=titles,
            calls_pbar=calls_pbar,
            spinner=True,
            max_tries_per_title=6,
        )
    else:
        # If OpenAI disabled, trust the skeleton's existing items (or empty).
        canon_items = canon.get("items", []) or []

    seeds["CanonicalTexts"] = {
        **canon,
        "titles": titles,
        "items": canon_items,
        "explanation": canon.get(
            "explanation",
            "Canon texts used to seed narrative voice, institutional posture, and ideology.",
        ),
    }

    # -------------------------
    # PersonaCore expansions
    # -------------------------
    if expand_persona_core:
        pc_cfg = seeds.get("PersonaCore", {}) or {}
        if can_call_openai:
            tqdm.write("==> Expanding PersonaCore")
        seeds["PersonaCore"] = build_persona_core_from_yaml(
            client=client,
            cache=cache,
            guard=guard,
            model=model,
            temperature=temperature,
            top_p=top_p,
            embedding_model=embedding_model,
            persona_core_cfg=pc_cfg,
            calls_pbar=calls_pbar,
            spinner=True,
            use_openai=can_call_openai,
        )

    # -------------------------
    # AdditionalTraits expansions
    # -------------------------
    if expand_traits:
        traits_cfg = seeds.get("AdditionalTraits", {}) or {}
        if can_call_openai:
            tqdm.write("==> Expanding AdditionalTraits")
        seeds["AdditionalTraits"] = build_additional_traits_from_yaml(
            client=client,
            cache=cache,
            guard=guard,
            model=model,
            temperature=temperature,
            top_p=top_p,
            embedding_model=embedding_model,
            traits_cfg=traits_cfg,
            calls_pbar=calls_pbar,
            spinner=True,
            use_openai=can_call_openai,
        )

    # -------------------------
    # PolicyDebates expansions (looped; seeds come from skeleton)
    # -------------------------
    if expand_policy_debates:
        pd_cfg = seeds.get("PolicyDebates", {}) or {}
        seed_debates = pd_cfg.get("debates", []) or []

        if can_call_openai:
            tqdm.write("==> Expanding PolicyDebates (looped)")
            debates_expanded = openai_expand_policy_debates_looped(
                client=client,cache=cache,
                guard=guard, model=model,
                seed_debates=seed_debates,
                temperature=temperature, top_p=top_p,
                calls_pbar=calls_pbar, spinner=True,
            )

        else:
            tqdm.write(f"PolicyDebates seeds loaded: {len(seed_debates)}")
            debates_expanded = pd_cfg.get("expanded_items", []) or pd_cfg.get("items", []) or []

        seeds["PolicyDebates"] = {
            **pd_cfg,
            "debates": seed_debates,
            "items": debates_expanded,
            "explanation": pd_cfg.get(
                "explanation",
                "Debates expanded into bullets and stances with party-conditioned priors.",
            ),
        }

    if calls_pbar is not None:
        calls_pbar.close()

    return {root_key: seeds}

# ---------------------------
# CLI
# ---------------------------
@dataclass
class Args:
    config: str
    out: str
    model: str
    cache_dir: str
    no_openai: bool
    no_expand_traits: bool
    no_expand_persona_core: bool
    no_expand_policy_debates: bool
    temperature: float
    top_p: float
    embedding_model: str


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to skeleton YAML.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output YAML path.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model id.")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="diskcache directory.")
    p.add_argument("--no-openai", action="store_true", help="Disable OpenAI calls (just re-dump YAML).")
    p.add_argument("--no-expand-traits", action="store_true", help="Do not expand AdditionalTraits.")
    p.add_argument("--no-expand-persona-core", action="store_true", help="Do not expand PersonaCore.")
    p.add_argument("--no-expand-policy-debates", action="store_true", help="Do not expand PolicyDebates.")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature (default 2.0).")
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="top_p (default 0.98).")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model for semantic dedup.")
    ns = p.parse_args()

    return Args(
        config=ns.config,
        out=ns.out,
        model=ns.model,
        cache_dir=ns.cache_dir,
        no_openai=ns.no_openai,
        no_expand_traits=ns.no_expand_traits,
        no_expand_persona_core=ns.no_expand_persona_core,
        no_expand_policy_debates=ns.no_expand_policy_debates,
        temperature=float(ns.temperature),
        top_p=float(ns.top_p),
        embedding_model=str(ns.embedding_model),
    )


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    out_path = Path(args.out).resolve()

    skeleton = load_yaml(config_path)

    use_openai = not args.no_openai
    client = OpenAI() if use_openai else None

    cache = Cache(args.cache_dir)
    guard = CacheLoopGuard()

    # YAML can contain GeneratorConfig too; we apply CLI overrides after loading YAML:
    # (we still pass embedding_model so it can be used for dedup expansions)
    data = build_from_skeleton(
        skeleton=skeleton,
        client=client,
        cache=cache,
        guard=guard,
        model=args.model,
        temperature=args.temperature,     # you wanted 2.0 everywhere; default is 2.0
        top_p=args.top_p,                 # default 0.98
        embedding_model=args.embedding_model,
        use_openai=use_openai,
        expand_traits=(not args.no_expand_traits),
        expand_persona_core=(not args.no_expand_persona_core),
        expand_policy_debates=(not args.no_expand_policy_debates),
    )

    dump_yaml(data, out_path)
    cache.close()

    print(f"Wrote: {out_path}")
    if args.no_openai:
        print("NOTE: OpenAI disabled → no enrichment/expansion was performed.")


if __name__ == "__main__":
    main()
