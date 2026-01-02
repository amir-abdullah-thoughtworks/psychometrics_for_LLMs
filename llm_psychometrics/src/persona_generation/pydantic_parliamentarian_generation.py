# src/persona_generation/pydantic_parliamentarian_generation.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfFolder
from openai import OpenAI
from pydantic import Field, create_model
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------
# Paths (repo-relative defaults)
# ----------------------------
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parents[2]  # .../src/persona_generation -> .../src -> repo root

DEFAULT_SEEDS_YAML = REPO_ROOT / "configs" / "parliament_seeds_enriched.yaml"
DEFAULT_DEMOGRAPHICS_CSV = REPO_ROOT / "data" / "demographics" / "balanced_us_police_officers.csv"


# ----------------------------
# Utils
# ----------------------------
def _norm(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _weighted_choice(rng: random.Random, items: List[Any], weights: List[float]) -> Any:
    if not items:
        raise ValueError("weighted_choice got empty items")
    if len(items) != len(weights):
        raise ValueError("weighted_choice items/weights mismatch")
    total = float(sum(max(0.0, float(w)) for w in weights))
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += max(0.0, float(w))
        if r <= acc:
            return it
    return items[-1]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _words_at_most(text: str, max_words: int) -> str:
    w = text.split()
    if len(w) <= max_words:
        return text.strip()
    return " ".join(w[:max_words]).strip()


# ----------------------------
# Demographics (from CSV)
# ----------------------------
@dataclass(frozen=True)
class Demographics:
    uuid: str
    name: str
    age: int
    sex: str
    marital_status: str
    ethnic_background: str


# ----------------------------
# Seed extraction helpers (robust to slight schema drift)
# ----------------------------
def _get_root_seed_obj(data: dict) -> dict:
    """
    Many of our seed YAMLs wrap the real payload under a single top-level key, e.g.
    BritishParliamentPersonaSeeds: {...}
    """
    # 1) Known wrappers
    for k in [
        "BritishParliamentPersonaSeeds",
        "ParliamentarianPersonaSeeds",
        "ParliamentPersonaSeeds",
        "ParliamentarianSeeds",
        "ParliamentSeeds",
    ]:
        v = data.get(k)
        if isinstance(v, dict):
            return v

    # 2) If there's exactly one top-level dict key and its value is a dict, treat that as root
    if isinstance(data, dict) and len(data) == 1:
        only_val = next(iter(data.values()))
        if isinstance(only_val, dict):
            return only_val

    # 3) Fall back
    return data


def _extract_memoirs(root: dict) -> Tuple[List[str], Dict[str, str]]:
    """
    Supports:
      A) MemoirSeeds: [title,...] + MemoirSummaries: {title: summary,...}
      B) Memoirs / CanonicalTexts / MemoirSeeds: [{title, summary}, ...]
      C) CanonicalTexts:
            titles: [title,...]
         (titles-only; summaries empty)
    """
    # ---- C) CanonicalTexts.titles (your current schema) ----
    ct = root.get("CanonicalTexts") or root.get("canonical_texts")
    if isinstance(ct, dict):
        titles = ct.get("titles") or ct.get("Titles")
        if isinstance(titles, list) and titles:
            t = [str(x).strip() for x in titles if str(x).strip()]
            if t:
                return t, {}  # titles-only: no summaries

    # ---- A) separate titles + summaries ----
    titles = root.get("MemoirSeeds") or root.get("memoir_seeds")
    summaries = root.get("MemoirSummaries") or root.get("memoir_summaries")
    if isinstance(titles, list) and titles:
        t = [str(x).strip() for x in titles if str(x).strip()]
        s = {}
        if isinstance(summaries, dict):
            s = {str(k).strip(): str(v).strip() for k, v in summaries.items()}
        if t:
            return t, s

    # ---- B) list of dicts ----
    for key in ["Memoirs", "memoirs", "CanonicalTexts", "canonical_texts", "MemoirSeeds"]:
        raw = root.get(key)
        if isinstance(raw, dict):
            # Some schemas might have CanonicalTexts.items: [{title, summary}, ...]
            raw = raw.get("items") or raw.get("texts") or raw.get("entries")
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            t: List[str] = []
            s: Dict[str, str] = {}
            for m in raw:
                title = str(m.get("title", "")).strip() or str(m.get("name", "")).strip()
                summ = str(m.get("summary", "")).strip()
                if title:
                    t.append(title)
                    if summ:
                        s[title] = summ
            if t:
                return t, s

    raise ValueError(
        "Could not find canonical/memoir titles in YAML. "
        "Expected CanonicalTexts.titles (titles-only), or MemoirSeeds, or list of {title,summary}."
    )


def _extract_party_priors(root: dict) -> Dict[str, float]:
    pa = root.get("PartyAffiliation") or root.get("party_affiliation") or {}
    priors = pa.get("priors") if isinstance(pa, dict) else None
    if not isinstance(priors, dict) or not priors:
        raise ValueError("PartyAffiliation.priors missing/empty in seeds YAML")

    out: Dict[str, float] = {}
    for party, meta in priors.items():
        if isinstance(meta, dict) and "base_probability" in meta:
            out[str(party)] = _safe_float(meta["base_probability"], 0.0)
        else:
            out[str(party)] = _safe_float(meta, 0.0)

    out = {k: v for k, v in out.items() if k and v >= 0}
    if not out:
        raise ValueError("No usable party base_probability values found")
    return out


def _extract_weighted_items(section: Any, value_key: str = "value") -> List[Tuple[str, float, str]]:
    """
    Expects:
      section: {items: [{value, weight, explanation?}, ...]} or [{value, weight, explanation?}, ...]
    Returns: [(value, weight, explanation), ...]
    """
    items = None
    if isinstance(section, dict):
        items = section.get("items")
    elif isinstance(section, list):
        items = section

    if not isinstance(items, list) or not items:
        return []

    out: List[Tuple[str, float, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        v = str(it.get(value_key, "")).strip()
        w = _safe_float(it.get("weight", 0.0), 0.0)
        e = str(it.get("explanation", "")).strip()
        if v:
            out.append((v, w, e))
    return out


def _extract_regions(root: dict) -> List[Tuple[str, float, str]]:
    reg = root.get("Regions") or root.get("regions") or {}
    items = reg.get("items") if isinstance(reg, dict) else None
    if not isinstance(items, dict) or not items:
        raise ValueError("Regions.items missing/empty in seeds YAML")
    out: List[Tuple[str, float, str]] = []
    for region_key, meta in items.items():
        if not isinstance(meta, dict):
            continue
        share = _safe_float(meta.get("population_share", 0.0), 0.0)
        expl = str(meta.get("explanation", "")).strip()
        out.append((str(region_key), share, expl))
    if not out:
        raise ValueError("No usable region entries found")
    return out


def _extract_parliamentary_styles(root: dict) -> List[str]:
    # Common patterns: list, or {items:[{value:...},...]}
    for key in ["ParliamentaryStyles", "parliamentary_styles"]:
        raw = root.get(key)
        if isinstance(raw, list):
            vals = [str(x).strip() for x in raw if str(x).strip()]
            if vals:
                return vals

    ps = root.get("ParliamentaryStyle") or root.get("parliamentary_style") or {}
    if isinstance(ps, dict) and isinstance(ps.get("items"), list):
        vals = []
        for it in ps["items"]:
            if isinstance(it, dict) and it.get("value"):
                vals.append(str(it["value"]).strip())
            else:
                vals.append(str(it).strip())
        vals = [v for v in vals if v]
        if vals:
            return vals

    raise ValueError("Parliamentary styles missing/empty in seeds YAML")


def _extract_media_dispositions(root: dict) -> List[str]:
    md = root.get("MediaDisposition") or root.get("media_disposition") or {}
    if isinstance(md, dict) and isinstance(md.get("items"), list):
        vals = []
        for it in md["items"]:
            if isinstance(it, dict) and it.get("value"):
                vals.append(str(it["value"]).strip())
            else:
                vals.append(str(it).strip())
        vals = [v for v in vals if v]
        if vals:
            return vals
    if isinstance(md, list):
        vals = [str(x).strip() for x in md if str(x).strip()]
        if vals:
            return vals
    # fallback
    return ["Media-shy", "Selective", "Media-savvy", "Performative", "Policy-focused"]


def _extract_additional_traits_and_rhetorical_register(root: dict) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    AdditionalTraits is usually:
      AdditionalTraits:
        RhetoricalRegister: {items:[{value, weight?},...]}
        SomeOtherTrait:     {items:[...]}
    We:
      - pull RhetoricalRegister out as a top-level persona field options
      - keep all other traits as additional_traits dict(trait_name -> [options])
    """
    raw = root.get("AdditionalTraits") or root.get("additional_traits") or {}
    if not isinstance(raw, dict):
        return {}, []

    additional: Dict[str, List[str]] = {}
    rhetorical: List[str] = []

    for trait_name, trait_section in raw.items():
        # allow list directly or {items:[...]}
        items = trait_section.get("items") if isinstance(trait_section, dict) else trait_section
        if not isinstance(items, list) or not items:
            continue

        opts: List[str] = []
        for it in items:
            if isinstance(it, dict) and "value" in it:
                opts.append(str(it["value"]).strip())
            else:
                opts.append(str(it).strip())
        opts = [o for o in opts if o]
        if not opts:
            continue

        if _norm(trait_name) == "rhetoricalregister":
            rhetorical = opts
        else:
            additional[str(trait_name)] = opts

    return additional, rhetorical


def _extract_policy_items(root: dict) -> List[dict]:
    """
    Your described structure:

      PolicyDebates:
        debates: [{name: Immigration}, ...]
      (somewhere else)
      items:
        - name: Immigration
          stances:
            - bullet: ...
              id: ...
              party_distribution: {Conservative: 0.85, ...}

    We try:
      1) if root has a dict with key 'items' that looks like this, use it
      2) else scan root values for any dict that has an 'items' list matching the pattern
      3) else scan root for any top-level 'items' list matching the pattern
    """
    def looks_like_policy_items(items: Any) -> bool:
        if not isinstance(items, list) or not items:
            return False
        ex = items[0]
        if not isinstance(ex, dict):
            return False
        if "name" not in ex or "stances" not in ex:
            return False
        st = ex.get("stances")
        if not isinstance(st, list) or not st:
            return False
        st0 = st[0]
        return isinstance(st0, dict) and ("id" in st0) and ("party_distribution" in st0)

    # direct at root
    if looks_like_policy_items(root.get("items")):
        return root["items"]

    # common container keys
    for key in ["PolicyDebatesExpanded", "PolicyDebatesItems", "PolicyDebateItems", "PolicyDebatesFull"]:
        sec = root.get(key)
        if isinstance(sec, dict) and looks_like_policy_items(sec.get("items")):
            return sec["items"]
        if looks_like_policy_items(sec):
            return sec  # if stored directly as list

    # scan root values
    for _, v in root.items():
        if isinstance(v, dict) and looks_like_policy_items(v.get("items")):
            return v["items"]
        if looks_like_policy_items(v):
            return v

    raise ValueError("Could not find expanded PolicyDebates items list in seeds YAML (list of {name, stances:[{id,party_distribution},...]}).")


def _build_policy_lookup(policy_items: List[dict]) -> Dict[str, List[dict]]:
    """
    Returns: issue_name -> list of stance dicts (each must include id, bullet, description, party_distribution)
    """
    out: Dict[str, List[dict]] = {}
    for issue in policy_items:
        if not isinstance(issue, dict):
            continue
        name = str(issue.get("name", "")).strip()
        stances = issue.get("stances")
        if not name or not isinstance(stances, list) or not stances:
            continue
        good: List[dict] = []
        for st in stances:
            if not isinstance(st, dict):
                continue
            sid = str(st.get("id", "")).strip()
            if not sid:
                continue
            pdist = st.get("party_distribution")
            if not isinstance(pdist, dict):
                continue
            good.append(st)
        if good:
            out[name] = good
    if not out:
        raise ValueError("Policy debates parsed, but no usable stance entries found.")
    return out


# ----------------------------
# Persona Generator
# ----------------------------
class ParliamentarianPersonaGenerator:
    """
    Pydantic-structured generation (like police generator), but with:
      - Demographics "cheated" from police_officers.csv (age>=30)
      - Policy stances per issue sampled using party priors, stored as dict issue -> list[stance_id]
      - Speech covers 3 randomly selected issues, grounded in memoir voice + parliamentary_style + rhetorical_register
      - persona_string excludes ONLY: memoir (canonical title), memoir_summary, memoir_narrative, archetype, archetype_description
    """

    def __init__(
        self,
        seeds_yaml: str,
        demographics_csv: str,
        version: str,
        model: str = "gpt-4o-mini",
        temperature: float = 1.2,
        top_p: float = 0.95,
        api_key: Optional[str] = None,
        rng_seed: int = 42,
    ):
        self.version = version
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.base_seed = int(rng_seed)
        self._rng = random.Random(self.base_seed)

        # ---- load seeds ----
        with open(seeds_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        root = _get_root_seed_obj(data)

        # ---- memoirs ----
        self.memoir_titles, self.memoir_summaries = _extract_memoirs(root)
        self._mem_offset = self._rng.randrange(len(self.memoir_titles))

        # ---- archetypes (grounding) ----
        raw_block = root["Archetypes"]  # must exist
        raw_archetypes = raw_block["items"]  # must exist

        if not isinstance(raw_archetypes, list) or not raw_archetypes:
            raise ValueError("Archetypes missing/empty in seeds YAML")

        self.archetypes: List[dict] = [
            a if isinstance(a, dict) else {"name": str(a), "description": ""}
            for a in raw_archetypes
        ]

        self._arch_offset = self._rng.randrange(len(self.archetypes))

        # ---- party priors ----
        self.party_priors = _extract_party_priors(root)

        # ---- occupations / education / regions ----
        occ = root.get("Occupations") or root.get("occupations") or {}
        self.occupations = _extract_weighted_items(occ, value_key="value")
        if not self.occupations:
            raise ValueError("Occupations.items missing/empty in seeds YAML")

        edu = root.get("Education") or root.get("education") or {}
        self.educ_secondary = _extract_weighted_items(edu.get("Secondary", []), value_key="value")
        self.educ_tertiary = _extract_weighted_items(edu.get("Tertiary", edu.get("University", [])), value_key="value")
        if not self.educ_secondary and not self.educ_tertiary:
            raise ValueError("Education.Secondary and Education.Tertiary both missing/empty in seeds YAML")

        self.regions = _extract_regions(root)

        # ---- styles / categories ----
        self.parliamentary_styles = _extract_parliamentary_styles(root)
        self.appearance_categories: Dict[str, List[str]] = root.get("AppearanceCategories") or root.get("appearance_categories") or {}
        self.behavior_categories: Dict[str, List[str]] = root.get("BehaviorCategories") or root.get("behavior_categories") or {}
        if not self.appearance_categories or not self.behavior_categories:
            raise ValueError("AppearanceCategories/BehaviorCategories missing/empty in seeds YAML")

        self.media_dispositions = _extract_media_dispositions(root)

        # ---- additional traits + rhetorical register (top level) ----
        self.additional_traits, self.rhetorical_register_options = _extract_additional_traits_and_rhetorical_register(root)
        if not self.rhetorical_register_options:
            # hard fail per your requirement (top-level field, used in speech)
            raise ValueError("AdditionalTraits.RhetoricalRegister missing/empty in seeds YAML")

        # ---- policy debates ----
        # We only need the expanded stance blocks to sample stances per issue.
        policy_items = _extract_policy_items(root)
        self.policy_lookup = _build_policy_lookup(policy_items)

        # ---- demographics CSV ----
        df = pd.read_csv(demographics_csv)
        required = {"uuid", "sex", "age", "first_name", "last_name", "marital_status", "ethnic_background"}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns in demographics CSV: {sorted(miss)}")

        df = df.dropna(subset=list(required)).copy()
        df["age"] = df["age"].astype(int)
        df = df[df["age"] >= 30].reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid rows after enforcing required columns and age>=30")
        self.df = df

    # ----------------------------
    # Deterministic pickers
    # ----------------------------
    def _pick_demographics_by_df_index(self, df_idx: int) -> Demographics:
        row = self.df.iloc[df_idx]
        uuid = str(row["uuid"]).strip()
        name = f"{str(row['first_name']).strip()} {str(row['last_name']).strip()}"
        return Demographics(
            uuid=uuid,
            name=name,
            age=int(row["age"]),
            sex=str(row["sex"]).strip(),
            marital_status=str(row["marital_status"]).strip(),
            ethnic_background=str(row["ethnic_background"]).strip(),
        )

    def _pick_memoir_by_index(self, idx: int) -> Tuple[str, str]:
        title = self.memoir_titles[(self._mem_offset + idx) % len(self.memoir_titles)]
        return title, self.memoir_summaries.get(title, "")

    def _compose_archetype_description(self, a: dict) -> str:
        """
        Expected archetype schema (from seeds YAML):
          - name: str
          - description: str
          - signature_tells: list[str] (optional)
          - strengths: list[str] (optional)
          - pitfalls: list[str] (optional)

        Returns a compact, prompt-ready description string.
        """

        def _fmt_list(label: str, xs) -> str:
            if not xs:
                return ""
            if isinstance(xs, list):
                xs = [str(x).strip() for x in xs if str(x).strip()]
                if not xs:
                    return ""
                return f"{label}: " + "; ".join(xs) + "."
            # fallback if YAML is malformed (string/dict)
            s = str(xs).strip()
            return f"{label}: {s}." if s else ""

        name = (a.get("name") or "").strip()
        desc = (a.get("description") or "").strip()

        parts: list[str] = []
        if name:
            parts.append(f"Archetype: {name}.")
        if desc:
            parts.append(desc if desc.endswith((".", "!", "?")) else desc + ".")

        sig = _fmt_list("Signature tells", a.get("signature_tells"))
        if sig:
            parts.append(sig)

        strengths = _fmt_list("Strengths", a.get("strengths"))
        if strengths:
            parts.append(strengths)

        pitfalls = _fmt_list("Pitfalls", a.get("pitfalls"))
        if pitfalls:
            parts.append(pitfalls)

        out = " ".join(parts).strip()
        return out or "Archetype: (unspecified)."


    def _pick_archetype_by_index(self, idx: int) -> Tuple[str, str]:
        a = self.archetypes[(self._arch_offset + idx) % len(self.archetypes)]
        name = str(a.get("name", "")).strip() or str(a)
        desc = self._compose_archetype_description(a)
        return name, desc

    def _pick_party_by_index(self, idx: int) -> str:
        rng = random.Random(self.base_seed + 101 * idx)
        parties = list(self.party_priors.keys())
        weights = [self.party_priors[p] for p in parties]
        return _weighted_choice(rng, parties, weights)

    def _pick_occupation_by_index(self, idx: int) -> Tuple[str, str]:
        rng = random.Random(self.base_seed + 103 * idx)
        vals = [v for (v, _, _) in self.occupations]
        wts = [w for (_, w, _) in self.occupations]
        i = int(_weighted_choice(rng, list(range(len(vals))), wts))
        return vals[i], self.occupations[i][2]

    def _pick_education_by_index(self, idx: int) -> Tuple[str, str, str, str]:
        rng = random.Random(self.base_seed + 107 * idx)

        sec_val, sec_exp = "", ""
        ter_val, ter_exp = "", ""

        if self.educ_secondary:
            s_vals = [v for (v, _, _) in self.educ_secondary]
            s_wts = [w for (_, w, _) in self.educ_secondary]
            si = int(_weighted_choice(rng, list(range(len(s_vals))), s_wts))
            sec_val, sec_exp = s_vals[si], self.educ_secondary[si][2]

        if self.educ_tertiary:
            t_vals = [v for (v, _, _) in self.educ_tertiary]
            t_wts = [w for (_, w, _) in self.educ_tertiary]
            ti = int(_weighted_choice(rng, list(range(len(t_vals))), t_wts))
            ter_val, ter_exp = t_vals[ti], self.educ_tertiary[ti][2]

        return sec_val, sec_exp, ter_val, ter_exp

    def _pick_region_by_index(self, idx: int) -> Tuple[str, str]:
        rng = random.Random(self.base_seed + 109 * idx)
        vals = [v for (v, _, _) in self.regions]
        wts = [w for (_, w, _) in self.regions]
        i = int(_weighted_choice(rng, list(range(len(vals))), wts))
        return vals[i], self.regions[i][2]

    def _pick_parliamentary_style_by_index(self, idx: int) -> str:
        rng = random.Random(self.base_seed + 113 * idx)
        return rng.choice(self.parliamentary_styles)

    def _pick_rhetorical_register_by_index(self, idx: int) -> str:
        rng = random.Random(self.base_seed + 141 * idx)
        return rng.choice(self.rhetorical_register_options)

    def _pick_appearance_random(self, idx: int) -> Tuple[str, List[str]]:
        rng = random.Random(self.base_seed + 127 * idx)
        cat = rng.choice(list(self.appearance_categories.keys()))
        seeds = self.appearance_categories.get(cat, []) or []
        k = min(5, len(seeds))
        examples = rng.sample(seeds, k) if k else []
        return cat, examples

    def _pick_behavior_random(self, idx: int) -> Tuple[str, List[str]]:
        rng = random.Random(self.base_seed + 131 * idx)
        cat = rng.choice(list(self.behavior_categories.keys()))
        seeds = self.behavior_categories.get(cat, []) or []
        k = min(5, len(seeds))
        examples = rng.sample(seeds, k) if k else []
        return cat, examples

    def _pick_media_disposition_by_index(self, idx: int) -> str:
        rng = random.Random(self.base_seed + 137 * idx)
        return rng.choice(self.media_dispositions)

    def _pick_additional_traits_by_index(self, idx: int) -> Dict[str, str]:
        rng = random.Random(self.base_seed + 139 * idx)
        out: Dict[str, str] = {}
        for trait, opts in (self.additional_traits or {}).items():
            if opts:
                out[trait] = rng.choice(opts)
        return out

    # ----------------------------
    # Policy stance sampling (per issue, ONE stance, party-weighted)
    # ----------------------------
    def _pick_policy_stances_by_party(self, idx: int, party: str) -> Dict[str, List[str]]:
        """
        For each issue: choose exactly ONE stance proportional to party_distribution[party].
        Store dict[issue_name -> list[stance_id]] (list singleton for your requested type).
        """
        rng = random.Random(self.base_seed + 149 * idx)
        out: Dict[str, List[str]] = {}

        for issue_name, stances in self.policy_lookup.items():
            stance_ids: List[str] = []
            weights: List[float] = []
            for st in stances:
                sid = str(st.get("id", "")).strip()
                pdist = st.get("party_distribution") or {}
                w = _safe_float(pdist.get(party, 0.0), 0.0) if isinstance(pdist, dict) else 0.0
                if sid:
                    stance_ids.append(sid)
                    weights.append(w)

            if stance_ids:
                chosen = str(_weighted_choice(rng, stance_ids, weights))
                out[issue_name] = [chosen]

        return out

    def _pick_speech_issues(self, idx: int, policy_stances: Dict[str, List[str]], k: int = 3) -> List[str]:
        rng = random.Random(self.base_seed + 157 * idx)
        issues = list(policy_stances.keys())
        if len(issues) <= k:
            return issues
        return rng.sample(issues, k)

    def _speech_targets_payload(self, speech_issues: List[str], policy_stances: Dict[str, List[str]]) -> Dict[str, List[dict]]:
        """
        Provide enough info for the model to write a coherent speech:
          issue -> [{stance_id, bullet, description}]
        (We only include the one selected stance per issue.)
        """
        out: Dict[str, List[dict]] = {}
        for issue in speech_issues:
            chosen_ids = set(policy_stances.get(issue, []) or [])
            if not chosen_ids:
                continue
            stances = self.policy_lookup.get(issue, [])
            picked = []
            for st in stances:
                sid = str(st.get("id", "")).strip()
                if sid in chosen_ids:
                    picked.append(
                        {
                            "stance_id": sid,
                            "bullet": str(st.get("bullet", "")).strip(),
                            "description": str(st.get("description", "")).strip(),
                        }
                    )
            if picked:
                out[issue] = picked
        return out

    # ----------------------------
    # Persona string formatting
    # ----------------------------
    def persona_row_to_string(self, row: Dict[str, Any]) -> str:
        """
        Line-separated key-value pairs.
        Excludes grounding fields only:
          memoir, memoir_summary, memoir_narrative, archetype, archetype_description
        """
        exclude = {
            "memoir",
            "memoir_summary",
            "memoir_narrative",
            "archetype",
            "archetype_description",
        }

        preferred_order = [
            "version",
            "uuid",
            "name",
            "age",
            "sex",
            "marital_status",
            "ethnic_background",
            "party_affiliation",
            "occupation",
            "education_secondary",
            "education_tertiary",
            "region",
            "parliamentary_style",
            "rhetorical_register",
            "appearance_category",
            "behavior_category",
            "media_disposition",
            "policy_stances",
            "speech_issues",
            "speech",
            "appearance",
            "behavior",
            "educational_vocational_history",
            "medical_developmental_history",
            "family_history",
            "presenting_problems",
            "thought_content",
            "insight_judgment",
            "cognition",
            "emotional_behavioral_functioning",
            "social_functioning",
            "additional_traits",
            "summary_of_psychological_profile",
        ]

        def clean_value(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False, sort_keys=True)
            return " ".join(str(v).strip().split())

        lines: List[str] = []
        for k in preferred_order:
            if k in exclude:
                continue
            if k not in row:
                continue
            v = row.get(k)
            if v is None:
                continue

            # Pretty formatting for presenting_problems list
            if k == "presenting_problems" and isinstance(v, list):
                v2 = [clean_value(x) for x in v if x is not None and str(x).strip()]
                if v2:
                    lines.append("presenting_problems:")
                    lines.extend([f"- {x}" for x in v2])
                continue

            s = clean_value(v)
            if not s:
                continue
            lines.append(f"{k}: {s}")

        extras = sorted([k for k in row.keys() if k not in set(preferred_order) and k not in exclude])
        for k in extras:
            v = row.get(k)
            if v is None:
                continue
            s = clean_value(v)
            if not s:
                continue
            lines.append(f"{k}: {s}")

        return "\n".join(lines)

    # ----------------------------
    # Generation
    # ----------------------------
    def generate_one(self, idx: int, df_idx: int) -> Optional[dict]:
        dem = self._pick_demographics_by_df_index(df_idx)

        memoir_title, memoir_summary = self._pick_memoir_by_index(idx)
        archetype_name, archetype_desc = self._pick_archetype_by_index(idx)

        party = self._pick_party_by_index(idx)
        occupation, occupation_expl = self._pick_occupation_by_index(idx)
        edu_sec, edu_sec_expl, edu_ter, edu_ter_expl = self._pick_education_by_index(idx)
        region, region_expl = self._pick_region_by_index(idx)
        parliamentary_style = self._pick_parliamentary_style_by_index(idx)
        rhetorical_register = self._pick_rhetorical_register_by_index(idx)
        appearance_cat, appearance_examples = self._pick_appearance_random(idx)
        behavior_cat, behavior_examples = self._pick_behavior_random(idx)
        media_disp = self._pick_media_disposition_by_index(idx)
        additional_traits = self._pick_additional_traits_by_index(idx)

        # policy stances across ALL issues (stored as dict issue -> [stance_id])
        policy_stances = self._pick_policy_stances_by_party(idx, party)

        # speech issues: pick 3 issues from the already-chosen stances
        speech_issues = self._pick_speech_issues(idx, policy_stances, k=3)
        speech_targets = self._speech_targets_payload(speech_issues, policy_stances)

        # ----------------------------
        # Pydantic schema (LLM generates ONLY the narrative/text fields + must echo fixed literals)
        # ----------------------------
        SeededParliamentarianPersonaSchema = create_model(  # type: ignore[assignment]
            "SeededParliamentarianPersonaSchema",
            version=(Literal[self.version], ...),
            uuid=(Literal[dem.uuid], ...),
            name=(Literal[dem.name], ...),
            age=(Literal[dem.age], ...),
            sex=(Literal[dem.sex], ...),
            marital_status=(Literal[dem.marital_status], ...),
            ethnic_background=(Literal[dem.ethnic_background], ...),

            party_affiliation=(Literal[party], ...),
            occupation=(Literal[occupation], ...),
            education_secondary=(Literal[edu_sec], ...),
            education_tertiary=(Literal[edu_ter], ...),
            region=(Literal[region], ...),
            parliamentary_style=(Literal[parliamentary_style], ...),
            rhetorical_register=(Literal[rhetorical_register], ...),
            appearance_category=(Literal[appearance_cat], ...),
            behavior_category=(Literal[behavior_cat], ...),
            media_disposition=(Literal[media_disp], ...),

            # Grounding (excluded from persona_string)
            memoir=(Literal[memoir_title], ...),
            memoir_summary=(str, Field(
                ...,
                description="Copy the selected memoir summary exactly as provided. If empty, output an empty string."
            )),
            memoir_narrative=(str, Field(
                ...,
                description=(
                    "Write ~200 words as a short memoir-like narrative in the voice and cadence suggested by the selected memoir. "
                    "Ground all later details in this narrative. Do not mention 'memoir' or 'canonical text'."
                ),
            )),
            archetype=(Literal[archetype_name], ...),
            archetype_description=(str, Field(
                ...,
                description="Copy the provided archetype description exactly; grounding only.",
            )),

            # Appearance + behavior (memoir voice)
            appearance=(str, Field(
                ...,
                description=(
                    "2–3 specialized sentences about the person's appearance, faithful to the memoir narrative voice. "
                    "Use the selected appearance category as the anchor; stay concrete."
                ),
            )),
            behavior=(str, Field(
                ...,
                description=(
                    "2–3 specialized sentences about the person's behavior/mannerisms, faithful to the memoir narrative voice. "
                    "Use the selected behavior category as the anchor; stay concrete."
                ),
            )),

            # Psychological grounding (INCLUDED in persona_string)
            educational_vocational_history=(str, Field(
                ...,
                description="30–50 words. Align with education and occupation and party; show training/trajectory effects.",
            )),
            medical_developmental_history=(str, Field(
                ...,
                description="30–50 words. Health/development context relevant to the narrative; only what’s needed.",
            )),
            family_history=(str, Field(
                ...,
                description="30–50 words. Relational dynamics consistent with narrative, ethnic background, and marital status.",
            )),
            presenting_problems=(List[str], Field(
                ...,
                description=(
                    "3–6 concise mental-health problem phrases describing THE PARLIAMENTARIAN. "
                    "Natural phrasing, not diagnoses, not generic political tropes. "
                    "Not all problems should be work/politics-related."
                ),
            )),
            thought_content=(str, Field(
                ...,
                description="25–45 words. What tends to occupy the person’s mind, drawn from the narrative; natural phrasing.",
            )),
            insight_judgment=(str, Field(
                ...,
                description="25–45 words. Practical decision-making and self-understanding suggested by the narrative.",
            )),
            cognition=(str, Field(
                ...,
                description="25–45 words. Observable thinking/recall/problem-solving implied by the narrative.",
            )),
            emotional_behavioral_functioning=(str, Field(
                ...,
                description="35–55 words. How they handle pressure and difficult feelings; show behavior, avoid labels.",
            )),
            social_functioning=(str, Field(
                ...,
                description="35–55 words. Patterns in closeness, trust, and participation with others; concrete cues.",
            )),
            summary_of_psychological_profile=(str, Field(
                ...,
                description=(
                    "75–105 words. Integrative summary using narrative + histories + functioning + problems, "
                    "implicitly framed by the archetype description (grounding). Do not explicitly name the archetype."
                ),
            )),

            # Speech (<=250 words) about the 3 selected issues/stances
            speech=(str, Field(
                ...,
                description=(
                    "Write a short speech (<=250 words) as the parliamentarian. "
                    "Use BOTH the parliamentary_style and rhetorical_register to set the tenor, cadence, and rhetorical devices. "
                    "Ground it in the memoir narrative voice. Cover ONLY the provided issues and their provided stances; "
                    "do not invent new issues or contradict the stance descriptions."
                ),
            )),
        )

        system_msg = (
            "You are generating a synthetic British parliamentarian persona.\n"
            "Rules:\n"
            "• You MUST follow the fixed literal fields exactly.\n"
            "• Keep everything consistent with the memoir narrative voice.\n"
            "• Do not mention 'archetype', 'memoir', 'canonical text', or generation instructions.\n"
            "• Presenting problems must be concise phrases, not diagnoses.\n"
            "• If the canonical summary is empty, infer voice/cadence from the title alone (do not invent a 'summary' field).\n"
        )

        user_msg = (
            "Fixed seeds you must respect:\n"
            f"- Party affiliation: {party}\n"
            f"- Occupation (pre-parliament): {occupation} (hint: {occupation_expl})\n"
            f"- Education secondary: {edu_sec} (hint: {edu_sec_expl})\n"
            f"- Education tertiary: {edu_ter} (hint: {edu_ter_expl})\n"
            f"- Region: {region} (hint: {region_expl})\n"
            f"- Parliamentary style: {parliamentary_style}\n"
            f"- Rhetorical register: {rhetorical_register}\n"
            f"- Appearance category: {appearance_cat}\n"
            f"  Examples: {appearance_examples}\n"
            f"- Behavior category: {behavior_cat}\n"
            f"  Examples: {behavior_examples}\n"
            f"- Media disposition: {media_disp}\n"
            f"- Additional traits (grounding only): {additional_traits}\n\n"
            "Speech targets (use these EXACTLY; do not invent issues/stances):\n"
            f"{json.dumps(speech_targets, ensure_ascii=False)}\n\n"
            "Canonical text selection (grounding only):\n"
            f"- title: {memoir_title}\n"
            f"- summary (may be empty): {memoir_summary}\n"
        )

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    text_format=SeededParliamentarianPersonaSchema,
                )
                out = resp.output_parsed

                # enforce exact grounding strings (copy as provided)
                out.memoir_summary = memoir_summary
                out.archetype_description = archetype_desc

                d = out.model_dump()

                # Add deterministic fields as explicit persona columns
                d["policy_stances"] = policy_stances  # dict[issue -> [stance_id]]
                d["speech_issues"] = speech_issues
                d["additional_traits"] = additional_traits

                # make persona_string/hash (excluding only the grounding fields)
                persona_str = self.persona_row_to_string(d)
                d["persona_string"] = persona_str
                d["persona_hash"] = stable_hash(persona_str)

                # word-limit hardening (just in case)
                d["memoir_narrative"] = _words_at_most(str(d.get("memoir_narrative", "")).strip(), 210)
                d["speech"] = _words_at_most(str(d.get("speech", "")).strip(), 260)

                return d

            except Exception as e:
                last_err = e
                time.sleep((0.5 * (2 ** attempt)) + random.random() * 0.25)

        print(f"[warn] generation skipped idx={idx} df_idx={df_idx}: {last_err}")
        return None


# ----------------------------
# Multiprocessing worker
# ----------------------------
def _worker_one(
    job: Tuple[int, int],
    seeds_yaml: str,
    demographics_csv: str,
    version: str,
    model: str,
    temperature: float,
    top_p: float,
    api_key: Optional[str],
    base_seed: int,
) -> Optional[dict]:
    idx, df_idx = job
    try:
        gen = ParliamentarianPersonaGenerator(
            seeds_yaml=seeds_yaml,
            demographics_csv=demographics_csv,
            version=version,
            model=model,
            temperature=temperature,
            top_p=top_p,
            api_key=api_key,
            rng_seed=base_seed,
        )
        return gen.generate_one(idx=idx, df_idx=df_idx)
    except Exception as e:
        print(f"[warn] worker failed idx={idx} df_idx={df_idx}: {e}")
        return None


# ----------------------------
# Push to HF (merge + dedup)
# ----------------------------
def push_personas_to_hub(
    records: List[dict],
    repo_id: str,
    config_name: str,
    hf_token: Optional[str],
    private: bool,
    commit_message: str,
):
    if not records:
        print("[warn] No records to push.")
        return

    if hf_token:
        HfFolder.save_token(hf_token)

    df_new = pd.DataFrame([dict(r) for r in records])

    # Load existing config train split, if any
    try:
        ds_existing = load_dataset(repo_id, name=config_name, split="train", token=hf_token)
        df_existing = ds_existing.to_pandas()
    except Exception:
        df_existing = pd.DataFrame(columns=list(df_new.columns))

    df_merged = pd.concat([df_existing, df_new], ignore_index=True)

    # Dedup by uuid (requested)
    if "uuid" in df_merged.columns:
        df_merged = df_merged.drop_duplicates(subset=["uuid"]).reset_index(drop=True)
    else:
        df_merged = df_merged.drop_duplicates().reset_index(drop=True)

    ds = Dataset.from_pandas(df_merged, preserve_index=False)
    dsd = DatasetDict({"train": ds})

    dsd.push_to_hub(
        repo_id,
        config_name=config_name,
        private=private,
        token=hf_token,
        commit_message=commit_message,
    )

    print(
        f"[ok] Pushed {len(df_new)} new rows; merged dataset now {len(df_merged)} rows "
        f"to {repo_id} (config={config_name})."
    )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds-yaml",
        type=str,
        default=str(DEFAULT_SEEDS_YAML),
        help="Path to parliament seeds YAML",
    )
    parser.add_argument(
        "--demographics-csv",
        type=str,
        default=str(DEFAULT_DEMOGRAPHICS_CSV),
        help="Path to CSV used for demographic cheating (uuid/sex/age/name/marital/ethnic).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="thoughtworks/parliamentary_personas",
        help="HF repo id to push to",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="HF dataset config name (bump to create a new config)",
    )
    parser.add_argument(
        "--num-personas",
        type=int,
        default=2000,
        help="If not debug: sample at most this many personas (seed=42).",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Debug mode: pushes only 20 rows (default: true). Use --no-debug to disable.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    args = parser.parse_args()

    # Load demographics upfront to choose indices (deterministic sampling)
    df = pd.read_csv(args.demographics_csv)
    required = {"uuid", "sex", "age", "first_name", "last_name", "marital_status", "ethnic_background"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in demographics CSV: {sorted(miss)}")
    df = df.dropna(subset=list(required)).copy()
    df["age"] = df["age"].astype(int)
    df = df[df["age"] >= 30].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after enforcing required columns and age>=30")

    # Determine which df indices to generate
    if args.debug:
        n = min(20, len(df))
        chosen_df_indices = list(range(n))
    else:
        rng = random.Random(42)
        n = min(int(args.num_personas), len(df))
        chosen_df_indices = rng.sample(list(range(len(df))), n)

    jobs: List[Tuple[int, int]] = [(i, df_idx) for i, df_idx in enumerate(chosen_df_indices)]

    api_key = args.openai_api_key
    hf_token = args.hf_token

    records: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _worker_one,job,
                args.seeds_yaml, args.demographics_csv,
                args.version, args.model,
                args.temperature, args.top_p, api_key, 42,  # base_seed fixed as requested
            )
            for job in jobs
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Generating personas"):
            r = fut.result()
            if r:
                records.append(r)

    print(f"[ok] Generated {len(records)} personas (requested {len(jobs)}).")

    commit_message = f"Add parliamentarian personas ({args.version}){' [debug]' if args.debug else ''}"
    push_personas_to_hub(
        records=records,
        repo_id=args.repo_id,
        config_name=args.version,
        hf_token=hf_token,
        private=args.private,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    main()
