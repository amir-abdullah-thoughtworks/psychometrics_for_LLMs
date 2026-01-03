# src/persona_generation/pydantic_parliamentarian_generation.py
from __future__ import annotations

import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings(
    "ignore",
    message=r".*NotOpenSSLWarning.*|.*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
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

import traceback
from datetime import datetime
from threading import Lock

_ERROR_LOG_PATH = Path("errors.log")
_ERROR_LOG_LOCK = Lock()

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
def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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


def _words_at_most(text: str, max_words: int) -> str:
    w = text.split()
    if len(w) <= max_words:
        return text.strip()
    return " ".join(w[:max_words]).strip()


def log_worker_error(*, idx: int, df_idx: int, exc: Exception, context: Optional[str] = None):
    ts = datetime.utcnow().isoformat()
    header = f"\n{'=' * 80}\n[{ts}] idx={idx} df_idx={df_idx}"
    if context:
        header += f" context={context}"
    header += "\n"

    tb = traceback.format_exc()
    with _ERROR_LOG_LOCK:
        with _ERROR_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(header)
            f.write(str(exc) + "\n")
            f.write(tb)


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
# STRICT extraction for AdditionalTraits (schema-exact)
# ----------------------------
_ADDITIONAL_TRAITS_ORDER: List[str] = [
    "RhetoricalRegister",
    "RelationshipToParty",
    "AttitudeToInstitutions",
    "CommitteeFocus",
    "ConstituencyType",
    "DonorLobbyExposure",
    "ScandalVulnerability",
    "MediaFootprint",
    "LeadershipAmbition",
    "CoalitionPosture",
    "ConstituencyServiceStyle",
]


# ----------------------------
# Other seed extraction helpers (kept as-is; not fully schema-locked)
# ----------------------------
def load_root(seeds_yaml: str) -> dict:
    with open(seeds_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Seed YAML must be a dict")

    if "BritishParliamentPersonaSeeds" not in data:
        raise ValueError("Top-level key 'BritishParliamentPersonaSeeds' missing")

    root = data["BritishParliamentPersonaSeeds"]
    if not isinstance(root, dict):
        raise ValueError("BritishParliamentPersonaSeeds must be a dict")

    return root



def extract_canonical_texts(root: dict) -> tuple[list[str], dict[str, str]]:
    ct = root["CanonicalTexts"]

    if not isinstance(ct, dict):
        raise ValueError("CanonicalTexts must be a dict")

    titles = ct["titles"]
    if not isinstance(titles, list) or not titles:
        raise ValueError("CanonicalTexts.titles must be a non-empty list")

    titles = [str(t).strip() for t in titles]
    if any(not t for t in titles):
        raise ValueError("CanonicalTexts.titles contains empty entries")

    summaries: dict[str, str] = {}

    # items are OPTIONAL but schema-valid
    items = ct.get("items", [])
    if items:
        if not isinstance(items, list):
            raise ValueError("CanonicalTexts.items must be a list")
        for it in items:
            if not isinstance(it, dict):
                raise ValueError("CanonicalTexts.items elements must be dicts")
            title = it["title"]
            summary = it["summary"]
            if title and summary:
                summaries[str(title)] = str(summary)

    return titles, summaries


def extract_party_priors(root: dict) -> dict[str, float]:
    pa = root["PartyAffiliation"]
    priors = pa["priors"]

    if not isinstance(priors, dict) or not priors:
        raise ValueError("PartyAffiliation.priors must be a non-empty dict")

    out: dict[str, float] = {}
    for party, block in priors.items():
        if not isinstance(block, dict):
            raise ValueError(f"PartyAffiliation.priors.{party} must be dict")
        out[party] = float(block["base_probability"])

    return out


def extract_regions(root: dict) -> list[tuple[str, float, str]]:
    reg = root["Regions"]
    items = reg["items"]
    out = []
    for region, meta in items.items():
        out.append(
            (region, float(meta["population_share"]), str(meta["explanation"]),)
        )
    return out

def extract_occupations(root: dict) -> list[tuple[str, float, str]]:
    items = root["Occupations"]["items"]
    return [
        (it["value"], float(it["weight"]), it["explanation"])
        for it in items
    ]

def extract_education(root: dict):
    edu = root["Education"]
    secondary = [
        (it["value"], float(it["weight"]), it["explanation"])
        for it in edu["Secondary"]
    ]
    tertiary = [
        (it["value"], float(it["weight"]), it["explanation"])
        for it in edu["Tertiary"]
    ]
    return secondary, tertiary

def extract_archetypes(root: dict) -> list[dict]:
    items = root["Archetypes"]["items"]
    for a in items:
        for k in ("name", "description", "signature_tells", "strengths", "pitfalls"):
            if k not in a:
                raise ValueError(f"Archetype missing key: {k}")
    return items

def extract_additional_traits(root: dict) -> dict[str, list[tuple[str, str]]]:
    at = root["AdditionalTraits"]

    if set(at.keys()) != set(_ADDITIONAL_TRAITS_ORDER):
        raise ValueError("AdditionalTraits keys do not match schema exactly")

    out: dict[str, list[tuple[str, str]]] = {}
    for name in _ADDITIONAL_TRAITS_ORDER:
        items = at[name]["items"]
        if not isinstance(items, list) or not items:
            raise ValueError(f"AdditionalTraits.{name}.items missing/empty")
        pairs: list[tuple[str, str]] = []
        for it in items:
            if not isinstance(it, dict):
                raise ValueError(f"AdditionalTraits.{name}.items must be list[dict]")
            v = str(it.get("value", "")).strip()
            e = str(it.get("explanation", "")).strip()
            if not v:
                raise ValueError(f"AdditionalTraits.{name}.items has empty value")
            if not e:
                raise ValueError(f"AdditionalTraits.{name}.items has empty explanation")
            pairs.append((v, e))
        out[name] = pairs
    return out

def extract_persona_core(root: dict):
    pc = root["PersonaCore"]

    def extract_block(name: str):
        items = pc[name]["items"]
        results = [(it["value"], it["explanation"]) for it in items]
        results = [result for result in results if result[0]]
        return results

    appearance = extract_block("Appearance")
    parliamentary_style = extract_block("ParliamentaryStyle")
    media_disposition = extract_block("MediaDisposition")


    return appearance, parliamentary_style, media_disposition

def extract_policy_debates(root: dict) -> dict[str, list[dict]]:
    items = root["PolicyDebates"]["items"]

    out = {}
    for issue in items:
        name = issue["name"]
        stances = issue["stances"]
        out[name] = stances
    return out

def extract_generator_config(root: dict) -> dict:
    return root["GeneratorConfig"]

# ----------------------------
# Persona Generator
# ----------------------------
class ParliamentarianPersonaGenerator:
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
        root = load_root(seeds_yaml)

        self.version = root["version"]

        self.memoir_titles, self.memoir_summaries = extract_canonical_texts(root)
        self.party_priors = extract_party_priors(root)
        self.regions = extract_regions(root)
        self.occupations = extract_occupations(root)
        self.educ_secondary, self.educ_tertiary = extract_education(root)
        self.archetypes = extract_archetypes(root)

        self._mem_offset = self._rng.randrange(len(self.memoir_titles))
        self._arch_offset = self._rng.randrange(len(self.archetypes))

        (
            self.appearance_items,
            self.parliamentary_styles,
            self.media_dispositions,
        ) = extract_persona_core(root)

        # print("DEBUG parliamentary_styles sample:", self.parliamentary_styles[:3])
        # print("DEBUG types:", [type(x) for x in self.parliamentary_styles[:3]])

        self.additional_trait_options = extract_additional_traits(root)
        self.policy_lookup = extract_policy_debates(root)

        clinical = root.get("ClinicalFewShots", {})
        if not isinstance(clinical, dict):
            raise ValueError("ClinicalFewShots must be a dict")

        def _req_list(name: str) -> List[str]:
            xs = clinical.get(name)["items"]
            if not isinstance(xs, list) or not xs:
                raise ValueError(f"ClinicalFewShots.{name} must be a non-empty list")
            return [str(x).strip() for x in xs if str(x).strip()]

        self.fewshot_presenting_problems = _req_list("presenting_problems")
        self.fewshot_thought_content = _req_list("thought_content")
        self.fewshot_insight_judgment = _req_list("insight_judgment")
        self.fewshot_cognition = _req_list("cognition")

        self.generator_config = extract_generator_config(root)

        # demographics CSV
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
        def fmt(label: str, xs) -> str:
            if not xs:
                return ""
            if isinstance(xs, list):
                xs = [str(x).strip() for x in xs if str(x).strip()]
                if not xs:
                    return ""
                return f"{label}: " + "; ".join(xs) + "."
            s = str(xs).strip()
            return f"{label}: {s}." if s else ""

        name = (a.get("name") or "").strip()
        desc = (a.get("description") or "").strip()
        parts: List[str] = []
        if name:
            parts.append(f"Archetype: {name}.")
        if desc:
            parts.append(desc if desc.endswith((".", "!", "?")) else desc + ".")
        s1 = fmt("signature_tells", a.get("signature_tells"))
        if s1:
            parts.append(s1)
        s2 = fmt("strengths", a.get("strengths"))
        if s2:
            parts.append(s2)
        s3 = fmt("pitfalls", a.get("pitfalls"))
        if s3:
            parts.append(s3)
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

    def _pick_media_disposition_by_index(self, idx: int) -> str:
        rng = random.Random(self.base_seed + 137 * idx)
        return rng.choice(self.media_dispositions)

    # ✅ LAW: pick ONE value for EACH AdditionalTraits key, promoted to top-level fields
    def _pick_additional_traits_top_level_by_index(self, idx: int) -> Dict[str, Dict[str, str]]:
        rng = random.Random(self.base_seed + 139 * idx)
        out: Dict[str, Dict[str, str]] = {}
        for trait_name in _ADDITIONAL_TRAITS_ORDER:
            pairs = self.additional_trait_options[trait_name]  # list[(value, explanation)]
            value, explanation = rng.choice(pairs)
            out[trait_name] = {"value": value, "explanation": explanation}
        return out

    def _pick_appearance_by_index(self, idx: int) -> Tuple[str, List[str]]:
        rng = random.Random(self.base_seed + 127 * idx)

        vals = [v for (v, _) in self.appearance_items]
        if not vals:
            raise ValueError("PersonaCore.Appearance.items yielded no values")

        chosen_value = rng.choice(vals)
        examples = rng.sample(vals, k=min(5, len(vals)))
        return chosen_value, examples


    # ----------------------------
    # Policy stance sampling (per issue, ONE stance, party-weighted)
    # ----------------------------
    def _pick_policy_stances_by_party(self, idx: int, party: str) -> Dict[str, List[str]]:
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
        exclude = {
            "memoir",
            "memoir_summary",
            "memoir_narrative",
            "version",
            "uuid",
            "archetype",
            "archetype_description",
        }

        # Put AdditionalTraits promoted keys in schema order, top-level
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
            # promoted AdditionalTraits
            *_ADDITIONAL_TRAITS_ORDER,
            "appearance_category",
            "media_disposition",
            "policy_stances",
            "speech_issues",
            "speech",
            "appearance",
            "educational_vocational_history",
            "medical_developmental_history",
            "family_history",
            "presenting_problems",
            "thought_content",
            "insight_judgment",
            "cognition",
            "emotional_behavioral_functioning",
            "social_functioning",
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

            if k == "presenting_problems" and isinstance(v, list):
                v2 = [clean_value(x) for x in v if x is not None and str(x).strip()]
                if v2:
                    lines.append("presenting_problems:")
                    lines.extend([f"- {x}" for x in v2])
                continue

            if k in _ADDITIONAL_TRAITS_ORDER and isinstance(v, dict):
                lines.append(f"{k}.value: {v.get('value', '')}")
                lines.append(f"{k}.explanation: {v.get('explanation', '')}")
                continue

            s = clean_value(v)
            if s:
                lines.append(f"{k}: {s}")

        extras = sorted([k for k in row.keys() if k not in set(preferred_order) and k not in exclude])
        for k in extras:
            v = row.get(k)
            if v is None:
                continue
            s = clean_value(v)
            if s:
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
        media_disp = self._pick_media_disposition_by_index(idx)

        # appearance chosen values
        appearance_value, appearance_examples = self._pick_appearance_by_index(idx)

        # ✅ AdditionalTraits: each is a dict {value, explanation}
        trait_values = self._pick_additional_traits_top_level_by_index(idx)

        # policy stances across ALL issues (stored as dict issue -> [stance_id])
        policy_stances = self._pick_policy_stances_by_party(idx, party)

        # speech issues: pick 3 issues from the already-chosen stances
        speech_issues = self._pick_speech_issues(idx, policy_stances, k=3)
        speech_targets = self._speech_targets_payload(speech_issues, policy_stances)

        # ----------------------------
        # Few-shot examples for clinical fields (3 per field)
        # Assumes you loaded these lists from YAML into:
        #   self.fewshot_presenting_problems: List[str]  (each is a single phrase)
        #   self.fewshot_thought_content: List[str]      (25–45 words)
        #   self.fewshot_insight_judgment: List[str]     (25–45 words)
        #   self.fewshot_cognition: List[str]            (25–45 words)
        # ----------------------------
        rng_fs = random.Random(self.base_seed + 1009 * idx)

        def pick_n(pool: List[str], n=3) -> List[str]:
            pool = [str(x).strip() for x in (pool or []) if str(x).strip()]
            if len(pool) <= n:
                return pool
            return rng_fs.sample(pool, n)

        fs_presenting = pick_n(getattr(self, "fewshot_presenting_problems", []))
        fs_thought = pick_n(getattr(self, "fewshot_thought_content", []))
        fs_insight = pick_n(getattr(self, "fewshot_insight_judgment", []))
        fs_cog = pick_n(getattr(self, "fewshot_cognition", []))

        # ----------------------------
        # Build dynamic Pydantic trait fields (top-level Literals)
        # Each trait becomes an object with {value, explanation} both literal
        # ----------------------------
        trait_fields: Dict[str, tuple[Any, Any]] = {}
        for trait_name in _ADDITIONAL_TRAITS_ORDER:
            chosen = trait_values[trait_name]
            val = str(chosen["value"])
            expl = str(chosen["explanation"])

            TraitModel = create_model(  # type: ignore[misc]
                f"{trait_name}Choice_{idx}",
                value=(Literal[val], ...),
                explanation=(Literal[expl], ...),
            )
            trait_fields[trait_name] = (TraitModel, ...)

        SeededParliamentarianPersonaSchema = create_model(  # type: ignore[assignment]
            f"SeededParliamentarianPersonaSchema_{idx}",
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

            appearance_category=(Literal[appearance_value], ...),
            media_disposition=(Literal[media_disp], ...),

            # ✅ promoted AdditionalTraits fields at top-level
            **trait_fields,

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

            appearance=(str, Field(
                ...,
                description=(
                    "2–3 specialized sentences about the person's appearance, faithful to the memoir narrative voice. "
                    "Anchor it in the fixed appearance_category value; stay concrete."
                ),
            )),

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

            # ✅ Now free-generated, but with 3 few-shot examples included in the prompt
            presenting_problems=(List[str], Field(
                ...,
                description=(
                    "Return 3–6 concise clinical presenting problems phrases describing the parliamentarian. "
                    "Use clinical/medical language (can include diagnostic formulations). "
                    "Avoid generic politics tropes; not all problems should be work/politics-related."
                ),
            )),
            thought_content=(str, Field(
                ...,
                description=(
                    "25–45 words. Write like a clinician documenting thought content (MSE style): themes, ruminations, "
                    "preoccupations, intrusions, cognitive style. Clinical/medical focus."
                ),
            )),
            insight_judgment=(str, Field(
                ...,
                description=(
                    "25–45 words. Clinical assessment of insight and judgment: awareness, attribution, decision-making, "
                    "stress effects, risk. Use professional tone."
                ),
            )),
            cognition=(str, Field(
                ...,
                description=(
                    "25–45 words. Clinical cognition/MSE style: attention, memory, executive function, processing speed, "
                    "cognitive flexibility; specify if variable under stress."
                ),
            )),

            emotional_behavioral_functioning=(str, Field(
                ...,
                description="35–55 words. How they handle pressure and difficult feelings; show behavior, avoid labels unless clinically warranted.",
            )),
            social_functioning=(str, Field(
                ...,
                description="35–55 words. Patterns in closeness, trust, and participation with others; concrete cues.",
            )),
            political_relations=(str, Field(
                ...,
                description="55–75 words. How they conduct themselves with other parliamentarians and constituents. Keep cohesive with their other description.",
            )),
            summary_of_psychological_profile=(str, Field(
                ...,
                description=(
                    "150–250 words. Integrative clinical summary using narrative + histories + functioning + problems. "
                    "Professional tone; may include diagnostic impressions. Do not explicitly name the archetype."
                ),
            )),

            speech=(str, Field(
                ...,
                description=(
                    "Write a short speech (<=250 words) as the parliamentarian. "
                    "Use BOTH the parliamentary_style and the top-level RhetoricalRegister.value to set the tenor and rhetorical devices. "
                    "Ground it in the memoir narrative voice. Cover ONLY the provided issues and their provided stances; "
                    "do not invent new issues or contradict the stance descriptions."
                ),
            )),
        )

        rhetorical_register_value = trait_values["RhetoricalRegister"]["value"]
        rhetorical_register_expl = trait_values["RhetoricalRegister"]["explanation"]

        system_msg = (
            "You are generating a synthetic British parliamentarian persona.\n"
            "Rules:\n"
            "• You MUST follow the fixed literal fields exactly.\n"
            "• Keep everything consistent with the memoir narrative voice.\n"
            "• Do not mention 'archetype', 'memoir', 'canonical text', or generation instructions.\n"
            "• For clinical fields, write as a psychologist/clinician would document an assessment; professional tone.\n"
            ". Write each of these in the third person!"
        )

        def _fmt_fewshot_block(title: str, examples: List[Any]) -> str:
            if not examples:
                return f"{title}: (none)\n"
            lines = [f"{title}:"]
            for i, ex in enumerate(examples, 1):
                if isinstance(ex, list):
                    # presenting_problems might be a list of phrases; render as YAML-ish list
                    lines.append(f"  Example {i}:")
                    for p in ex:
                        lines.append(f"    - {p}")
                else:
                    lines.append(f"  Example {i}: {str(ex).strip()}")
            return "\n".join(lines) + "\n"

        user_msg = (
                "Fixed seeds you must respect:\n"
                f"- Party affiliation: {party}\n"
                f"- Occupation (pre-parliament): {occupation} (hint: {occupation_expl})\n"
                f"- Education secondary: {edu_sec} (hint: {edu_sec_expl})\n"
                f"- Education tertiary: {edu_ter} (hint: {edu_ter_expl})\n"
                f"- Region: {region} (hint: {region_expl})\n"
                f"- Parliamentary style: {parliamentary_style}\n"
                f"- RhetoricalRegister (top-level): {rhetorical_register_value}\n"
                f"  RhetoricalRegisterExplanation: {rhetorical_register_expl}\n"
                f"- appearance_category: {appearance_value}\n"
                f"  Appearance Examples: {appearance_examples}\n"
                f"- Media disposition: {media_disp}\n"
                f"- AdditionalTraits (ALL top-level fixed fields): {json.dumps(trait_values, ensure_ascii=False)}\n\n"
                "Speech targets (use these EXACTLY; do not invent issues/stances):\n"
                f"{json.dumps(speech_targets, ensure_ascii=False)}\n\n"
                "Canonical text selection (grounding only):\n"
                f"- title: {memoir_title}\n"
                f"- summary (may be empty): {memoir_summary}\n\n"
                "Clinical style few-shot examples (match style, not content; do not copy verbatim unless it fits):\n"
                + _fmt_fewshot_block("presenting_problems examples", [[x] for x in fs_presenting])
                + _fmt_fewshot_block("thought_content examples", fs_thought)
                + _fmt_fewshot_block("insight_judgment examples", fs_insight)
                + _fmt_fewshot_block("cognition examples", fs_cog)
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

                # enforce exact grounding strings
                out.memoir_summary = memoir_summary
                out.archetype_description = archetype_desc

                d = out.model_dump()

                # deterministic columns
                d["policy_stances"] = policy_stances
                d["speech_issues"] = speech_issues

                # persona_string/hash
                persona_str = self.persona_row_to_string(d)
                d["persona_string"] = persona_str
                d["persona_hash"] = stable_hash(persona_str)

                # word-limit hardening
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
        log_worker_error(idx=idx, df_idx=df_idx, exc=e, context="worker_generate_one")
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
    print(df_new["parliamentary_style"].map(type).value_counts().head(10))

    try:
        ds_existing = load_dataset(repo_id, name=config_name, split="train", token=hf_token)
        df_existing = ds_existing.to_pandas()
    except Exception:
        df_existing = pd.DataFrame(columns=list(df_new.columns))

    df_merged = pd.concat([df_existing, df_new], ignore_index=True)

    bad = df_merged[df_merged["parliamentary_style"].map(lambda x: not isinstance(x, str))]
    print(bad[["uuid", "name", "parliamentary_style"]].head(8))
    print(bad["parliamentary_style"].map(type).value_counts())

    if "persona_hash" in df_merged.columns:
        df_merged = df_merged.drop_duplicates(subset=["persona_hash"]).reset_index(drop=True)
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
    parser.add_argument("--seeds-yaml", type=str, default=str(DEFAULT_SEEDS_YAML))
    parser.add_argument("--demographics-csv", type=str, default=str(DEFAULT_DEMOGRAPHICS_CSV))
    parser.add_argument("--repo-id", type=str, default="thoughtworks/parliamentary_personas")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--num-personas", type=int, default=2200)
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Debug mode: pushes only 8 rows (default: true). Use --no-debug to disable.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    args = parser.parse_args()

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

    if args.debug:
        n = min(8, len(df))
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
                _worker_one,
                job,
                args.seeds_yaml,
                args.demographics_csv,
                args.version,
                args.model,
                args.temperature,
                args.top_p,
                api_key,
                42,
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
