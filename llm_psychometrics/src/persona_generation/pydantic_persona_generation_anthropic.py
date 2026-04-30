from __future__ import annotations

import hashlib
import json
import time
import random
import traceback

from pathlib import Path

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

import pandas as pd
import yaml
from tqdm import tqdm
import anthropic
from pydantic import BaseModel, Field, create_model
from datasets import Dataset, load_dataset

from sentence_transformers import SentenceTransformer

from typing import Set

_PERSONA_SYSTEM_MSG = (
    "You are an expert clinical interviewer and psychological profiler.\n"
    "Generate a detailed, realistic persona strictly adhering to the schema and length guidance.\n"
    "Avoid caricature or stereotypes; allow subtle contradictions with archetype for realism; avoid repetition.\n"
    "Return valid structured output matching the provided schema exactly.\n"
    "\n"
    "STYLE & GROUNDING (do NOT output this list):\n"
    "1) The memoir_narrative is canonical grounding. Write it as a concrete, sensory, scene-level story (180–250 words).\n"
    "   All fields must align with its facts and tone; if conflicts arise with archetype, prefer narrative. If conflicts arise with demographics, pick the demographics.\n"
    "2) Treat the archetype as a loose orientation. Do NOT quote or paraphrase it; never list 'Core trait/Focus/Strengths/Challenges'.\n"
    "3) Do not reuse ≥5 consecutive words from inputs (archetype description or memoir summary). Rephrase and localize details to the scene.\n"
    "4) Favor specificity (who/what/where/when) over generic traits; vary wording across sections.\n"
    "5) Persona should be internally consistent between fields. "
    "6) Use natural phrasing; do not feel compelled to use section labels or taxonomy words "
    "(e.g., 'stress,' 'trauma,' 'coping,' 'abstraction,' 'obsession'). Prefer specific, scene-derived wording."
)


def _norm(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _load_existing_version_keys(version: str) -> Set[str]:
    """
    Load ./{version}.jsonl if present and return a set of row keys:
      norm(name)|age|norm(sex)|norm(location)
    """
    p = Path(f"{version}.jsonl")
    if not p.exists():
        return set()

    keys: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    keys.add(obj['uuid'])
            except Exception:
                continue
    return keys

@dataclass(frozen=True)
class Demographics:
    name: str
    age: int
    sex: str
    location: str
    education_level: str
    bachelors_field: str
    ethnic_background: str
    marital_status: str

# =========================
# Persona Generator (Class)
# =========================
class PersonaGenerator:
    """
    Generates ONE persona via Anthropic Claude Structured Outputs (tool use) with a Pydantic schema
    whose constrained fields (version, name, age, location, archetype, archetype_description,
    memoir, memoir_summary, appearance_category, behavior_category) are Literals of the selected values.

    Multiprocess-friendly:
      - Use generate_one(idx) with index-based selection:
          demographics row = idx % len(df)
          archetype        = idx % len(archetypes)
          memoir           = idx % len(memoir_titles)
          appearance/behavior categories are sampled with a RNG seeded by (base_seed + c * idx).

    Requires:
      - populated_police_seeds.yaml with:
          PoliceOfficerPersonaSeeds:
            archetypes: [ {name: "...", core_trait: "...", focus: "...", strengths: [...], challenges: [...] }, ... ]
            MemoirSeeds: ["...", ...]
            MemoirSummaries: { "<memoir title>": "<~20-word summary>", ... }
            AppearanceCategories: { "<category>": ["seed1", ...], ... }
            BehaviorCategories: { "<category>": ["seed1", ...], ... }
      - balanced_us_police_officers.csv with columns (strict):
          sex, age, city, state, first_name, last_name
      - sentence-transformers model: Qwen/Qwen3-Embedding-0.6B
    """

    # Generated text fields eligible for embeddings (seeds/demographics excluded)
    _GENERATED_TEXT_FIELDS = [
        "educational_vocational_history",
        "appearance",
        "behavior",
        "mood_affect",
        "speech",
        "thought_content",
        "insight_judgment",
        "cognition",
        "medical_developmental_history",
        "family_history",
        "emotional_behavioral_functioning",
        "social_functioning",
        "summary_of_psychological_profile",
    ]

    def __init__(
        self,
        populated_seeds_yaml: str,
        balanced_officers_csv: str,
        version: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 1.0,
        top_p: float = 0.98,
        api_key: Optional[str] = None,
        rng_seed: Optional[int] = None,
    ):
        self.version = version
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.base_seed = rng_seed if rng_seed is not None else 1337
        self._rng = random.Random(self.base_seed)

        self._embedder: Optional[SentenceTransformer] = None  # lazy-loaded per process

        # ---- load seeds ----
        with open(populated_seeds_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
        root = data.get("PoliceOfficerPersonaSeeds", data) or {}

        # archetypes as list[dict]
        raw_archetypes = root.get("archetypes", [])
        if not isinstance(raw_archetypes, list) or not raw_archetypes:
            raise ValueError("No archetypes found in populated_police_seeds.yaml")
        self.archetypes: List[dict] = []
        for a in raw_archetypes:
            self.archetypes.append(a if isinstance(a, dict) else {"name": str(a)})

        # memoirs + summaries
        self.memoir_titles: List[str] = root.get("MemoirSeeds", []) or []
        self.memoir_summaries: Dict[str, str] = root.get("MemoirSummaries", {}) or {}
        if not self.memoir_titles:
            raise ValueError("No MemoirSeeds found in populated_police_seeds.yaml")
        missing = [m for m in self.memoir_titles if m not in self.memoir_summaries]
        if missing:
            raise ValueError(f"Missing MemoirSummaries for: {missing}")

        # appearance / behavior categories
        self.appearance_categories: Dict[str, List[str]] = root.get("AppearanceCategories") or {}
        self.behavior_categories: Dict[str, List[str]] = root.get("BehaviorCategories") or {}
        if not self.appearance_categories or not self.behavior_categories:
            raise ValueError("AppearanceCategories/BehaviorCategories not found or empty.")

        # ---- officers (strict) ----
        self.df = pd.read_csv(balanced_officers_csv)

        self._arch_offset = self._rng.randrange(len(self.archetypes))
        self._mem_offset = self._rng.randrange(len(self.memoir_titles))

        required = {"sex", "age", "city", "state", "first_name", "last_name", "education_level",
                    "marital_status","ethnic_background"}
        miss = required - set(self.df.columns)
        if miss:
            raise ValueError(f"Missing required columns in CSV: {sorted(miss)}")
        self.df = self.df.dropna(subset=list(required)).reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No valid rows after enforcing required columns.")
        self.df["age"] = self.df["age"].astype(int)
        self.df = self.df[(self.df["age"] >= 21) & (self.df["age"] <= 70)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No rows with age in [21, 70].")

    # -------- embedder (lazy) --------
    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        return self._embedder

    def build_concat_and_embedding(self, rec: dict) -> tuple[str, list[float]]:
        """
        Build a single concatenated text from generated fields only (not literals),
        in the form 'key: value\\n' per line; then return its embedding.
        Never raises; returns ('', []) if empty or on embedding issues.
        """
        lines: list[str] = []
        for key in self._GENERATED_TEXT_FIELDS:
            if key not in rec:
                continue
            val = rec[key]
            if val is None:
                continue

            # Presenting problems may be a list -> join
            if key == "presenting_problems":
                if isinstance(val, list):
                    # keep concise, inline semicolons
                    val_str = "; ".join([str(x).strip() for x in val if str(x).strip()])
                else:
                    val_str = str(val).strip()
            else:
                val_str = str(val).strip()

            if val_str:
                lines.append(f"{key}: {val_str}")

        concat_text = "\n".join(lines) + ("\n" if lines else "")
        if not concat_text.strip():
            return "", []

        # Embedding
        emb_model = self._get_embedder()
        if emb_model is None:
            return concat_text, []
        try:
            vec = emb_model.encode([concat_text], convert_to_numpy=False)[0]
            vec = list(map(float, vec))
        except Exception:
            vec = []
        return concat_text, vec

    # -------- archetype helpers --------
    def _compose_archetype_description(self, a: dict) -> str:
        parts = []
        if a.get("core_trait"):
            parts.append(f"Core trait: {a['core_trait']}.")
        if a.get("focus"):
            parts.append(f"Focus: {a['focus']}.")
        if a.get("strengths"):
            s = a["strengths"]
            parts.append("Strengths: " + ("; ".join(s) if isinstance(s, list) else str(s)) + ".")
        if a.get("challenges"):
            c = a["challenges"]
            parts.append("Challenges: " + ("; ".join(c) if isinstance(c, list) else str(c)) + ".")
        return " ".join(parts).strip() or f"Archetype: {a.get('name','(unspecified)')}."

    def get_row(self, idx: int):
        return self.df.iloc[idx]

    # -------- index-based pickers (no shared state) --------
    def _pick_demographics_by_index(self, idx: int) -> Demographics:
        row = self.get_row(idx)
        name = f"{str(row['first_name']).strip()} {str(row['last_name']).strip()}"
        age = int(row["age"])
        location = f"{str(row['city']).strip()}, {str(row['state']).strip()}"
        education_level = str(row["education_level"]).strip()
        sex = str(row["sex"]).strip()
        bachelors_field = str(row["bachelors_field"]).strip()
        ethnic_background = str(row["ethnic_background"]).strip()
        marital_status = str(row["marital_status"]).strip()
        return Demographics(
            name=name,
            age=age,
            sex=sex,
            location=location,
            education_level=education_level,
            bachelors_field=bachelors_field,
            ethnic_background=ethnic_background,
            marital_status=marital_status,
        )

    def _pick_archetype_by_index(self, idx: int) -> Tuple[str, str]:
        a = self.archetypes[(self._arch_offset + idx) % len(self.archetypes)]
        return a["name"], self._compose_archetype_description(a)

    def _pick_memoir_by_index(self, idx: int) -> Tuple[str, str]:
        title = self.memoir_titles[(self._mem_offset + idx) % len(self.memoir_titles)]
        return title, self.memoir_summaries[title]

    def _pick_appearance_random(self, idx: int) -> Tuple[str, List[str]]:
        rng = random.Random(self.base_seed + 17 * idx)
        cat = rng.choice(list(self.appearance_categories.keys()))
        seeds = self.appearance_categories.get(cat, []) or []
        k = min(5, len(seeds))
        examples = rng.sample(seeds, k) if k else []
        return cat, examples

    def _pick_behavior_random(self, idx: int) -> Tuple[str, List[str]]:
        rng = random.Random(self.base_seed + 23 * idx)
        cat = rng.choice(list(self.behavior_categories.keys()))
        seeds = self.behavior_categories.get(cat, []) or []
        k = min(5, len(seeds))
        examples = rng.sample(seeds, k) if k else []
        return cat, examples

    def embed_generated_fields(self, rec: dict) -> List[float]:
        """
        Return ONE embedding vector for the concatenated generated text only.
        Includes:
          - all fields in _GENERATED_TEXT_FIELDS
          - all items in presenting_problems
        Excludes seeds/demographics/categories entirely.
        """
        parts: List[str] = []

        # Generated prose fields
        for k in self._GENERATED_TEXT_FIELDS:
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        # Presenting problems (list[str])
        probs = rec.get("presenting_problems") or []
        if isinstance(probs, list):
            probs = [p for p in probs if isinstance(p, str) and p.strip()]
            if probs:
                parts.append("; ".join(probs))

        combined = "\n\n".join(parts).strip()
        if not combined:
            return []  # nothing to embed

        emb_model = self._get_embedder()
        vec = emb_model.encode([combined], convert_to_numpy=False)[0]
        return list(map(float, vec))

    @staticmethod
    def persona_row_to_string(row: dict) -> str:
        """
        Convert a persona dict into a line-separated "persona_string".

        - Uses a preferred key order (labels prettified).
        - Injects presenting problems right after mood_affect.
        - Supports BOTH:
            * `presenting_problems: List[str]` (your current schema), and
            * numbered keys ("1","2",...) from older datasets.
        """
        preferred_order = [
            "name",
            "age",
            "sex",
            "location",
            "ethnic_background",
            "marital_status",
            "appearance",
            "behavior",
            "speech",
            "mood_affect",
            "educational_vocational_history",
            "medical_developmental_history",
            "family_history",
            "thought_content",
            "insight_judgment",
            "cognition",
            "emotional_behavioral_functioning",
            "social_functioning",
            "summary_of_psychological_profile",
        ]

        def pretty_label(k: str) -> str:
            return k.replace("_", " ").strip()

        def clean_value(v: Any) -> str:
            s = str(v).strip()
            return " ".join(s.split())

        # --- presenting problems (new schema: list[str]) ---
        presenting_lines: list[str] = []
        probs = row.get("presenting_problems")

        if isinstance(probs, list):
            probs = [clean_value(p) for p in probs if p is not None and str(p).strip()]
            if probs:
                presenting_lines.append("presenting_problems:")
                presenting_lines.extend([f"- {p}" for p in probs])

        # --- presenting problems (back-compat: numbered keys) ---
        if not presenting_lines:
            presenting = []
            for k, v in row.items():
                if isinstance(k, str) and k.isdigit():
                    if v is not None and str(v).strip():
                        presenting.append((int(k), clean_value(v)))
            presenting.sort(key=lambda x: x[0])
            if presenting:
                presenting_lines.append("presenting_problems:")
                presenting_lines.extend([f"- {p[1]}" for p in presenting])

        lines: list[str] = []
        for k in preferred_order:
            if k in row and row[k] is not None and str(row[k]).strip():
                lines.append(f"{pretty_label(k)}: {clean_value(row[k])}")

            # inject presenting problems right after mood_affect
            if k == "mood_affect" and presenting_lines:
                lines.extend(presenting_lines)

        return "\n".join(lines)


    # -------- main generation (with retries) --------
    def generate_one(self, idx: int) -> Optional[dict]:
        archetype_name, archetype_desc = self._pick_archetype_by_index(idx)
        memoir_title, memoir_summary = self._pick_memoir_by_index(idx)

        row = self.get_row(idx)
        uuid = row["uuid"]

        dem = self._pick_demographics_by_index(idx)

        appearance_cat, appearance_examples = self._pick_appearance_random(idx)
        behavior_cat, behavior_examples = self._pick_behavior_random(idx)

        # dynamic schema with Literals of selected values (model sees exact values)
        SeededPersonaSchema = create_model(  # type: ignore[assignment]
            "SeededPersonaSchema",

            # 1) Hard constraints (literals; EXCEPT archetype which is moved later)
            version=(Literal[self.version], ...),
            name=(Literal[dem.name], ...),
            age=(Literal[dem.age], ...),
            sex=(Literal[dem.sex], ...),
            location=(Literal[dem.location], ...),
            education_level=(Literal[dem.education_level], ...),
            bachelors_field=(Literal[dem.bachelors_field], ...),
            ethnic_background=(Literal[dem.ethnic_background], ...),
            marital_status=(Literal[dem.marital_status], ...),
            appearance_category=(Literal[appearance_cat], ...),
            behavior_category=(Literal[behavior_cat], ...),
            memoir=(Literal[memoir_title], ...),
            # Keep as free text; we overwrite with the exact provided summary post-parse
            memoir_summary=(str, Field(..., description="Summary of the selected memoir; copy as provided.")),

            memoir_narrative=(str, Field(
                ...,
                description=(
                    "Vivid, scene-level narrative (180–250 words) in the selected memoir's milieu. "
                    "This is the canonical grounding; all other fields must align with it."
                )
            )),
            archetype=(Literal[archetype_name], ...),
            archetype_description=(str, Field(
                ...,
                description="Concise description of the archetype as provided; use ONLY to inform the summary; do not paraphrase earlier."
            )),
            appearance=(str, Field(..., description="Observational, sensory description of appearance (10–30 words).")),
            behavior=(str, Field(...,
                                 description="Behavioral cues, posture, interaction style, responsiveness (10–30 words).")),
            speech=(str, Field(..., description="Speech register, rhythm, formality, coherence (10–30 words).")),
            mood_affect=(str, Field(..., description="Mood/affect, tone modulation, emotional nuance (10–30 words).")),
            educational_vocational_history=(str, Field(
                ...,
                description="30–50 words. Align with education_level and bachelors_field; show training/trajectory effects."
            )),
            medical_developmental_history=(str, Field(
                ...,
                description="30–50 words. Health/development context relevant to the scene; only what's needed."
            )),
            family_history=(str, Field(
                ...,
                description="30–50 words. Relational dynamics consistent with narrative, ethnic background, and marital status."
            )),
            presenting_problems=(List[str], Field(
                ...,
                description=(
                    "3–6 mental-health problem phrases describing THE OFFICER (for example., 'insomnia/fragmented sleep', "
                    "'anger rumination', 'avoidance of reminders'). "
                    "Do not copy the examples. Include at least two problems not tied to their police work or to archetype."
                )
            )),
            thought_content=(str, Field(
                ...,
                description="25–45 words. What tends to occupy the person's mind, drawn from the narrative; natural phrasing."
            )),
            insight_judgment=(str, Field(
                ...,
                description="25–45 words. Practical decision-making and self-understanding suggested by the scene."
            )),
            cognition=(str, Field(
                ...,
                description="25–45 words. Observable thinking/recall/problem-solving implied by the narrative."
            )),
            emotional_behavioral_functioning=(str, Field(
                ...,
                description="35–55 words. How the person handles pressure and difficult feelings; show behavior, avoid labels."
            )),
            social_functioning=(str, Field(
                ...,
                description="35–55 words. Patterns in closeness, trust, and participation with others; concrete cues."
            )),
            summary_of_psychological_profile=(str, Field(
                ...,
                description="75–105 words. Integrative summary using the narrative + histories + functioning + problems, framed by the archetype description."
            )),
        )

        def fmt_examples(title: str, items: List[str]) -> str:
            return f"{title} examples:\n" + ("\n".join(f"- {ex}" for ex in items) if items else "(none)")

        system_msg = _PERSONA_SYSTEM_MSG

        user_msg = (
            f"Selected archetype: {archetype_name}\n"
            f"Archetype description (guidance only — DO NOT copy or paraphrase): {archetype_desc}\n"
            f"Selected memoir: {memoir_title}\n"
            f"Memoir summary (guidance only — DO NOT copy or paraphrase): {memoir_summary}\n"
            f"Demographics (USE EXACT VALUES): name={dem.name}; age={dem.age}; location={dem.location}\n"
            f"Education level: education_level={dem.education_level}\n\n"
            f"Appearance category: {appearance_cat}\n{fmt_examples('Appearance', appearance_examples)}\n\n"
            f"Behavior category: {behavior_cat}\n{fmt_examples('Behavior', behavior_examples)}\n\n"
            "Instructions:\n"
            "• First, craft the memoir_narrative (180–250 words) in style and setting of selected memoir as a vivid scene, but alter as needed to be consistent with specified demographics.\n"
            "• Then write every other field so it is consistent with that narrative and the exact demographics.\n"
            "• If narrative and demographics conflict when filling a field, prefer the specified demographics"
            "• Do not mention 'archetype', 'memoir', or 'summary' in the prose; the writing must stand alone.\n"
            "• Include `presenting_problems` as 3–6 concise items consistent with the psychological profile."
        )

        tool = {
            "name": "generate_persona",
            "description": "Generate a structured LEO persona matching the provided schema exactly.",
            "input_schema": SeededPersonaSchema.model_json_schema(),
        }

        # retry with jitter for transient errors
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_msg}],
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "generate_persona"},
                    temperature=self.temperature,
                )
                tool_use_block = next(b for b in resp.content if b.type == "tool_use")
                out = SeededPersonaSchema(**tool_use_block.input)
                # Overwrite to ensure exact ground truth values regardless of model output
                out.archetype_description = archetype_desc
                out.memoir_summary = memoir_summary
                d = out.model_dump()
                d["uuid"] = row["uuid"]
                persona_string = self.persona_row_to_string(d)
                d["persona_string"] = persona_string
                d["persona_hash"] = stable_hash(persona_string)

                return d
            except Exception as e:
                last_err = e
                time.sleep((0.5 * (2 ** attempt)) + random.random() * 0.25)

        # give up quietly: skip this row
        print(f"[warn] generation skipped for idx={idx}: {last_err}")
        with open("logs.txt", "a") as f:
            f.write(f"[warn] generation skipped for idx={idx}: {last_err}")
        return None

    def generate_one_from_seed(self, seed: dict) -> Optional[dict]:
        """
        Re-generate a persona using the exact seeds from an already-generated record.

        All Literal-constrained fields (uuid, demographics, archetype, memoir,
        appearance_category, behavior_category) are pinned to the values in `seed`.
        The prose fields are regenerated fresh by Claude.
        Appearance/behavior examples are re-sampled from the YAML using a hash of
        the uuid so the sampling is deterministic per persona.
        """
        uuid = str(seed["uuid"])
        archetype_name = seed["archetype"]
        archetype_desc = seed["archetype_description"]
        memoir_title = seed["memoir"]
        memoir_summary = seed["memoir_summary"]
        appearance_cat = seed["appearance_category"]
        behavior_cat = seed["behavior_category"]

        dem = Demographics(
            name=seed["name"],
            age=int(seed["age"]),
            sex=seed["sex"],
            location=seed["location"],
            education_level=seed["education_level"],
            bachelors_field=seed["bachelors_field"],
            ethnic_background=seed["ethnic_background"],
            marital_status=seed["marital_status"],
        )

        # Use the original appearance/behavior prose directly from the seed record
        appearance_ref = seed.get("appearance", "")
        behavior_ref = seed.get("behavior", "")

        SeededPersonaSchema = create_model(  # type: ignore[assignment]
            "SeededPersonaSchema",
            version=(Literal[self.version], ...),
            name=(Literal[dem.name], ...),
            age=(Literal[dem.age], ...),
            sex=(Literal[dem.sex], ...),
            location=(Literal[dem.location], ...),
            education_level=(Literal[dem.education_level], ...),
            bachelors_field=(Literal[dem.bachelors_field], ...),
            ethnic_background=(Literal[dem.ethnic_background], ...),
            marital_status=(Literal[dem.marital_status], ...),
            appearance_category=(Literal[appearance_cat], ...),
            behavior_category=(Literal[behavior_cat], ...),
            memoir=(Literal[memoir_title], ...),
            memoir_summary=(str, Field(..., description="Summary of the selected memoir; copy as provided.")),
            memoir_narrative=(str, Field(..., description="Vivid, scene-level narrative (180–250 words) in the selected memoir's milieu. This is the canonical grounding; all other fields must align with it.")),
            archetype=(Literal[archetype_name], ...),
            archetype_description=(str, Field(..., description="Concise description of the archetype as provided; use ONLY to inform the summary; do not paraphrase earlier.")),
            appearance=(str, Field(..., description="Observational, sensory description of appearance (10–30 words).")),
            behavior=(str, Field(..., description="Behavioral cues, posture, interaction style, responsiveness (10–30 words).")),
            speech=(str, Field(..., description="Speech register, rhythm, formality, coherence (10–30 words).")),
            mood_affect=(str, Field(..., description="Mood/affect, tone modulation, emotional nuance (10–30 words).")),
            educational_vocational_history=(str, Field(..., description="30–50 words. Align with education_level and bachelors_field; show training/trajectory effects.")),
            medical_developmental_history=(str, Field(..., description="30–50 words. Health/development context relevant to the scene; only what's needed.")),
            family_history=(str, Field(..., description="30–50 words. Relational dynamics consistent with narrative, ethnic background, and marital status.")),
            presenting_problems=(List[str], Field(..., description="3–6 mental-health problem phrases describing THE OFFICER. Do not copy the examples. Include at least two problems not tied to their police work or to archetype.")),
            thought_content=(str, Field(..., description="25–45 words. What tends to occupy the person's mind, drawn from the narrative; natural phrasing.")),
            insight_judgment=(str, Field(..., description="25–45 words. Practical decision-making and self-understanding suggested by the scene.")),
            cognition=(str, Field(..., description="25–45 words. Observable thinking/recall/problem-solving implied by the narrative.")),
            emotional_behavioral_functioning=(str, Field(..., description="35–55 words. How the person handles pressure and difficult feelings; show behavior, avoid labels.")),
            social_functioning=(str, Field(..., description="35–55 words. Patterns in closeness, trust, and participation with others; concrete cues.")),
            summary_of_psychological_profile=(str, Field(..., description="75–105 words. Integrative summary using the narrative + histories + functioning + problems, framed by the archetype description.")),
        )

        system_msg = _PERSONA_SYSTEM_MSG

        app_ref_line = f"Original appearance (for style reference only — do NOT copy): {appearance_ref}" if appearance_ref else ""
        beh_ref_line = f"Original behavior (for style reference only — do NOT copy): {behavior_ref}" if behavior_ref else ""

        user_msg = (
            f"Selected archetype: {archetype_name}\n"
            f"Archetype description (guidance only — DO NOT copy or paraphrase): {archetype_desc}\n"
            f"Selected memoir: {memoir_title}\n"
            f"Memoir summary (guidance only — DO NOT copy or paraphrase): {memoir_summary}\n"
            f"Demographics (USE EXACT VALUES): name={dem.name}; age={dem.age}; location={dem.location}\n"
            f"Education level: education_level={dem.education_level}\n\n"
            f"Appearance category: {appearance_cat}\n{app_ref_line}\n\n"
            f"Behavior category: {behavior_cat}\n{beh_ref_line}\n\n"
            "Instructions:\n"
            "• First, craft the memoir_narrative (180–250 words) in style and setting of selected memoir as a vivid scene, but alter as needed to be consistent with specified demographics.\n"
            "• Then write every other field so it is consistent with that narrative and the exact demographics.\n"
            "• If narrative and demographics conflict when filling a field, prefer the specified demographics"
            "• Do not mention 'archetype', 'memoir', or 'summary' in the prose; the writing must stand alone.\n"
            "• Include `presenting_problems` as 3–6 concise items consistent with the psychological profile."
        )

        tool = {
            "name": "generate_persona",
            "description": "Generate a structured LEO persona matching the provided schema exactly.",
            "input_schema": SeededPersonaSchema.model_json_schema(),
        }

        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_msg}],
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "generate_persona"},
                    temperature=self.temperature,
                )
                tool_use_block = next(b for b in resp.content if b.type == "tool_use")
                out = SeededPersonaSchema(**tool_use_block.input)
                out.archetype_description = archetype_desc
                out.memoir_summary = memoir_summary
                d = out.model_dump()
                d["uuid"] = uuid
                persona_string = self.persona_row_to_string(d)
                d["persona_string"] = persona_string
                d["persona_hash"] = stable_hash(persona_string)
                return d
            except Exception as e:
                last_err = e
                time.sleep((0.5 * (2 ** attempt)) + random.random() * 0.25)

        print(f"[warn] generation skipped for uuid={uuid}: {last_err}")
        with open("logs.txt", "a") as f:
            f.write(f"[warn] generation skipped for uuid={uuid}: {last_err}")
        return None

def _load_seeds(source: str, hf_token: Optional[str] = None) -> List[dict]:
    """
    Load existing personas from a JSONL file or a HuggingFace dataset path
    (format: 'owner/repo' or 'owner/repo:config:split').
    Returns a list of persona dicts.
    """
    p = Path(source)
    if p.exists() and p.suffix == ".jsonl":
        records = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} seeds from {source}")
        return records

    # HuggingFace dataset: 'owner/repo', 'owner/repo:config', or 'owner/repo:config:split'
    parts = source.split(":")
    repo_id = parts[0]
    config = parts[1] if len(parts) > 1 else None
    split = parts[2] if len(parts) > 2 else "train"
    ds = load_dataset(repo_id, name=config, split=split, token=hf_token)
    records = [dict(row) for row in ds]
    print(f"Loaded {len(records)} seeds from HF dataset {source}")
    return records


def run_batch_from_seeds(
    source: str,
    populated_seeds_yaml: str,
    balanced_officers_csv: str,
    out_jsonl: str,
    version: str,
    workers: int = 10,
    model: str = "claude-sonnet-4-6",
    temperature: float = 1.0,
    top_p: float = 0.98,
    api_key: Optional[str] = None,
    base_seed: int = 1337,
    hf_token: Optional[str] = None,
    hf_repo_id: str = "thoughtworks/psychometric_personas_temp",
    hf_private: bool = True,
    push_to_hub_flag: bool = True,
):
    """
    Re-generate personas from an existing set, pinning all seed fields exactly.
    `source` is a JSONL file path or HF dataset string ('owner/repo:config:split').
    Already-generated UUIDs found in `out_jsonl` are skipped.
    """
    seeds = _load_seeds(source, hf_token=hf_token)

    existing_keys = _load_existing_version_keys(version)
    seeds = [s for s in seeds if str(s["uuid"]) not in existing_keys]
    print(f"Scheduling {len(seeds)} personas (skipped {len(_load_seeds(source, hf_token)) - len(seeds)} already done).")

    if not seeds:
        print("Nothing to generate.")
        return

    embed_helper = PersonaGenerator(
        populated_seeds_yaml=populated_seeds_yaml,
        balanced_officers_csv=balanced_officers_csv,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        rng_seed=base_seed,
        version=version,
    )

    records: List[dict] = []
    skipped = 0

    def _seed_worker(seed: dict) -> Optional[dict]:
        try:
            gen = PersonaGenerator(
                populated_seeds_yaml=populated_seeds_yaml,
                balanced_officers_csv=balanced_officers_csv,
                model=model,
                temperature=temperature,
                top_p=top_p,
                api_key=api_key,
                rng_seed=base_seed,
                version=version,
            )
            return gen.generate_one_from_seed(seed)
        except Exception:
            with open("logs.txt", "a") as f_out:
                f_out.write(traceback.format_exc())
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex, open(out_jsonl, "a", encoding="utf-8") as f:
        futures = {ex.submit(_seed_worker, seed): seed for seed in seeds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Personas"):
            rec = fut.result()
            if not rec:
                skipped += 1
                continue
            rec["version"] = version
            concat_text, concat_vec = embed_helper.build_concat_and_embedding(rec)
            rec["concat_field"] = concat_text
            rec["concat_embedding"] = concat_vec
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} personas to {out_jsonl} (skipped {skipped}).")

    if push_to_hub_flag and records:
        push_personas_to_hub(
            records=records,
            repo_id=hf_repo_id,
            hf_token=hf_token,
            private=hf_private,
            commit_message=f"append personas from seeds ({version})",
        )
        print("Push complete.")


def _worker_one(
    i: int,
    populated_seeds_yaml: str,
    balanced_officers_csv: str,
    model: str,
    temperature: float,
    top_p: float,
    api_key: Optional[str],
    base_seed: int,
) -> Optional[dict]:
    try:
        gen = PersonaGenerator(
            populated_seeds_yaml=populated_seeds_yaml,
            balanced_officers_csv=balanced_officers_csv,
            model=model,
            temperature=temperature,
            top_p=top_p,
            api_key=api_key,
            rng_seed=base_seed,
        )
        m = gen.generate_one(i)
        return m
    except Exception as e:
        print(f"[warn] worker skipped idx={i}: {e}")
        return None

def push_personas_to_hub(
    records: List[dict],
    repo_id: str,
    hf_token: Optional[str],
    private: bool,
    commit_message: str,
):
    if hf_token:
        HfApi().set_access_token(hf_token)

    # compute uids and merge with existing
    new_records = []
    for r in records:
        r = dict(r)
        new_records.append(r)

    try:
        ds_existing = load_dataset(repo_id, split="train", token=hf_token)
        df_existing = ds_existing.to_pandas()
    except Exception:
        print("Existing dataset not found, pushing new instead.")
        df_existing = pd.DataFrame(columns=list(new_records[0].keys()) if new_records else ["uuid"])

    df_new = pd.DataFrame(new_records)
    if "temp" in repo_id.lower():
        df_merged = df_new
    else:
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)

    df_merged = df_merged.drop_duplicates(subset=["uuid"]).reset_index(drop=True)

    ds_merged = Dataset.from_pandas(df_merged, preserve_index=False)
    ds_merged.push_to_hub(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        commit_message=commit_message,
    )

def run_batch(
    populated_seeds_yaml: str,
    balanced_officers_csv: str,
    count: int,
    out_jsonl: str,
    version: str,
    workers: int = 10,
    model: str = "claude-sonnet-4-6",
    temperature: float = 2.0,
    top_p: float = 0.98,
    api_key: Optional[str] = None,
    base_seed: int = 1337,
    # HF settings
    hf_repo_id: str = "thoughtworks/psychometric_personas_temp",
    hf_token: Optional[str] = None,
    hf_private: bool = True,
    push_to_hub_flag: bool = True,
):
    """
    Concurrent generation (~threads). Before scheduling, checks ./{version}.jsonl
    for already-generated rows (by CSV demographics key) and skips them.
    """
    # helper used for: offsets/df, building concat/embedding later
    embed_helper = PersonaGenerator(
        populated_seeds_yaml=populated_seeds_yaml,
        balanced_officers_csv=balanced_officers_csv,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        rng_seed=base_seed,
        version=version,
    )

    # 1) Load existing keys from ./{version}.jsonl
    existing_keys = _load_existing_version_keys(version)
    print(f"Found {len(existing_keys)} existing keys")

    # 2) Plan which indices to schedule (skip if CSV row already present)
    planned_indices: List[int] = []
    pre_skipped = 0

    for i in tqdm(range(count)):
        try:
            row = embed_helper.get_row(i)
            key = row["uuid"]
        except Exception as e:
            print(f"Error {e} on {i}")
        if key in existing_keys:
            pre_skipped += 1
            continue
        planned_indices.append(i)
        # guard against scheduling same row twice within the same run
        existing_keys.add(key)

    if not planned_indices:
        print(f"No new rows to generate; all {count} planned items were already present in ./{version}.jsonl")
        return

    print(f"Pre-skip (already in ./{version}.jsonl): {pre_skipped}")
    print(f"Scheduling {len(planned_indices)} new generations out of requested {count}.")

    records: List[dict] = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as ex, open(out_jsonl, "a", encoding="utf-8") as f:
        futures = []
        for i in planned_indices:
            fut = ex.submit(
                _thread_worker,
                i,
                populated_seeds_yaml,
                balanced_officers_csv,
                model,
                temperature,
                top_p,
                api_key,
                base_seed,
                version,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            rec = fut.result()
            if not rec:
                print(f"Received error")
                skipped += 1
                continue

            rec["version"] = version

            # concat text + embedding from generated fields only
            concat_text, concat_vec = embed_helper.build_concat_and_embedding(rec)
            rec["concat_field"] = concat_text
            rec["concat_embedding"] = concat_vec

            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} personas to {out_jsonl} (skipped {skipped}, pre-skipped {pre_skipped}).")

    if push_to_hub_flag and records:
        print(f"Pushing to HF dataset: {hf_repo_id} (append + dedupe + overwrite)")
        push_personas_to_hub(
            records=records,
            repo_id=hf_repo_id,
            hf_token=hf_token,
            private=hf_private,
            commit_message=f"append personas ({version}, deduped)",
        )
        print("Push complete.")

def _thread_worker(
    i: int,
    populated_seeds_yaml: str,
    balanced_officers_csv: str,
    model: str,
    temperature: float,
    top_p: float,
    api_key: Optional[str],
    base_seed: int,
    version: str,
) -> Optional[dict]:
    """
    Thread worker: constructs its own PersonaGenerator (client per thread),
    generates ONE persona for index i, and returns a plain dict (or None on skip).
    """
    try:
        gen = PersonaGenerator(
            populated_seeds_yaml=populated_seeds_yaml,
            balanced_officers_csv=balanced_officers_csv,
            model=model,
            temperature=temperature,
            top_p=top_p,
            api_key=api_key,
            rng_seed=base_seed,
            version=version,
        )
        m = gen.generate_one(i)  # may return None (skip)
        return m
    except Exception:
        with open("logs.txt", "a") as f_out:
            f_out.write(traceback.format_exc())
        return None


if __name__ == "__main__":
    import argparse, os

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[1]

    parser = argparse.ArgumentParser(description="Generate LEO personas locally via Anthropic Claude.")
    parser.add_argument("--n", type=int, default=2, help="Number of personas to generate (ignored when --from-personas is set)")
    parser.add_argument("--seeds-yaml", default=str(ROOT / "configs" / "populated_police_seeds.yaml"))
    parser.add_argument("--officers-csv", default=str(ROOT / "data" / "demographics" / "balanced_us_police_officers.csv"))
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--version", default="local_test")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument(
        "--from-personas",
        default=None,
        help="JSONL file or HF dataset ('owner/repo:config:split') of existing personas to re-generate from their seeds.",
    )
    args = parser.parse_args()

    gen = PersonaGenerator(
        populated_seeds_yaml=args.seeds_yaml,
        balanced_officers_csv=args.officers_csv,
        model=args.model,
        version=args.version,
        api_key=args.api_key,
    )

    if args.from_personas:
        seeds = _load_seeds(args.from_personas)
        for i, seed in enumerate(seeds):
            print(f"\n{'='*60}\nPersona {i+1} (uuid={seed['uuid']})\n{'='*60}")
            persona = gen.generate_one_from_seed(seed)
            if persona is None:
                print("Skipped (error)")
            else:
                concat_text, concat_vec = gen.build_concat_and_embedding(persona)
                persona["concat_field"] = concat_text
                persona["concat_embedding"] = concat_vec
                print(json.dumps(persona, indent=2, ensure_ascii=False))
    else:
        for i in range(args.n):
            print(f"\n{'='*60}\nPersona {i+1}\n{'='*60}")
            persona = gen.generate_one(i)
            if persona is None:
                print("Skipped (duplicate or error)")
            else:
                rec = persona if isinstance(persona, dict) else persona.model_dump()
                concat_text, concat_vec = gen.build_concat_and_embedding(rec)
                rec["concat_field"] = concat_text
                rec["concat_embedding"] = concat_vec
                print(json.dumps(rec, indent=2, ensure_ascii=False))
