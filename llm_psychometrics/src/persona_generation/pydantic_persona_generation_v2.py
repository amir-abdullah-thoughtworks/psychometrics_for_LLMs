from __future__ import annotations

import os
import json
import time
import random
from pathlib import Path
from typing import List, Optional, Literal, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from datasets import Dataset, load_dataset
from utils.hf_utils import dedupe_persona_records, push_personas_to_hub
from sentence_transformers import SentenceTransformer


# =========================
# Persona Generator (Class)
# =========================
class PersonaGenerator:
    """
    Generates ONE persona via OpenAI Structured Outputs with a Pydantic schema
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
        "memoir_narrative",
        "appearance",
        "behavior",
        "mood_affect",
        "speech",
        "thought_content",
        "insight_judgment",
        "cognition",
        "medical_developmental_history",
        "family_history",
        "educational_vocational_history",
        "emotional_behavioral_functioning",
        "social_functioning",
        "summary_of_psychological_profile",
    ]

    def __init__(
        self,
        populated_seeds_yaml: str,
        balanced_officers_csv: str,
        model: str = "gpt-4.1-mini",
        temperature: float = 2.0,
        top_p: float = 0.98,
        api_key: Optional[str] = None,
        rng_seed: Optional[int] = None,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.base_seed = rng_seed if rng_seed is not None else 1337
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
        required = {"sex", "age", "city", "state", "first_name", "last_name"}
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

    # -------- index-based pickers (no shared state) --------
    def _pick_demographics_by_index(self, idx: int) -> Tuple[str, int, str]:
        row = self.df.iloc[idx % len(self.df)]
        name = f"{str(row['first_name']).strip()} {str(row['last_name']).strip()}"
        age = int(row["age"])
        location = f"{str(row['city']).strip()}, {str(row['state']).strip()}"
        return name, age, location

    def _pick_archetype_by_index(self, idx: int) -> Tuple[str, str]:
        a = self.archetypes[idx % len(self.archetypes)]
        return a["name"], self._compose_archetype_description(a)

    def _pick_memoir_by_index(self, idx: int) -> Tuple[str, str]:
        title = self.memoir_titles[idx % len(self.memoir_titles)]
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

    # -------- main generation (with retries) --------
    def generate_one(self, idx: int) -> BaseModel:
        archetype_name, archetype_desc = self._pick_archetype_by_index(idx)
        memoir_title, memoir_summary = self._pick_memoir_by_index(idx)
        name, age, location = self._pick_demographics_by_index(idx)
        appearance_cat, appearance_examples = self._pick_appearance_random(idx)
        behavior_cat, behavior_examples = self._pick_behavior_random(idx)

        # dynamic schema with Literals of selected values (model sees exact values)
        SeededPersonaSchema = create_model(  # type: ignore[assignment]
            "SeededPersonaSchema",
            # constrained (exact) fields
            version=(Literal["v0"], ...),
            archetype=(Literal[archetype_name], ...),
            archetype_description=(Literal[archetype_desc], ...),
            memoir=(Literal[memoir_title], ...),
            memoir_summary=(Literal[memoir_summary], ...),
            name=(Literal[name], ...),
            age=(Literal[age], ...),
            location=(Literal[location], ...),
            appearance_category=(Literal[appearance_cat], ...),
            behavior_category=(Literal[behavior_cat], ...),

            # model-generated fields
            memoir_narrative=(str, Field(
                ...,
                description="~30-word snippet in the selected memoir’s style; consistent with archetype and demographics."
            )),
            presenting_problems=(List[str], Field(..., description="3–6 concise items.")),

            appearance=(str, Field(..., description="Observational, sensory description of appearance (10–20 words).")),
            behavior=(str, Field(..., description="Behavioral cues, posture, interaction style, responsiveness (10–20 words).")),
            mood_affect=(str, Field(..., description="Mood/affect, tone modulation, emotional nuance (10–20 words).")),
            speech=(str, Field(..., description="Speech register, rhythm, formality, coherence (10–20 words).")),
            thought_content=(str, Field(..., description="Internal reflection, logic, obsessions, themes (10–20 words).")),
            insight_judgment=(str, Field(..., description="Clinical phrasing of insight and judgment (10–20 words).")),
            cognition=(str, Field(..., description="Memory, abstraction, coherence, cognitive style (10–20 words).")),

            medical_developmental_history=(str, Field(..., description="Medical/developmental history, chronic/stress-related issues (30–50 words).")),
            family_history=(str, Field(..., description="Family history, generational details, substance patterns, relational dynamics (30–50 words).")),
            educational_vocational_history=(str, Field(..., description="Education, job trajectory, training, affiliations (30–50 words).")),

            emotional_behavioral_functioning=(str, Field(..., description="Stress, trauma, anger processing, coping mechanisms (30–50 words).")),
            social_functioning=(str, Field(..., description="Relationship style, trust, group affiliation, isolation/connection (30–50 words).")),

            summary_of_psychological_profile=(str, Field(..., description="Integrative clinical summary: diagnostic impressions, resilience, risks, prognosis (50–80 words).")),
        )

        def fmt_examples(title: str, items: List[str]) -> str:
            return f"{title} examples:\n" + ("\n".join(f"- {ex}" for ex in items) if items else "(none)")

        system_msg = (
            "You are an expert clinical interviewer and psychological profiler. "
            "Generate a detailed, realistic persona strictly adhering to the schema and its length guidance. "
            "Avoid caricature or stereotypes; include subtle contradictions for realism. Avoid repetition. "
            "Return valid structured output matching the provided schema exactly."
        )
        user_msg = (
            f"Selected archetype: {archetype_name}\n"
            f"Archetype description: {archetype_desc}\n"
            f"Selected memoir: {memoir_title}\n"
            f"Memoir summary: {memoir_summary}\n"
            f"Demographics (USE EXACT VALUES): name={name}; age={age}; location={location}\n\n"
            f"Appearance category: {appearance_cat}\n{fmt_examples('Appearance', appearance_examples)}\n\n"
            f"Behavior category: {behavior_cat}\n{fmt_examples('Behavior', behavior_examples)}\n\n"
            "Include `presenting_problems` as 3–6 concise items consistent with the profile."
        )

        # retry with jitter for transient errors
        last_err = None
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
                    text_format=SeededPersonaSchema,
                )
                return resp.output_parsed
            except Exception as e:
                last_err = e
                time.sleep((0.5 * (2 ** attempt)) + random.random() * 0.25)
        raise RuntimeError(f"Generation failed after retries: {last_err}")

# =========================
# Multiprocess Runner
# =========================
def _worker(args) -> dict:
    idx, paths, model, temp, top_p, api_key, seed = args
    gen = PersonaGenerator(
        populated_seeds_yaml=paths["yaml"],
        balanced_officers_csv=paths["csv"],
        model=model,
        temperature=temp,
        top_p=top_p,
        api_key=api_key,
        rng_seed=seed,
    )
    persona = gen.generate_one(idx)
    rec = persona.model_dump()
    rec.setdefault("version", "v0")
    # Single embedding over concatenated generated fields:
    rec["generated_text_embedding"] = gen.embed_generated_fields(rec)
    return rec


def run_batch(
    populated_seeds_yaml: str,
    balanced_officers_csv: str,
    count: int,
    out_jsonl: str,
    workers: int = 4,
    model: str = "gpt-4.1-mini",
    temperature: float = 2.0,
    top_p: float = 0.98,
    api_key: Optional[str] = None,
    base_seed: int = 1337,
    # HF settings
    hf_repo_id: str = "thoughtworks/psychometric_personas",
    hf_token: Optional[str] = None,
    hf_private: bool = False,
    push_to_hub_flag: bool = True,
):
    """
    Generate `count` personas across `workers` processes, write JSONL locally,
    then append+dedupe+overwrite to HF dataset.
    """
    paths = {"yaml": populated_seeds_yaml, "csv": balanced_officers_csv}
    records: List[dict] = []
    tasks = []

    with ProcessPoolExecutor(max_workers=workers) as ex, open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(count):
            args = (i, paths, model, temperature, top_p, api_key, base_seed)
            tasks.append(ex.submit(_worker, args))

        for fut in as_completed(tasks):
            rec = fut.result()
            rec.setdefault("version", "v0")
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} personas to {out_jsonl}")

    if push_to_hub_flag:
        print(f"Pushing to HF dataset: {hf_repo_id} (append + dedupe + overwrite)")
        push_personas_to_hub(
            records=records,
            repo_id=hf_repo_id,
            hf_token=hf_token,
            private=hf_private,
            commit_message="append personas (v0, deduped)",
        )
        print("Push complete.")


# =========================
# Example usage (commented)
# =========================
# if __name__ == "__main__":
#     run_batch(
#         populated_seeds_yaml="populated_police_seeds.yaml",
#         balanced_officers_csv="balanced_us_police_officers.csv",
#         count=100,
#         out_jsonl="personas.jsonl",
#         workers=4,  # start here; try 6–8 if rate limits allow
#         model="gpt-4.1-mini",
#         temperature=2.0,
#         top_p=0.98,
#         api_key=os.getenv("OPENAI_API_KEY"),
#         hf_repo_id="thoughtworks/psychometric_personas",
#         hf_token=os.getenv("HUGGINGFACE_TOKEN"),
#         hf_private=False,
#         push_to_hub_flag=True,
#     )
