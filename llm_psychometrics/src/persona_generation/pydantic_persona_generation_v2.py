import random
from pathlib import Path
from typing import List, Optional, Literal

import pandas as pd
import yaml
from openai import OpenAI
from pydantic import BaseModel, Field


class PersonaGenerator:
    """
    Generates ONE persona via OpenAI Structured Outputs with a Pydantic schema
    whose constrained fields (name, age, location, archetype, memoir) are Literals
    of the specific values sampled for this run.

    Requirements:
      - populated_police_seeds.yaml with:
          PoliceOfficerPersonaSeeds:
            archetypes: [ {name: "...", ...}, ... ]  OR  ["...", ...]
            MemoirSeeds: ["...", ...]
            MemoirSummaries: { "<memoir title>": "<~20-word summary>", ... }
      - balanced_us_police_officers.csv with columns:
          sex, age, city, state, first_name, last_name
        (ages must be 21–70; rows outside are excluded from sampling)
    """

    def __init__(
        self,
        populated_seeds_yaml: str | Path,
        balanced_officers_csv: str | Path,
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
        self.rng = random.Random(rng_seed)

        # ---- load seeds ----
        with open(populated_seeds_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
        root = data.get("PoliceOfficerPersonaSeeds", data) or {}

        # archetypes -> list[str] of names
        raw_archetypes = root.get("archetypes", [])
        if not isinstance(raw_archetypes, list) or not raw_archetypes:
            raise ValueError("No archetypes found in populated_police_seeds.yaml")
        self.archetype_names: List[str] = [
            (a["name"] if isinstance(a, dict) and "name" in a else str(a)) for a in raw_archetypes
        ]

        # memoirs + summaries
        self.memoir_titles: List[str] = root.get("MemoirSeeds", []) or []
        self.memoir_summaries: dict = root.get("MemoirSummaries", {}) or {}
        if not self.memoir_titles:
            raise ValueError("No MemoirSeeds found in populated_police_seeds.yaml")

        # Ensure every memoir has a summary (no fallbacks)
        missing = [m for m in self.memoir_titles if m not in self.memoir_summaries]
        if missing:
            raise ValueError(f"Missing MemoirSummaries for: {missing}")

        # ---- load officers (STRICT columns, STRICT age range) ----
        self.df = pd.read_csv(balanced_officers_csv)
        required = {"sex", "age", "city", "state", "first_name", "last_name"}
        miss = required - set(self.df.columns)
        if miss:
            raise ValueError(f"Missing required columns in CSV: {sorted(miss)}")

        # drop rows with nulls in required cols
        self.df = self.df.dropna(subset=list(required)).reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No valid rows after enforcing required columns.")

        # enforce int age and filter to [21, 70]
        try:
            self.df["age"] = self.df["age"].astype(int)
        except Exception as e:
            raise ValueError("Column 'age' must be integers.") from e
        self.df = self.df[(self.df["age"] >= 21) & (self.df["age"] <= 70)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No rows with age in [21, 70].")

    def _pick_demographics(self) -> tuple[str, int, str]:
        row = self.df.sample(n=1, random_state=self.rng.randrange(1_000_000)).iloc[0]
        name = f"{str(row['first_name']).strip()} {str(row['last_name']).strip()}"
        age = int(row["age"])
        location = f"{str(row['city']).strip()}, {str(row['state']).strip()}"
        return name, age, location

    def _pick_archetype(self) -> str:
        return self.rng.choice(self.archetype_names)

    def _pick_memoir(self) -> tuple[str, str]:
        m = self.rng.choice(self.memoir_titles)
        return m, self.memoir_summaries[m]

    def generate_one(self) -> BaseModel:
        """
        Returns a Pydantic instance parsed from structured output.
        Constrained fields are Literals of the sampled values.
        """
        # ---- sample seeds ----
        archetype_name = self._pick_archetype()
        memoir_title, memoir_summary = self._pick_memoir()
        name, age, location = self._pick_demographics()

        # ---- define the constrained schema *after* selection ----
        class SeededPersonaSchema(BaseModel):
            # Constrained fields (Literals of sampled values)
            archetype: Literal[archetype_name]
            memoir: Literal[memoir_title]
            name: Literal[name]
            age: Literal[age]
            location: Literal[location]

            # Added fields
            memoir_summary: str
            presenting_problems: List[str]

            # Behavioral & Psychological Descriptors
            appearance: str = Field(..., description="Observational, sensory description of appearance (30–60 words).")
            behavior: str = Field(..., description="Behavioral cues, posture, interaction style, responsiveness (30–60 words).")
            mood_affect: str = Field(..., description="Mood/affect, tone modulation, emotional nuance (30–60 words).")
            speech: str = Field(..., description="Speech register, rhythm, formality, coherence (30–60 words).")
            thought_content: str = Field(..., description="Internal reflection, logic, obsessions, themes (30–60 words).")
            insight_judgment: str = Field(..., description="Clinical phrasing of insight and judgment (30–60 words).")
            cognition: str = Field(..., description="Memory, abstraction, coherence, cognitive style (30–60 words).")

            # Life History Segments
            medical_developmental_history: str = Field(..., description="Medical/developmental history, chronic/stress-related issues (100–150 words).")
            family_history: str = Field(..., description="Family history, generational details, substance patterns, relational dynamics (100–150 words).")
            educational_vocational_history: str = Field(..., description="Education, job trajectory, training, affiliations (100–150 words).")

            # Functional Assessments
            emotional_behavioral_functioning: str = Field(..., description="How persona processes stress, trauma, anger, coping mechanisms (100–150 words).")
            social_functioning: str = Field(..., description="Relationship style, trust, group affiliation, isolation/connection (100–150 words).")

            # Summary
            summary_of_psychological_profile: str = Field(..., description="Integrative clinical summary: diagnostic impressions, resilience, risks, prognosis (150–250 words).")

        # ---- prompt (no extra seeded_content; use memoir summary) ----
        system_msg = (
            "You are an expert clinical interviewer and psychological profiler. "
            "Generate a detailed, realistic persona strictly adhering to the schema and its length guidance. "
            "Avoid caricature or stereotypes; include subtle contradictions for realism. "
            "Return valid structured output matching the provided schema exactly."
        )
        user_msg = (
            f"Selected archetype: {archetype_name}\n"
            f"Selected memoir: {memoir_title}\n"
            f"Memoir summary: {memoir_summary}\n"
            f"Demographics (USE EXACT VALUES): name={name}; age={age}; location={location}\n\n"
            "Include `presenting_problems` as 3–6 concise items consistent with the profile."
        )

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


# Convenience function
def generate_persona(
    populated_seeds_yaml: str | Path = "populated_police_seeds.yaml",
    balanced_officers_csv: str | Path = "balanced_us_police_officers.csv",
    model: str = "gpt-4.1-mini",
    temperature: float = 2.0,
    top_p: float = 0.98,
    api_key: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> BaseModel:
    gen = PersonaGenerator(
        populated_seeds_yaml=populated_seeds_yaml,
        balanced_officers_csv=balanced_officers_csv,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        rng_seed=rng_seed,
    )
    return gen.generate_one()

# Example:
# p = generate_persona(
#     populated_seeds_yaml="/path/to/populated_police_seeds.yaml",
#     balanced_officers_csv="/path/to/balanced_us_police_officers.csv",
#     model="gpt-4.1-mini",
#     rng_seed=1337,
# )
# print(p.model_dump_json(indent=2, ensure_ascii=False))
