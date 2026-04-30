from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
import yaml
from datasets import Dataset
from huggingface_hub import HfFolder
from openai import OpenAI
from pydantic import Field, create_model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ABLATION_CONFIGS = {
    "full",
    "no_attribute_injections",
    "no_memoir_grounding",
    "no_demographic_grounding",
    "no_archetype_grounding",
}


def _norm(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_existing_version_keys(version: str) -> Set[str]:
    """
    Load ./{version}.jsonl if present and return a set of UUID-like keys
    used to prevent duplicate generation within the same ablation/version run.
    """
    p = Path(f"{version}.jsonl")
    if not p.exists():
        return set()

    keys: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "uuid" in obj:
                    keys.add(str(obj["uuid"]))
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


class PersonaGenerator:
    """
    Generates one persona via OpenAI Structured Outputs with a Pydantic schema.

    Supports five ablation modes:
      - full
      - no_attribute_injections
      - no_memoir_grounding
      - no_demographic_grounding
      - no_archetype_grounding

    Intended usage:
      - Call this script once per ablation config from a wrapper script.
      - Each run pushes to a separate Hugging Face dataset config.
    """

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
        ablation_config: str = "full",
        model: str = "gpt-4.1-mini",
        temperature: float = 2.0,
        top_p: float = 0.98,
        api_key: Optional[str] = None,
        rng_seed: Optional[int] = None,
    ):
        if ablation_config not in ABLATION_CONFIGS:
            raise ValueError(
                f"Invalid ablation_config={ablation_config}. "
                f"Must be one of {sorted(ABLATION_CONFIGS)}"
            )

        self.version = version
        self.ablation_config = ablation_config
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.base_seed = rng_seed if rng_seed is not None else 1337
        self._rng = random.Random(self.base_seed)
        self._embedder: Optional[SentenceTransformer] = None

        with open(populated_seeds_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        root = data.get("PoliceOfficerPersonaSeeds", data) or {}

        raw_archetypes = root.get("archetypes", [])
        if not isinstance(raw_archetypes, list) or not raw_archetypes:
            raise ValueError("No archetypes found in populated_police_seeds.yaml")
        self.archetypes: List[dict] = []
        for a in raw_archetypes:
            self.archetypes.append(a if isinstance(a, dict) else {"name": str(a)})

        self.memoir_titles: List[str] = root.get("MemoirSeeds", []) or []
        self.memoir_summaries: Dict[str, str] = root.get("MemoirSummaries", {}) or {}
        if not self.memoir_titles:
            raise ValueError("No MemoirSeeds found in populated_police_seeds.yaml")
        missing = [m for m in self.memoir_titles if m not in self.memoir_summaries]
        if missing:
            raise ValueError(f"Missing MemoirSummaries for: {missing}")

        self.appearance_categories: Dict[str, List[str]] = root.get("AppearanceCategories") or {}
        self.behavior_categories: Dict[str, List[str]] = root.get("BehaviorCategories") or {}
        if not self.appearance_categories or not self.behavior_categories:
            raise ValueError("AppearanceCategories/BehaviorCategories not found or empty.")

        self.df = pd.read_csv(balanced_officers_csv)

        self._arch_offset = self._rng.randrange(len(self.archetypes))
        self._mem_offset = self._rng.randrange(len(self.memoir_titles))

        required = {
            "uuid",
            "sex",
            "age",
            "city",
            "state",
            "first_name",
            "last_name",
            "education_level",
            "bachelors_field",
            "ethnic_background",
            "marital_status",
        }
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

    def _use_attribute_injections(self) -> bool:
        return self.ablation_config != "no_attribute_injections"

    def _use_memoir_grounding(self) -> bool:
        return self.ablation_config != "no_memoir_grounding"

    def _use_demographic_grounding(self) -> bool:
        return self.ablation_config != "no_demographic_grounding"

    def _use_archetype_grounding(self) -> bool:
        return self.ablation_config != "no_archetype_grounding"

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        return self._embedder

    def build_concat_and_embedding(self, rec: dict) -> tuple[str, list[float]]:
        lines: list[str] = []
        for key in self._GENERATED_TEXT_FIELDS:
            if key not in rec:
                continue
            val = rec[key]
            if val is None:
                continue
            val_str = str(val).strip()
            if val_str:
                lines.append(f"{key}: {val_str}")

        probs = rec.get("presenting_problems")
        if isinstance(probs, list):
            probs = [str(x).strip() for x in probs if str(x).strip()]
            if probs:
                lines.append(f"presenting_problems: {'; '.join(probs)}")

        concat_text = "\n".join(lines) + ("\n" if lines else "")
        if not concat_text.strip():
            return "", []

        try:
            emb_model = self._get_embedder()
            vec = emb_model.encode([concat_text], convert_to_numpy=False)[0]
            return concat_text, list(map(float, vec))
        except Exception:
            return concat_text, []

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
        return " ".join(parts).strip() or f"Archetype: {a.get('name', '(unspecified)')}."

    def get_row(self, idx: int):
        return self.df.iloc[idx % len(self.df)]

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

    @staticmethod
    def persona_row_to_string(row: dict) -> str:
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
            return " ".join(str(v).strip().split())

        lines: List[str] = []
        for k in preferred_order:
            if k in row and row[k] is not None and str(row[k]).strip():
                lines.append(f"{pretty_label(k)}: {clean_value(row[k])}")
            if k == "mood_affect":
                probs = row.get("presenting_problems") or []
                if isinstance(probs, list) and probs:
                    lines.append("presenting_problems:")
                    lines.extend([f"- {clean_value(p)}" for p in probs if str(p).strip()])

        return "\n".join(lines)

    def generate_one(self, idx: int) -> Optional[dict]:
        row = self.get_row(idx)
        uuid = str(row["uuid"])

        dem = self._pick_demographics_by_index(idx)
        archetype_name, archetype_desc = self._pick_archetype_by_index(idx)
        memoir_title, memoir_summary = self._pick_memoir_by_index(idx)
        appearance_cat, appearance_examples = self._pick_appearance_random(idx)
        behavior_cat, behavior_examples = self._pick_behavior_random(idx)

        if not self._use_demographic_grounding():
            dem = Demographics(
                name="Jordan Blake",
                age=39,
                sex="Male",
                location="Columbus, Ohio",
                education_level="Bachelor's degree",
                bachelors_field="Criminal Justice",
                ethnic_background="White",
                marital_status="Married",
            )

        if not self._use_archetype_grounding():
            archetype_name = "None"
            archetype_desc = "No archetype guidance provided."

        if not self._use_memoir_grounding():
            memoir_title = "None"
            memoir_summary = "No memoir grounding provided."

        if not self._use_attribute_injections():
            appearance_cat = "None"
            appearance_examples = []
            behavior_cat = "None"
            behavior_examples = []

        schema_name = f"SeededPersonaSchema_{self.ablation_config}_{idx}"

        SeededPersonaSchema = create_model(
            schema_name,
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
            memoir_summary=(str, Field(..., description="Summary of the selected memoir.")),
            memoir_narrative=(
                str,
                Field(
                    ...,
                    description=(
                        "Vivid, scene-level narrative (180–250 words) in the selected memoir’s milieu. "
                        "This is the canonical grounding unless memoir grounding is ablated."
                    ),
                ),
            ),
            archetype=(Literal[archetype_name], ...),
            archetype_description=(
                str,
                Field(
                    ...,
                    description="Concise description of the archetype as provided.",
                ),
            ),
            appearance=(str, Field(..., description="Observational, sensory description of appearance (10–30 words).")),
            behavior=(str, Field(..., description="Behavioral cues, posture, interaction style, responsiveness (10–30 words).")),
            speech=(str, Field(..., description="Speech register, rhythm, formality, coherence (10–30 words).")),
            mood_affect=(str, Field(..., description="Mood/affect, tone modulation, emotional nuance (10–30 words).")),
            educational_vocational_history=(
                str,
                Field(..., description="30–50 words. Align with education_level and bachelors_field."),
            ),
            medical_developmental_history=(
                str,
                Field(..., description="30–50 words. Health/development context relevant to the scene."),
            ),
            family_history=(
                str,
                Field(..., description="30–50 words. Relational dynamics consistent with the broader persona."),
            ),
            presenting_problems=(
                List[str],
                Field(
                    ...,
                    description=(
                        "3–6 mental-health problem phrases describing the officer. "
                        "Include at least two problems not tied directly to police work or archetype."
                    ),
                ),
            ),
            thought_content=(str, Field(..., description="25–45 words. What tends to occupy the person’s mind.")),
            insight_judgment=(str, Field(..., description="25–45 words. Practical decision-making and self-understanding.")),
            cognition=(str, Field(..., description="25–45 words. Observable thinking/recall/problem-solving.")),
            emotional_behavioral_functioning=(
                str,
                Field(..., description="35–55 words. How the person handles pressure and difficult feelings."),
            ),
            social_functioning=(
                str,
                Field(..., description="35–55 words. Patterns in closeness, trust, and participation with others."),
            ),
            summary_of_psychological_profile=(
                str,
                Field(..., description="75–105 words. Integrative summary of the psychological profile."),
            ),
        )

        def fmt_examples(title: str, items: List[str]) -> str:
            return f"{title} examples:\n" + ("\n".join(f"- {ex}" for ex in items) if items else "(none)")

        system_msg = (
            "You are an expert clinical interviewer and psychological profiler.\n"
            "Generate a detailed, realistic persona strictly adhering to the schema and length guidance.\n"
            "Avoid caricature or stereotypes; allow subtle contradictions for realism; avoid repetition.\n"
            "Return valid structured output matching the provided schema exactly.\n"
            "\n"
            "STYLE & GROUNDING (do NOT output this list):\n"
            "1) Write memoir_narrative as a concrete, sensory, scene-level story (180–250 words).\n"
            "2) If memoir grounding is provided, align other fields to that narrative unless demographics force a correction.\n"
            "3) If archetype grounding is provided, treat it as a loose orientation rather than a template to copy.\n"
            "4) Do not reuse ≥5 consecutive words from the prompt. Rephrase and localize details.\n"
            "5) Favor specificity over generic traits; vary wording across sections.\n"
            "6) Keep the persona internally consistent."
        )

        user_parts: List[str] = [f"Ablation config: {self.ablation_config}"]

        if self._use_archetype_grounding():
            user_parts.append(
                f"Selected archetype: {archetype_name}\n"
                f"Archetype description (guidance only — DO NOT copy or paraphrase): {archetype_desc}"
            )
        else:
            user_parts.append("No archetype grounding is provided for this run.")

        if self._use_memoir_grounding():
            user_parts.append(
                f"Selected memoir: {memoir_title}\n"
                f"Memoir summary (guidance only — DO NOT copy or paraphrase): {memoir_summary}"
            )
        else:
            user_parts.append("No memoir grounding is provided for this run.")

        if self._use_demographic_grounding():
            user_parts.append(
                "Demographics (USE EXACT VALUES): "
                f"name={dem.name}; age={dem.age}; sex={dem.sex}; location={dem.location}; "
                f"education_level={dem.education_level}; bachelors_field={dem.bachelors_field}; "
                f"ethnic_background={dem.ethnic_background}; marital_status={dem.marital_status}"
            )
        else:
            user_parts.append(
                "Demographic grounding is ablated. Use the schema-constrained values as given, "
                "but do not treat them as an explicit grounding block."
            )

        if self._use_attribute_injections():
            user_parts.append(
                f"Appearance category: {appearance_cat}\n{fmt_examples('Appearance', appearance_examples)}\n\n"
                f"Behavior category: {behavior_cat}\n{fmt_examples('Behavior', behavior_examples)}"
            )
        else:
            user_parts.append("No appearance or behavior seed examples are provided for this run.")

        user_parts.append(
            "Instructions:\n"
            "• First, craft the memoir_narrative (180–250 words) as a vivid scene.\n"
            "• Then write every other field so it is consistent with that narrative and any provided grounding.\n"
            "• Do not mention 'archetype', 'memoir', or 'summary' in the prose.\n"
            "• Include `presenting_problems` as 3–6 concise items consistent with the psychological profile."
        )

        user_msg = "\n\n".join(user_parts)

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
                    text_format=SeededPersonaSchema,
                )
                out = resp.output_parsed

                out.archetype_description = archetype_desc
                out.memoir_summary = memoir_summary

                d = out.model_dump()
                d["uuid"] = uuid
                d["version"] = self.version
                d["ablation_config"] = self.ablation_config

                persona_string = self.persona_row_to_string(d)
                d["persona_string"] = persona_string
                d["persona_hash"] = stable_hash(persona_string)
                return d
            except Exception as e:
                last_err = e
                time.sleep((0.5 * (2 ** attempt)) + random.random() * 0.25)

        print(f"[warn] generation skipped for idx={idx}: {last_err}")
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"[warn] generation skipped for idx={idx}: {last_err}\n")
        return None


def push_personas_to_hub(
    records: List[dict],
    repo_id: str,
    config_name: str,
    hf_token: Optional[str],
    private: bool,
    commit_message: str,
) -> None:
    if hf_token:
        HfFolder.save_token(hf_token)

    df = pd.DataFrame(records)
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        split="train",
        token=hf_token,
        private=private,
        commit_message=commit_message,
    )


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
    ablation_config: str,
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
            version=version,
            ablation_config=ablation_config,
        )
        return gen.generate_one(i)
    except Exception:
        with open("logs.txt", "a", encoding="utf-8") as f_out:
            f_out.write(traceback.format_exc() + "\n")
        return None


def run_batch(
    populated_seeds_yaml: str,
    balanced_officers_csv: str,
    count: int,
    out_jsonl: str,
    version: str,
    ablation_config: str = "full",
    workers: int = 10,
    model: str = "gpt-4.1-mini",
    temperature: float = 2.0,
    top_p: float = 0.98,
    api_key: Optional[str] = None,
    base_seed: int = 1337,
    hf_repo_id: str = "thoughtworks/ablation_psychometrics_personas",
    hf_token: Optional[str] = None,
    hf_private: bool = True,
    push_to_hub_flag: bool = True,
) -> None:
    embed_helper = PersonaGenerator(
        populated_seeds_yaml=populated_seeds_yaml,
        balanced_officers_csv=balanced_officers_csv,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        rng_seed=base_seed,
        version=version,
        ablation_config=ablation_config,
    )

    existing_keys = _load_existing_version_keys(version)
    print(f"Found {len(existing_keys)} existing keys in {version}.jsonl")

    planned_indices: List[int] = []
    pre_skipped = 0

    for i in tqdm(range(count), desc="Planning"):
        try:
            row = embed_helper.get_row(i)
            key = f"{ablation_config}::{row['uuid']}"
        except Exception as e:
            print(f"Error while planning idx={i}: {e}")
            continue

        if key in existing_keys:
            pre_skipped += 1
            continue

        planned_indices.append(i)
        existing_keys.add(key)

    if not planned_indices:
        print(f"No new rows to generate for version={version}, ablation={ablation_config}.")
        return

    print(f"Pre-skipped: {pre_skipped}")
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
                ablation_config,
            )
            futures.append(fut)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            rec = fut.result()
            if not rec:
                skipped += 1
                continue

            concat_text, concat_vec = embed_helper.build_concat_and_embedding(rec)
            rec["concat_field"] = concat_text
            rec["concat_embedding"] = concat_vec

            records.append(rec)

            jsonl_rec = dict(rec)
            jsonl_rec["uuid"] = f"{ablation_config}::{rec['uuid']}"
            f.write(json.dumps(jsonl_rec, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(records)} personas to {out_jsonl} "
        f"(skipped {skipped}, pre-skipped {pre_skipped})."
    )

    if push_to_hub_flag and records:
        print(f"Pushing to HF dataset: {hf_repo_id}, config={ablation_config}")
        push_personas_to_hub(
            records=records,
            repo_id=hf_repo_id,
            config_name=ablation_config,
            hf_token=hf_token,
            private=hf_private,
            commit_message=f"append personas ({version}, {ablation_config})",
        )
        print("Push complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--populated-seeds-yaml", type=str, required=True)
    parser.add_argument("--balanced-officers-csv", type=str, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument(
        "--ablation-config",
        type=str,
        default="full",
        choices=sorted(ABLATION_CONFIGS),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-seed", type=int, default=1337)
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="thoughtworks/ablation_psychometrics_personas",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--no-push-to-hub", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    run_batch(
        populated_seeds_yaml=args.populated_seeds_yaml,
        balanced_officers_csv=args.balanced_officers_csv,
        count=args.count,
        out_jsonl=args.out_jsonl,
        version=args.version,
        ablation_config=args.ablation_config,
        workers=args.workers,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        api_key=api_key,
        base_seed=args.base_seed,
        hf_repo_id=args.hf_repo_id,
        hf_token=hf_token,
        hf_private=args.hf_private,
        push_to_hub_flag=not args.no_push_to_hub,
    )
    
