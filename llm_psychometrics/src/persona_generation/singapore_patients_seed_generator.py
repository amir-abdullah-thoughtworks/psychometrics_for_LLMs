#!/usr/bin/env python3
"""
Singapore patient persona seed generator (Digital Health Coach), full-fledged.

What this generates (YAML):
- Regions (Singapore planning areas / macro-regions) with sampling priors
- Occupations (top-level): {value, explanation, weight}
- Education (top-level): {value, explanation, weight}
- PrimaryLanguage, Ethnicity
- HealthProblems (common), HealthGoals, CoachingMotivations
- PreferredCommunicationStyle, AttitudesTowardsAuthority
- EmotionalTics, StressCoping, HealthLiteracy, TechComfort
- DietPattern, ActivityBaseline, SleepPattern
- FamilyContext, CareAccessPattern, CoachChannelPreference
- PersonaCore: Appearance/InteractionStyle/MediaDisposition expanded to 25 items each (OpenAI)
- AdditionalTraits: each subcategory expanded to >=10 items (OpenAI)
- HealthCoachingTopics: catalog of topics; each topic has bullet_points and >=4 stances per bullet
  (e.g., "Weight management" -> stances for calorie tracking vs portioning vs mindful eating, etc.)

Uses:
- OpenAI responses.parse with Pydantic structured outputs
- Semantic de-dup with embeddings + retry/fill to quota

Requires:
  pip install openai pydantic pyyaml

Defaults:
  model = gpt-4o
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import yaml
from pydantic import BaseModel, Field, conlist, model_validator
from openai import OpenAI


# ---------------------------
# Defaults / constants
# ---------------------------

DEFAULT_MODEL = "gpt-4o"
DEFAULT_OUT = "singapore_health_coach_patient_seeds.yaml"

EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------
# Pydantic output models (Structured Outputs via responses.parse)
# ---------------------------

class OptionItem(BaseModel):
    value: str = Field(..., description="Categorical label.")
    explanation: str = Field(..., description="1–2 sentences describing what it signals and how it manifests.")
    weight: Optional[float] = Field(None, description="Optional sampling weight/probability.")


class OptionsOut(BaseModel):
    items: List[OptionItem]


class TopicBulletStance(BaseModel):
    bullet: str = Field(..., description="Exact bullet point this stance addresses.")
    id: str = Field(..., description="snake_case id for the stance.")
    description: str = Field(..., description="1–2 sentence stance description, behaviorally grounded.")
    # Here we don’t do party distribution; we do population propensity by segment.
    # Segments are flexible buckets useful for sampling.
    segment_distribution: Dict[str, float] = Field(
        ...,
        description=(
            "Nonnegative weights across segments (keys must match provided segment list). "
            "Represents prevalence/propensity within each segment (not survey data)."
        ),
    )
    rationale: str = Field(..., description="1–3 sentence intuition for the segment_distribution.")


class CoachingTopicExpanded(BaseModel):
    name: str
    description: str = Field(..., description="One-paragraph description of the coaching topic.")
    bullet_points: conlist(str, min_length=4, max_length=12) = Field(
        ..., description="Key sub-issues for this topic; written as bullets."
    )
    stances: List[TopicBulletStance]

    @model_validator(mode="after")
    def _validate_stances_per_bullet(self):
        counts: Dict[str, int] = {b: 0 for b in self.bullet_points}
        for s in self.stances:
            if s.bullet in counts:
                counts[s.bullet] += 1
        missing = [b for b, c in counts.items() if c < 4]
        if missing:
            raise ValueError(f"Not enough stances for bullets: {missing}")
        return self


class CoachingTopicsOut(BaseModel):
    topics: conlist(CoachingTopicExpanded, min_length=15, max_length=15)


# ---------------------------
# Embedding + semantic de-dup
# ---------------------------

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

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _norm_key(s: str) -> str:
    return " ".join(s.strip().casefold().split())

def dedup_options_semantic(
    client: Optional[OpenAI],
    items: List[OptionItem],
    sim_threshold: float = 0.90,
) -> List[OptionItem]:
    """
    1) Exact de-dup by normalized value.
    2) Semantic de-dup by embedding (value + explanation).
    """
    seen = set()
    exact_kept: List[OptionItem] = []
    for it in items:
        k = _norm_key(it.value)
        if k not in seen:
            seen.add(k)
            exact_kept.append(it)

    if client is None or len(exact_kept) <= 1:
        return exact_kept

    texts = [f"{it.value} :: {it.explanation}" for it in exact_kept]
    vecs = embed_texts(client, texts)

    kept: List[OptionItem] = []
    kept_vecs: List[List[float]] = []
    for it, v in zip(exact_kept, vecs):
        if any(cosine(v, kv) >= sim_threshold for kv in kept_vecs):
            continue
        kept.append(it)
        kept_vecs.append(v)
    return kept


# ---------------------------
# Prompt helpers (novelty fill)
# ---------------------------

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
- Keep items Singapore / clinical / health-coaching plausible where relevant
- No profanity, no slurs, no real-person accusations
"""

def prompt_expand_category_fill(category_name: str, description: str, need_n: int, existing_values: List[str]) -> str:
    existing = "\n".join([f"- {v}" for v in existing_values])
    return f"""
We are expanding the persona seed category: {category_name}

Category purpose:
{description}

We already have these items (DO NOT repeat; do not paraphrase into near-synonyms):
{existing}

Task:
- Produce {need_n} NEW items that are clearly distinct from the above.
- Each item must include:
  - value
  - explanation (1–2 sentences)

Novelty constraints:
- Avoid near-duplicates in wording or meaning.
- Prefer concrete, observable cues (phrasing, habits, routines, reactions).
- Keep Singapore/health-coach plausible.
"""

def prompt_expand_trait_fill(subcategory: str, description: str, need_n: int, existing_values: List[str]) -> str:
    existing = "\n".join([f"- {v}" for v in existing_values])
    return f"""
We are expanding the AdditionalTraits subcategory: {subcategory}

Purpose:
{description}

We already have these items (DO NOT repeat; do not paraphrase into near-synonyms):
{existing}

Task:
- Produce {need_n} NEW items that are clearly distinct from the above.
- Each item must include:
  - value
  - explanation (1–2 sentences)

Novelty constraints:
- Make items meaningfully different (not just rewordings).
- Prefer traits that change coaching dynamics (adherence, disclosure, motivation).
"""


# ---------------------------
# OpenAI parse wrappers with retry/fill
# ---------------------------

def openai_expand_options_fill(
    client: OpenAI,
    model: str,
    base_prompt: str,
    min_items: int,
    *,
    sim_threshold: float = 0.90,
    max_rounds: int = 4,
    oversample_factor: float = 1.6,
    fill_prompt_fn=None,
    category_name: str = "",
    category_description: str = "",
) -> List[OptionItem]:
    if fill_prompt_fn is None:
        raise ValueError("fill_prompt_fn must be provided.")

    pool: List[OptionItem] = []

    def do_parse(prompt: str, target: int) -> List[OptionItem]:
        want = max(target, int(math.ceil(target * oversample_factor)))
        sys = "You generate structured persona seed options for Singapore health coaching. Return diverse, concrete, distinct items."
        user = prompt + f"\n\nReturn at least {want} items."
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.8,
            top_p=1.0,
            text_format=OptionsOut,
        )
        return resp.output_parsed.items

    # Round 0
    pool.extend(do_parse(base_prompt, min_items))
    pool = dedup_options_semantic(client, pool, sim_threshold=sim_threshold)

    round_i = 1
    while len(pool) < min_items and round_i <= max_rounds:
        need = min_items - len(pool)
        existing_vals = [it.value for it in pool]
        fill_prompt = fill_prompt_fn(category_name, category_description, need, existing_vals)
        pool.extend(do_parse(fill_prompt, need))
        pool = dedup_options_semantic(client, pool, sim_threshold=sim_threshold)
        round_i += 1
        time.sleep(0.2)

    return pool[:min_items]

def normalize_dist(d: Dict[str, float], keys: List[str], eps: float = 1e-9) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in d.items() if k in keys}
    s = sum(cleaned.values())
    if s < eps:
        # uniform over first few keys
        base = keys[: min(5, len(keys))]
        return {k: round(1.0 / len(base), 4) for k in base}
    return {k: round(v / s, 4) for k, v in cleaned.items()}

def prompt_expand_coaching_topics(
    seed_topics: List[Dict[str, Any]],
    segments: List[str],
) -> str:
    return f"""
You are expanding a catalog of 15 digital health coaching topics for Singapore patients.

For EACH topic:
- Provide a 1-paragraph description
- Provide 4–12 bullet_points (sub-issues). Each bullet must be specific.
- For EACH bullet point, generate AT LEAST 4 discrete stances.
  Each stance must include:
  - bullet (exact bullet text it addresses)
  - id (snake_case)
  - description (1–2 sentences; behaviorally grounded)
  - segment_distribution: nonnegative weights across these segments only:
    {segments}
  - rationale: 1–3 sentences justifying the segment_distribution

Important:
- These are plausible priors (not survey results). Spread probability when uncertain.
- Keep stances meaningfully different, not just rewordings.
- Make sure each bullet has at least 4 stances.

Seed topics (names and rough scope):
{seed_topics}
"""

def openai_expand_coaching_topics(
    client: OpenAI,
    model: str,
    seed_topics: List[Dict[str, Any]],
    segments: List[str],
) -> List[Dict[str, Any]]:
    system_msg = (
        "You expand health coaching topics into bullet sub-issues and stance sets, "
        "including plausible segment-conditioned prevalence distributions."
    )
    resp = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": system_msg},
               {"role": "user", "content": prompt_expand_coaching_topics(seed_topics, segments)}],
        temperature=0.6,
        top_p=1.0,
        text_format=CoachingTopicsOut,
    )
    parsed: CoachingTopicsOut = resp.output_parsed

    out: List[Dict[str, Any]] = []
    for t in parsed.topics:
        tt = t.model_dump()
        for s in tt["stances"]:
            s["segment_distribution"] = normalize_dist(s["segment_distribution"], segments)
        out.append(tt)
    return out


# ---------------------------
# Top-level static frames: Regions / Occupations / Education
# ---------------------------

def regions_sg() -> Dict[str, Any]:
    # Practical sampling priors (not official stats). Tune to your needs.
    # You can swap to “planning areas” at finer granularity if you want.
    return {
        "Central": {"population_share": 0.22, "explanation": "Dense central corridor; higher access to clinics, time pressure, high cost-of-living sensitivity."},
        "East": {"population_share": 0.26, "explanation": "Large residential areas; commuting patterns; strong mall/food-court eating contexts."},
        "NorthEast": {"population_share": 0.20, "explanation": "Family-heavy estates; multi-generational households are common; school-and-care routines salient."},
        "North": {"population_share": 0.14, "explanation": "Mix of older and newer estates; shift-work clusters; access varies by neighbourhood."},
        "West": {"population_share": 0.18, "explanation": "Industry + new towns; worksite routines and canteen food patterns salient."},
    }

def occupations_sg() -> List[Dict[str, Any]]:
    return [
        {"value": "Office_Professional", "explanation": "Sedentary hours, meeting-driven days; health efforts hinge on schedule control and stress eating.", "weight": 0.24},
        {"value": "Service_Retail", "explanation": "Standing and irregular breaks; meals are opportunistic; fatigue affects exercise adherence.", "weight": 0.12},
        {"value": "Healthcare_Worker", "explanation": "Shift work and caregiving load; strong health knowledge but constrained personal bandwidth.", "weight": 0.08},
        {"value": "Driver_Delivery", "explanation": "Long seated stretches; reliance on hawker/fast options; musculoskeletal pain patterns common.", "weight": 0.07},
        {"value": "Skilled_Technician", "explanation": "Hands-on work; variable physical load; injury prevention and recovery matter.", "weight": 0.07},
        {"value": "Student", "explanation": "Routine depends on term schedule; sleep and stress swings influence habits.", "weight": 0.10},
        {"value": "Homemaker_Caregiver", "explanation": "Care logistics dominate; motivation often tied to family wellbeing and energy levels.", "weight": 0.10},
        {"value": "Retired", "explanation": "More time but routines can be rigid; chronic conditions and mobility constraints shape goals.", "weight": 0.12},
        {"value": "Self_Employed", "explanation": "Unpredictable hours; autonomy helps, but consistency is hard.", "weight": 0.06},
        {"value": "Other", "explanation": "Catch-all to preserve heterogeneity and unusual life paths.", "weight": 0.04},
    ]

def education_sg() -> Dict[str, Any]:
    return {
        "HighestEducation": [
            {"value": "PrimaryOrLess", "explanation": "May prefer concrete, action-first coaching; higher risk of health-literacy gaps.", "weight": 0.10},
            {"value": "Secondary", "explanation": "Comfortable with straightforward explanations; benefits from examples and routines.", "weight": 0.28},
            {"value": "ITEOrDiploma", "explanation": "Practical, skills-focused; responds well to applied guidance and structured plans.", "weight": 0.28},
            {"value": "University", "explanation": "Often comfortable with rationale and metrics; may over-intellectualise behaviour change.", "weight": 0.30},
            {"value": "Postgraduate", "explanation": "High information tolerance; strong preference for evidence and autonomy.", "weight": 0.04},
        ]
    }


# ---------------------------
# High-value patient categories (seed + expand)
# ---------------------------

PATIENT_CATEGORIES_PURPOSE: Dict[str, str] = {
    "PrimaryLanguage": "Language used for coaching. Impacts idioms, politeness, comprehension, and trust.",
    "Ethnicity": "Cultural food norms, family dynamics, health beliefs, and holiday routines.",
    "PreferredCommunicationStyle": "How the patient wants guidance delivered: directive vs collaborative vs reflective, etc.",
    "AttitudesTowardsAuthority": "Comfort with clinician authority, deference, questioning, and shared decision-making.",
    "EmotionalTics": "Small affective patterns that show up under stress: humour deflection, catastrophising, minimising, etc.",
    "HealthProblemsCommon": "Common issues relevant to lifestyle coaching: metabolic, MSK pain, sleep, stress, etc.",
    "HealthGoals": "What success looks like: stamina, labs, weight, pain reduction, sleep, confidence, etc.",
    "CoachingMotivations": "Why they care: kids, work performance, confidence, longevity, rehab, etc.",
    "DietPattern": "Everyday eating context in Singapore: hawker routines, late dinners, sweet drinks, etc.",
    "ActivityBaseline": "Starting activity level and movement context (commute, stairs, weekends).",
    "SleepPattern": "Sleep regularity, latency, screen use, shift-work patterns, insomnia cues.",
    "HealthLiteracy": "Ability to interpret health info, labels, and tradeoffs; affects how much explanation helps.",
    "TechComfort": "Comfort with apps, trackers, reminders, telehealth; influences intervention design.",
    "CareAccessPattern": "How they use care: polyclinic vs GP vs specialist vs none; affects expectations and adherence.",
    "CoachChannelPreference": "Preferred coaching channel: WhatsApp-like chat, calls, in-app messages, group sessions, etc.",
    "FamilyContext": "Household dynamics and caregiving load that shape time, food, and stress.",
    "PainAndRehabContext": "If relevant: injury history, physio adherence style, fear-avoidance, pacing.",
}

PATIENT_CATEGORIES_SEEDS: Dict[str, List[tuple[str, str]]] = {
    "PrimaryLanguage": [
        ("English_dominant", "Prefers coaching in English and tends to use health terms comfortably."),
        ("Mandarin_dominant", "More fluent in Mandarin; prefers culturally resonant food and family examples."),
        ("Malay_dominant", "More fluent in Malay; values respectful tone and practical routines."),
        ("Tamil_dominant", "More fluent in Tamil; benefits from clear steps and family-aligned framing."),
        ("Mixed_code_switching", "Comfortably mixes languages; responds well to short, flexible prompts."),
    ],
    "Ethnicity": [
        ("Chinese", "Food norms may include rice/noodles, shared dishes, festive feasting cycles."),
        ("Malay", "Community eating and family routines can shape meal timing and choices."),
        ("Indian", "Rich spice profiles and carb staples can be navigated with portion and preparation strategies."),
        ("Eurasian_Other", "Mixed cultural norms; routines vary widely and can be personalised quickly."),
    ],
    "PreferredCommunicationStyle": [
        ("Direct_and_structured", "Wants clear instructions, checklists, and concrete next steps."),
        ("Collaborative_planning", "Prefers shared decision-making and co-created goals."),
        ("Reflective_coaching", "Responds to questions that build insight and intrinsic motivation."),
        ("Data_and_metrics", "Likes tracking, targets, and evidence-based explanations."),
        ("Encouraging_and_gentle", "Needs emotional safety; responds to affirmation and pacing."),
    ],
    "AttitudesTowardsAuthority": [
        ("Deferential", "Treats clinician advice as final; may hesitate to admit non-adherence."),
        ("Respectful_but_questioning", "Will ask for rationale and alternatives; prefers shared decisions."),
        ("Skeptical", "Low trust in generic advice; needs specificity and credibility cues."),
        ("Independent", "Prefers autonomy; reacts poorly to prescriptive tone."),
    ],
    "EmotionalTics": [
        ("Humour_deflection", "Jokes when uncomfortable; avoids direct discussion of guilt or fear."),
        ("Minimising", "Downplays symptoms; tends to delay action until it becomes urgent."),
        ("Catastrophising", "Spirals quickly under setbacks; needs reassurance and step-down plans."),
        ("All_or_nothing", "Overcommits then drops; benefits from micro-habits and relapse planning."),
        ("People_pleasing", "Agrees readily but struggles to say no; needs permission to negotiate plans."),
    ],
    "HealthProblemsCommon": [
        ("Weight_management", "Struggles with gradual weight gain tied to routine eating and low activity."),
        ("Prediabetes_or_T2_risk", "Concerned about sugar control; benefits from carbohydrate strategy and activity."),
        ("Hypertension_risk", "Salt and stress interact; lifestyle consistency is key."),
        ("Chronic_back_or_knee_pain", "Movement is limited by pain; pacing and rehab adherence matter."),
        ("Poor_sleep_quality", "Irregular sleep and screen time; sleep hygiene and stress reduction help."),
        ("High_stress_burnout", "Stress drives eating and low activity; needs coping tools and boundaries."),
    ],
    "HealthGoals": [
        ("Play_with_kids_more", "Wants stamina and mobility to participate with children without pain or fatigue."),
        ("Weight_loss_steady", "Wants gradual, maintainable loss rather than crash dieting."),
        ("Reduce_pain_and_move_confidently", "Wants less fear around movement and better daily function."),
        ("Improve_labs", "Targets HbA1c, lipids, BP; motivated by measurable outcomes."),
        ("Sleep_better", "Wants consistent sleep and more daytime energy."),
    ],
    "CoachingMotivations": [
        ("Family_responsibility", "Wants to stay healthy to care for family and avoid burdening others."),
        ("Confidence_and_body_image", "Wants to feel comfortable socially and in clothing."),
        ("Work_performance", "Wants sharper focus and less fatigue at work."),
        ("Avoid_future_complications", "Motivated by long-term risk avoidance and longevity."),
        ("Pain_rehab_priority", "Motivated by getting back to sport/work tasks without flare-ups."),
    ],
    "DietPattern": [
        ("Hawker_lunch_regular", "Lunch often from hawker centres; portioning and drink choices matter."),
        ("Sweet_drinks_habit", "Sugary beverages are a frequent default; substitution strategies help."),
        ("Late_dinners", "Dinner timing is late due to commute; affects sleep and snacking."),
        ("Frequent_snacking", "Convenience snacks during work/study; needs environment redesign."),
        ("Home_cooked_majority", "More control at home; cooking methods and portions drive outcomes."),
    ],
    "ActivityBaseline": [
        ("Mostly_sedentary", "Low daily steps; exercise feels like a big lift."),
        ("Some_walking_commute", "Walks to MRT/bus; can build from existing routine."),
        ("Weekend_active_only", "Active bursts on weekends; weekday consistency is the challenge."),
        ("Physically_demanding_job", "Work provides movement but may cause fatigue or pain."),
    ],
    "SleepPattern": [
        ("Consistent_but_short", "Regular schedule but insufficient duration; needs time budgeting."),
        ("Irregular_bedtime", "Bedtime shifts; habit loops and wind-down routines help."),
        ("Sleep_onset_insomnia", "Takes long to fall asleep; cognitive and environment strategies help."),
        ("Shift_work", "Rotating schedule; needs adaptable routines and recovery prioritisation."),
    ],
    "HealthLiteracy": [
        ("Low_confidence", "Finds health info confusing; benefits from plain language and one change at a time."),
        ("Practical_understanding", "Understands basics; wants simple rules and meal templates."),
        ("High_literacy", "Comfortable with nuance; prefers rationale and personalised tradeoffs."),
    ],
    "TechComfort": [
        ("Low_tech", "Prefers simple messages; may avoid apps and tracking."),
        ("Basic_smartphone", "Comfortable with chat reminders and simple logs."),
        ("App_power_user", "Likes trackers, wearables, and dashboards."),
    ],
    "CareAccessPattern": [
        ("Polyclinic_regular", "Uses subsidised public care; trusts standard pathways and protocols."),
        ("GP_first", "Uses private GP for convenience; expects responsiveness."),
        ("Specialist_followup", "Already in specialist care; coaching must align with clinical plan."),
        ("Avoids_care", "Delays visits; needs trust-building and low-friction steps."),
    ],
    "CoachChannelPreference": [
        ("Chat_async", "Likes WhatsApp-style short messages and quick check-ins."),
        ("Scheduled_calls", "Prefers voice calls; wants deeper conversation and accountability."),
        ("In_app_program", "Prefers structured modules; likes progress visuals."),
        ("Group_support", "Motivated by peers; likes challenges and shared goals."),
    ],
    "FamilyContext": [
        ("Multi_generational_home", "Shared meals and caregiving; negotiation and cultural food strategies matter."),
        ("Young_kids_at_home", "Time-poor; routines must fit school and caregiving demands."),
        ("Single_adult_household", "More autonomy but can lack support and structure."),
        ("Elder_care_responsibility", "High stress load; sleep and self-care often compromised."),
    ],
    "PainAndRehabContext": [
        ("Fear_avoidance", "Avoids movement due to pain fear; needs graded exposure and reassurance."),
        ("Overdo_then_crash", "Pushes hard on good days then flares; needs pacing strategies."),
        ("Physio_adherent", "Follows rehab plan well; benefits from progressive structure."),
        ("Low_follow_through", "Starts rehab but drops; needs friction reduction and habit cues."),
    ],
}


# ---------------------------
# PersonaCore + AdditionalTraits (high richness)
# ---------------------------

PERSONA_CORE_PURPOSE: Dict[str, str] = {
    "Appearance": "Observable appearance cues relevant to patient life: workwear, comfort needs, injury supports, etc.",
    "InteractionStyle": "How they interact with the coach: disclosure, defensiveness, warmth, pacing, boundaries.",
    "MediaDisposition": "How they consume health info: TikTok vs long-form, influencer trust, scepticism.",
}

PERSONA_CORE_SEEDS: Dict[str, List[tuple[str, str]]] = {
    "Appearance": [
        ("Office_attire_neat", "Looks put-together for work; signals routine, structure, and social accountability."),
        ("Comfort_first", "Prioritises comfort; may signal pain, fatigue, or long commuting days."),
        ("Sporty_accessories", "Wears sporty items or shoes; suggests openness to activity identity."),
        ("Visible_support_brace", "Knee/back support or tape; signals rehab context and movement constraints."),
    ],
    "InteractionStyle": [
        ("Warm_and_chatty", "Shares context easily; responds well to relational rapport."),
        ("Reserved_minimal", "Gives short answers; needs gentle prompting and trust-building."),
        ("Highly_analytical", "Asks ‘why’; wants mechanisms and evidence."),
        ("Defensive_about_weight", "Sensitive to judgement; needs shame-free language."),
        ("Eager_but_inconsistent", "Enthusiastic starts then drops; needs relapse planning."),
    ],
    "MediaDisposition": [
        ("Influencer_driven", "Tries trends; benefits from myth-busting and safe substitutions."),
        ("Evidence_focused", "Trusts official guidelines; prefers credible sources and clarity."),
        ("Skeptical_of_health_claims", "Dislikes hype; needs concrete proof and personal relevance."),
        ("Short_form_scroller", "Consumes quick tips; needs tiny tasks and reminders."),
    ],
}

ADDITIONAL_TRAITS_SEEDS: Dict[str, Dict[str, Any]] = {
    "DisclosureStyle": {
        "purpose": "How candid the patient is about lapses, cravings, or fears.",
        "seed_values": [
            ("Very_candid", "Shares setbacks quickly; enables faster plan adjustment."),
            ("Selective", "Holds back sensitive details; needs reassurance and privacy cues."),
            ("Underreports_lapses", "Minimises non-adherence; needs neutral check-ins and normalisation."),
        ],
    },
    "AccountabilityPreference": {
        "purpose": "How much external accountability they want from the coach.",
        "seed_values": [
            ("High_accountability", "Wants firm check-ins and clear expectations."),
            ("Gentle_accountability", "Wants supportive reminders without pressure."),
            ("Autonomy_first", "Prefers tools and choices; reacts poorly to nagging."),
        ],
    },
    "StressCoping": {
        "purpose": "How stress shows up in eating, sleep, and activity.",
        "seed_values": [
            ("Stress_eating", "Uses snacks/sweets for relief; benefits from coping replacements."),
            ("Shutdown_and_withdraw", "Avoids tasks under stress; needs tiny, low-friction steps."),
            ("Overworking", "Sacrifices sleep and meals; needs boundary and recovery planning."),
        ],
    },
    "SocialSupport": {
        "purpose": "Availability of help from family/friends and how it affects adherence.",
        "seed_values": [
            ("Strong_support", "Family supports changes; can coordinate meals and activity."),
            ("Mixed_support", "Some supportive, some sabotaging; needs negotiation strategies."),
            ("Low_support", "Mostly alone; needs self-structure and community options."),
        ],
    },
    "FoodEnvironment": {
        "purpose": "How environment drives food choices (canteen, hawker, delivery apps).",
        "seed_values": [
            ("Canteen_default", "Worksite meals are set; needs menu heuristics."),
            ("Delivery_app_heavy", "Convenience-driven; needs substitution and friction tactics."),
            ("Home_cooking_norm", "More control; needs recipe/portion guidance."),
        ],
    },
    "TimeStructure": {
        "purpose": "How predictable the day is; impacts habit formation.",
        "seed_values": [
            ("Very_structured", "Routine-friendly; can implement schedules."),
            ("Semi_structured", "Some anchors; needs flexible plans."),
            ("Chaotic", "Unpredictable; needs ‘if-then’ plans and fallback options."),
        ],
    },
    "PainSensitivity": {
        "purpose": "How pain affects movement confidence and adherence.",
        "seed_values": [
            ("High_sensitivity", "Avoids discomfort; needs graded exposure and reassurance."),
            ("Moderate", "Can progress with good pacing."),
            ("Low", "Less constrained by pain; may overdo and need recovery planning."),
        ],
    },
    "AuthorityComfort": {
        "purpose": "Nuanced variant of authority attitude: comfort being corrected or challenged.",
        "seed_values": [
            ("Comfortable_with_direct_feedback", "Prefers blunt clarity; doesn’t take correction personally."),
            ("Sensitive_to_correction", "Feels judged easily; needs careful phrasing."),
            ("Needs_face_saving", "Prefers indirect suggestions to maintain dignity."),
        ],
    },
    "GoalHorizon": {
        "purpose": "Short-term vs long-term orientation for goals and rewards.",
        "seed_values": [
            ("Short_term_rewards", "Needs quick wins and visible progress."),
            ("Balanced", "Responds to weekly goals with occasional long-term reminders."),
            ("Long_term_horizon", "Can invest patiently; likes trajectories and risk framing."),
        ],
    },
    "MessageTonePreference": {
        "purpose": "Tone for coaching messages: playful, formal, encouraging, tough-love.",
        "seed_values": [
            ("Playful", "Responds to humour and friendly challenges."),
            ("Professional_formal", "Prefers respectful, concise, clinician-like tone."),
            ("Warm_encouraging", "Needs affirmation and compassion."),
            ("Tough_love", "Responds to firmness and accountability cues."),
        ],
    },
}


# ---------------------------
# Segments (for topic stance distributions)
# ---------------------------

def segments_for_sampling() -> List[str]:
    # Useful buckets for distributions. Adjust to your generation pipeline.
    return [
        "YoungAdults_18_35",
        "Midlife_36_55",
        "Older_56_plus",
        "ShiftWorkers",
        "DeskWorkers",
        "Caregivers",
        "HighHealthLiteracy",
        "LowHealthLiteracy",
        "PainLimited",
        "WeightLossFocused",
    ]


# ---------------------------
# Coaching topics (15) — like “policy debates” but health-focused
# ---------------------------

def seed_coaching_topics_15() -> List[Dict[str, Any]]:
    return [
        {"name": "WeightManagement"},
        {"name": "PrediabetesAndGlycemicControl"},
        {"name": "HypertensionAndSaltStress"},
        {"name": "CholesterolAndHeartHealth"},
        {"name": "ChronicPainAndRehabAdherence"},
        {"name": "SleepHygiene"},
        {"name": "StressAndEmotionalEating"},
        {"name": "PhysicalActivityHabits"},
        {"name": "DietInHawkerAndCanteenSettings"},
        {"name": "SugaryDrinksAndSnacking"},
        {"name": "MedicationAdherenceAndBeliefs"},
        {"name": "SmokingVapingReduction"},
        {"name": "AlcoholModeration"},
        {"name": "SocialSupportAndFamilyNegotiation"},
        {"name": "SustainingChangeAndRelapsePlanning"},
    ]


# ---------------------------
# Builders
# ---------------------------

def to_plain_options(items: List[OptionItem]) -> List[Dict[str, Any]]:
    return [it.model_dump() for it in items]

def build_persona_core(client: Optional[OpenAI], model: str, use_openai: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cat, purpose in PERSONA_CORE_PURPOSE.items():
        seed_items = [OptionItem(value=v, explanation=e) for v, e in PERSONA_CORE_SEEDS[cat]]

        if use_openai and client is not None:
            pool: List[OptionItem] = list(seed_items)
            avoid = [s.value for s in seed_items]

            base_prompt = prompt_expand_category(cat, purpose, n=25, avoid=avoid)
            generated = openai_expand_options_fill(
                client=client,
                model=model,
                base_prompt=base_prompt,
                min_items=25,
                sim_threshold=0.90,
                max_rounds=5,
                oversample_factor=1.8,
                fill_prompt_fn=prompt_expand_category_fill,
                category_name=cat,
                category_description=purpose,
            )

            pool.extend(generated)
            pool = dedup_options_semantic(client, pool, sim_threshold=0.90)

            out[cat] = {"items": to_plain_options(pool[:25]), "explanation": purpose}
        else:
            out[cat] = {"items": [s.model_dump() for s in seed_items], "explanation": purpose}

    return out

def build_additional_traits(client: Optional[OpenAI], model: str, use_openai: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for subcat, spec in ADDITIONAL_TRAITS_SEEDS.items():
        seed_items = [OptionItem(value=v, explanation=e) for v, e in spec["seed_values"]]

        if use_openai and client is not None:
            pool: List[OptionItem] = list(seed_items)
            avoid = [s.value for s in seed_items]

            base_prompt = prompt_expand_category(subcat, spec["purpose"], n=10, avoid=avoid)
            generated = openai_expand_options_fill(
                client=client,
                model=model,
                base_prompt=base_prompt,
                min_items=10,
                sim_threshold=0.90,
                max_rounds=5,
                oversample_factor=1.8,
                fill_prompt_fn=prompt_expand_trait_fill,
                category_name=subcat,
                category_description=spec["purpose"],
            )

            pool.extend(generated)
            pool = dedup_options_semantic(client, pool, sim_threshold=0.90)

            out[subcat] = {"items": to_plain_options(pool[:10]), "explanation": spec["purpose"]}
        else:
            out[subcat] = {"items": [s.model_dump() for s in seed_items], "explanation": spec["purpose"]}

    return out

def build_patient_categories(client: Optional[OpenAI], model: str, use_openai: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cat, purpose in PATIENT_CATEGORIES_PURPOSE.items():
        seeds = PATIENT_CATEGORIES_SEEDS.get(cat, [])
        seed_items = [OptionItem(value=v, explanation=e) for v, e in seeds]

        # Choose default targets: rich lists for key categories; smaller for others.
        target = 20 if cat in {"HealthProblemsCommon", "HealthGoals", "CoachingMotivations"} else 15
        if cat in {"PrimaryLanguage", "Ethnicity"}:
            target = 10

        if use_openai and client is not None:
            pool: List[OptionItem] = list(seed_items)
            avoid = [s.value for s in seed_items]

            base_prompt = prompt_expand_category(cat, purpose, n=target, avoid=avoid)
            generated = openai_expand_options_fill(
                client=client,
                model=model,
                base_prompt=base_prompt,
                min_items=target,
                sim_threshold=0.90,
                max_rounds=5,
                oversample_factor=1.8,
                fill_prompt_fn=prompt_expand_category_fill,
                category_name=cat,
                category_description=purpose,
            )

            pool.extend(generated)
            pool = dedup_options_semantic(client, pool, sim_threshold=0.90)
            out[cat] = {"items": to_plain_options(pool[:target]), "explanation": purpose}
        else:
            out[cat] = {"items": [s.model_dump() for s in seed_items], "explanation": purpose}

    return out

def build_yaml(client: Optional[OpenAI], model: str, use_openai: bool) -> Dict[str, Any]:
    segments = segments_for_sampling()
    topics_expanded = (
        openai_expand_coaching_topics(client, model, seed_coaching_topics_15(), segments)
        if (use_openai and client is not None)
        else []
    )

    return {
        "SingaporeHealthCoachPatientSeeds": {
            "version": "1.0",
            "Regions": {
                "items": regions_sg(),
                "explanation": "Sampling frame for residence/clinic access/commute context; population_share is a practical prior.",
            },
            "Occupations": {
                "items": occupations_sg(),
                "explanation": "Prevalent work contexts shaping time, stress, and movement; weights are sampling priors.",
            },
            "Education": {
                "items": education_sg(),
                "explanation": "Education priors for sampling; influences health literacy and communication preferences.",
            },

            "PatientCategories": build_patient_categories(client, model, use_openai),
            "PersonaCore": build_persona_core(client, model, use_openai),
            "AdditionalTraits": build_additional_traits(client, model, use_openai),

            "HealthCoachingTopics": {
                "segments": segments,
                "items": topics_expanded,
                "explanation": (
                    "Each topic includes bullet sub-issues and >=4 stances per bullet. "
                    "Stances include segment-conditioned prevalence priors (plausible, not survey data)."
                ),
            },
        }
    }


# ---------------------------
# CLI
# ---------------------------

@dataclass
class Args:
    out: str
    model: str
    no_openai: bool

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT, help="Output YAML path.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model id (default: gpt-4o).")
    p.add_argument("--no-openai", action="store_true", help="Disable OpenAI calls (partial YAML).")
    ns = p.parse_args()
    return Args(out=ns.out, model=ns.model, no_openai=ns.no_openai)

def main() -> None:
    args = parse_args()
    use_openai = not args.no_openai

    client = OpenAI() if use_openai else None
    data = build_yaml(client=client, model=args.model, use_openai=use_openai)

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote: {args.out}")
    if args.no_openai:
        print("NOTE: OpenAI disabled → no expansions for categories; topics list will be empty.")

if __name__ == "__main__":
    main()
