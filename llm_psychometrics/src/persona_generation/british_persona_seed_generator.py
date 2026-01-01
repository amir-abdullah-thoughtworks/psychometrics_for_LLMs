#!/usr/bin/env python3
"""
British Parliament persona seed generator (Westminster), full-fledged.

What this generates (YAML):
- CanonicalTexts.items: exactly 100 books, each with summary + ideology_tags
- Archetypes.items: 10 archetypes (police-style)
- PartyAffiliation priors
- Regions: population_share + explanation
- Occupations (top-level): {value, explanation, weight}
- Education (top-level): secondary + tertiary {value, explanation, weight}
- PersonaCore: Appearance/ParliamentaryStyle/MediaDisposition expanded to 25 items each (OpenAI)
- AdditionalTraits: each subcategory expanded to >=10 items (OpenAI)
- PolicyDebates: each debate has bullet_points; each bullet gets >=4 stances with party_distribution (OpenAI)
- PartyDistribution is estimated using OpenAI prior knowledge (not polling)

Requires:
  pip install openai pydantic pyyaml

OpenAI model default:
  gpt-4o  (Structured Outputs supported) :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import math
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import yaml
from pydantic import BaseModel, Field, conlist, field_validator, model_validator
from openai import OpenAI

# ---------------------------
# Embedding-based semantic de-dup + retry fill helpers
# ---------------------------

EMBEDDING_MODEL = "text-embedding-3-small"  # switch to -large for stricter semantics

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
    # casefold + collapse whitespace
    return " ".join(s.strip().casefold().split())

def dedup_options_semantic(
    client: Optional[OpenAI],
    items: List[OptionItem],
    sim_threshold: float = 0.88,
) -> List[OptionItem]:
    """
    De-dupes:
      1) exact duplicates by normalized `value`
      2) semantic near-duplicates by embedding (value + explanation)

    Keeps earlier items preferentially.
    If client is None -> only exact de-dup.
    """
    # 1) exact dedup by value
    seen = set()
    exact_kept: List[OptionItem] = []
    for it in items:
        k = _norm_key(it.value)
        if k not in seen:
            seen.add(k)
            exact_kept.append(it)

    if client is None or len(exact_kept) <= 1:
        return exact_kept

    # 2) semantic dedup (value + explanation is more stable than value-only)
    texts = [f"{it.value} :: {it.explanation}" for it in exact_kept]
    vecs = embed_texts(client, texts)

    kept: List[OptionItem] = []
    kept_vecs: List[List[float]] = []

    for it, v in zip(exact_kept, vecs):
        too_close = False
        for kv in kept_vecs:
            if cosine(v, kv) >= sim_threshold:
                too_close = True
                break
        if not too_close:
            kept.append(it)
            kept_vecs.append(v)

    return kept



def prompt_expand_category_fill(
    category_name: str,
    description: str,
    need_n: int,
    existing_values: List[str],
) -> str:
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
"""


def prompt_expand_trait_fill(
    subcategory: str,
    description: str,
    need_n: int,
    existing_values: List[str],
) -> str:
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
"""



# ---------------------------
# Defaults / constants
# ---------------------------

DEFAULT_MODEL = "gpt-4o"  # stronger model; valid in Responses API :contentReference[oaicite:2]{index=2}
DEFAULT_OUT = "british_parliament_persona_seeds.yaml"

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
# Pydantic output models (Structured Outputs via responses.parse)
# ---------------------------

class OptionItem(BaseModel):
    value: str = Field(..., description="Categorical label.")
    explanation: str = Field(..., description="1–2 sentences explaining what it signals or how it manifests.")
    weight: Optional[float] = Field(None, description="Optional sampling weight/probability.")


class OptionsOut(BaseModel):
    items: List[OptionItem]


class PartyDistributionOut(BaseModel):
    party_distribution: Dict[Party, float] = Field(
        ...,
        description="Nonnegative prevalence weights of the stance within each party."
    )
    rationale: str = Field(..., description="1–3 sentences explaining the intuition.")


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


class Canon100Out(BaseModel):
    items: conlist(CanonItem, min_length=100, max_length=100)


class DebateBulletStance(BaseModel):
    bullet: str = Field(..., description="The specific bullet point the stance addresses.")
    id: str = Field(..., description="snake_case id for the stance.")
    description: str = Field(..., description="1–2 sentences describing the stance.")
    party_distribution: Dict[Party, float]
    rationale: str = Field(..., description="1–3 sentences explaining party_distribution intuition.")


class PolicyDebateExpanded(BaseModel):
    name: str
    description: str = Field(..., description="One-paragraph overall debate description.")
    bullet_points: conlist(str, min_length=4, max_length=12) = Field(
        ..., description="Key sub-issues for the debate; written as bullets."
    )
    stances: List[DebateBulletStance]

    @model_validator(mode="after")
    def _validate_stances_per_bullet(self):
        # Ensure >=4 stances per bullet point.
        counts: Dict[str, int] = {b: 0 for b in self.bullet_points}
        for s in self.stances:
            if s.bullet in counts:
                counts[s.bullet] += 1
        missing = [b for b, c in counts.items() if c < 4]
        if missing:
            raise ValueError(f"Not enough stances for bullets: {missing}")
        return self


class PolicyDebatesOut(BaseModel):
    debates: conlist(PolicyDebateExpanded, min_length=20, max_length=20)


# ---------------------------
# Utility
# ---------------------------

def normalize_dist(d: Dict[str, float], eps: float = 1e-9) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in d.items() if k in PARTIES}
    s = sum(cleaned.values())
    if s < eps:
        majors = ["Conservative", "Labour", "LiberalDemocrats", "SNP", "Green"]
        return {p: round(1.0 / len(majors), 4) for p in majors}
    return {k: round(v / s, 4) for k, v in cleaned.items()}


def to_plain_options(items: List[OptionItem]) -> List[Dict[str, Any]]:
    return [it.model_dump() for it in items]


# ---------------------------
# Seed archetypes (police-style list)
# ---------------------------

def archetypes() -> List[Dict[str, Any]]:
    return [
        {
            "name": "The Proceduralist",
            "description": "Wins by rules, timing, and amendment craft rather than moral heat or media volume.",
            "signature_tells": ["Quotes procedure mid-debate", "Uses committees as battleground", "Treats wording changes as victories"],
            "strengths": ["Scrutiny through detail", "Institutional fluency", "Effective bill-shaping"],
            "pitfalls": ["Can sound pedantic", "Low public resonance", "Seen as ‘process over people’"],
        },
        {
            "name": "The Constituency Champion",
            "description": "Frames national questions through local consequences; casework and delivery dominate priorities.",
            "signature_tells": ["Leads with constituent stories", "Presses for timelines", "Talks in local-service language"],
            "strengths": ["Authenticity", "Grounded intuition", "Local trust"],
            "pitfalls": ["Parochial framing", "Less system design focus", "Pork-barrel critique risk"],
        },
        {
            "name": "The Media Gladiator",
            "description": "Optimised for the clip: sharp lines, conflict instincts, relentless message discipline.",
            "signature_tells": ["Turns answers into headlines", "Attacks framing first", "Treats PMQs as main arena"],
            "strengths": ["Agenda-setting", "Rapid response", "High-visibility leverage"],
            "pitfalls": ["Thin on implementation", "Escalates conflict", "Overfits news cycle"],
        },
        {
            "name": "The Policy Technocrat",
            "description": "Mechanism-first and evidence-first; persuasive via design, budgets, and delivery constraints.",
            "signature_tells": ["References pilots/evaluation", "Asks ‘how implemented?’", "Prefers briefs to theatrics"],
            "strengths": ["Expert credibility", "Implementation realism", "Strong committee performance"],
            "pitfalls": ["Low emotional resonance", "Can seem detached", "Underestimates symbolism"],
        },
        {
            "name": "The Faction Whisperer",
            "description": "Internal operator who builds blocs, brokers deals, and trades loyalty for concessions.",
            "signature_tells": ["Knows incentives", "Moves behind scenes", "Uses coded signals"],
            "strengths": ["Coalition craft", "Party intelligence", "Negotiation leverage"],
            "pitfalls": ["Seen as cynical", "Leak vulnerability", "Distrusted by outsiders"],
        },
        {
            "name": "The Moral Tribune",
            "description": "Values-driven; treats politics as conscience with clear red lines and moral language.",
            "signature_tells": ["Conscience framing", "Principle rebellions", "Ethical rhetoric"],
            "strengths": ["Integrity signal", "Value coalition building", "Clarity"],
            "pitfalls": ["Rigid negotiation", "Polarising", "Ignores second-order effects"],
        },
        {
            "name": "The Cross-Party Fixer",
            "description": "Quiet amendment-winner who accumulates reform via trust, reciprocity, and pragmatism.",
            "signature_tells": ["Strange-bedfellow alliances", "Avoids personal attacks", "Treats wins as incremental"],
            "strengths": ["Durable reform", "Trust-building", "Low-drama effectiveness"],
            "pitfalls": ["Invisible to voters", "Looks unprincipled to purists", "Outflanked by loud factions"],
        },
        {
            "name": "The Culture-Warrior",
            "description": "Symbolic combatant; politics as identity conflict and boundary-setting.",
            "signature_tells": ["High-salience wedges", "Speech/symbol obsession", "Threat framing"],
            "strengths": ["Mobilisation", "Media attention", "Sticky frames"],
            "pitfalls": ["Policy thinness", "Escalation spiral", "Coalition collapse"],
        },
        {
            "name": "The Devolution Hawk",
            "description": "Constitution-first worldview; union/devolution/sovereignty dominate interpretive frame.",
            "signature_tells": ["Returns debates to settlement", "Legitimacy language", "Institution-as-identity framing"],
            "strengths": ["Strategic coherence", "Institutional foresight", "Identity politics instincts"],
            "pitfalls": ["Single-issue tunnel vision", "Hard compromises", "Centre-periphery flare-ups"],
        },
        {
            "name": "The Party Ladder-Climber",
            "description": "Promotion-minded; disciplined, risk-managed positions, visible loyalty, careful media behaviour.",
            "signature_tells": ["No improvisation", "Avoids landmines", "‘Responsibility’ language"],
            "strengths": ["Reliability", "Message discipline", "Ministerial suitability"],
            "pitfalls": ["Opportunism perception", "Low authenticity", "Avoids necessary conflict"],
        },
    ]


# ---------------------------
# Top-level Regions / Occupations / Education (static, but can also be expanded later)
# ---------------------------

def regions_top_level() -> Dict[str, Any]:
    # Synthetic sampling priors; tune to your target population base.
    return {
        "England_London": {"population_share": 0.19, "explanation": "Greater London; dense media proximity, housing salience, high diversity."},
        "England_SouthEast": {"population_share": 0.14, "explanation": "Commuter belt; planning battles, tax sensitivity, transport salience."},
        "England_Midlands": {"population_share": 0.14, "explanation": "Mixed urban/post-industrial; regeneration and manufacturing legacy politics."},
        "England_North": {"population_share": 0.18, "explanation": "Large post-industrial/urban mix; services, inequality, and ‘levelling up’ narratives."},
        "England_SouthWest": {"population_share": 0.09, "explanation": "Rural/coastal mix; farming, tourism, infrastructure constraints."},
        "Scotland": {"population_share": 0.08, "explanation": "Distinct party system and devolution; independence/union questions remain central."},
        "Wales": {"population_share": 0.05, "explanation": "Devolved governance; language/culture politics and regional inequality focus."},
        "NorthernIreland": {"population_share": 0.03, "explanation": "Power-sharing context; identity and constitutional settlement uniquely salient."},
    }


def occupations_top_level() -> List[Dict[str, Any]]:
    # Weights are illustrative priors for sampling synthetic biographies.
    return [
        {"value": "Lawyer", "explanation": "Adversarial argument training; comfortable with precedent, scrutiny, and drafting language.", "weight": 0.20},
        {"value": "CivilServant", "explanation": "Understands machinery of government; implementation realism and process fluency.", "weight": 0.12},
        {"value": "Business_Executive", "explanation": "Frames policy as competitiveness and tradeoffs; strong delivery/metrics orientation.", "weight": 0.14},
        {"value": "LocalCouncillor", "explanation": "Casework instincts and local delivery mindset; strong council/service familiarity.", "weight": 0.12},
        {"value": "Academic", "explanation": "Evidence-first posture; tends to cite studies and prefer mechanism-level policy design.", "weight": 0.09},
        {"value": "Journalist", "explanation": "Narrative and agenda-setting instincts; skilled at framing and timing.", "weight": 0.07},
        {"value": "Union_Official", "explanation": "Worker-centred worldview; coalition-building via organised labour networks.", "weight": 0.06},
        {"value": "NGO_Advocate", "explanation": "Mission-driven; moral language and stakeholder coalition work.", "weight": 0.06},
        {"value": "Military_Officer", "explanation": "Duty/security framing; prefers clarity and hierarchy in decisions.", "weight": 0.04},
        {"value": "Healthcare_Professional", "explanation": "Service-delivery realism; patient/outcome framing and workforce concerns.", "weight": 0.04},
        {"value": "Teacher_Educator", "explanation": "Institutional empathy; strong views on standards, access, and social mobility.", "weight": 0.04},
        {"value": "Other", "explanation": "Catch-all to preserve heterogeneity and enable unusual biographies.", "weight": 0.06},
    ]


def education_top_level() -> Dict[str, Any]:
    return {
        "Secondary": [
            {"value": "Comprehensive", "explanation": "Broad public schooling; often signals mainstream social exposure.", "weight": 0.62},
            {"value": "Grammar", "explanation": "Selective schooling; often aligns with meritocracy narratives.", "weight": 0.18},
            {"value": "Private", "explanation": "Elite schooling; signals networks and high cultural capital.", "weight": 0.20},
        ],
        "Tertiary": [
            {"value": "RussellGroup", "explanation": "Research-intensive university; policy-literate and technocratic dispositions common.", "weight": 0.45},
            {"value": "Oxbridge", "explanation": "Elite networks, rhetorical confidence, high institutional fluency.", "weight": 0.18},
            {"value": "Post92", "explanation": "Often vocational or locally grounded; practicality and access framing.", "weight": 0.25},
            {"value": "NoUniversity", "explanation": "Politics framed through work/lived experience; authenticity emphasised.", "weight": 0.12},
        ],
    }


# ---------------------------
# OpenAI prompts for expansions
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
- Keep items UK/Westminster-plausible where relevant
- No profanity, no slurs, no real-person accusations
"""


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
"""


def prompt_expand_canon_100(seed_items: List[Dict[str, Any]]) -> str:
    return f"""
Curate exactly 100 canon items for UK parliament persona seeding.

Each item must include:
- title, author, year (or null)
- category (one of: ParliamentaryHistory, PoliticalMemoir, PoliticalPhilosophy, PoliticalBiography,
  PoliticalJournalism, ConstitutionalLaw, DevolutionAndUnion, BrexitAndEurope, CampaigningAndSpin, PoliticalFiction)
- summary (1–2 sentences; persona-forming)
- ideology_tags (must cover economic + constitutional/secession/devolution + Brexit/Europe dimensions)

Coverage constraints:
- Strong coverage of constitutional reform + devolution/union/secession topics
- Strong Brexit/Europe coverage across hard/soft/rejoin/alignment perspectives
- At least 10 PoliticalFiction / satire-adjacent Westminster culture items
- Broad ideological spread

Seed items (may revise and expand from these):
{seed_items}
"""


def prompt_expand_policy_debates_20(seed_debates: List[Dict[str, Any]]) -> str:
    return f"""
You are expanding a catalog of 20 UK policy debates for synthetic MP persona generation.

For EACH debate:
- Provide a 1-paragraph description
- Provide 4–12 bullet_points (sub-issues). Each bullet must be specific.
- For EACH bullet point, generate AT LEAST 4 discrete stances.
  Each stance must include:
  - bullet (exact bullet text it addresses)
  - id (snake_case)
  - description (1–2 sentences)
  - party_distribution: nonnegative weights across these parties only:
    {PARTIES}
  - rationale: 1–3 sentences justifying the party_distribution

Important:
- These are plausible priors (not polling). Spread probability when uncertain.
- Keep stances meaningfully different, not just rewordings.
- Make sure each bullet has at least 4 stances.

Seed debates (names and rough scope):
{seed_debates}
"""


# ---------------------------
# OpenAI parse wrappers (Pydantic Structured Outputs)
# ---------------------------

def openai_expand_options_fill(
    client: OpenAI,
    model: str,
    base_prompt: str,
    min_items: int,
    *,
    # semantic dedup settings
    sim_threshold: float = 0.88,
    # retry strategy
    max_rounds: int = 4,
    oversample_factor: float = 2.0,
    # prompt rewriter for subsequent rounds
    fill_prompt_fn=None,
    category_name: str = "",
    category_description: str = "",
) -> List[OptionItem]:
    """
    Generate options with retries:
      round 0: base_prompt
      round 1..: fill_prompt_fn(category_name, category_description, need_n, existing_values)

    Each round:
      - generate ~need * oversample_factor items
      - merge into pool
      - semantic dedup
      - stop if >= min_items
    """
    if fill_prompt_fn is None:
        raise ValueError("fill_prompt_fn must be provided for retry/fill behaviour.")

    pool: List[OptionItem] = []

    def do_parse(prompt: str, target: int) -> List[OptionItem]:
        # Ask for more than required to survive dedup
        want = max(target, int(math.ceil(target * oversample_factor)))
        sys = "You generate structured persona seed options. Return diverse, concrete, distinct items."
        user = prompt + f"\n\nReturn at least {want} items."
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.8,
            top_p=1.0,
            text_format=OptionsOut,
        )
        out: OptionsOut = resp.output_parsed
        return out.items

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

        # small backoff in case of rate limits
        time.sleep(0.2)

    if len(pool) < min_items:
        # last-resort: return what we got; caller can decide to raise
        return pool

    return pool[:min_items]



def openai_expand_canon_100(client: OpenAI, model: str, seed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_msg = "You curate a UK political canon for persona seeding; output exactly 100 items."
    resp = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": system_msg},
               {"role": "user", "content": prompt_expand_canon_100(seed_items)}],
        temperature=0.6,
        top_p=1.0,
        text_format=Canon100Out,
    )
    parsed: Canon100Out = resp.output_parsed
    return [it.model_dump() for it in parsed.items]


def openai_expand_policy_debates(client: OpenAI, model: str, seed_debates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_msg = (
        "You expand policy debates into bullet sub-issues and stance sets, "
        "including plausible party-conditioned prevalence distributions."
    )
    resp = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": system_msg},
               {"role": "user", "content": prompt_expand_policy_debates_20(seed_debates)}],
        temperature=0.6,
        top_p=1.0,
        text_format=PolicyDebatesOut,
    )
    parsed: PolicyDebatesOut = resp.output_parsed
    # Normalize distributions defensively.
    debates = []
    for d in tqdm(parsed.debates, desc="Expanding policy debates"):
        dd = d.model_dump()
        for s in dd["stances"]:
            s["party_distribution"] = normalize_dist(s["party_distribution"])
        debates.append(dd)
    return debates


# ---------------------------
# Base seeds (for expansion)
# ---------------------------

def base_canon_seed() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Erskine May: Parliamentary Practice",
            "author": "Erskine May",
            "year": None,
            "category": "ParliamentaryHistory",
            "summary": "The core guide to Westminster procedure; forms procedural, institution-minded MPs who argue through rules and precedent.",
            "ideology_tags": ["institutionalist"],
        },
        {
            "title": "The English Constitution",
            "author": "Walter Bagehot",
            "year": 1867,
            "category": "ConstitutionalLaw",
            "summary": "A classic account of constitutional functions and legitimacy; forms pragmatic, systems-minded parliamentarians.",
            "ideology_tags": ["institutionalist", "constitutional_realism"],
        },
        {
            "title": "On Liberty",
            "author": "John Stuart Mill",
            "year": 1859,
            "category": "PoliticalPhilosophy",
            "summary": "Canonical argument on freedom and limits of coercion; informs civil-libertarian instincts and speech-oriented rhetoric.",
            "ideology_tags": ["liberal", "civil_libertarian"],
        },
        {
            "title": "House of Cards",
            "author": "Michael Dobbs",
            "year": 1989,
            "category": "PoliticalFiction",
            "summary": "Satirical thriller of ambition and manoeuvre; seeds cynical media instincts and internal power-awareness.",
            "ideology_tags": ["realpolitik", "party_machine"],
        },
    ]


def seed_debates_20() -> List[Dict[str, Any]]:
    # Seed is just names; the model will generate bullets and stances per bullet.
    return [
        {"name": "Immigration"},
        {"name": "ClimateAndEnergy"},
        {"name": "NHSAndSocialCare"},
        {"name": "TaxationAndSpending"},
        {"name": "HousingAndPlanning"},
        {"name": "PolicingAndJustice"},
        {"name": "Education"},
        {"name": "WelfareAndBenefits"},
        {"name": "Transport"},
        {"name": "ForeignPolicy"},
        {"name": "DefenceAndSecurity"},
        {"name": "DigitalPrivacy"},
        {"name": "AIRegulation"},
        {"name": "TradeAndIndustry"},
        {"name": "AgricultureAndRural"},
        {"name": "FreeSpeechAndCulture"},
        {"name": "ElectoralReform"},
        {"name": "DevolutionAndUnion"},
        {"name": "BrexitAndEurope"},
        {"name": "ReligiousFreedomAndSecularism"},
    ]


# ---------------------------
# Static “AdditionalTraits” structure (expanded to >=10 items each by OpenAI)
# ---------------------------

ADDITIONAL_TRAITS_SEEDS: Dict[str, Dict[str, Any]] = {
    "RhetoricalRegister": {
        "purpose": "How they argue: tone, structure, persuasion style.",
        "seed_values": [
            ("Legalistic", "Argues through definitions, precedent, and procedural constraints."),
            ("Moralising", "Frames issues as right vs wrong; appeals to conscience and values."),
            ("Technocratic", "Emphasises implementation detail and measurable outcomes."),
            ("Populist", "Frames elites as detached; speaks for ‘ordinary people’."),
            ("Narrative-driven", "Uses stories and constituency anecdotes to anchor policy."),
        ],
    },
    "RelationshipToParty": {
        "purpose": "How they relate to leadership, factions, and discipline.",
        "seed_values": [
            ("Loyalist", "Defends leadership line; values discipline and collective responsibility."),
            ("Soft rebel", "Breaks ranks selectively; cultivates independence without burning bridges."),
            ("Factional organiser", "Builds internal blocs; uses causes to consolidate influence."),
            ("Leadership aspirant", "Positions for promotion; careful about timing and risk."),
        ],
    },
    "AttitudeToInstitutions": {
        "purpose": "How they view constitutional norms and institutional legitimacy.",
        "seed_values": [
            ("Institutional conservative", "Treats norms as stabilising; prefers continuity."),
            ("Reformist", "Wants change through procedure and negotiated settlement."),
            ("Abolitionist", "Sees some institutions as illegitimate; prefers replacement."),
            ("Pragmatic incrementalist", "Focuses on tractable steps; accepts imperfect compromises."),
        ],
    },

    # Extras you wanted earlier (keep, and expand to >=10 each):
    "CommitteeFocus": {
        "purpose": "Which policy areas they specialise in and use as their influence channel.",
        "seed_values": [
            ("Treasury/Finance", "Budgets, taxation, fiscal rules, and macro framing."),
            ("Home Affairs/Justice", "Policing, courts, prisons, migration enforcement."),
            ("Health & Social Care", "NHS capacity, workforce, waiting lists, social care."),
            ("Foreign Affairs/Defence", "Alliances, procurement, and geopolitical risk."),
            ("Constitution/Devolution", "Union settlement, electoral rules, Lords reform."),
            ("Environment/Energy", "Net-zero, energy security, industrial transition."),
        ],
    },
    "ConstituencyType": {
        "purpose": "The voter environment that shapes instincts, language, and priorities.",
        "seed_values": [
            ("Urban metropolitan", "Diversity, housing pressure, transit salience."),
            ("Post-industrial town", "Regeneration, skills, ‘left behind’ narratives."),
            ("Rural/agricultural", "Farming, land use, sparse services."),
            ("Coastal", "Seasonal economy, infrastructure, identity politics."),
            ("Affluent commuter belt", "Tax sensitivity, planning conflicts, schools."),
            ("Devolved-national seat", "National identity and constitutional salience."),
        ],
    },
    "DonorLobbyExposure": {
        "purpose": "How external networks shape priorities and framing.",
        "seed_values": [
            ("Minimal donor exposure", "Low external influence; relies on grassroots and local party."),
            ("Industry-aligned networks", "Sector ties; emphasises competitiveness and regulation cost."),
            ("Union-aligned networks", "Labour ties; prioritises worker protections."),
            ("Think-tank ecosystem", "Policy shaped by reports and frameworks; ‘ideas’ oriented."),
        ],
    },
    "ScandalVulnerability": {
        "purpose": "What kinds of risks or weak points plausibly shape caution and behaviour.",
        "seed_values": [
            ("None visible", "Careful compliance and disciplined personal conduct."),
            ("Expenses/administrative risk", "Complex travel/staffing; vulnerable to procedural mishaps."),
            ("Personal-life scrutiny", "High-profile or tabloid interest; narrative risk."),
            ("Historical statements", "Old quotes/posts susceptible to recontextualisation."),
            ("Factional rivalries", "Internal enemies; vulnerable to leaks and briefings."),
        ],
    },
    "MediaFootprint": {
        "purpose": "Where influence happens: chamber, local press, national broadcast, or online.",
        "seed_values": [
            ("Backbench low profile", "Influence is internal; rarely trends."),
            ("Regional media regular", "Strong local press; constituency-first language."),
            ("National media figure", "High scrutiny; disciplined messaging."),
            ("Online influencer MP", "Social-first; mobilisation through virality."),
        ],
    },
    "LeadershipAmbition": {
        "purpose": "Their incentive landscape: policy mastery, ministerial ladder, or internal power.",
        "seed_values": [
            ("Content specialist", "Aims for policy mastery, not leadership power."),
            ("Ministerial ladder", "Promotion track; loyalty and caution rewarded."),
            ("Party leadership", "Long-term brand building; faction management focus."),
            ("Kingmaker/whip influence", "Prefers internal leverage; trades votes and amendments."),
        ],
    },
    "CoalitionPosture": {
        "purpose": "How they negotiate across parties and factions.",
        "seed_values": [
            ("Cross-party builder", "Seeks durable alliances for reform."),
            ("Partisan combatant", "Avoids giving opponents wins; conflict as strategy."),
            ("Issue-by-issue pragmatist", "Supports based on merit and local impact."),
        ],
    },
    "ConstituencyServiceStyle": {
        "purpose": "How they ‘do the job’ locally: casework, organising, or business/service advocacy.",
        "seed_values": [
            ("Casework maximalist", "Measures success by resolved constituent problems."),
            ("Community organiser", "Builds local campaigns; mobilisation oriented."),
            ("Business liaison", "Works with employers/councils; jobs and investment focus."),
            ("Public services advocate", "Champions hospitals/schools/policing; delivery focus."),
        ],
    },
}


# ---------------------------
# Party affiliation priors
# ---------------------------

def party_affiliation_priors() -> Dict[str, Any]:
    priors = {
        "Conservative": 0.35,
        "Labour": 0.33,
        "LiberalDemocrats": 0.08,
        "SNP": 0.05,
        "Green": 0.02,
        "PlaidCymru": 0.01,
        "DUP": 0.01,
        "SinnFein": 0.01,
        "SDLP": 0.01,
        "Alliance": 0.01,
        "Independent": 0.02,
    }
    s = sum(priors.values())
    return {
        k: {"base_probability": round(v / s, 4), "explanation": "Sampling prior for synthetic persona generation."}
        for k, v in priors.items()
    }


# ---------------------------
# PersonaCore seeds + expansion to 25 each
# ---------------------------

PERSONA_CORE_PURPOSE = {
    "Appearance": "Observable presentation choices that signal class, professionalism, and media orientation.",
    "ParliamentaryStyle": "How they operate in Westminster: procedure, theatre, scrutiny, coalition, conflict.",
    "MediaDisposition": "How they relate to media: avoidance, broadcast skill, social strategy, long-form seriousness.",
}

PERSONA_CORE_SEEDS = {
    "Appearance": [
        ("Conservative tailoring, muted colours", "Signals institutional respect and low-drama professionalism."),
        ("Constituency-worn suits with visible wear", "Projects groundedness and travel-heavy routine."),
        ("Television-ready polish", "Optimised for broadcast: crisp presentation and rehearsed phrasing."),
        ("Deliberately anti-elite presentation", "Plain styling used to reject metropolitan technocratic norms."),
    ],
    "ParliamentaryStyle": [
        ("Procedural and rule-focused", "Persuades through process: amendments, precedent, committee leverage."),
        ("Rhetorical and performative", "Optimised for chamber theatre and memorable lines."),
        ("Data-driven and technocratic", "Frames debates in mechanisms, metrics, and delivery constraints."),
        ("Populist and combative", "Uses ‘people vs system’ framing; confrontation-forward."),
        ("Conciliatory cross-party operator", "Builds coalitions quietly; trades concessions for incremental wins."),
    ],
    "MediaDisposition": [
        ("Avoids media except formal statements", "Influence is internal; prefers committee and constituency work."),
        ("Frequent broadcast commentator", "Uses media to set narrative; interview-hardened."),
        ("Social-media-native operator", "Thinks in shareable frames; rapid-response style."),
        ("Long-form essay and speech focused", "Prefers serious venues; detail over soundbites."),
    ],
}


# ---------------------------
# Assembly
# ---------------------------
def build_persona_core(client: Optional[OpenAI], model: str, use_openai: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    print(f"Building persona core")
    for cat, purpose in tqdm(PERSONA_CORE_PURPOSE.items(), desc=f"Building persona core"):
        seed_items = [OptionItem(value=v, explanation=e) for v, e in PERSONA_CORE_SEEDS[cat]]

        if use_openai and client is not None:
            # Start the pool with seeds so they are guaranteed included.
            pool: List[OptionItem] = list(seed_items)

            avoid = [s.value for s in seed_items]
            base_prompt = prompt_expand_category(cat, purpose, n=25, avoid=avoid)

            # Expand to quota with semantic dedup + novelty retry
            generated = openai_expand_options_fill(
                client=client,
                model=model,
                base_prompt=base_prompt,
                min_items=25,
                sim_threshold=0.88,
                max_rounds=5,
                oversample_factor=1.8,
                fill_prompt_fn=prompt_expand_category_fill,
                category_name=cat,
                category_description=purpose,
            )

            pool.extend(generated)
            pool = dedup_options_semantic(client, pool, sim_threshold=0.88)

            if len(pool) < 25:
                # You can choose to raise instead, but returning partial is sometimes fine
                # raise ValueError(f"Could not reach 25 items for {cat}; got {len(pool)}")
                pass

            out[cat] = {"items": to_plain_options(pool[:25]), "explanation": purpose}
        else:
            out[cat] = {"items": [s.model_dump() for s in seed_items], "explanation": purpose}

    return out



def build_additional_traits(client: Optional[OpenAI], model: str, use_openai: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    print(f"Building additional traits")
    for subcat, spec in tqdm(ADDITIONAL_TRAITS_SEEDS.items(), desc=f"Building additional traits"):
        seed_items = [OptionItem(value=v, explanation=e) for v, e in spec["seed_values"]]

        if use_openai and client is not None:
            pool: List[OptionItem] = list(seed_items)
            avoid = [s.value for s in seed_items]

            base_prompt = prompt_expand_additional_trait(subcat, spec["purpose"], n=10, avoid=avoid)
            generated = openai_expand_options_fill(
                client=client,
                model=model,
                base_prompt=base_prompt,
                min_items=10,
                sim_threshold=0.88,
                max_rounds=5,
                oversample_factor=1.8,
                fill_prompt_fn=prompt_expand_trait_fill,
                category_name=subcat,
                category_description=spec["purpose"],
            )

            pool.extend(generated)
            pool = dedup_options_semantic(client, pool, sim_threshold=0.88)

            out[subcat] = {"items": to_plain_options(pool[:10]), "explanation": spec["purpose"]}
        else:
            out[subcat] = {"items": [s.model_dump() for s in seed_items], "explanation": spec["purpose"]}

    return out



def build_yaml(client: Optional[OpenAI], model: str, use_openai: bool) -> Dict[str, Any]:
    # Canonical texts
    canon_seed = base_canon_seed()
    print(f"Expanding canon items")
    canon_items = openai_expand_canon_100(client, model, canon_seed) if (use_openai and client) else canon_seed

    # Policy debates expanded (20 debates, bullets + >=4 stances per bullet)

    print(f"Expanding debate items")
    debates_expanded = openai_expand_policy_debates(client, model, seed_debates_20()) if (use_openai and client) else []

    return {
        "BritishParliamentPersonaSeeds": {
            "version": "1.2",
            "CanonicalTexts": {
                "items": canon_items,
                "explanation": "Canon texts used to seed narrative voice, institutional posture, and ideology.",
            },
            "Archetypes": {
                "items": archetypes(),
                "explanation": "High-level parliamentary personality templates to guide consistent persona generation.",
            },
            "PartyAffiliation": party_affiliation_priors(),

            # Top-level blocks you requested
            "Regions": {
                "items": regions_top_level(),
                "explanation": "Sampling frame for region-of-origin or constituency flavour; population_share used for priors.",
            },
            "Occupations": {
                "items": occupations_top_level(),
                "explanation": "Pre-parliament careers used to seed biography and competence cues; weights are sampling priors.",
            },
            "Education": {
                "items": education_top_level(),
                "explanation": "Education priors split into secondary and tertiary backgrounds with weights.",
            },

            # Expanded persona core
            "PersonaCore": build_persona_core(client, model, use_openai),

            # Expanded additional traits (each >= 10)
            "AdditionalTraits": build_additional_traits(client, model, use_openai),

            # Expanded debates
            "PolicyDebates": {
                "items": debates_expanded,
                "explanation": "Each debate includes bullet sub-issues and >=4 stances per bullet with party-conditioned prevalence priors.",
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
        print("NOTE: OpenAI disabled → no expansions (canon<100, categories unexpanded, debates empty).")


if __name__ == "__main__":
    main()
