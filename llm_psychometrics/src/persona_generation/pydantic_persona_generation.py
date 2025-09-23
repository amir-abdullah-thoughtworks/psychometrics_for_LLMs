from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

# Define a Pydantic schema for a literal "true" or "false"
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

list_of_names = [
    "Amir Rahman",
    "Jamal Carter",
    "Samantha Lee",
    "Priya Patel",
    "David Kim",
    "Maria Gonzalez",
    "Chen Wei",
    "Olivia Smith",
    "Marcus Johnson",
    "Fatima Al-Farsi",
    "Elena Petrova",
    "Noah Williams",
    "Ava Brown",
    "Liam Jones",
    "Sophia Garcia",
    "Ethan Miller",
    "Isabella Martinez",
    "Mason Davis",
    "Mia Rodriguez",
    "Logan Wilson",
    "Charlotte Anderson",
    "Benjamin Thomas",
    "Harper Moore",
    "Lucas Taylor",
    "Ella Jackson",
    "Henry White",
    "Grace Harris",
    "Alexander Martin",
    "Chloe Thompson",
    "Sebastian Clark"
]
specified_age = [53]
Location = ["Baltimore, MD"]
archtypes = [
    "The Professional (or Service-Oriented Officer): Core trait: Emphasizes community service and procedural justice. Focus: De-escalation, fairness, empathy, following policy. Often aligns with: Community policing values. Strengths: Builds trust, effective in diverse communities. Challenges: May be frustrated in high-conflict departments.",
    "The Enforcer (or Crime-Fighter): Core trait: Views the role as controlling crime and disorder. Focus: Making arrests, asserting authority, zero-tolerance. Often aligns with: Traditional or paramilitary models of policing. Strengths: Confident under pressure, decisive. Challenges: May escalate situations; less focused on community trust.",
    "The Reciprocator (or Nice Cop): Core trait: Seeks harmony and avoids confrontation. Focus: Helping others, mediating conflict. Strengths: Calms tensions, builds rapport. Challenges: May struggle with assertiveness or dangerous situations.",
    "The Avoider (or Passive Officer): Core trait: Minimizes involvement, avoids proactive enforcement. Focus: Doing the minimum required; reactive rather than proactive. Strengths: Stays out of trouble, avoids unnecessary escalation. Challenges: Low initiative; potential liability if avoidance leads to neglect.",
    "The Tough Cop (or Authoritarian): Core trait: Believes respect comes through command and control. Focus: Hierarchy, obedience, visible power. Strengths: Handles high-stress, compliance-driven environments. Challenges: Risk of excessive force, civil liberties concerns.",
    "The Problem Solver: Core trait: Sees policing as a mix of investigation, mediation, and public service. Focus: Resolving root causes of issues (e.g., disputes, addiction, disorder). Strengths: Effective in community policing and multidisciplinary teams. Challenges: Time-intensive; requires organizational support."
]

import random

#chosen_name = random.sample(list_of_names, 1)
#chosen_archtype = random.sample(archtypes, 1)
chosen_name = ["Guy Dude"]
chosen_archtype = ["The Problem Solver: Core trait: Sees policing as a mix of investigation, mediation, and public service. Focus: Resolving root causes of issues (e.g., disputes, addiction, disorder). Strengths: Effective in community policing and multidisciplinary teams. Challenges: Time-intensive; requires organizational support."]
print(chosen_name)


class SeededPersonaSchema(BaseModel):
    # Demographic Fields
    archtype: Literal[*chosen_archtype]
    name: Literal[*chosen_name]
    age: Literal[*specified_age]
    location: Literal[*specified_age]
    # Behavioral and Psychological Descriptors
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



class PersonaSchema(BaseModel):
    # Demographic Fields
    name: Literal[*list_of_names]
    age: Literal[
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70
    ] = Field(..., description="Age of the persona (21–70); ensure logical consistency with profile.")
    location: str = Field(..., description="U.S. city or fictional equivalent reflecting the persona's cultural/professional environment.")

    # Behavioral and Psychological Descriptors
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

# Set up OpenAI client to use your vLLM server
client = OpenAI()


after_action_report = """
It was just after midnight when the call came in—a suspected burglary in progress at a small electronics store on the edge of the district. The air was thick with humidity as I pulled up, headlights off, scanning the storefront for movement. My partner, Torres, radioed for backup while I approached the side alley, boots crunching softly on broken glass. 

Inside, the faint beam of a flashlight flickered between shelves. I steadied my breathing, recalling training: slow, deliberate, announce presence. "Police! Come out with your hands where I can see them!" The figure froze, then bolted toward the rear exit. Instinct took over—I gave chase, adrenaline sharpening my focus. The suspect slipped on a patch of oil, crashing into a stack of crates. I closed the distance, cuffed him, and read his rights as he caught his breath, eyes darting with fear and resignation.

After securing the scene and confirming no accomplices, I took a moment to survey the damage—shattered display cases, scattered merchandise, the lingering sense of violation. As dawn crept over the city, I filed my report, the weight of responsibility settling in. Another night, another story added to the ledger—a reminder of why I chose this badge, and the quiet resolve it demands.
"""

chat_response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[
        {"role": "system", "content": (
            "You are an expert clinical interviewer and psychological profiler. "
            "Your task is to generate a detailed, realistic persona profile following the schema below. "
            "For each section and field, strictly adhere to the content style and length guidelines. "
            "Write in natural, nuanced, and human-like language, simulating clinical notes and psychological assessments. "
            "Avoid caricature or stereotypes; include subtle contradictions and complexity for realism. "
            "Demographic Fields: "
            "- Name: multicultural, human-like, randomly generated full name (optionally include preferred name in parentheses). "
            "- Age: Integer 21–70, logically consistent with life history. "
            "- Location: U.S. city or fictional equivalent, reflecting cultural/professional context. "
            "Behavioral and Psychological Descriptors (30–60 words each): "
            "- Appearance: Observational, sensory details. "
            "- Behavior: Posture, interaction style, responsiveness. "
            "- Mood/Affect: Tone modulation, emotional nuance. "
            "- Speech: Register, rhythm, formality, coherence. "
            "- Thought Content: Internal reflection, logic, obsessions, themes. "
            "- Insight/Judgment: Clinical phrasing, situational insight, blind spots. "
            "- Cognition: Memory, abstraction, coherence, cognitive style. "
            "Life History Segments (100–150 words each): "
            "- Medical/Developmental History: Chronic/stress-related issues, exclude major psychiatric illness unless specified. "
            "- Family History: Generational details, substance patterns, relational dynamics. "
            "- Educational/Vocational History: Education, job trajectory, training, affiliations. "
            "Functional Assessments (100–150 words each): "
            "- Emotional/Behavioral Functioning: Stress, trauma, anger processing, coping mechanisms. "
            "- Social Functioning: Relationship style, trust, group affiliation, isolation/connection. "
            "Summary (150–250 words): "
            "- Summary of Psychological Profile: Integrative clinical summary, diagnostic impressions, resilience, risks, prognosis. "
            "Ensure all outputs are richly detailed, clinically plausible, and stylistically consistent with professional psychological documentation."
        )},
        {"role": "user", "content": after_action_report},
    ],
    temperature=2.0,
    top_p=0.98,
    text_format=SeededPersonaSchema,
)


print(chat_response.output_parsed)

