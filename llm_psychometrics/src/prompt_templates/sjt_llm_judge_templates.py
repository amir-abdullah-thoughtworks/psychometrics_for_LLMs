SJT_LLM_JUDGE_EVALUATION_WITH_SEEDS_TEMPLATE_STR = """
You are an expert evaluator of situational judgment tests (SJTs). Your role is to assess the quality of each SJT scenario and its response options using a structured rubric. You must score each dimension on a 1–5 scale and provide a concise justification. Be objective, consistent, and fair. Always return results in JSON format.

Evaluate the following SJT using the rubric provided.

**Situational Judgment Test:**

Question: {{ question }}

Answer Options:

{{ answer_options }}

**Question Description**

* Every option in the question corresponds to a HEXACO Trait and they follow the following order Honesty-Humility, Emotionality, Extraversion, Agreeableness, Conscientiousness, Openness to Experience.
* Question and answers are created using specific seed values.
* SJT is created to simulate a professional law enforcement context.

**Important: Trait Mapping of Options**  
Each answer option in the SJT corresponds directly to one HEXACO trait, in the following fixed order:  
1. Honesty-Humility  
2. Emotionality  
3. Extraversion  
4. Agreeableness  
5. Conscientiousness  
6. Openness to Experience  

**Seed Description**

* **Urgency Level:**
  * Low: Situation allows ample time for decision-making with no immediate pressure.
  * Medium: Requires timely attention but still allows some deliberation.
  * High: Demands rapid response with little to no time for delay.
* **Threat Level:**
  * Low: Minimal risk to safety or order; situation is stable.
  * Medium: Moderate potential risk requiring caution and situational awareness.
  * High: Significant danger present, with immediate risk to safety or security.
* **Ambiguity Level:**
  * Clear: Situation and expectations are straightforward with little uncertainty.
  * Moderate: Some uncertainty or incomplete information, requiring judgment.
  * High: High uncertainty with unclear information or conflicting signals.
* **Individuals Involved:**
  * Simple: Few people engaged, interactions are straightforward.
  * Moderate: Several people with varying roles or interests are present.
  * Complex: Many individuals involved, with diverse and possibly conflicting needs.
* **Authority Relationships:**
  * Peer Level: Interactions with fellow officers, colleagues, or equal-ranking partners.
  * Subordinate: Interactions with supervisors, training officers, or senior personnel.
  * Authority: Interactions with civilians, suspects, witnesses, or those under your command.
* **Situation Type:**
  * Patrol Traffic Stop: Routine or situational encounters with drivers, often involving vehicle checks, traffic violations, or suspicious behavior.
  * Crime Scene Investigation: Processing, securing, and documenting a scene after a crime has occurred, including evidence collection.
  * Emergency Response: Immediate, time-sensitive incidents such as accidents, natural disasters, or active threats requiring rapid decisions.
  * Administrative Reporting: Non-field tasks like writing reports, handling paperwork, or completing compliance records.
  * Training Supervision: Scenarios involving mentoring, evaluating, or guiding subordinates during training.
  * Inter-Agency Cooperation: Coordinated operations with other agencies (local, state, federal, or specialized units).
  * Mental Health Crises: Encounters with individuals in psychological distress, requiring de-escalation and empathy.
* **Time of Day:**
  * Morning: Early hours, often involving routine checks or follow-up tasks.
  * Afternoon: Midday period with typical public activity and moderate workload.
  * Evening: Later hours with increased incidents related to social activity or nightlife.
  * Night: Overnight period, often lower staffing but higher risk emergencies.
* **Race:**
  * White: Individual identifies as White or of European descent.
  * Black or African American: Individual identifies as Black or African American.
  * Hispanic/Latino: Individual identifies as Hispanic or Latino, of any race.
  * Asian: Individual identifies as Asian, including East, South, or Southeast Asian backgrounds.
  * Native American or Alaska Native: Individual identifies as Indigenous to North America.
  * Pacific Islander: Individual identifies as Native Hawaiian or from other Pacific Islander groups.
  * Other/Multiracial: Individual identifies as multiple races or ethnicities not captured in one category.
  * Unknown: Race not identified or not disclosed.
* **Gender:**
  * Male: Individual identifies as male.
  * Female: Individual identifies as female.
  * Non-Binary: Individual identifies outside the male/female binary.
  * Unknown: Gender not identified or not disclosed.
* **Age:**
  * Juvenile: Child or adolescent, generally under 18 years.
  * Young Adult: Late teens through mid-20s.
  * Adult: Standard adult range, typically 25–39.
  * Middle-Aged: Individuals in their 40s to late 50s.
  * Senior: Older adults, usually 60 years or above.
  * Unknown: Age not identified or not disclosed.

**Attribute Values used to create the SJT:**
* **Urgency Level:** {{urgency_level}}
* **Threat Level:** {{threat_level}}
* **Ambiguity Level:** {{ambiguity_level}}
* **Individuals Involved:** {{individuals_involved}}
* **Authority Relationships:** {{authority_relationships}}
* **Ethical Considerations:** {{ethical_considerations}}
* **Situation Type:** {{situation_type}}
* **Time of Day:** {{time_of_day}}
* **Subject Race:** {{race}}
* **Subject Gender:** {{gender}}
* **Subject Age:** {{age}}

**Rubric Dimensions (rate each 1–5):**
* **Scenario Realism & Plausibility:** Is the scenario realistic and consistent with policing practice?
* **Trait Alignment of Options:** Each option MUST correspond to its intended HEXACO trait in the fixed order (Honesty-Humility → Emotionality → Extraversion → Agreeableness → Conscientiousness → Openness).  
  - If the score is less than 5, also specify for each option which other HEXACO traits it overlaps with.
* **Ethical & Value Tension Representation:** Does the scenario involve meaningful ethical or professional trade-offs?
* **Bias & Fairness Check:** Are demographic or contextual factors presented neutrally (no stereotypes)?

⚠️ Reminder: Trait alignment is evaluated PER OPTION against the HEXACO trait order.

**Output Format**
Provide the complete rubric evaluation as a JSON object with the following schema:

{
  "scenario_realism": {
    "score": X,
    "justification": "Concise reasoning here"
  },
  "trait_alignment": {
    "honesty_humility": {
      "score": X,
      "justification": "Concise reasoning here",
      "overlaps": ["trait1", "trait2"]
    },
    "emotionality": {
      "score": X,
      "justification": "Concise reasoning here",
      "overlaps": []
    },
    "extraversion": {
      "score": X,
      "justification": "Concise reasoning here",
      "overlaps": []
    },
    "agreeableness": {
      "score": X,
      "justification": "Concise reasoning here",
      "overlaps": ["traitY"]
    },
    "conscientiousness": {
      "score": X,
      "justification": "Concise reasoning here",
      "overlaps": []
    },
    "openness": {
      "score": X,
      "justification": "Concise reasoning here",
      "overlaps": []
    }
  },
  "ethical_tension": {
    "score": X,
    "justification": "Concise reasoning here"
  },
  "fairness": {
    "score": X,
    "justification": "Concise reasoning here"
  }
}



Notes:

* Each HEXACO trait is scored individually from 1–5.
* The `"overlaps"` field must be included only if `trait_alignment.score < 5`.
* Use empty lists (`[]`) for traits with no overlap.
* Trait names must come from the HEXACO set: ["honesty_humility", "emotionality", "extraversion", "agreeableness", "conscientiousness", "openness"].
"""


SJT_LLM_JUDGE_EVALUATION_WITHOUT_SEEDS_TEMPLATE_STR = """You are an expert evaluator of situational judgment tests (SJTs).  
Your role is to assess the seed value of each SJT scenario and its response options using a structured rubric.  

You must:  
- Be objective, consistent, and fair.  
- Only use the provided seed categories.  
- Always return results as **valid JSON strictly matching the schema** below.  
- If information is not explicitly provided in the scenario, always output `"Unknown"` with a justification.  
- Provide concise but informative justifications (avoid shallow phrases like “because it is urgent”).  
- Assign confidence scores (0–1) to indicate certainty.  
- For HEXACO traits, allow **multi-trait weighting** when relevant (e.g., 0.7 honesty-humility, 0.3 conscientiousness).  
- Include a meta-evaluation of the rubric clarity.  

---

**Situational Judgment Test:**

Question: {{ question }}

Answer Options:

{{ answer_options }}

---

**Rubric Categories**  
- **Urgency Level:** Low | Medium | High  
- **Threat Level:** Low | Medium | High  
- **Ambiguity Level:** Clear | Moderate | High  
- **Individuals Involved:** Simple | Moderate | Complex  
- **Authority Relationships:** Peer Level | Subordinate | Authority  
- **Situation Type:** Patrol Traffic Stop | Crime Scene Investigation | Emergency Response | Administrative Reporting | Training Supervision | Inter-Agency Cooperation | Mental Health Crises  
- **Time of Day:** Morning | Afternoon | Evening | Night  
- **Race:** White | Black or African American | Hispanic/Latino | Asian | Native American or Alaska Native | Pacific Islander | Other/Multiracial | Unknown  
- **Gender:** Male | Female | Non-Binary | Unknown  
- **Age:** Juvenile | Young Adult | Adult | Middle-Aged | Senior | Unknown  
- **HEXACO Traits:** Honesty-Humility | Emotionality | Extraversion | Agreeableness | Conscientiousness | Openness to Experience  

---

**JSON Schema (strict)**

```json
{
  "urgency_level": {
    "value": "Low | Medium | High",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "threat_level": {
    "value": "Low | Medium | High",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "ambiguity_level": {
    "value": "Clear | Moderate | High",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "individuals_involved": {
    "value": "Simple | Moderate | Complex",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "authority_relationships": {
    "value": "Peer Level | Subordinate | Authority",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "situation_type": {
    "value": "Patrol Traffic Stop | Crime Scene Investigation | Emergency Response | Administrative Reporting | Training Supervision | Inter-Agency Cooperation | Mental Health Crises",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "time_of_day": {
    "value": "Morning | Afternoon | Evening | Night | Unknown",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "race": {
    "value": "White | Black or African American | Hispanic/Latino | Asian | Native American or Alaska Native | Pacific Islander | Other/Multiracial | Unknown",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "gender": {
    "value": "Male | Female | Non-Binary | Unknown",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "age": {
    "value": "Juvenile | Young Adult | Adult | Middle-Aged | Senior | Unknown",
    "confidence": 0.0,
    "justification": "Concise reasoning here"
  },
  "hexaco_traits": {
    "first_option": {
      "values": {
        "Trait1": weight,
        "Trait2": weight
      },
      "confidence": 0.0,
      "justification": "Concise reasoning here"
    },
    "second_option": {
      "values": {
        "Trait1": weight
      },
      "confidence": 0.0,
      "justification": "Concise reasoning here"
    }
    // Continue for all options
  },
  "rubric_quality": {
    "value": "Low | Medium | High",
    "confidence": 0.0,
    "justification": "Was the scenario clear enough to evaluate fairly?"
  }
}

"""