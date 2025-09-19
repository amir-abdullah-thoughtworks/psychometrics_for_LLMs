"""
ab_seed_builder.py

Single-class utility that:
- Loads your YAML
- Expands ONLY AppearanceCategories and BehaviorCategories (to N items each)
- Leaves all other top-level keys (e.g., MemoirSeeds) completely unchanged
- Writes populated_<filename>.yaml

Usage:
    export OPENAI_API_KEY=sk-...
    python ab_seed_builder.py /path/to/police_persona_seed.yaml

Optional env:
    OPENAI_MODEL=gpt-4o-mini
    TARGET_COUNT=20
    MIN_WORDS=4
    MAX_WORDS=10
    OPENAI_TEMPERATURE=1.2
    OPENAI_TOP_P=0.98
    MAX_RETRIES=3
    RETRY_BACKOFF=1.5
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from openai import OpenAI


class ABPersonaSeedBuilder:
    """
    One class to:
      - load/save YAML
      - build few-shot prompts
      - call OpenAI (JSON-only)
      - validate & dedupe items
      - expand ONLY AppearanceCategories and BehaviorCategories to target count
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        target_count: int = 20,
        min_words: int = 4,
        max_words: int = 10,
        temperature: float = 1.2,
        top_p: float = 0.98,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
    ):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        self.target_count = int(os.environ.get("TARGET_COUNT", target_count))
        self.min_words = int(os.environ.get("MIN_WORDS", min_words))
        self.max_words = int(os.environ.get("MAX_WORDS", max_words))
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", temperature))
        self.top_p = float(os.environ.get("OPENAI_TOP_P", top_p))
        self.max_retries = int(os.environ.get("MAX_RETRIES", max_retries))
        self.retry_backoff_s = float(os.environ.get("RETRY_BACKOFF", retry_backoff_s))

        # The ONLY sections this class will expand:
        self.sections = ("AppearanceCategories", "BehaviorCategories")

        # Light guardrails for neutral/non-identitarian content
        self._banned_identity = re.compile(
            r"\b(race|religion|ethnicity|catholic|jewish|muslim|gay|straight|lgbt|asian|black|white|latino)\b",
            re.I,
        )
        self._banned_fetish = re.compile(
            r"\b(fetish|bloodlust|arsenal|assault rifle|massacre)\b", re.I
        )

    # ---------- I/O ----------

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_yaml(self, data: Dict[str, Any], input_path: Path) -> Path:
        out_path = input_path.parent / f"populated_{input_path.name}"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"✅ Written: {out_path}")
        return out_path

    # ---------- Utils ----------

    @staticmethod
    def _normalize(s: str) -> str:
        s = " ".join(s.split()).strip()
        if s.endswith("."):
            s = s[:-1].strip()
        return s

    @staticmethod
    def _word_count(s: str) -> int:
        return len([w for w in s.split() if w])

    def _valid_item(self, s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        if self._banned_identity.search(s):
            return False
        if self._banned_fetish.search(s):
            return False
        wc = self._word_count(s)
        return self.min_words <= wc <= self.max_words

    def _few_shot_context(self, categories: Dict[str, List[str]], max_cats: int = 4, k: int = 5) -> str:
        """Take up to 4 categories × 5 examples each from same section."""
        parts = []
        for cat, seeds in categories.items():
            if seeds:
                sample = "\n".join(f"- {self._normalize(x)}" for x in seeds[:k])
                parts.append(f"{cat}:\n{sample}")
                if len(parts) >= max_cats:
                    break
        return "\n".join(parts) if parts else "(no exemplars available)"

    def _build_prompt(self, section: str, category: str, existing: List[str], need: int, few_shot: str) -> str:
        rules = (
            "You generate SHORT, concrete seeds for police officer personas.\n"
            f"- Each item is {self.min_words}–{self.max_words} words; one concise phrase.\n"
            "- Avoid psychometric/diagnostic labels, stereotypes, or slurs.\n"
            "- Avoid explicit identity attributes (race, religion, etc.).\n"
            "- Neutral, professional tone; avoid weapon fetishization.\n"
            "- Do NOT repeat or trivially vary earlier items.\n"
            f"- Return ONLY a JSON array of strings, length exactly {need}.\n"
        )
        ctx = (
            f"\nFew-shot exemplars from this section:\n{few_shot}\n\n"
            f"Target: {section} → {category}\n"
            f"Existing ({len(existing)}): {json.dumps(existing, ensure_ascii=False)}\n"
            f"Generate {need} NEW, distinct items."
        )
        return rules + ctx

    def _call_openai_json(self, prompt: str) -> List[str]:
        """Call OpenAI with JSON-only instruction; retry with backoff."""
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    messages=[
                        {"role": "system", "content": "Respond ONLY with a JSON array of strings. No prose."},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = resp.choices[0].message.content.strip()
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("Expected a JSON list.")
                return [self._normalize(x) for x in data if isinstance(x, str)]
            except Exception as e:
                last_err = e
                wait = self.retry_backoff_s * attempt
                print(f"⚠️ OpenAI call failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} attempts: {last_err}")

    def _clean_items(self, candidates: List[str], existing_pool: List[str]) -> List[str]:
        """Normalize, validate, and dedupe against existing_pool."""
        pool_norm = {self._normalize(x).lower() for x in existing_pool if isinstance(x, str)}
        out, seen = [], set()
        for s in candidates:
            s = self._normalize(s)
            if not self._valid_item(s):
                continue
            key = s.lower()
            if key in seen or key in pool_norm:
                continue
            seen.add(key)
            out.append(s)
        return out

    # ---------- Expansion (ONLY these sections) ----------

    def _expand_section(self, seeds_root: Dict[str, Any], section_key: str) -> bool:
        """Expand each category in a section to target_count, leave others untouched."""
        section = seeds_root.get(section_key, {})
        if not isinstance(section, dict):
            return False

        changed = False
        # Build a section-wide pool to discourage cross-category duplicates
        section_pool: List[str] = []
        for vals in section.values():
            if vals:
                section_pool.extend(vals)

        few_shot = self._few_shot_context(section)

        for category, current in section.items():
            current = current or []
            if len(current) >= self.target_count:
                continue

            need = self.target_count - len(current)
            prompt = self._build_prompt(section_key, category, current, need, few_shot)

            accepted: List[str] = []
            tries = 0
            while len(accepted) < need and tries < self.max_retries * 2:
                tries += 1
                proposed = self._call_openai_json(prompt)
                cleaned = self._clean_items(proposed, existing_pool=section_pool + current + accepted)
                accepted.extend(cleaned)
                if len(accepted) < need:
                    prompt = self._build_prompt(section_key, category, current + accepted, need - len(accepted), few_shot)

            if accepted:
                section[category] = current + accepted[:need]
                section_pool.extend(accepted[:need])
                changed = True
                print(f"✓ {section_key} → {category}: +{len(accepted[:need])} (total {len(section[category])})")
            else:
                print(f"… {section_key} → {category}: no valid additions (kept {len(current)})")

        seeds_root[section_key] = section
        return changed

    # ---------- Public API ----------

    def build(self, yaml_path: str | Path) -> Path:
        """Expand ONLY AppearanceCategories and BehaviorCategories; never modify other keys."""
        yaml_path = Path(yaml_path).expanduser().resolve()
        data = self._load_yaml(yaml_path)

        # Work strictly inside top-level PoliceOfficerPersonaSeeds
        root = data.get("PoliceOfficerPersonaSeeds")
        if not isinstance(root, dict):
            raise ValueError("Top-level key 'PoliceOfficerPersonaSeeds' must be a mapping.")

        any_changes = False
        for section_key in self.sections:
            if section_key in root:
                print(f"▶ Expanding section: {section_key}")
                any_changes |= self._expand_section(root, section_key)
            else:
                print(f"ℹ️ Section '{section_key}' not present; skipping.")

        # Write back ONLY the modified sections to the same root
        data["PoliceOfficerPersonaSeeds"] = root
        return self._save_yaml(data, yaml_path)


# ---------- CLI ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ab_seed_builder.py /path/to/police_persona_seed.yaml")
        sys.exit(2)

    builder = ABPersonaSeedBuilder()
    out = builder.build(sys.argv[1])
    print(f"Done → {out}")
