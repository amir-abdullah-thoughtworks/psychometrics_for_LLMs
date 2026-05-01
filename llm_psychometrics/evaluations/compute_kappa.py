#!/usr/bin/env python3
"""Compute Cohen's kappa between Human, GPT, and Claude raters for all rubric fields."""
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

EVAL_DIR = Path(__file__).parent

with open(EVAL_DIR / "human_persona_reviews.json") as f:
    human = json.load(f)
with open(EVAL_DIR / "open_ai_persona_reviews.json") as f:
    gpt = json.load(f)
with open(EVAL_DIR / "anthropic_persona_reviews.json") as f:
    claude = json.load(f)

common = sorted(set(human) & set(gpt) & set(claude))
SKIP = {"uuid", "persona_string", "persona_hash", "name"}
fields = [k for k in human[common[0]] if k not in SKIP]

def ratings(data, uuids, field):
    out = []
    for uid in uuids:
        v = data[uid].get(field)
        try:
            out.append(int(float(v)))
        except (TypeError, ValueError):
            out.append(None)
    return out

def safe_kappa(a, b):
    try:
        return cohen_kappa_score(a, b)
    except Exception:
        return float("nan")

rows = []
for field in fields:
    h = ratings(human, common, field)
    g = ratings(gpt, common, field)
    c = ratings(claude, common, field)
    triples = [(hv, gv, cv) for hv, gv, cv in zip(h, g, c) if all(x is not None for x in (hv, gv, cv))]
    hv, gv, cv = zip(*triples)
    rows.append((field, safe_kappa(hv, gv), safe_kappa(hv, cv), safe_kappa(gv, cv), len(triples)))

rows.sort(key=lambda x: x[2], reverse=True)

print(f"{'Field':<40} {'H vs GPT':>10} {'H vs Claude':>11} {'GPT vs Cla':>10} {'n':>5}")
print("-" * 80)
for field, k_hg, k_hc, k_gc, n in rows:
    print(f"{field:<40} {k_hg:>10.3f} {k_hc:>11.3f} {k_gc:>10.3f} {n:>5}")
