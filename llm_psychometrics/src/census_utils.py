
import csv
import os
import re
import pandas as pd

from collections import defaultdict, Counter
from typing import Dict, Tuple, Iterable
from joblib import Memory
from dataclasses import dataclass

from tqdm import tqdm

@dataclass
class PoliceDemographics:
    male_ratio: float = 0.86
    female_ratio: float = 0.14

def _folder_signature(folder: str) -> Tuple[Tuple[str, int, int], ...]:
    """
    Simple signature: tuple of (filename, mtime, size) for all *.txt files in folder.
    If any file changes, the signature changes and the cache recomputes.
    """
    sig = []
    for fn in sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt")):
        p = os.path.join(folder, fn)
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        sig.append((fn, int(st.st_mtime), int(st.st_size)))
    return tuple(sig)

def build_first_name_distributions(
    folder: str = "first_names_by_states",
    current_year: int = 2025,
    cache_dir = None,
) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """
    Persistent-cached build:
      (state, gender, age) -> {first_name: probability, ...}

    Lines look like:
        STATE,GENDER,BIRTH_YEAR,FIRST_NAME,COUNT
        AZ,F,1910,Mary,74
    """
    if cache_dir is None:
        cache_dir = os.path.join(folder, ".cache_joblib")
    memory = Memory(cache_dir, verbose=0)

    @memory.cache
    def _compute(folder: str, current_year: int, signature: Iterable[Tuple[str, int, int]]):
        counts = defaultdict(Counter)  # (state, gender, age) -> Counter(name -> count)
        for fname in tqdm(os.listdir(folder)):
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(folder, fname)

            # Try UTF-8 then latin-1
            for enc in ("utf-8", "latin-1"):
                try:
                    with open(path, "r", encoding=enc, newline="") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) < 5:
                                continue
                            state = row[0].strip().upper()
                            gender = row[1].strip().upper()
                            y = row[2].strip()
                            name = row[3].strip()
                            c = row[4].strip()
                            try:
                                birth_year = int(y)
                                count = int(c)
                            except ValueError:
                                continue
                            if gender not in ("M", "F"):
                                continue
                            age = current_year - birth_year
                            if age < 0:
                                continue
                            counts[(state, gender, age)][name] += count
                    break
                except UnicodeDecodeError:
                    continue

        # Normalize to probabilities
        dists: Dict[Tuple[str, str, int], Dict[str, float]] = {}
        for key, name_counts in counts.items():
            total = sum(name_counts.values())
            if total > 0:
                dists[key] = {n: c / total for n, c in name_counts.items()}
        return dists

    signature = _folder_signature(folder)
    return _compute(folder, current_year, signature)


def locations_distribution_by_state(
        csv_path: str,
        year: int = 2024,
        location_sumlev: int = 162,  # 162 = Places; change to 050 for counties if desired
):
    """
    Return a nested dict of per-state location distributions (dropping 'United States' and state aggregates).

    Output shape:
    {
        "Utah": {"Salt Lake City": 0.0662, "West Valley City": 0.0420, ...},
        "California": {"Los Angeles": 0.093..., "San Diego": ...},
        ...
    }
    Probabilities for each state sum to 1 across its locations.
    """
    # Robust read (Census files often need latin-1 + python engine)
    df = pd.read_csv(csv_path, encoding="latin-1", engine="python")

    # Drop national aggregate if present
    if "STNAME" in df.columns:
        df = df[df["STNAME"].ne("United States")]
    if "NAME" in df.columns:
        df = df[df["NAME"].ne("United States")]

    # Keep only desired geography level (drop state/county aggregates)
    if "SUMLEV" in df.columns:
        df = df[df["SUMLEV"] == location_sumlev]

    # Choose population column for requested year; fallback to latest available
    pop_col = f"POPESTIMATE{year}"
    if pop_col not in df.columns:
        pop_cols = sorted([c for c in df.columns if c.startswith("POPESTIMATE")])
        if not pop_cols:
            raise ValueError("No POPESTIMATE* columns found.")
        pop_col = pop_cols[-1]

    # Clean trailing legal designations from place names (e.g., "Salt Lake City city" -> "Salt Lake City")
    suffixes = [
        r"city and borough", r"charter township", r"charter town",
        r"metropolitan government", r"consolidated government",
        r"metro government", r"urban county", r"city-county consolidated government",
        r"municipality", r"borough", r"village", r"plantation", r"town", r"city", r"cdp", r"balance"
    ]
    suffix_pattern = re.compile(r"\s+(?:" + "|".join(suffixes) + r")$", flags=re.IGNORECASE)
    balance_paren_pattern = re.compile(r"\s*\(balance\)$", flags=re.IGNORECASE)

    def clean_place(name: str) -> str:
        s = str(name).strip()
        # Remove "(balance)" if present
        s = balance_paren_pattern.sub("", s)
        # Iteratively strip trailing legal suffixes
        while suffix_pattern.search(s):
            s = suffix_pattern.sub("", s).strip()
        return s

    df["clean_place"] = df["NAME"].apply(clean_place)

    # Aggregate by state and cleaned place name (in case multiple rows roll up to the same cleaned name)
    grouped = (
        df.groupby(["STNAME", "clean_place"], as_index=False)[pop_col]
        .sum()
        .rename(columns={pop_col: "weight"})
    )

    # Build per-state normalized distributions
    result = {}
    for state, g in grouped.groupby("STNAME"):
        total = g["weight"].sum()
        dist = {row["clean_place"]: (row["weight"] / total if total > 0 else 0.0)
                for _, row in g.iterrows()}
        result[state] = dist

    return result
# Show top 10 for Utah and California as a quick sanity check
def top_k(d, k=10):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])

def load_state_age_distribution_full(csv_path: str, year: int = 2024, min_age: int = 21, max_age: int = 65):
    """
    Returns:
        dict with per-state data and a 'state_distribution' entry:
        {
            state: {
                "gender": {"male": x, "female": 1-x},
                "total_weight": total_pop_in_band,
                "male":   {age: p_age_given_male, ...},
                "female": {age: p_age_given_female, ...}
            },
            ...
            "state_distribution": {state: p_state, ...}  # proportional to total_weight
        }
    """
    df = pd.read_csv(csv_path)
    col = f"POPEST{year}_CIV"
    df = df[["NAME", "SEX", "AGE", col]]

    # Restrict ages
    df = df[(df["AGE"] >= min_age) & (df["AGE"] <= max_age)]

    # Keep male(1) and female(2)
    sex_map = {1: "male", 2: "female"}
    df = df[df["SEX"].isin(sex_map.keys())].copy()
    df["SEX"] = df["SEX"].map(sex_map)

    result = {}
    state_weights = {}
    for state, g_state in df.groupby("NAME"):
        if state == 'United States':
            continue

        out = {}

        # totals by sex for gender weighting
        totals_by_sex = g_state.groupby("SEX")[col].sum()
        male_total = float(totals_by_sex.get("male", 0.0))
        female_total = float(totals_by_sex.get("female", 0.0))
        denom = male_total + female_total if (male_total + female_total) > 0 else 1.0
        male_prop = male_total / denom
        out["gender"] = {"male": male_prop, "female": 1.0 - male_prop}
        out["total_weight"] = male_total + female_total
        state_weights[state] = out["total_weight"]

        # conditional age distributions by sex
        for sex, g_sex in g_state.groupby("SEX"):
            age_probs = (g_sex.set_index("AGE")[col] / g_sex[col].sum()).to_dict()
            out[sex] = age_probs

        result[state] = out

    # Add state_distribution key
    total_all = sum(state_weights.values())
    if total_all > 0:
        state_distribution = {s: w / total_all for s, w in state_weights.items()}
    else:
        state_distribution = {s: 0 for s in state_weights}
    result["state_distribution"] = state_distribution

    return result

def location_distribution_for_state(
    csv_path: str,
    state,  # state name (e.g., "Utah") or numeric FIPS code (e.g., 49)
    year: int = 2024,
    location_sumlev: int = 162,  # 162 = Places (cities/towns). Use 050 for counties, etc.
):
    """
    Return a probability distribution over *locations within a given state*,
    proportional to POPESTIMATE{year} for the chosen geography level.
    """
    df = pd.read_csv(csv_path, encoding="latin-1", engine="python")
    if "STNAME" in df.columns:
        df = df[df["STNAME"].ne("United States")]
    if "NAME" in df.columns:
        df = df[df["NAME"].ne("United States")]
    if "SUMLEV" in df.columns:
        df = df[df["SUMLEV"] == location_sumlev]
    if isinstance(state, int):
        df_state = df[df["STATE"] == state]
    else:
        df_state = df[df["STNAME"].str.casefold() == str(state).casefold()]
    if df_state.empty:
        raise ValueError("No rows found for the requested state and SUMLEV.")
    pop_col = f"POPESTIMATE{year}"
    if pop_col not in df_state.columns:
        pop_cols = sorted([c for c in df_state.columns if c.startswith("POPESTIMATE")])
        if not pop_cols:
            raise ValueError("No POPESTIMATE* columns found in the file.")
        pop_col = pop_cols[-1]
    grouped = df_state.groupby("NAME", as_index=False)[pop_col].sum().rename(columns={pop_col: "weight"})
    total = grouped["weight"].sum()
    dist = {row["NAME"]: (row["weight"] / total if total > 0 else 0.0) for _, row in grouped.iterrows()}
    return dist