import pandas as pd

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