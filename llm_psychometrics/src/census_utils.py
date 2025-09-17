import pandas as pd


def load_state_age_distribution_with_gender(csv_path: str, year: int = 2024, min_age: int = 21, max_age: int = 65):
    """
    Returns:
        dict: {
            state: {
                "gender": {"male": x, "female": 1-x},  # x weighted by total male/female pop in [min_age, max_age]
                "male":   {age: p_age_given_male, ...}, # normalized over ages min_age..max_age
                "female": {age: p_age_given_female, ...}
            },
            ...
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
    for state, g_state in df.groupby("NAME"):
        out = {}

        # totals by sex for gender weighting
        totals_by_sex = g_state.groupby("SEX")[col].sum()
        male_total = float(totals_by_sex.get("male", 0.0))
        female_total = float(totals_by_sex.get("female", 0.0))
        denom = male_total + female_total if (male_total + female_total) > 0 else 1.0
        male_prop = male_total / denom
        out["gender"] = {"male": male_prop, "female": 1.0 - male_prop}

        # conditional age distributions by sex
        for sex, g_sex in g_state.groupby("SEX"):
            # Normalize across ages to make a probability distribution p(age | sex, state)
            age_probs = (g_sex.set_index("AGE")[col] / g_sex[col].sum()).to_dict()
            out[sex] = age_probs

        result[state] = out

    return result