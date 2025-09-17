import pandas as pd


def load_state_age_distribution(csv_path: str, year: int = 2024, min_age: int = 21, max_age: int = 65):
    """
    Parse a census-like CSV into a dictionary of the form:
    {
        state: {
            "male": {age: probability, ...},
            "female": {age: probability, ...}
        }
    }

    Args:
        csv_path (str): Path to the CSV file.
        year (int): Which year column to use (default 2024).
        min_age (int): Minimum age to include (default 21).
        max_age (int): Maximum age to include (default 65).

    Returns:
        dict: Nested dictionary of state -> sex -> {age: probability}
    """
    df = pd.read_csv(csv_path)
    col = f"POPEST{year}_CIV"

    # keep relevant columns
    df = df[["NAME", "SEX", "AGE", col]]

    # filter ages
    df = df[(df["AGE"] >= min_age) & (df["AGE"] <= max_age)]

    # map sex codes
    sex_map = {1: "male", 2: "female"}
    df = df[df["SEX"].isin(sex_map.keys())]
    df["SEX"] = df["SEX"].map(sex_map)

    result = {}
    for state, group in df.groupby("NAME"):
        result[state] = {}
        for sex, g in group.groupby("SEX"):
            probs = (g.set_index("AGE")[col] / g[col].sum())
            result[state][sex] = probs.to_dict()

    return result
