import pandas as pd
import re

def balance_officers_csv(
    in_path: str,
    out_path: str = "balanced_us_police_officers.csv",
    male_ratio: float = 0.86,
    seed: int = 1337,
) -> dict:
    """
    Rebalance merged_us_police_officers.csv to ~male_ratio : (1-male_ratio) by
    keeping all males and down-sampling females. Saves to out_path.

    Returns a dict with counts and achieved ratios.
    """
    if not (0.0 < male_ratio < 1.0):
        raise ValueError("male_ratio must be between 0 and 1 (e.g., 0.86).")

    df = pd.read_csv(in_path)

    # Locate sex/gender column
    sex_col = "sex" if "sex" in df.columns else next(
        (c for c in df.columns if re.search(r"gender", c, re.I)), None
    )
    if sex_col is None:
        raise ValueError("No 'sex' or 'gender*' column found in the CSV.")

    # Normalize labels
    labels = df[sex_col].astype(str).str.strip().str.lower()
    male_mask   = labels.isin({"m", "male"})
    female_mask = labels.isin({"f", "female"})

    df_m = df[male_mask]
    df_f = df[female_mask]

    m = len(df_m)
    if m == 0:
        raise ValueError("No male rows found; cannot construct an 86/14 split by down-sampling females.")

    # Target # of females to hit desired ratio when keeping all males
    target_f = round(m * ((1.0 - male_ratio) / male_ratio))
    target_f = min(target_f, len(df_f))

    df_f_bal = df_f.sample(n=target_f, random_state=seed) if target_f > 0 else df_f.iloc[0:0]

    # Combine and (optionally) shuffle for neutrality
    df_balanced = pd.concat([df_m, df_f_bal], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save
    df_balanced.to_csv(out_path, index=False)

    # Report
    n_m = len(df_m)
    n_f = len(df_f_bal)
    total = n_m + n_f
    ratios = {"male": (n_m / total if total else 0.0),
              "female": (n_f / total if total else 0.0)}

    return {
        "out_path": out_path,
        "counts": {"male": n_m, "female": n_f, "total": total},
        "ratios": ratios,
    }