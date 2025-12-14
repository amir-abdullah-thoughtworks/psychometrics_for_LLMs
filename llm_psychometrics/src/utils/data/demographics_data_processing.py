import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, Iterable, Optional

def balance_officers_csv(
    in_path: str,
    male_ratio: float = 0.86,
    seed: int = 1337,
    dedupe_exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    - Loads merged_us_police_officers.csv from `in_path`
    - Removes duplicate rows while specifically OMITTING any 'uuid' column from the dedupe key
    - Rebalances to ~male_ratio : (1 - male_ratio) by keeping all males and down-sampling females
    - Writes 'balanced_us_police_officers.csv' to the SAME FOLDER as `in_path`
    - Prints old/new ratios and returns a small stats dict
    """
    if not (0.0 < male_ratio < 1.0):
        raise ValueError("male_ratio must be between 0 and 1 (e.g., 0.86).")

    in_path = Path(in_path)
    out_path = in_path.with_name("balanced_us_police_officers.csv")

    df = pd.read_csv(in_path)

    # ----------------- De-duplicate (omit any 'uuid' column) -----------------
    cols = list(df.columns)
    # Always exclude any column named exactly 'uuid' (case-insensitive)
    exclude = {c for c in cols if c.lower() == "uuid"}
    # Allow user to exclude more columns if desired
    if dedupe_exclude:
        exclude |= {c for c in cols if c in dedupe_exclude}

    subset = [c for c in cols if c not in exclude]
    n_before = len(df)
    if subset:
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        subset_label = f"SUBSET({', '.join(subset)})"
    else:
        subset_label = "SKIPPED (no columns left after excluding 'uuid')"
    duplicates_removed = n_before - len(df)
    print(f"De-duplication: removed {duplicates_removed} row(s) using {subset_label}.")

    # ----------------- Locate sex/gender column (case-insensitive) ----------
    # Prefer a 'sex' column; otherwise any column containing 'gender'
    lower_map = {c.lower(): c for c in df.columns}
    if "sex" in lower_map:
        sex_col = lower_map["sex"]
    else:
        candidates = [c for c in df.columns if re.search(r"gender", c, re.I)]
        sex_col = candidates[0] if candidates else None
    if sex_col is None:
        raise ValueError("No 'sex' or 'gender*' column found in the CSV.")

    # Normalize labels
    labels = df[sex_col].astype(str).str.strip().str.lower()
    male_mask   = labels.isin({"m", "male"})
    female_mask = labels.isin({"f", "female"})

    # ----------------- BEFORE (pre-balance) ratios ---------------------------
    n_m_old = int(male_mask.sum())
    n_f_old = int(female_mask.sum())
    total_old = n_m_old + n_f_old
    old_ratios = {
        "male":   (n_m_old / total_old if total_old else 0.0),
        "female": (n_f_old / total_old if total_old else 0.0),
    }
    print(f"Before balancing (recognized male/female only): male={n_m_old}, female={n_f_old}, total={total_old}")
    print(f"Old ratios -> male: {old_ratios['male']:.3f}, female: {old_ratios['female']:.3f}")

    # ----------------- Balance (keep all males, down-sample females) --------
    df_m = df[male_mask]
    df_f = df[female_mask]

    if len(df_m) == 0:
        raise ValueError("No male rows found; cannot construct target split by down-sampling females.")

    target_f = round(len(df_m) * ((1.0 - male_ratio) / male_ratio))
    target_f = min(target_f, len(df_f))
    df_f_bal = df_f.sample(n=target_f, random_state=seed) if target_f > 0 else df_f.iloc[0:0]

    df_balanced = pd.concat([df_m, df_f_bal], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ----------------- Save to SAME FOLDER as input --------------------------
    df_balanced.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    # ----------------- AFTER (post-balance) ratios ---------------------------
    n_m_new = len(df_m)
    n_f_new = len(df_f_bal)
    total_new = n_m_new + n_f_new
    new_ratios = {
        "male":   (n_m_new / total_new if total_new else 0.0),
        "female": (n_f_new / total_new if total_new else 0.0),
    }
    print(f"After balancing: male={n_m_new}, female={n_f_new}, total={total_new}")
    print(f"New ratios  -> male: {new_ratios['male']:.3f}, female: {new_ratios['female']:.3f}")

    return {
        "out_path": str(out_path),
        "duplicates_removed": duplicates_removed,
        "dedupe_subset_used": subset if subset else [],
        "before": {"counts": {"male": n_m_old, "female": n_f_old, "total": total_old}, "ratios": old_ratios},
        "after":  {"counts": {"male": n_m_new, "female": n_f_new, "total": total_new}, "ratios": new_ratios},
    }

# Example:
# stats = balance_officers_csv("path/to/merged_us_police_officers.csv")
# print(stats)
