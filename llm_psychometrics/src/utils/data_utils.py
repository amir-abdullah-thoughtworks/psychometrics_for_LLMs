import pandas as pd
import re
from typing import Iterable, Optional, Dict, Any

def balance_officers_csv(
    in_path: str,
    out_path: str = "balanced_us_police_officers.csv",
    male_ratio: float = 0.86,
    seed: int = 1337,
    dedupe_on: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Load merged_us_police_officers.csv, optionally de-duplicate rows, and rebalance to ~male_ratio : (1-male_ratio)
    by keeping all males and down-sampling females. Prints old/new ratios and saves to out_path.

    Args:
        in_path: Path to input CSV.
        out_path: Path to write balanced CSV (duplicates removed).
        male_ratio: Target male proportion (default 0.86).
        seed: RNG seed for deterministic sampling/shuffle.
        dedupe_on: Column names to consider for duplicate detection. If None:
                   - use ['uuid'] if present; otherwise, use all columns (exact row duplicates).

    Returns:
        Dict with before/after counts/ratios and duplicates_removed.
    """
    if not (0.0 < male_ratio < 1.0):
        raise ValueError("male_ratio must be between 0 and 1 (e.g., 0.86).")

    df = pd.read_csv(in_path)

    # --- De-duplicate ---------------------------------------------------------
    if dedupe_on is None:
        dedupe_on = ["uuid"] if "uuid" in df.columns else None  # fall back to all-cols exact duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=dedupe_on).reset_index(drop=True) if dedupe_on else df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = n_before - len(df)
    print(f"De-duplication: removed {duplicates_removed} duplicate row(s) "
          f"using subset={list(dedupe_on) if dedupe_on else 'ALL_COLUMNS'}.")

    # --- Locate sex/gender column --------------------------------------------
    sex_col = "sex" if "sex" in df.columns else next(
        (c for c in df.columns if re.search(r"gender", c, re.I)), None
    )
    if sex_col is None:
        raise ValueError("No 'sex' or 'gender*' column found in the CSV.")

    # Normalize labels
    labels = df[sex_col].astype(str).str.strip().str.lower()
    male_mask   = labels.isin({"m", "male"})
    female_mask = labels.isin({"f", "female"})

    # --- BEFORE (pre-balance) -------------------------------------------------
    n_m_old = int(male_mask.sum())
    n_f_old = int(female_mask.sum())
    total_old = n_m_old + n_f_old
    old_ratios = {
        "male":   (n_m_old / total_old if total_old else 0.0),
        "female": (n_f_old / total_old if total_old else 0.0),
    }
    print(f"Before balancing (recognized male/female only): male={n_m_old}, female={n_f_old}, total={total_old}")
    print(f"Old ratios -> male: {old_ratios['male']:.3f}, female: {old_ratios['female']:.3f}")

    # --- Build M/F frames for balancing --------------------------------------
    df_m = df[male_mask]
    df_f = df[female_mask]

    if len(df_m) == 0:
        raise ValueError("No male rows found; cannot construct target split by down-sampling females.")

    # Target females to hit desired ratio while keeping all males
    target_f = round(len(df_m) * ((1.0 - male_ratio) / male_ratio))
    target_f = min(target_f, len(df_f))

    df_f_bal = df_f.sample(n=target_f, random_state=seed) if target_f > 0 else df_f.iloc[0:0]

    # Combine and shuffle for neutrality
    df_balanced = pd.concat([df_m, df_f_bal], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save
    df_balanced.to_csv(out_path, index=False)

    # --- AFTER (post-balance) -------------------------------------------------
    n_m_new = len(df_m)
    n_f_new = len(df_f_bal)
    total_new = n_m_new + n_f_new
    new_ratios = {
        "male":   (n_m_new / total_new if total_new else 0.0),
        "female": (n_f_new / total_new if total_new else 0.0),
    }
    print(f"Saved {out_path}")
    print(f"After balancing: male={n_m_new}, female={n_f_new}, total={total_new}")
    print(f"New ratios  -> male: {new_ratios['male']:.3f}, female: {new_ratios['female']:.3f}")

    return {
        "out_path": out_path,
        "duplicates_removed": duplicates_removed,
        "before": {"counts": {"male": n_m_old, "female": n_f_old, "total": total_old}, "ratios": old_ratios},
        "after":  {"counts": {"male": n_m_new, "female": n_f_new, "total": total_new}, "ratios": new_ratios},
    }

# Example:
# stats = balance_officers_csv("merged_us_police_officers.csv", "balanced_us_police_officers.csv")
# stats = balance_officers_csv("merged_us_police_officers.csv", dedupe_on=["uuid"])  # explicit
