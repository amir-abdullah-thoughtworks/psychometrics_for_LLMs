import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import login
from scipy.spatial.distance import jensenshannon


# ── Helper Functions ──────────────────────────────────────────────────────────
def write_to_json(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(file, f, indent=2)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_hf_dataset(dataset_path, name="analysis", split="train"):
    dataset = load_dataset(dataset_path, name=name)
    return dataset[split].to_pandas()


def compute_js_stability(trait_dist):
    js_scores = {}
    for persona, g in trait_dist.groupby(level=0):
        dist = g.values
        pairwise = []
        for i in range(len(dist)):
            for j in range(i + 1, len(dist)):
                js = jensenshannon(dist[i], dist[j]) ** 2
                pairwise.append(js)
        js_scores[persona] = np.mean(pairwise)
    return pd.Series(js_scores, name="js_divergence")


def compute_icc(stats_series: pd.Series) -> pd.DataFrame:
    """Compute ICC2 for a per-persona, per-iter stats Series."""
    dist_df = stats_series.reset_index()
    icc_results = []
    for trait in [stats_series.name]:
        df_trait = dist_df[["persona_uuid", "iter", trait]].rename(columns={trait: "score"})
        icc = pg.intraclass_corr(
            data=df_trait,
            targets="persona_uuid",
            raters="iter",
            ratings="score",
        )
        icc2 = icc[icc["Type"] == "ICC(A,1)"]["ICC"].values[0]
        icc_results.append({"trait": trait, "ICC": icc2})
    return pd.DataFrame(icc_results)


def get_z_score_summary_df(base_df: pd.DataFrame, persona_df: pd.DataFrame, option_cols: List[str]) -> pd.DataFrame:
    df_list = []
    for col in option_cols:
        z_score = (base_df[col].item() - persona_df[col].mean()) / persona_df[col].std()
        df_list.append(
            {
                "col": col,
                "persona_mean": np.round(persona_df[col].mean(), 3).item(),
                "persona_std": np.round(persona_df[col].std(), 3).item(),
                "base_score": np.round(base_df[col].item(), 3).item(),
                "z_score": np.round(-z_score, 3).item(),
            }
        )
    return pd.DataFrame(df_list)


# ── Dataset Configuration ─────────────────────────────────────────────────────
RESPONSE_DATASET_PATH = "thoughtworks/gemma_psychometrics_personas_responses"

PERSONA_EMOBENCH_RESPONSE_DATASET_CONFIG = "analysis_emo_bench"
BASE_EMOBENCH_RESPONSE_DATASET_CONFIG = "base_emo_bench"

PERSONA_TRUTHFULQA_RESPONSE_DATASET_CONFIG = "analysis_truthfulqa_mc"
BASE_TRUTHFULQA_RESPONSE_DATASET_CONFIG = "base_truthfulqa_mc"

# Commented out — not used in current analysis
# PERSONA_SJT_RESPONSE_DATASET_CONFIG = "analysis_sjt"
# BASE_SJT_RESPONSE_DATASET_CONFIG = "base_sjt"
# PERSONA_ADVBENCH_RESPONSE_DATASET_CONFIG = "analysis_advbench"
# BASE_ADVBENCH_RESPONSE_DATASET_CONFIG = "base_advbench"

PERSONA_DATASET_PATH = "thoughtworks/psychometric_personas"
PERSONA_DATASET_CONFIG = "analysis"

RESULTS_DIR = "../../experiment_results/analysis_2026MarARR/"
ANON_DATA_DIR = "../../annonymous_data/"

ARCHETYPE_VALS = [
    "The Problem Solver / Public Servant",
    "The Enforcer (Crime-Fighter)",
    "The Avoider (Unconfident Officer)",
    "The Professional (Service-Oriented Officer)",
    "The Avoider (Lazy Officer)",
    "The Problem Solver / Investigator",
    "The Tough Cop (Authoritarian)",
    "The Reciprocator (Nice Cop)",
]


# ── Data Loading ──────────────────────────────────────────────────────────────
login(token=os.environ.get("HF_TOKEN", ""))

personas = get_hf_dataset(PERSONA_DATASET_PATH, PERSONA_DATASET_CONFIG)
# personas.to_csv(os.path.join(ANON_DATA_DIR, "persona_data.csv"), index=False)

sjts = get_hf_dataset("thoughtworks/psychometric_sjts_analysis", "analysis")
# sjts.to_csv(os.path.join(ANON_DATA_DIR, "sjt_data.csv"), index=False)

emobench_persona_responses = get_hf_dataset(RESPONSE_DATASET_PATH, PERSONA_EMOBENCH_RESPONSE_DATASET_CONFIG)
truthfulqa_persona_responses = get_hf_dataset(RESPONSE_DATASET_PATH, PERSONA_TRUTHFULQA_RESPONSE_DATASET_CONFIG)

# Base response datasets — uncomment to load when available
# emobench_base_responses = get_hf_dataset(RESPONSE_DATASET_PATH, BASE_EMOBENCH_RESPONSE_DATASET_CONFIG)
# truthfulqa_base_responses = get_hf_dataset(RESPONSE_DATASET_PATH, BASE_TRUTHFULQA_RESPONSE_DATASET_CONFIG)
# sjt_persona_responses = get_hf_dataset(RESPONSE_DATASET_PATH, PERSONA_SJT_RESPONSE_DATASET_CONFIG)
# sjt_base_responses = get_hf_dataset(RESPONSE_DATASET_PATH, BASE_SJT_RESPONSE_DATASET_CONFIG)


# ── Section: EmoBench — Stability & ICC ───────────────────────────────────────
emobench_persona_stats = emobench_persona_responses.groupby(["persona_uuid", "iter"])["is_correct"].mean()

print("EmoBench JS Stability (mean):", compute_js_stability(emobench_persona_stats).mean())

emobench_icc_results = compute_icc(emobench_persona_stats)
print("\nEmoBench ICC Results:")
print(emobench_icc_results)


# ── Section: EmoBench — Persona Accuracy from Responses ──────────────────────
emobench_persona_dist = pd.crosstab(
    [emobench_persona_responses["persona_uuid"], emobench_persona_responses["question_hash"]],
    emobench_persona_responses["is_correct"],
).reset_index()
emobench_persona_dist.columns = ["persona_uuid", "question_hash", "False", "True"]

option_cols = ["True", "False"]
emobench_persona_dist["most_frequent"] = emobench_persona_dist[option_cols].idxmax(axis=1)
emobench_persona_dist["answer_num"] = emobench_persona_dist["most_frequent"].map({"True": 1, "False": 0})

persona_emobench_accuracy = (
    emobench_persona_dist.groupby("persona_uuid")["answer_num"]
    .mean()
    .reset_index(name="emobench_proportion_correct")
)
persona_emobench_accuracy["emobench_rank"] = (
    persona_emobench_accuracy["emobench_proportion_correct"].rank(method="dense", ascending=False).astype(int)
)
# persona_emobench_accuracy.to_csv(os.path.join(RESULTS_DIR, "persona500_emobench_rank.csv"), index=False)

# Base emobench accuracy — requires emobench_base_responses to be loaded
# emobench_base_dist = pd.crosstab(
#     [emobench_base_responses["persona_uuid"], emobench_base_responses["question_hash"]],
#     emobench_base_responses["is_correct"],
# ).reset_index()
# emobench_base_dist.columns = ["persona_uuid", "question_hash", "False", "True"]
# emobench_base_dist["most_frequent"] = emobench_base_dist[option_cols].idxmax(axis=1)
# emobench_base_dist["answer_num"] = emobench_base_dist["most_frequent"].map({"True": 1, "False": 0})
# base_emobench_accuracy = (
#     emobench_base_dist.groupby("persona_uuid")["answer_num"]
#     .mean()
#     .reset_index(name="emobench_proportion_correct")
# )
# base_emobench_accuracy["emobench_rank"] = (
#     base_emobench_accuracy["emobench_proportion_correct"].rank(method="dense", ascending=False).astype(int)
# )
# base_emobench_accuracy.to_csv(os.path.join(RESULTS_DIR, "base_emobench_rank.csv"), index=False)


# ── Section: TruthfulQA — Stability & ICC ────────────────────────────────────
truthfulqa_persona_stats = (
    truthfulqa_persona_responses.groupby(["persona_uuid", "iter"])["is_correct_canonical"].mean()
)

print("\nTruthfulQA JS Stability (mean):", compute_js_stability(truthfulqa_persona_stats).mean())

truthfulqa_icc_results = compute_icc(truthfulqa_persona_stats)
print("\nTruthfulQA ICC Results:")
print(truthfulqa_icc_results)

# TruthfulQA accuracy from responses — requires truthfulqa_base_responses and crosstab persona dist
# truthfulqa_persona_dist = pd.crosstab(
#     [truthfulqa_persona_responses["persona_uuid"], truthfulqa_persona_responses["question_hash"]],
#     truthfulqa_persona_responses["is_correct_canonical"],
# ).reset_index()
# truthfulqa_persona_dist.columns = ["persona_uuid", "question_hash", "False", "True"]
# truthfulqa_persona_dist["most_frequent"] = truthfulqa_persona_dist[answer_cols].idxmax(axis=1)
# truthfulqa_persona_dist["answer_num"] = truthfulqa_persona_dist["most_frequent"].map({"True": 1, "False": 0})
# persona_truthfulqa_accuracy = (
#     truthfulqa_persona_dist.groupby("persona_uuid")["answer_num"]
#     .mean()
#     .reset_index(name="truthful_proportion_correct")
# )
# persona_truthfulqa_accuracy.to_csv(os.path.join(RESULTS_DIR, "persona_truthfulqa_rank.csv"), index=False)

# truthfulqa_base_dist = pd.crosstab(
#     [truthfulqa_base_responses["persona_uuid"], truthfulqa_base_responses["question_hash"]],
#     truthfulqa_base_responses["is_correct_canonical"],
# ).reset_index()
# truthfulqa_base_dist.columns = ["persona_uuid", "question_hash", "False", "True"]
# truthfulqa_base_dist["most_frequent"] = truthfulqa_base_dist[answer_cols].idxmax(axis=1)
# truthfulqa_base_dist["answer_num"] = truthfulqa_base_dist["most_frequent"].map({"True": 1, "False": 0})
# base_truthfulqa_accuracy = (
#     truthfulqa_base_dist.groupby("persona_uuid")["answer_num"]
#     .mean()
#     .reset_index(name="truthful_proportion_correct")
# )
# base_truthfulqa_accuracy["truthfulqa_rank"] = (
#     base_truthfulqa_accuracy["truthful_proportion_correct"].rank(method="dense", ascending=False).astype(int)
# )
# base_truthfulqa_accuracy.to_csv(os.path.join(RESULTS_DIR, "base_truthfulqa_rank.csv"), index=False)


# ── Load Pre-Computed CSVs ────────────────────────────────────────────────────
persona_sjt_score = pd.read_csv(os.path.join(RESULTS_DIR, "persona500_sjt_score.csv"))
base_sjt_score = pd.read_csv(os.path.join(RESULTS_DIR, "base_sjt_score.csv"))
persona_hexaco_score = pd.read_csv(os.path.join(RESULTS_DIR, "persona500_hexaco_scores.csv"))
base_hexaco_score = pd.read_csv(os.path.join(RESULTS_DIR, "base_hexaco_scores.csv"))
persona_emobench_rank = pd.read_csv(os.path.join(RESULTS_DIR, "persona500_emobench_rank.csv"))
base_emobench_rank = pd.read_csv(os.path.join(RESULTS_DIR, "base_emobench_rank.csv"))
persona_truthfulqa_rank = pd.read_csv(os.path.join(RESULTS_DIR, "persona_truthfulqa_rank.csv"))
base_truthfulqa_rank = pd.read_csv(os.path.join(RESULTS_DIR, "base_truthfulqa_rank.csv"))

persona_hexaco_score.fillna(0, inplace=True)


# ── Section: Overall Z-Score Comparisons ─────────────────────────────────────
sjt_cols = [
    "agreeableness_option",
    "conscientiousness_option",
    "emotionality_option",
    "extraversion_option",
    "honesty_humility_option",
    "openness_option",
]
hexaco_cols = ["agreeableness", "conscientiousness", "emotionality", "extraversion", "honest-humility", "openness to experience"]

print("\n=== Z-Score Summary: SJT Scores ===")
print(get_z_score_summary_df(base_sjt_score, persona_sjt_score, sjt_cols))

print("\n=== Z-Score Summary: HEXACO Scores ===")
print(get_z_score_summary_df(base_hexaco_score, persona_hexaco_score, hexaco_cols))

print("\n=== Z-Score Summary: EmoBench ===")
print(get_z_score_summary_df(base_emobench_rank, persona_emobench_rank, ["emobench_proportion_correct"]))

print("\n=== Z-Score Summary: TruthfulQA ===")
print(get_z_score_summary_df(base_truthfulqa_rank, persona_truthfulqa_rank, ["truthful_proportion_correct"]))


# ── Section: Correlation Analysis — SJT/HEXACO vs Benchmarks ─────────────────
persona_hexaco_score.columns = [
    "honest_humility_hexaco",
    "emotionality_hexaco",
    "extraversion_hexaco",
    "agreeableness_hexaco",
    "conscientiousness_hexaco",
    "openness_hexaco",
    "altruism_hexaco",
    "persona_uuid",
]
persona_sjt_score.columns = [
    "persona_uuid",
    "agreeableness_sjt",
    "conscientiousness_sjt",
    "emotionality_sjt",
    "extraversion_sjt",
    "honesty_humility_sjt",
    "openness_sjt",
]

persona_sjt_score = pd.merge(persona_sjt_score, persona_emobench_rank, on="persona_uuid")
persona_sjt_score = pd.merge(persona_sjt_score, persona_truthfulqa_rank, on="persona_uuid")
persona_sjt_score = pd.merge(persona_sjt_score, persona_hexaco_score, on="persona_uuid")

trait_cols = [
    "agreeableness_sjt",
    "conscientiousness_sjt",
    "emotionality_sjt",
    "extraversion_sjt",
    "honesty_humility_sjt",
    "openness_sjt",
    "honest_humility_hexaco",
    "emotionality_hexaco",
    "extraversion_hexaco",
    "agreeableness_hexaco",
    "conscientiousness_hexaco",
    "openness_hexaco",
]
for col in trait_cols:
    persona_sjt_score[col + "_rank"] = (
        persona_sjt_score[col].rank(method="dense", ascending=False).astype(int)
    )

rank_cols = [col for col in persona_sjt_score.columns if "rank" in col]

plt.figure(figsize=(10, 10))
sns.heatmap(
    persona_sjt_score[rank_cols].corr(method="spearman"),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
)
plt.title("Correlation Between SJT Traits and EmoBench & TruthfulQA Ranks")
plt.tight_layout()
plt.show()


# ── Section: Archetype-Level Comparisons ─────────────────────────────────────
persona_sjt_score = pd.merge(
    persona_sjt_score, personas[["uuid", "archetype"]], left_on="persona_uuid", right_on="uuid"
)
persona_hexaco_score = pd.merge(
    persona_hexaco_score, personas[["uuid", "archetype"]], left_on="persona_uuid", right_on="uuid"
)
persona_emobench_rank = pd.merge(
    persona_emobench_rank, personas[["uuid", "archetype"]], left_on="persona_uuid", right_on="uuid"
)
persona_truthfulqa_rank = pd.merge(
    persona_truthfulqa_rank, personas[["uuid", "archetype"]], left_on="persona_uuid", right_on="uuid"
)

sjt_col_rename = {
    "agreeableness_sjt": "agreeableness_option",
    "conscientiousness_sjt": "conscientiousness_option",
    "emotionality_sjt": "emotionality_option",
    "extraversion_sjt": "extraversion_option",
    "honesty_humility_sjt": "honesty_humility_option",
    "openness_sjt": "openness_option",
}
hexaco_col_rename = {
    "agreeableness_hexaco": "agreeableness",
    "conscientiousness_hexaco": "conscientiousness",
    "emotionality_hexaco": "emotionality",
    "extraversion_hexaco": "extraversion",
    "honest_humility_hexaco": "honest-humility",
    "openness_hexaco": "openness to experience",
}

print("\n=== Z-Score by Archetype: SJT Scores ===")
for archetype in ARCHETYPE_VALS:
    print(f"\n{archetype}")
    archetype_df = persona_sjt_score[persona_sjt_score["archetype"] == archetype].rename(columns=sjt_col_rename)
    print(get_z_score_summary_df(base_sjt_score, archetype_df, sjt_cols))

print("\n=== Z-Score by Archetype: HEXACO Scores ===")
for archetype in ARCHETYPE_VALS:
    print(f"\n{archetype}")
    archetype_df = persona_hexaco_score[persona_hexaco_score["archetype"] == archetype].rename(columns=hexaco_col_rename)
    print(get_z_score_summary_df(base_hexaco_score, archetype_df, hexaco_cols))

print("\n=== Z-Score by Archetype: EmoBench ===")
for archetype in ARCHETYPE_VALS:
    print(f"\n{archetype}")
    archetype_df = persona_emobench_rank[persona_emobench_rank["archetype"] == archetype]
    print(get_z_score_summary_df(base_emobench_rank, archetype_df, ["emobench_proportion_correct"]))

print("\n=== Z-Score by Archetype: TruthfulQA ===")
for archetype in ARCHETYPE_VALS:
    print(f"\n{archetype}")
    archetype_df = persona_truthfulqa_rank[persona_truthfulqa_rank["archetype"] == archetype]
    print(get_z_score_summary_df(base_truthfulqa_rank, archetype_df, ["truthful_proportion_correct"]))
