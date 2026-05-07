import json
import os
import random
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import login
from scipy.spatial.distance import cdist, jensenshannon


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


# calculate_hexaco_score and get_trait_scores depend on generation_config and
# hexaco_eval loaded below — defined after config loading.


# ── Dataset Configuration ─────────────────────────────────────────────────────
RESPONSE_DATASET_PATH = "thoughtworks/gemma_psychometrics_personas_responses"

PERSONA_SJT_RESPONSE_DATASET_CONFIG = "analysis_sjt"
BASE_SJT_RESPONSE_DATASET_CONFIG = "base_sjt"

PERSONA_HEXACO_RESPONSE_DATASET_CONFIG = "analysis_hexaco"
BASE_HEXACO_RESPONSE_DATASET_CONFIG = "base_hexaco"

PERSONA_DATASET_PATH = "thoughtworks/psychometric_personas"
PERSONA_DATASET_CONFIG = "analysis"

RESULTS_DIR = "../../experiment_results/analysis_2026MarARR/"

SJT_OPTION_COLS = [
    "agreeableness_option",
    "conscientiousness_option",
    "emotionality_option",
    "extraversion_option",
    "honesty_humility_option",
    "openness_option",
]
HEXACO_TRAIT_LIST = [
    "honest-humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness to experience",
    "altruism",
]
HEXACO_OPTION_COLS = ["Agree", "Disagree", "Neutral", "Strongly agree", "Strongly disagree"]


# ── Data Loading ──────────────────────────────────────────────────────────────
login(token=os.environ.get("HF_TOKEN", ""))

with open("../../configs/generation_config.yaml", "r") as f:
    generation_config = yaml.safe_load(f)

with open("../../psychometric_tests/hexaco_100_eval.yaml", "r") as f:
    hexaco_eval = yaml.safe_load(f)


# HEXACO scoring functions — depend on generation_config and hexaco_eval
def calculate_hexaco_score(trait, subtrait, answers, likert_scale=generation_config["likert_scale"]):
    answers = [answer if answer in likert_scale else "Do not wish to answer" for answer in answers]
    answers = pd.Series(answers)
    subtrait_dict = hexaco_eval[trait][subtrait]
    indices = [idx - 1 for idx in subtrait_dict["indices"]]
    trait_answers = answers[indices]
    non_refused_answers = trait_answers[
        ~trait_answers.isin(["Do not wish to answer", "Do not wish to answer."])
    ]
    answer_indices = [likert_scale.index(answer) + 1 for answer in non_refused_answers]
    true_answer_indices = [
        6 - idx if reverse else idx
        for idx, reverse in zip(answer_indices, subtrait_dict["reverse"])
    ]
    return true_answer_indices


def get_trait_scores(answer, likert_scale=generation_config["likert_scale"]):
    trait_score = []
    for trait in hexaco_eval.keys():
        trait_true_answer_indices = []
        for subtrait in hexaco_eval[trait].keys():
            trait_true_answer_indices.extend(
                calculate_hexaco_score(trait, subtrait, answer, likert_scale)
            )
        trait_score.append(np.round(np.mean(trait_true_answer_indices).item(), 3).item())
    return trait_score


personas = get_hf_dataset(PERSONA_DATASET_PATH, PERSONA_DATASET_CONFIG)

hexaco_base_responses = get_hf_dataset(RESPONSE_DATASET_PATH, BASE_HEXACO_RESPONSE_DATASET_CONFIG)
hexaco_persona_responses = get_hf_dataset(RESPONSE_DATASET_PATH, PERSONA_HEXACO_RESPONSE_DATASET_CONFIG)

sjt_base_responses = get_hf_dataset(RESPONSE_DATASET_PATH, BASE_SJT_RESPONSE_DATASET_CONFIG)
sjt_persona_responses = get_hf_dataset(RESPONSE_DATASET_PATH, PERSONA_SJT_RESPONSE_DATASET_CONFIG)


# ── Section: SJT — Crosstabs ──────────────────────────────────────────────────
dist = pd.crosstab(
    [sjt_persona_responses["persona_uuid"], sjt_persona_responses["question_hash"]],
    sjt_persona_responses["normalized_answer"],
).reset_index()

base_dist = pd.crosstab(
    [sjt_base_responses["persona_uuid"], sjt_base_responses["question_hash"]],
    sjt_base_responses["normalized_answer"],
).reset_index()


# ── Section: SJT — Stability & ICC ───────────────────────────────────────────
trait_stats = (
    sjt_persona_responses.groupby(["persona_uuid", "iter"])["normalized_answer"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

base_trait_stats = (
    sjt_base_responses.groupby(["persona_uuid", "iter"])["normalized_answer"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

persona_js_stability = compute_js_stability(trait_stats)
base_js_stability = compute_js_stability(base_trait_stats)

print("SJT JS Stability — persona mean:", np.round(persona_js_stability.mean(), 3).item())
print("SJT JS Stability — base mean:   ", np.round(base_js_stability.mean(), 3).item())

trait_dist = trait_stats.reset_index()
sjt_icc_results = []
for trait in trait_stats.columns:
    df_trait = trait_dist[["persona_uuid", "iter", trait]].rename(columns={trait: "score"})
    icc = pg.intraclass_corr(
        data=df_trait, targets="persona_uuid", raters="iter", ratings="score"
    )
    icc2 = icc[icc["Type"] == "ICC(A,1)"]["ICC"].values[0]
    sjt_icc_results.append({"trait": trait, "ICC": icc2})
sjt_icc_results = pd.DataFrame(sjt_icc_results)
print("\nSJT ICC Results:")
print(sjt_icc_results)

stability_metrics_summary = {
    "JS_divergence_mean": np.round(persona_js_stability.mean(), 3).item(),
    "JS_divergence_std": np.round(persona_js_stability.std(), 3).item(),
    "ICC_mean": np.round(sjt_icc_results["ICC"].mean(), 3).item(),
}
print("\nSJT Stability Metrics Summary:", stability_metrics_summary)


# ── Section: SJT — Tie Detection & Train/Test Split ───────────────────────────
row_max = dist[SJT_OPTION_COLS].max(axis=1)
dist["most_frequent"] = dist[SJT_OPTION_COLS].idxmax(axis=1)
dist["tie_flag"] = dist[SJT_OPTION_COLS].eq(row_max, axis=0).sum(axis=1) > 1
dist["tied_options"] = dist[SJT_OPTION_COLS].apply(lambda r: r.index[r == r.max()].tolist(), axis=1)

base_row_max = base_dist[SJT_OPTION_COLS].max(axis=1)
base_dist["most_frequent"] = base_dist[SJT_OPTION_COLS].idxmax(axis=1)
base_dist["tie_flag"] = base_dist[SJT_OPTION_COLS].eq(base_row_max, axis=0).sum(axis=1) > 1
base_dist["tied_options"] = base_dist[SJT_OPTION_COLS].apply(
    lambda r: r.index[r == r.max()].tolist(), axis=1
)

print("\nPersona tie_flag distribution:")
print(dist["tie_flag"].value_counts() / dist.shape[0])
print("\nBase tie_flag distribution:")
print(base_dist["tie_flag"].value_counts() / base_dist.shape[0])

dropped_dist = dist[~dist["tie_flag"]]
base_dropped_dist = base_dist[~base_dist["tie_flag"]]

total_sjt_ids = list(dist["question_hash"].drop_duplicates())
random.seed(42)
random.shuffle(total_sjt_ids)
split_idx = int(0.8 * len(total_sjt_ids))
train_sjts = total_sjt_ids[:split_idx]
test_sjts = total_sjt_ids[split_idx:]
print(f"\nTrain SJTs: {len(train_sjts)}, Test SJTs: {len(test_sjts)}")

train_dist = dropped_dist[dropped_dist["question_hash"].isin(train_sjts)]
test_dist = dropped_dist[dropped_dist["question_hash"].isin(test_sjts)]


# ── Section: SJT — Score Computation ─────────────────────────────────────────
sjt_score = (
    dropped_dist.groupby("persona_uuid")["most_frequent"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)
train_sjt_score = (
    train_dist.groupby("persona_uuid")["most_frequent"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)
test_sjt_score = (
    test_dist.groupby("persona_uuid")["most_frequent"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)
base_sjt_score = (
    base_dropped_dist.groupby("persona_uuid")["most_frequent"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)

# sjt_score.to_csv(os.path.join(RESULTS_DIR, "persona500_sjt_score.csv"), index=False)
# base_sjt_score.to_csv(os.path.join(RESULTS_DIR, "base_sjt_score.csv"), index=False)


# ── Section: SJT — Identity Retrieval ────────────────────────────────────────
train_persona_id = train_sjt_score["persona_uuid"]
test_persona_id = test_sjt_score["persona_uuid"]

X_train = train_sjt_score[SJT_OPTION_COLS].values
X_test = test_sjt_score[SJT_OPTION_COLS].values

js_dist = cdist(X_test, X_train, metric="jensenshannon")
sim_matrix = 1 - js_dist

best_idx = sim_matrix.argmax(axis=1)
most_similar_id = train_persona_id.iloc[best_idx].values

ranks = []
top10_flags = []
percentiles = []
N = len(train_persona_id)

for i, test_id in enumerate(test_persona_id):
    sims = sim_matrix[i]
    sorted_idx = np.argsort(-sims)

    if test_id in train_persona_id.values:
        true_idx = np.where(train_persona_id.values == test_id)[0][0]
        rank = np.where(sorted_idx == true_idx)[0][0] + 1
        percentile = 1 - (rank - 1) / N
        top10 = percentile >= 0.90
    else:
        rank = None
        percentile = None
        top10 = False

    ranks.append(rank)
    percentiles.append(percentile)
    top10_flags.append(top10)

similarity_result = test_sjt_score[["persona_uuid"]].copy()
similarity_result["most_similar_train_id"] = most_similar_id
similarity_result["self_rank"] = ranks
similarity_result["percentiles"] = percentiles
similarity_result["top10_percentile_flag"] = top10_flags

print(
    "\nProportion of test personas in top-10 percentile:",
    similarity_result["top10_percentile_flag"].sum() / similarity_result.shape[0],
)

similarity_result = (
    pd.merge(
        similarity_result,
        personas[["uuid", "archetype"]],
        left_on="persona_uuid",
        right_on="uuid",
    )
    .rename(columns={"archetype": "persona_archetype"})
    .pipe(
        lambda df: pd.merge(
            df,
            personas[["uuid", "archetype"]],
            left_on="most_similar_train_id",
            right_on="uuid",
        ).rename(columns={"archetype": "most_similar_train_archetype"})
    )
    .drop(["uuid_x", "uuid_y"], axis=1)
)

archetype_matches = (
    similarity_result["persona_archetype"] == similarity_result["most_similar_train_archetype"]
).sum()
print(f"Archetype matches (most similar): {archetype_matches} / {len(similarity_result)}")


# ── Section: SJT — Trait Shift from Base to Persona ──────────────────────────
trait_sjt_score_diff = train_sjt_score[SJT_OPTION_COLS] - base_sjt_score[SJT_OPTION_COLS].iloc[0]
trait_sjt_score_diff["persona_uuid"] = train_sjt_score["persona_uuid"]
trait_sjt_score_diff = pd.merge(
    trait_sjt_score_diff, personas[["uuid", "archetype"]], left_on="persona_uuid", right_on="uuid"
)

print("\nMedian trait shift by archetype:")
print(trait_sjt_score_diff.groupby("archetype")[SJT_OPTION_COLS].median())
print("\nTrait shift descriptive stats:")
print(trait_sjt_score_diff[SJT_OPTION_COLS].describe())


# ── Section: HEXACO — Per-Iter Trait Scores ───────────────────────────────────
base_hexaco_str_answers = (
    hexaco_base_responses.groupby(["persona_id", "iter"])["answer"]
    .apply(list)
    .reset_index(name="answers")
)
persona_hexaco_str_answers = (
    hexaco_persona_responses.groupby(["persona_id", "iter"])["answer"]
    .apply(list)
    .reset_index(name="answers")
)

base_hexaco_trait_df = pd.DataFrame(
    base_hexaco_str_answers.apply(lambda x: get_trait_scores(x["answers"]), axis=1).to_list(),
    index=base_hexaco_str_answers.index,
    columns=HEXACO_TRAIT_LIST,
)
base_hexaco_trait_df["persona_id"] = base_hexaco_str_answers["persona_id"]
base_hexaco_trait_df["iter"] = base_hexaco_str_answers["iter"]

persona_hexaco_trait_df = pd.DataFrame(
    persona_hexaco_str_answers.apply(lambda x: get_trait_scores(x["answers"]), axis=1).to_list(),
    index=persona_hexaco_str_answers.index,
    columns=HEXACO_TRAIT_LIST,
)
persona_hexaco_trait_df["persona_id"] = persona_hexaco_str_answers["persona_id"]
persona_hexaco_trait_df["iter"] = persona_hexaco_str_answers["iter"]

# Set persona_id as index for compute_js_stability (groups by level 0)
persona_hexaco_trait_df.index = persona_hexaco_trait_df["persona_id"]
persona_hexaco_trait_df.drop("persona_id", axis=1, inplace=True)


# ── Section: HEXACO — Stability & ICC ────────────────────────────────────────
persona_hexaco_js = compute_js_stability(persona_hexaco_trait_df)
base_hexaco_js = compute_js_stability(base_hexaco_trait_df)

print("\nHEXACO JS Stability — persona mean:", np.round(persona_hexaco_js.mean(), 3).item())
print("HEXACO JS Stability — base mean:   ", np.round(base_hexaco_js.mean(), 3).item())

persona_hexaco_trait_df.fillna(0, inplace=True)

# Use a distinct variable for the ICC input df (avoids overwrite by crosstab below)
persona_hexaco_icc_df = persona_hexaco_trait_df.reset_index()

hexaco_icc_trait_cols = [
    "honest-humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness to experience",
]
hexaco_icc_results = []
for trait in hexaco_icc_trait_cols:
    df_trait = persona_hexaco_icc_df[["persona_id", "iter", trait]].rename(columns={trait: "score"})
    icc = pg.intraclass_corr(
        data=df_trait, targets="persona_id", raters="iter", ratings="score"
    )
    icc2 = icc[icc["Type"] == "ICC(A,1)"]["ICC"].values[0]
    hexaco_icc_results.append({"trait": trait, "ICC": icc2})
hexaco_icc_results = pd.DataFrame(hexaco_icc_results)
print("\nHEXACO ICC Results:")
print(hexaco_icc_results)


# ── Section: HEXACO — Aggregate Scores from Most Frequent Answers ─────────────
persona_hexaco_dist = pd.crosstab(
    [hexaco_persona_responses["persona_id"], hexaco_persona_responses["question_idx"]],
    hexaco_persona_responses["answer"],
).reset_index()

base_hexaco_dist = pd.crosstab(
    [hexaco_base_responses["persona_id"], hexaco_base_responses["question_idx"]],
    hexaco_base_responses["answer"],
).reset_index()

hexaco_persona_row_max = persona_hexaco_dist[HEXACO_OPTION_COLS].max(axis=1)
persona_hexaco_dist["most_frequent"] = persona_hexaco_dist[HEXACO_OPTION_COLS].idxmax(axis=1)
persona_hexaco_dist["tie_flag"] = (
    persona_hexaco_dist[HEXACO_OPTION_COLS].eq(hexaco_persona_row_max, axis=0).sum(axis=1) > 1
)
persona_hexaco_dist["tied_options"] = persona_hexaco_dist[HEXACO_OPTION_COLS].apply(
    lambda r: r.index[r == r.max()].tolist(), axis=1
)

base_hexaco_row_max = base_hexaco_dist[HEXACO_OPTION_COLS].max(axis=1)
base_hexaco_dist["most_frequent"] = base_hexaco_dist[HEXACO_OPTION_COLS].idxmax(axis=1)
base_hexaco_dist["tie_flag"] = (
    base_hexaco_dist[HEXACO_OPTION_COLS].eq(base_hexaco_row_max, axis=0).sum(axis=1) > 1
)
base_hexaco_dist["tied_options"] = base_hexaco_dist[HEXACO_OPTION_COLS].apply(
    lambda r: r.index[r == r.max()].tolist(), axis=1
)

base_hexaco_grouped_str_answers = (
    base_hexaco_dist.groupby("persona_id")["most_frequent"].apply(list).reset_index(name="answers")
)
persona_hexaco_grouped_str_answers = (
    persona_hexaco_dist.groupby("persona_id")["most_frequent"]
    .apply(list)
    .reset_index(name="answers")
)

base_hexaco_grouped_trait_df = pd.DataFrame(
    base_hexaco_grouped_str_answers.apply(lambda x: get_trait_scores(x["answers"]), axis=1).to_list(),
    index=base_hexaco_grouped_str_answers.index,
    columns=HEXACO_TRAIT_LIST,
)
base_hexaco_grouped_trait_df["persona_id"] = base_hexaco_grouped_str_answers["persona_id"]

persona_hexaco_grouped_trait_df = pd.DataFrame(
    persona_hexaco_grouped_str_answers.apply(
        lambda x: get_trait_scores(x["answers"]), axis=1
    ).to_list(),
    index=persona_hexaco_grouped_str_answers.index,
    columns=HEXACO_TRAIT_LIST,
)
persona_hexaco_grouped_trait_df["persona_id"] = persona_hexaco_grouped_str_answers["persona_id"]

# persona_hexaco_grouped_trait_df.to_csv(os.path.join(RESULTS_DIR, "persona500_hexaco_scores.csv"), index=False)
# base_hexaco_grouped_trait_df.to_csv(os.path.join(RESULTS_DIR, "base_hexaco_scores.csv"), index=False)


# ── Section: SJT–HEXACO Correlation Heatmap ──────────────────────────────────
sjt_col_order = [
    "agreeableness_option",
    "conscientiousness_option",
    "emotionality_option",
    "extraversion_option",
    "honesty_humility_option",
    "openness_option",
]
hexaco_col_order = [
    "agreeableness",
    "conscientiousness",
    "emotionality",
    "extraversion",
    "honest-humility",
    "openness to experience",
]

corr_matrix = pd.DataFrame(
    {
        col: train_sjt_score[sjt_col_order].corrwith(persona_hexaco_grouped_trait_df[col])
        for col in hexaco_col_order
    }
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
)
plt.title("Correlation Between SJT and Hexaco Traits")
plt.tight_layout()
plt.show()
