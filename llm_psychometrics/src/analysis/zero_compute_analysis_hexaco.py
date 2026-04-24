import json
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import login
from sklearn.cluster import KMeans
from umap.umap_ import UMAP


# ── Config Loading ────────────────────────────────────────────────────────────
with open("../../configs/generation_config.yaml", "r") as f:
    generation_config = yaml.safe_load(f)

with open("../../psychometric_tests/hexaco_100_eval.yaml", "r") as f:
    hexaco_eval = yaml.safe_load(f)


# ── Helper Functions ──────────────────────────────────────────────────────────
def write_to_json(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(file, f, indent=2)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_model_name(filename, file_str):
    return filename.replace("_hexaco_answers_", "").replace(".json", "").replace(file_str, "")


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


def load_personas(hf_persona_path, n_persona_sample):
    print(f"Loading Personas from {hf_persona_path}")
    hf_persona_dataset = load_dataset(hf_persona_path)
    total_persona_df = hf_persona_dataset["train"].to_pandas()
    persona_datasets = total_persona_df.groupby("archetype").sample(
        n=n_persona_sample, random_state=42
    )
    print(f"No of Personas: {persona_datasets.shape[0]}")
    return persona_datasets


def hexaco_aggregation_by_attribute(hexaco_df, persona_datasets, attribute):
    persona_df = persona_datasets[["uuid", attribute]].drop_duplicates()
    attribute_hexaco = pd.merge(
        hexaco_df, persona_df, left_on="persona_id", right_on="uuid", how="inner"
    )
    return attribute_hexaco.drop("uuid", axis=1).groupby([attribute, "model_name"], as_index=False).mean()


def categorize_age(age):
    if pd.isna(age):
        return "unknown"
    elif age < 18:
        return "juvenile"
    elif 18 <= age < 25:
        return "young_adult"
    elif 25 <= age < 40:
        return "adult"
    elif 40 <= age < 60:
        return "middle_aged"
    else:
        return "senior"


def get_preferred_trait_df(attribute_df, attribute, trait_list):
    attribute_df = attribute_df.copy()
    attribute_df["preferred_trait"] = attribute_df[trait_list].idxmax(axis=1)
    return attribute_df[[attribute, "model_name", "preferred_trait"]].pivot(
        index=attribute, columns="model_name", values="preferred_trait"
    )


# ── Constants ─────────────────────────────────────────────────────────────────
ZERO_COMPUTE_DATA_DIR = "../../experiment_results/zero_compute_analysis_hexaco/"
ANALYSIS_RESULTS_DIR = os.path.join(ZERO_COMPUTE_DATA_DIR, "analysis_results")
FILE_STR = "huggingface"

HEXACO_MAPPING = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
    "Do not wish to answer": -3,
}
TRAIT_LIST = [
    "honest-humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness to experience",
    "altruism",
]

os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
login(token=os.environ.get("HF_TOKEN", ""))
persona_datasets = load_personas("thoughtworks/psychometric_personas", n_persona_sample=25)
persona_datasets["age_category"] = persona_datasets["age"].apply(categorize_age)
archetype_df = persona_datasets[["uuid", "archetype"]].drop_duplicates()


# ── Section: Per-Model Analysis ───────────────────────────────────────────────
# Merged from notebook cells 7, 10, and 23:
#   - Eigenvalue histogram of raw HEXACO answers (cell 7)
#   - UMAP + KMeans clustering plot (cell 7)
#   - HEXACO trait correlation heatmap (cell 7)
#   - Accumulate hf_hexaco_trait_persona_df across models (cell 10)
#   - UMAP + KMeans plot coloured by archetype (cell 23)

hf_hexaco_trait_persona_df = pd.DataFrame()

for filename in sorted(os.listdir(ZERO_COMPUTE_DATA_DIR)):
    if not filename.endswith(".json"):
        continue

    print(f"\nProcessing: {filename}")
    model_name = get_model_name(filename, FILE_STR)

    hf_persona_hexaco_results = read_json(os.path.join(ZERO_COMPUTE_DATA_DIR, filename))
    raw_rows = [
        {"persona_id": key, "answers": hf_persona_hexaco_results[key]["answers"][0]}
        for key in hf_persona_hexaco_results.keys()
    ]
    persona_hexaco_str_answers_df = pd.DataFrame(raw_rows)
    persona_hexaco_str_answers_df = pd.DataFrame(
        persona_hexaco_str_answers_df["answers"].to_list(),
        columns=list(range(0, 100)),
        index=persona_hexaco_str_answers_df["persona_id"],
    )
    persona_hexaco_df = persona_hexaco_str_answers_df.replace(HEXACO_MAPPING)

    # Eigenvalue histogram
    plt.figure()
    plt.hist(np.linalg.eigvals(np.cov(persona_hexaco_df.values.T)))
    plt.title(f"Eigen Value Histogram - Hexaco Answers - {model_name}")
    plt.savefig(
        os.path.join(ANALYSIS_RESULTS_DIR, f"Eigen_histogram_hexaco_{model_name}"), dpi=300
    )
    plt.close()

    # UMAP + KMeans
    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_reduced = umap_model.fit_transform(persona_hexaco_df)

    kmeans_model = KMeans(n_clusters=10, random_state=42)
    kmeans_model.fit(X_reduced)
    cluster_labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    # Plain KMeans scatter
    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap="viridis", marker="o")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200)
    plt.title(f"Interfactor Hexaco Clusters {model_name}")
    plt.savefig(
        os.path.join(ANALYSIS_RESULTS_DIR, f"interfactor_clusters_hexaco_{model_name}"), dpi=300
    )
    plt.close()

    # Archetype-coloured KMeans scatter
    archetype_labels = pd.merge(
        archetype_df,
        pd.Series(persona_hexaco_df.index, name="persona_id"),
        left_on="uuid",
        right_on="persona_id",
        how="inner",
    )["archetype"]
    archetype_codes = archetype_labels.astype("category").cat.codes
    archetype_categories = archetype_labels.astype("category").cat.categories

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_reduced[:, 0], X_reduced[:, 1], c=archetype_codes, cmap="viridis", marker="o"
    )
    centroids_scatter = ax.scatter(
        centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    handles, _ = scatter.legend_elements(prop="colors")
    legend2 = ax.legend(
        [centroids_scatter] + handles,
        ["Centroids"] + list(archetype_categories),
        title="Archetypes",
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        fontsize=10,
        title_fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            ANALYSIS_RESULTS_DIR,
            f"interfactor_clusters_hexaco_color_by_archetype_{model_name}",
        ),
        bbox_inches="tight",
        bbox_extra_artists=(legend2,),
        dpi=300,
    )
    plt.close()

    # Trait scores + correlation heatmap
    hexaco_trait_persona_df = pd.DataFrame(
        persona_hexaco_str_answers_df.apply(lambda x: get_trait_scores(x), axis=1).to_list(),
        index=persona_hexaco_str_answers_df.index,
        columns=TRAIT_LIST,
    )

    plt.figure()
    sns.heatmap(
        np.corrcoef(hexaco_trait_persona_df.T - 3),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        cbar=True,
        xticklabels=hexaco_trait_persona_df.columns,
        yticklabels=hexaco_trait_persona_df.columns,
    )
    plt.title(f"Overall Trait Correlation Plot Hexaco - {model_name}")
    plt.savefig(
        os.path.join(ANALYSIS_RESULTS_DIR, f"trait_correlation_hexaco_{model_name}"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Accumulate across models
    hexaco_trait_persona_df["model_name"] = model_name
    hf_hexaco_trait_persona_df = pd.concat(
        [hf_hexaco_trait_persona_df, hexaco_trait_persona_df], axis=0
    )

# hf_hexaco_trait_persona_df.reset_index().to_csv(
#     os.path.join(ZERO_COMPUTE_DATA_DIR, "hf_hexaco_trait_persona_df.csv"), index=False
# )

print(f"\nTotal rows in hf_hexaco_trait_persona_df: {hf_hexaco_trait_persona_df.shape[0]}")


# ── Section: Attribute-Level HEXACO Profiles ──────────────────────────────────
print("\n=== HEXACO Profile by Archetype ===")
archetype_hexaco = hexaco_aggregation_by_attribute(
    hf_hexaco_trait_persona_df, persona_datasets, attribute="archetype"
)
print(archetype_hexaco)
print("\nPreferred trait by archetype:")
print(get_preferred_trait_df(archetype_hexaco, "archetype", TRAIT_LIST))

print("\n=== HEXACO Profile by Ethnic Background ===")
ethnic_hexaco = hexaco_aggregation_by_attribute(
    hf_hexaco_trait_persona_df, persona_datasets, attribute="ethnic_background"
)
print(ethnic_hexaco)
print("\nPreferred trait by ethnic background:")
print(get_preferred_trait_df(ethnic_hexaco, "ethnic_background", TRAIT_LIST))

print("\n=== HEXACO Profile by Age Category ===")
age_category_hexaco = hexaco_aggregation_by_attribute(
    hf_hexaco_trait_persona_df, persona_datasets, attribute="age_category"
)
print("\nPreferred trait by age category:")
print(get_preferred_trait_df(age_category_hexaco, "age_category", TRAIT_LIST))
