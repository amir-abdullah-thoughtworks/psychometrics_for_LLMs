import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import login
from sklearn.cluster import KMeans
from umap.umap_ import UMAP


# ── Constants ─────────────────────────────────────────────────────────────────
ZERO_COMPUTE_DATA_DIR = "../../experiment_results/zero_compute_analysis_sjt/"
ANALYSIS_RESULTS_DIR = os.path.join(ZERO_COMPUTE_DATA_DIR, "analysis_results")
FILE_STR = "huggingface"

TRAIT_MAP = {
    "1": "Honesty-Humility",
    "2": "Emotionality",
    "3": "Extraversion",
    "4": "Agreeableness",
    "5": "Conscientiousness",
    "6": "Openness",
}

os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def write_to_json(file, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(file, f, indent=2)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_model_name(filename, file_str):
    return filename.replace("_sjt_answers_", "").replace(".json", "").replace(file_str, "")


def get_true_indices_v1(data):
    resolved_traits = []
    true_answer_indices = []
    for answer, shuffled_indices in zip(data["answers"][0], data["config"]["answer_index"][0]):
        true_index = str(shuffled_indices[int(answer) - 1] + 1)
        if true_index in TRAIT_MAP:
            resolved_traits.append(TRAIT_MAP[true_index])
            true_answer_indices.append(true_index)
    return true_answer_indices, resolved_traits


def get_true_indices(data):
    resolved_traits = []
    true_answer_indices = []
    for answer, shuffled_indices in zip(data["answers"][0], data["config"]["answer_index"]):
        true_index = str(shuffled_indices[int(answer) - 1] + 1)
        if true_index in TRAIT_MAP:
            resolved_traits.append(TRAIT_MAP[true_index])
            true_answer_indices.append(true_index)
    return true_answer_indices, resolved_traits


def get_trait_distribution(data):
    _, resolved_traits = get_true_indices(data)
    trait_counts = Counter(resolved_traits)
    total = sum(trait_counts.values())
    trait_summary = {
        trait: {
            "count": trait_counts[trait],
            "proportion": round(trait_counts[trait] / total, 3) if total > 0 else 0,
        }
        for trait in TRAIT_MAP.values()
    }
    return [trait_summary[key]["proportion"] for key in trait_summary]


def get_trait_distribution_v1(data):
    _, resolved_traits = get_true_indices_v1(data)
    trait_counts = Counter(resolved_traits)
    total = sum(trait_counts.values())
    trait_summary = {
        trait: {
            "count": trait_counts[trait],
            "proportion": round(trait_counts[trait] / total, 3) if total > 0 else 0,
        }
        for trait in TRAIT_MAP.values()
    }
    return [trait_summary[key]["proportion"] for key in trait_summary]


def load_personas(hf_persona_path, n_persona_sample):
    print(f"Loading Personas from {hf_persona_path}")
    hf_persona_dataset = load_dataset(hf_persona_path)
    total_persona_df = hf_persona_dataset["train"].to_pandas()
    persona_datasets = total_persona_df.groupby("archetype").sample(n=n_persona_sample, random_state=42)
    print(f"No of Personas: {persona_datasets.shape[0]}")
    return persona_datasets


def load_sjts(hf_sjt_path, n_sjt_sample):
    print(f"Loading SJTs from {hf_sjt_path}")
    hf_sjt_dataset = load_dataset(hf_sjt_path)
    total_sjt_df = hf_sjt_dataset["train"].to_pandas()
    sjt_datasets = total_sjt_df.groupby("template_no").sample(n=n_sjt_sample, random_state=42)
    print(f"No of SJTs: {sjt_datasets.shape[0]}")
    return sjt_datasets


def get_trait_distribution_df(file_dir, file_str):
    hf_sjt_trait_distributions_df = pd.DataFrame()
    for filename in os.listdir(file_dir):
        if ".json" not in filename:
            continue
        print(filename)
        model_name = get_model_name(filename, file_str)
        hf_persona_sjt_results = read_json(os.path.join(file_dir, filename))
        persona_sjt_str_answers_df = pd.DataFrame(
            [
                {"persona_id": key, "answers": get_trait_distribution(hf_persona_sjt_results[key])}
                for key in hf_persona_sjt_results
            ]
        )
        persona_sjt_str_answers_df = pd.DataFrame(
            persona_sjt_str_answers_df["answers"].to_list(),
            columns=list(TRAIT_MAP.values()),
            index=persona_sjt_str_answers_df["persona_id"],
        )
        persona_sjt_str_answers_df["model_name"] = model_name
        hf_sjt_trait_distributions_df = pd.concat(
            [hf_sjt_trait_distributions_df, persona_sjt_str_answers_df], axis=0
        )
    return hf_sjt_trait_distributions_df


def sjt_aggregation_by_attribute(sjt_df, attribute):
    persona_df = persona_datasets[["uuid", attribute]].drop_duplicates()
    attribute_sjt = pd.merge(sjt_df, persona_df, left_on="persona_id", right_on="uuid", how="inner")
    return attribute_sjt.drop("uuid", axis=1).groupby([attribute, "model_name"], as_index=False).mean()


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


def get_preferred_trait_df(attribute_df, attribute):
    attribute_df = attribute_df.copy()
    attribute_df["preferred_trait"] = attribute_df[list(TRAIT_MAP.values())].idxmax(axis=1)
    return attribute_df[[attribute, "model_name", "preferred_trait"]].pivot(
        index=attribute, columns="model_name", values="preferred_trait"
    )


# ── Data Loading ──────────────────────────────────────────────────────────────
login(token=os.environ.get("HF_TOKEN", ""))
persona_datasets = load_personas("thoughtworks/psychometric_personas", n_persona_sample=25)
persona_datasets["age_category"] = persona_datasets["age"].apply(categorize_age)
archetype_uuid_df = persona_datasets[["uuid", "archetype"]].drop_duplicates()


# ── Section: Per-Model Analysis ───────────────────────────────────────────────
# Merged from notebook cells 7 and 16:
#   - Eigenvalue histogram of normalized one-hot SJT answers (cell 7)
#   - Plain KMeans scatter (cell 7)
#   - Trait correlation heatmap (cell 7)
#   - UMAP + KMeans scatter coloured by archetype (cell 16)

for filename in sorted(os.listdir(ZERO_COMPUTE_DATA_DIR)):
    if not filename.endswith(".json"):
        continue

    print(f"\nProcessing: {filename}")
    model_name = get_model_name(filename, FILE_STR)
    hf_persona_sjt_results = read_json(os.path.join(ZERO_COMPUTE_DATA_DIR, filename))

    persona_sjt_str_answers_df = pd.DataFrame(
        [
            {"persona_id": key, "answers": get_true_indices(hf_persona_sjt_results[key])[0]}
            for key in hf_persona_sjt_results
        ]
    )
    persona_sjt_str_answers_df = pd.DataFrame(
        persona_sjt_str_answers_df["answers"].to_list(),
        columns=list(range(0, 500)),
        index=persona_sjt_str_answers_df["persona_id"],
    )

    n_options = 6
    persona_sjt_answers_one_hot_df = pd.concat(
        [
            pd.get_dummies(persona_sjt_str_answers_df[col], prefix=col).reindex(
                columns=[f"{col}_{i + 1}" for i in range(n_options)], fill_value=0
            )
            for col in persona_sjt_str_answers_df.columns
        ],
        axis=1,
    )
    persona_sjt_answers_one_hot_df = persona_sjt_answers_one_hot_df.astype(int)

    persona_sjt_answers_normalized_df = (
        persona_sjt_answers_one_hot_df - persona_sjt_answers_one_hot_df.mean()
    ) / persona_sjt_answers_one_hot_df.std()
    persona_sjt_answers_normalized_df.fillna(0, inplace=True)

    # Eigenvalue histogram
    plt.figure()
    plt.hist(np.linalg.eigvals(np.cov(persona_sjt_answers_normalized_df.values.T)))
    plt.title(f"Eigen Value Histogram - SJT Answers - {model_name}")
    plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, f"Eigen_histogram_sjt_{model_name}"), dpi=300)
    plt.close()

    # UMAP + KMeans
    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_reduced = umap_model.fit_transform(persona_sjt_answers_normalized_df)

    kmeans_model = KMeans(n_clusters=10, random_state=42)
    kmeans_model.fit(X_reduced)
    cluster_labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    # Plain KMeans scatter
    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap="viridis", marker="o")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200)
    plt.title(f"Interfactor SJT Clusters {model_name}")
    plt.savefig(os.path.join(ANALYSIS_RESULTS_DIR, f"interfactor_clusters_sjt_{model_name}"), dpi=300)
    plt.close()

    # Archetype-coloured KMeans scatter
    archetype_labels = pd.merge(
        archetype_uuid_df,
        pd.Series(persona_sjt_answers_normalized_df.index, name="persona_id"),
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
    centroids_scatter = ax.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200)
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
        os.path.join(ANALYSIS_RESULTS_DIR, f"interfactor_clusters_sjt_color_by_archetype_{model_name}"),
        bbox_inches="tight",
        bbox_extra_artists=(legend2,),
        dpi=300,
    )
    plt.close()

    # Trait correlation heatmap
    traits = sorted(set(col.split("_")[1] for col in persona_sjt_answers_normalized_df.columns))
    trait_means = pd.DataFrame(
        {
            f"trait_{trait}": persona_sjt_answers_normalized_df.filter(regex=rf"_{trait}$").mean(axis=1)
            for trait in traits
        }
    )

    plt.figure()
    sns.heatmap(
        np.corrcoef(trait_means.T),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        cbar=True,
        xticklabels=list(TRAIT_MAP.values()),
        yticklabels=list(TRAIT_MAP.values()),
    )
    plt.title(f"Overall Trait Correlation Plot SJT - {model_name}")
    plt.savefig(
        os.path.join(ANALYSIS_RESULTS_DIR, f"trait_correlation_sjt_{model_name}"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# ── Section: Trait Distribution DataFrame ────────────────────────────────────
hf_sjt_trait_distributions_df = get_trait_distribution_df(ZERO_COMPUTE_DATA_DIR, FILE_STR)
print(f"\nTotal rows in hf_sjt_trait_distributions_df: {hf_sjt_trait_distributions_df.shape[0]}")


# ── Section: Attribute-Level SJT Profiles ────────────────────────────────────
print("\n=== SJT Profile by Archetype ===")
archetype_sjt_df = sjt_aggregation_by_attribute(hf_sjt_trait_distributions_df, attribute="archetype")
archetype_sjt_df["avoider_label"] = np.where(
    archetype_sjt_df["archetype"].isin(
        ["The Avoider (Lazy Officer)", "The Avoider (Unconfident Officer)"]
    ),
    1,
    0,
)
print(archetype_sjt_df)
print("\nSJT scores by avoider label and model:")
print(
    archetype_sjt_df.drop(["archetype"], axis=1, errors="ignore")
    .groupby(["avoider_label", "model_name"])
    .mean()
)
print("\nSJT scores by model:")
print(
    archetype_sjt_df.drop(["archetype"], axis=1, errors="ignore")
    .groupby("model_name")
    .mean()
)
print("\nPreferred trait by archetype:")
print(get_preferred_trait_df(archetype_sjt_df, "archetype"))

print("\n=== SJT Profile by Ethnic Background ===")
ethnic_background_df = sjt_aggregation_by_attribute(
    hf_sjt_trait_distributions_df, attribute="ethnic_background"
)
print(ethnic_background_df.describe())

ethnic_conc_df = ethnic_background_df.pivot(
    columns=["model_name"], index="ethnic_background"
)["Conscientiousness"]
ethnic_conc_df["minority_label"] = np.where(
    ethnic_conc_df.index.isin(
        ["colombian", "east asian", "hispanic or latino other", "southeast asian"]
    ),
    1,
    0,
)
print("\nConscientiousness by minority label:")
print(ethnic_conc_df.groupby("minority_label").mean())
print("\nPreferred trait by ethnic background:")
print(get_preferred_trait_df(ethnic_background_df, "ethnic_background"))

print("\n=== SJT Profile by Age Category ===")
age_category_df = sjt_aggregation_by_attribute(
    hf_sjt_trait_distributions_df, attribute="age_category"
)
print(age_category_df)


# ── Section: SJT Dataset and Threat-Level Indexing ───────────────────────────
sjt_datasets = load_sjts("thoughtworks/psychometric_SJTs", n_sjt_sample=25).reset_index()
high_threat_sjt_idx = [
    i for i, config in enumerate(sjt_datasets["config"]) if config["threat_level"] == "high"
]
low_threat_sjt_idx = [
    i for i, config in enumerate(sjt_datasets["config"]) if config["threat_level"] == "low"
]
print(f"\nHigh-threat SJTs: {len(high_threat_sjt_idx)}, Low-threat SJTs: {len(low_threat_sjt_idx)}")
