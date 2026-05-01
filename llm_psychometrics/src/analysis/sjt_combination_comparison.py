import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import login
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp, wasserstein_distance

# ── Configuration ──────────────────────────────────────────────────────────────
SJT_OPTION_COLS = [
    "agreeableness_option",
    "conscientiousness_option",
    "emotionality_option",
    "extraversion_option",
    "honesty_humility_option",
    "openness_option",
]

MODEL_CONFIGS = {
    "gemma": {
        # "claude_sjt_claude_persona": {
        #     "hf_path": "thoughtworks/gemma_psychometrics_personas_responses",
        #     "hf_config": "cmp_anthropic_personas_cmp_anthropic_sjts",
        # },
        "claude_sjt_gpt_persona": {
            "hf_path": "thoughtworks/gemma_psychometrics_personas_responses",
            "hf_config": "cmp_openai_personas_cmp_anthropic_sjts",
        },
        "gpt_sjt_claude_persona": {
            "hf_path": "thoughtworks/gemma_psychometrics_personas_responses",
            "hf_config": "cmp_anthropic_personas_cmp_openai_sjts",
        },
        # "gpt_sjt_gpt_persona": {
        #     "hf_path": "thoughtworks/gemma_psychometrics_personas_responses",
        #     "hf_config": "cmp_openai_personas_cmp_openai_sjts",
        # },
    },
    "qwen": {
        # "claude_sjt_claude_persona": {
        #     "hf_path": "thoughtworks/psychometric_personas_responses",
        #     "hf_config": "TODO",
        # },
        "claude_sjt_gpt_persona": {
            "hf_path": "thoughtworks/psychometric_personas_responses",
            "hf_config": "gpt_persona_claude_sjt",
        },
        "gpt_sjt_claude_persona": {
            "hf_path": "thoughtworks/psychometric_personas_responses",
            "hf_config": "claude_persona_gpt_sjt",
        },
        # "gpt_sjt_gpt_persona": {
        #     "hf_path": "thoughtworks/psychometric_personas_responses",
        #     "hf_config": "TODO",
        # },
    },
}

RESULTS_DIR = "../../experiment_results/sjt_combination_comparison/"


# ── Helpers ────────────────────────────────────────────────────────────────────
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


def compute_sjt_score(responses_df):
    """Crosstab → tie-drop → per-persona normalized trait proportions."""
    present_cols = [
        c for c in SJT_OPTION_COLS if c in responses_df["normalized_answer"].unique()
    ]
    dist = pd.crosstab(
        [responses_df["persona_uuid"], responses_df["question_hash"]],
        responses_df["normalized_answer"],
    ).reset_index()
    for col in SJT_OPTION_COLS:
        if col not in dist.columns:
            dist[col] = 0
    row_max = dist[SJT_OPTION_COLS].max(axis=1)
    dist["most_frequent"] = dist[SJT_OPTION_COLS].idxmax(axis=1)
    dist["tie_flag"] = dist[SJT_OPTION_COLS].eq(row_max, axis=0).sum(axis=1) > 1
    dropped = dist[~dist["tie_flag"]]
    score = (
        dropped.groupby("persona_uuid")["most_frequent"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in SJT_OPTION_COLS:
        if col not in score.columns:
            score[col] = 0.0
    return score[["persona_uuid"] + SJT_OPTION_COLS]


def compute_stability_metrics(responses_df):
    """JS divergence + ICC2 per trait, following reference code exactly."""
    trait_stats = (
        responses_df.groupby(["persona_uuid", "iter"])["normalized_answer"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    for col in SJT_OPTION_COLS:
        if col not in trait_stats.columns:
            trait_stats[col] = 0.0

    js_stability = compute_js_stability(trait_stats)

    trait_dist = trait_stats.reset_index()
    icc_results = []
    for trait in SJT_OPTION_COLS:
        df_trait = trait_dist[["persona_uuid", "iter", trait]].rename(
            columns={trait: "score"}
        )
        try:
            icc = pg.intraclass_corr(
                data=df_trait,
                targets="persona_uuid",
                raters="iter",
                ratings="score",
            )
            icc2 = icc[icc["Type"] == "ICC2"]["ICC"].values[0]
        except Exception:
            icc2 = float("nan")
        icc_results.append({"trait": trait, "ICC": icc2})

    icc_df = pd.DataFrame(icc_results)

    summary = {
        "JS_divergence_mean": np.round(js_stability.mean(), 3).item(),
        "JS_divergence_std": np.round(js_stability.std(), 3).item(),
        "ICC_mean": np.round(icc_df["ICC"].mean(), 3).item(),
    }
    return summary, icc_df


def compute_entropy(sjt_score_df):
    mean_vec = sjt_score_df[SJT_OPTION_COLS].mean().values
    mean_vec = mean_vec / mean_vec.sum()
    return np.round(entropy(mean_vec), 3).item()


def compute_pairwise_js_between_conditions(scores_by_cond):
    """4×4 JS divergence matrix between mean trait vectors of each condition."""
    cond_names = list(scores_by_cond.keys())
    mean_vecs = {
        c: scores_by_cond[c][SJT_OPTION_COLS].mean().values for c in cond_names
    }
    matrix = pd.DataFrame(index=cond_names, columns=cond_names, dtype=float)
    for a, b in itertools.product(cond_names, repeat=2):
        va = mean_vecs[a] / (mean_vecs[a].sum() or 1)
        vb = mean_vecs[b] / (mean_vecs[b].sum() or 1)
        matrix.loc[a, b] = np.round(jensenshannon(va, vb) ** 2, 4)
    return matrix


def compute_pairwise_ks_wasserstein(scores_by_cond):
    """Pairwise KS test + Wasserstein distance per trait between condition pairs."""
    cond_names = list(scores_by_cond.keys())
    rows = []
    for a, b in itertools.combinations(cond_names, 2):
        row = {"condition_a": a, "condition_b": b}
        for trait in SJT_OPTION_COLS:
            sa = scores_by_cond[a][trait].values
            sb = scores_by_cond[b][trait].values
            ks_stat, ks_p = ks_2samp(sa, sb)
            wd = wasserstein_distance(sa, sb)
            short = trait.replace("_option", "")
            row[f"{short}_ks_stat"] = np.round(ks_stat, 4)
            row[f"{short}_ks_p"] = np.round(ks_p, 4)
            row[f"{short}_wasserstein"] = np.round(wd, 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_grouped_bar(scores_by_cond, model_name, out_dir):
    records = []
    for cond, df in scores_by_cond.items():
        means = df[SJT_OPTION_COLS].mean()
        for trait, val in means.items():
            records.append(
                {"condition": cond, "trait": trait.replace("_option", ""), "mean": val}
            )
    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 5))
    traits = [c.replace("_option", "") for c in SJT_OPTION_COLS]
    x = np.arange(len(traits))
    cond_names = list(scores_by_cond.keys())
    width = 0.8 / len(cond_names)
    for i, cond in enumerate(cond_names):
        vals = (
            plot_df[plot_df["condition"] == cond]
            .set_index("trait")
            .loc[traits, "mean"]
            .values
        )
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cond)
    ax.set_xticks(x)
    ax.set_xticklabels(traits, rotation=20, ha="right")
    ax.set_ylabel("Mean trait proportion")
    ax.set_title(f"{model_name}: Mean SJT Trait Distribution by Condition")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_grouped_bar.png"), dpi=150)
    plt.close()


def plot_js_heatmap(js_matrix, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        js_matrix.astype(float),
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{model_name}: Pairwise JS Divergence Between Conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_js_heatmap.png"), dpi=150)
    plt.close()


def plot_boxplots(scores_by_cond, model_name, out_dir):
    records = []
    for cond, df in scores_by_cond.items():
        melted = df[["persona_uuid"] + SJT_OPTION_COLS].melt(
            id_vars="persona_uuid", var_name="trait", value_name="score"
        )
        melted["condition"] = cond
        records.append(melted)
    plot_df = pd.concat(records, ignore_index=True)
    plot_df["trait"] = plot_df["trait"].str.replace("_option", "")

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=plot_df, x="trait", y="score", hue="condition", ax=ax)
    ax.set_title(f"{model_name}: Per-Persona SJT Score Distribution by Condition")
    ax.set_xlabel("")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_boxplot.png"), dpi=150)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
login(token=os.environ.get("HF_TOKEN", ""))
os.makedirs(RESULTS_DIR, exist_ok=True)

cross_model_stability = {}

for model_name, conditions in MODEL_CONFIGS.items():
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'=' * 70}")

    scores_by_cond = {}
    stability_rows = []
    icc_by_cond = {}

    for cond_name, cfg in conditions.items():
        print(f"\n  Loading condition: {cond_name}")
        try:
            responses = get_hf_dataset(cfg["hf_path"], cfg["hf_config"])
            sjt_score = compute_sjt_score(responses)
            scores_by_cond[cond_name] = sjt_score

            stab_summary, icc_df = compute_stability_metrics(responses)
            ent = compute_entropy(sjt_score)
            stab_summary["entropy"] = ent
            stab_summary["condition"] = cond_name
            stability_rows.append(stab_summary)
            icc_by_cond[cond_name] = icc_df.set_index("trait")["ICC"]
        except Exception as e:
            print(f"  [SKIP] {cond_name} — {e}")

    # ── Stability summary table ────────────────────────────────────────────────
    stability_df = pd.DataFrame(stability_rows).set_index("condition")
    print(f"\n--- Stability Summary: {model_name} ---")
    print(stability_df.to_string())

    # ── Mean trait distribution table ─────────────────────────────────────────
    mean_trait_rows = []
    for cond, df in scores_by_cond.items():
        row = {"condition": cond}
        for col in SJT_OPTION_COLS:
            short = col.replace("_option", "")
            row[f"{short}_mean"] = np.round(df[col].mean(), 4)
            row[f"{short}_std"] = np.round(df[col].std(), 4)
        mean_trait_rows.append(row)
    mean_trait_df = pd.DataFrame(mean_trait_rows).set_index("condition")
    print(f"\n--- Mean Trait Distribution (mean ± std): {model_name} ---")
    print(mean_trait_df.to_string())

    # ── Per-trait ICC table ────────────────────────────────────────────────────
    icc_table = pd.DataFrame(icc_by_cond).T
    icc_table.index.name = "condition"
    icc_table.columns = [c.replace("_option", "") for c in icc_table.columns]
    print(f"\n--- ICC2 per Trait: {model_name} ---")
    print(icc_table.to_string())

    # ── Pairwise JS divergence table ──────────────────────────────────────────
    js_matrix = compute_pairwise_js_between_conditions(scores_by_cond)
    print(f"\n--- Pairwise JS Divergence Between Conditions: {model_name} ---")
    print(js_matrix.to_string())

    # ── Pairwise KS + Wasserstein table ───────────────────────────────────────
    pairwise_df = compute_pairwise_ks_wasserstein(scores_by_cond)
    print(f"\n--- Pairwise KS Statistic + Wasserstein Distance: {model_name} ---")
    print(pairwise_df.to_string(index=False))

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    stability_df.to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_stability_summary.csv")
    )
    mean_trait_df.to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_mean_trait_distribution.csv")
    )
    icc_table.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_icc_per_trait.csv"))
    js_matrix.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_pairwise_js.csv"))
    pairwise_df.to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_pairwise_ks_wasserstein.csv"),
        index=False,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_grouped_bar(scores_by_cond, model_name, RESULTS_DIR)
    plot_js_heatmap(js_matrix, model_name, RESULTS_DIR)
    plot_boxplots(scores_by_cond, model_name, RESULTS_DIR)

    cross_model_stability[model_name] = stability_df

# ── Cross-model comparison ─────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("CROSS-MODEL STABILITY COMPARISON (Gemma vs Qwen)")
print(f"{'=' * 70}")
metric_cols = ["JS_divergence_mean", "JS_divergence_std", "ICC_mean", "entropy"]
available_models = [
    m for m in cross_model_stability if not cross_model_stability[m].empty
]
if len(available_models) >= 2:
    gemma_vs_qwen = pd.concat(
        {m: cross_model_stability[m][metric_cols] for m in available_models},
        axis=1,
    )
    gemma_vs_qwen.index.name = "condition"
    print(gemma_vs_qwen.to_string())
    gemma_vs_qwen.to_csv(os.path.join(RESULTS_DIR, "gemma_vs_qwen_stability.csv"))


# ── GPT vs Claude source effect summary ───────────────────────────────────────
print(f"\n{'=' * 70}")
print("SOURCE EFFECT SUMMARY (GPT vs Claude, SJT source and Persona source)")
print(f"{'=' * 70}")

SJT_SOURCE_GROUPS = {
    "claude_sjt": ["claude_sjt_claude_persona", "claude_sjt_gpt_persona"],
    "gpt_sjt": ["gpt_sjt_claude_persona", "gpt_sjt_gpt_persona"],
}
PERSONA_SOURCE_GROUPS = {
    "claude_persona": ["claude_sjt_claude_persona", "gpt_sjt_claude_persona"],
    "gpt_persona": ["claude_sjt_gpt_persona", "gpt_sjt_gpt_persona"],
}

for model_name, stab_df in cross_model_stability.items():
    if stab_df.empty:
        continue
    print(f"\n  Model: {model_name.upper()}")

    for group_label, groups in [
        ("SJT Source", SJT_SOURCE_GROUPS),
        ("Persona Source", PERSONA_SOURCE_GROUPS),
    ]:
        rows = []
        for source_name, cond_list in groups.items():
            present = [c for c in cond_list if c in stab_df.index]
            if not present:
                continue
            avg = stab_df.loc[present, metric_cols].mean()
            avg.name = source_name
            rows.append(avg)
        if rows:
            summary = pd.DataFrame(rows)
            summary.index.name = group_label
            print(f"\n  --- {group_label} effect ---")
            print(summary.to_string())
            summary.to_csv(
                os.path.join(
                    RESULTS_DIR,
                    f"{model_name}_{group_label.lower().replace(' ', '_')}_effect.csv",
                )
            )
