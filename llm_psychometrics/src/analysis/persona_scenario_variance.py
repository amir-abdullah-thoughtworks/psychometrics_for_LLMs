"""Label-invariant variance decomposition for persona-conditioned SJT responses.

The SJT response is nominal, so assigning the six HEXACO categories integer
scores and decomposing the variance of those integers is not invariant to label
order. This script instead represents each response as a one-hot vector and
fits a balanced, crossed random-effects decomposition:

    y[p, j, r] = mean + persona[p] + scenario[j]
                 + persona_scenario[p, j] + error[p, j, r]

The method-of-moments estimates use the trace of the multivariate covariance,
which treats all response categories symmetrically. The permutation test
shuffles complete repeated-response blocks among persona labels separately
within every scenario. This preserves scenario response distributions and
within-pair variability while removing consistent persona identity across
scenarios.

By default, the script loads the public Hugging Face response dataset used in
the paper. A local CSV or Parquet file can be supplied with ``--input``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATASET = "thoughtworks/gemma_psychometrics_personas_responses"
DEFAULT_CONFIG = "analysis_sjt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Local .csv or .parquet response file. Defaults to Hugging Face.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--persona-column", default="persona_hash")
    parser.add_argument("--scenario-column", default="question_hash")
    parser.add_argument("--response-column", default="normalized_answer")
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path at which to save the reported estimates.",
    )
    return parser.parse_args()


def load_responses(args: argparse.Namespace) -> pd.DataFrame:
    if args.input is not None:
        suffix = args.input.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(args.input)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(args.input)
        raise ValueError("--input must be a CSV or Parquet file")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading from Hugging Face requires the 'datasets' package. "
            "Install it or pass a local file with --input."
        ) from exc

    return load_dataset(
        args.dataset,
        args.config,
        split=args.split,
    ).to_pandas()


def build_cell_probabilities(
    responses: pd.DataFrame,
    persona_column: str,
    scenario_column: str,
    response_column: str,
) -> tuple[np.ndarray, list[str], int]:
    required = {persona_column, scenario_column, response_column}
    missing = required.difference(responses.columns)
    if missing:
        raise ValueError(f"Response data is missing columns: {sorted(missing)}")
    if responses[list(required)].isna().any().any():
        raise ValueError("Persona, scenario, and response columns cannot contain nulls")

    persona_codes, personas = pd.factorize(responses[persona_column], sort=True)
    scenario_codes, scenarios = pd.factorize(responses[scenario_column], sort=True)
    response_codes, categories = pd.factorize(responses[response_column], sort=True)

    n_personas = len(personas)
    n_scenarios = len(scenarios)
    n_categories = len(categories)
    cell_counts = np.zeros((n_personas, n_scenarios, n_categories), dtype=np.int32)
    np.add.at(cell_counts, (persona_codes, scenario_codes, response_codes), 1)

    repetitions = cell_counts.sum(axis=2)
    unique_repetitions = np.unique(repetitions)
    if len(unique_repetitions) != 1 or unique_repetitions[0] == 0:
        raise ValueError(
            "The crossed variance estimator requires a complete, balanced design. "
            f"Observed repetitions per persona-scenario cell: {unique_repetitions.tolist()}"
        )

    n_repetitions = int(unique_repetitions[0])
    probabilities = cell_counts.astype(np.float64) / n_repetitions
    return probabilities, [str(category) for category in categories], n_repetitions


def variance_components(
    cell_probabilities: np.ndarray,
    n_repetitions: int,
) -> dict[str, float]:
    n_personas, n_scenarios, _ = cell_probabilities.shape
    if n_personas < 2 or n_scenarios < 2 or n_repetitions < 2:
        raise ValueError("At least two personas, scenarios, and repetitions are required")

    grand_mean = cell_probabilities.mean(axis=(0, 1))
    persona_means = cell_probabilities.mean(axis=1)
    scenario_means = cell_probabilities.mean(axis=0)
    interaction = (
        cell_probabilities
        - persona_means[:, None, :]
        - scenario_means[None, :, :]
        + grand_mean
    )

    ss_persona = (
        n_scenarios
        * n_repetitions
        * np.square(persona_means - grand_mean).sum()
    )
    ss_scenario = (
        n_personas
        * n_repetitions
        * np.square(scenario_means - grand_mean).sum()
    )
    ss_interaction = n_repetitions * np.square(interaction).sum()

    # For a cell with response probabilities q and R one-hot observations,
    # the within-cell sum of squares is R * (1 - sum(q_k ** 2)).
    ss_within = n_repetitions * (
        1.0 - np.square(cell_probabilities).sum(axis=2)
    ).sum()

    ms_persona = ss_persona / (n_personas - 1)
    ms_scenario = ss_scenario / (n_scenarios - 1)
    ms_interaction = ss_interaction / (
        (n_personas - 1) * (n_scenarios - 1)
    )
    ms_within = ss_within / (
        n_personas * n_scenarios * (n_repetitions - 1)
    )

    components = {
        "persona": (ms_persona - ms_interaction)
        / (n_scenarios * n_repetitions),
        "scenario": (ms_scenario - ms_interaction)
        / (n_personas * n_repetitions),
        "persona_x_scenario": (ms_interaction - ms_within) / n_repetitions,
        "residual": ms_within,
    }
    total = sum(components.values())
    if total <= 0:
        raise ValueError("Estimated total variance is not positive")

    return {
        **{f"{name}_variance": value for name, value in components.items()},
        **{
            f"{name}_percent": 100.0 * value / total
            for name, value in components.items()
        },
        "total_variance": total,
        "persona_to_scenario_ratio": components["persona"] / components["scenario"],
    }


def persona_permutation_test(
    cell_probabilities: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[str, float | int]:
    if n_permutations < 1:
        raise ValueError("--permutations must be at least 1")

    n_personas, n_scenarios, n_categories = cell_probabilities.shape
    grand_mean = cell_probabilities.mean(axis=(0, 1))
    observed = np.square(
        cell_probabilities.mean(axis=1) - grand_mean
    ).sum()

    rng = np.random.default_rng(seed)
    null_statistics = np.empty(n_permutations)
    for permutation_index in range(n_permutations):
        persona_sums = np.zeros((n_personas, n_categories), dtype=np.float64)
        for scenario_index in range(n_scenarios):
            shuffled_personas = rng.permutation(n_personas)
            persona_sums += cell_probabilities[
                shuffled_personas,
                scenario_index,
                :,
            ]
        permuted_means = persona_sums / n_scenarios
        null_statistics[permutation_index] = np.square(
            permuted_means - grand_mean
        ).sum()

    exceedances = int(np.count_nonzero(null_statistics >= observed))
    corrected_p = (exceedances + 1) / (n_permutations + 1)
    return {
        "permutations": n_permutations,
        "permutation_seed": seed,
        "observed_persona_statistic": float(observed),
        "null_mean": float(null_statistics.mean()),
        "null_standard_deviation": float(null_statistics.std(ddof=1)),
        "null_maximum": float(null_statistics.max()),
        "exceedances": exceedances,
        "permutation_p_value": float(corrected_p),
    }


def print_results(results: dict[str, Any]) -> None:
    print("Label-invariant crossed variance decomposition")
    print("------------------------------------------------")
    for name in (
        "persona",
        "scenario",
        "persona_x_scenario",
        "residual",
    ):
        print(
            f"{name:24s} "
            f"variance={results[f'{name}_variance']:.6f}  "
            f"percent={results[f'{name}_percent']:.2f}%"
        )
    print(f"{'total':24s} variance={results['total_variance']:.6f}")
    print(
        "persona/scenario ratio:  "
        f"{results['persona_to_scenario_ratio']:.6f}"
    )
    print()
    print("Block-permutation test of the persona main effect")
    print("-------------------------------------------------")
    print(f"permutations:             {results['permutations']}")
    print(f"exceedances:              {results['exceedances']}")
    print(f"finite-sample p-value:    {results['permutation_p_value']:.8f}")


def main() -> None:
    args = parse_args()
    responses = load_responses(args)
    cell_probabilities, categories, n_repetitions = build_cell_probabilities(
        responses,
        args.persona_column,
        args.scenario_column,
        args.response_column,
    )
    n_personas, n_scenarios, _ = cell_probabilities.shape

    results: dict[str, Any] = {
        "n_personas": n_personas,
        "n_scenarios": n_scenarios,
        "n_repetitions": n_repetitions,
        "n_categories": len(categories),
        "categories": categories,
    }
    results.update(variance_components(cell_probabilities, n_repetitions))
    results.update(
        persona_permutation_test(
            cell_probabilities,
            args.permutations,
            args.seed,
        )
    )
    print_results(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
