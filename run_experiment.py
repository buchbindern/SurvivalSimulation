"""
Run the full survival-model comparison experiment (CoxPH, RSF, GBSA, DeepSurv)
across censoring levels, and save results as a summary CSV + figures.

This replaces the notebook-driven workflow: same underlying src/ pipeline
(data_generation -> models -> simulation -> evaluation -> plotting), but
invoked as a script so it can be run, reviewed, and reproduced without
opening Jupyter.

Usage:
    python run_experiment.py
    python run_experiment.py --n_samples 1000 --n_repeats 50 --m 8
    python run_experiment.py --n_repeats 5 --output_dir results/smoke_test   # fast sanity check
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.models import MODEL_TYPES
from src.plotting import plot_censoring_performance, plot_model_comparison, plot_results
from src.simulation import run_simulation


def save_summary_table(results: dict, path: str) -> pd.DataFrame:
    """Flatten the {model: {"mean": [...]}} structure into one tidy CSV:
    one row per (model, censoring level), one column per metric.
    """
    rows = []
    for model_type in MODEL_TYPES:
        for mean_df in results[model_type]["mean"]:
            row = mean_df.iloc[0].to_dict()
            row["model"] = model_type
            rows.append(row)
    summary = pd.DataFrame(rows).set_index(["model", "censoring"])
    summary.to_csv(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the survival model comparison experiment.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Samples per generated dataset.")
    parser.add_argument("--m", type=int, default=8, help="Number of covariates.")
    parser.add_argument("--n_repeats", type=int, default=100, help="Repeated datasets per censoring level.")
    parser.add_argument("--time_points", type=int, default=10, help="Evaluation time points for time-dependent metrics.")
    parser.add_argument("--baseline_hazard", type=float, default=0.1, help="Baseline hazard rate for data generation.")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="results", help="Where to write the summary CSV and figures/.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Running simulation: n_samples={args.n_samples}, m={args.m}, n_repeats={args.n_repeats}")
    print(f"Models: {MODEL_TYPES}")

    results = run_simulation(
        n_samples=args.n_samples,
        m=args.m,
        n_repeats=args.n_repeats,
        time_points=args.time_points,
        baseline_hazard=args.baseline_hazard,
        random_state=args.random_state,
    )

    summary_path = os.path.join(args.output_dir, "model_comparison.csv")
    summary = save_summary_table(results, summary_path)
    print("\nSummary (mean across repeats, by model and censoring level):\n")
    print(summary.round(4).to_string())
    print(f"\nSaved summary table to {summary_path}")

    plot_model_comparison(results, save_path=os.path.join(figures_dir, "model_comparison.png"))
    plot_censoring_performance(results, save_path=os.path.join(figures_dir, "censoring_performance.png"))
    for model_type in MODEL_TYPES:
        plot_results(results, model_type=model_type, save_path=os.path.join(figures_dir, f"{model_type}_detail.png"))

    print(f"Saved figures to {figures_dir}/")


if __name__ == "__main__":
    main()