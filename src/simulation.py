"""
Experiment orchestration.

Runs the survival-model comparison: for each censoring level, repeatedly
generate data, fit each model, and evaluate it, then aggregate the metrics
(mean and standard deviation) across repeats.

This module only coordinates -- the data, models, and metrics live in
:mod:`data_generation`, :mod:`models`, and :mod:`evaluation` respectively.
"""

import numpy as np
import pandas as pd

from src.data_generation import generate_survival_data
from src.models import build_model, MODEL_TYPES
from src.evaluation import evaluate_model, METRIC_COLUMNS


# Right-censoring proportions to sweep over.
CENSORING_LEVELS = (0.0, 0.25, 0.5)


def run_simulation(n_samples, m, n_repeats=100, time_points=10,
                   baseline_hazard=0.1, random_state=42):
    """
    Compare survival models across several censoring levels.

    For each level in :data:`CENSORING_LEVELS`, generate ``n_repeats`` synthetic
    datasets, fit Cox PH, Random Survival Forest, and Gradient Boosting, and
    evaluate each. Repeats whose censoring optimization fails to converge are
    skipped. Metrics are aggregated (mean and sample std) across the surviving
    repeats.

    Parameters
    ----------
    n_samples : int
        Samples per generated dataset.
    m : int
        Number of covariates.
    n_repeats : int
        Repeated datasets per censoring level.
    time_points : int
        Evaluation time points for the time-dependent metrics.
    baseline_hazard : float
        Baseline hazard rate for the data-generating process.
    random_state : int
        Seed for the shared random number generator (controls data generation
        and the stochastic models, for reproducibility).

    Returns
    -------
    dict
        ``{model_type: {"mean": [DataFrame, ...], "std": [DataFrame, ...]}}``,
        with one DataFrame per censoring level. Columns follow
        :data:`evaluation.METRIC_COLUMNS`.
    """
    rnd = np.random.RandomState(random_state)
    results = {mt: {"mean": [], "std": []} for mt in MODEL_TYPES}

    for cens in CENSORING_LEVELS:
        # Collect one metric dict per (repeat, model) for this censoring level.
        records = {mt: [] for mt in MODEL_TYPES}

        for _ in range(n_repeats):
            (X, survival_test, survival_train, actual_c, converged,
             hazard_ratio, risk_scores, baseline_mean_auc, eval_times) = \
                generate_survival_data(
                    n_samples, m,
                    baseline_hazard=baseline_hazard,
                    percentage_cens=cens,
                    rnd=rnd,
                    time_points=time_points,
                )

            if not converged:
                continue  # skip repeats where target censoring wasn't reached

            for mt in MODEL_TYPES:
                model = build_model(mt, random_state=rnd)
                model.fit(X, survival_test)
                metrics = evaluate_model(
                    model, X, survival_train, survival_test,
                    time_points, actual_c, baseline_mean_auc,
                )
                records[mt].append(metrics)

        # Aggregate across repeats for this censoring level.
        for mt in MODEL_TYPES:
            df = pd.DataFrame(records[mt], columns=METRIC_COLUMNS)
            mean_row = pd.DataFrame(
                {col: [np.mean(df[col].values)] for col in METRIC_COLUMNS}
            )
            std_row = pd.DataFrame(
                {col: [np.std(df[col].values, ddof=1)] for col in METRIC_COLUMNS}
            )
            results[mt]["mean"].append(mean_row)
            results[mt]["std"].append(std_row)

    return results