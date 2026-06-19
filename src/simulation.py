"""
Experiment orchestration.

For each censoring level, repeatedly: generate a dataset, split it into train
and test, fit each model on the training split, and evaluate it on the held-out
test split. Metrics are aggregated (mean and sample std) across repeats.

This module only coordinates -- the data, models, and metrics live in
:mod:`data_generation`, :mod:`models`, and :mod:`evaluation`.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_generation import generate_survival_data
from src.models import build_model, MODEL_TYPES
from src.evaluation import evaluate_model, METRIC_COLUMNS


# Right-censoring proportions to sweep over.
CENSORING_LEVELS = (0.0, 0.25, 0.5)

# Fraction of each dataset held out for evaluation.
TEST_SIZE = 0.3


def run_simulation(n_samples, m, n_repeats=100, time_points=10,
                   baseline_hazard=0.1, random_state=42):
    """
    Compare survival models across several censoring levels, out of sample.

    For each level in :data:`CENSORING_LEVELS`, generate ``n_repeats`` datasets,
    split each into train/test (stratified on the event indicator when possible),
    fit Cox PH, Random Survival Forest, and Gradient Boosting on the training
    split, and score them on the held-out test split. Non-converged datasets are
    skipped. Metrics are averaged (with sample std) across the surviving repeats.

    Parameters
    ----------
    n_samples : int
        Samples per generated dataset (before the train/test split).
    m : int
        Number of covariates.
    n_repeats : int
        Repeated datasets per censoring level.
    time_points : int
        Evaluation time points for the time-dependent metrics.
    baseline_hazard : float
        Baseline hazard rate for the data-generating process.
    random_state : int
        Seed for the shared RNG (data generation, the split, and the
        stochastic models), for reproducibility.

    Returns
    -------
    dict
        ``{model_type: {"mean": [DataFrame, ...], "std": [DataFrame, ...]}}``,
        one DataFrame per censoring level, columns per
        :data:`evaluation.METRIC_COLUMNS`.
    """
    rnd = np.random.RandomState(random_state)
    results = {mt: {"mean": [], "std": []} for mt in MODEL_TYPES}

    for cens in CENSORING_LEVELS:
        records = {mt: [] for mt in MODEL_TYPES}

        for _ in range(n_repeats):
            X, y, true_risk, converged, hazard_ratio = generate_survival_data(
                n_samples, m,
                baseline_hazard=baseline_hazard,
                percentage_cens=cens,
                rnd=rnd,
            )
            if not converged:
                continue

            # Stratify on the event indicator when both classes are present.
            events = y["event"]
            stratify = events if (events.sum() >= 2 and (~events).sum() >= 2) else None

            X_tr, X_te, y_tr, y_te, _, risk_te = train_test_split(
                X, y, true_risk,
                test_size=TEST_SIZE,
                random_state=rnd,
                stratify=stratify,
            )

            # Restrict the test set to the training follow-up window: the
            # IPCW-based metrics estimate the censoring distribution from the
            # training split and are undefined beyond its largest observed time.
            within_followup = y_te["time"] < y_tr["time"].max()
            X_te = X_te[within_followup]
            y_te = y_te[within_followup]
            risk_te = risk_te[within_followup]

            for mt in MODEL_TYPES:
                model = build_model(mt, random_state=rnd)
                model.fit(X_tr, y_tr)
                metrics = evaluate_model(
                    model, X_te, y_tr, y_te, risk_te, time_points
                )
                records[mt].append(metrics)

        for mt in MODEL_TYPES:
            df = pd.DataFrame(records[mt], columns=METRIC_COLUMNS)
            if len(df) == 0:
                # Every repeat at this censoring level failed to converge
                # (only realistically possible at very low n_repeats) --
                # skip it with a clear message instead of crashing on
                # np.std of an empty array.
                print(
                    f"Warning: 0/{n_repeats} datasets converged for model "
                    f"'{mt}' at censoring={cens}; skipping this level. "
                    f"Increase n_repeats if this persists."
                )
                continue
            mean_row = pd.DataFrame(
                {col: [np.mean(df[col].values)] for col in METRIC_COLUMNS}
            )
            std_row = pd.DataFrame(
                {col: [np.std(df[col].values, ddof=1) if len(df) > 1 else 0.0] for col in METRIC_COLUMNS}
            )
            results[mt]["mean"].append(mean_row)
            results[mt]["std"].append(std_row)

    return results