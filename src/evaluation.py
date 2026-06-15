"""
Survival model evaluation.

Computes the metrics used to compare models on a single generated dataset:

- Harrell's concordance index (rank accuracy, ignores censoring distribution)
- Uno's concordance index (IPCW-corrected, robust to censoring)
- Mean time-dependent (cumulative/dynamic) AUC
- Integrated Brier score (calibration + discrimination over time)

Ground-truth quantities that come from the data-generating process rather than
from the model -- the oracle concordance (``Actual C``) and oracle AUC
(``Baseline AUC``) -- are passed in and recorded alongside, so each row is a
complete, self-describing record.

Note on naming: ``survival_train`` / ``survival_test`` follow scikit-survival's
convention for the reference and evaluation label sets. In this simulation the
model is fit and scored on the same covariates (an in-sample metrics study, not
a held-out prediction benchmark); ``survival_train`` is the fully uncensored
label set used by the IPCW-based metrics as the censoring reference.
"""

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

# Metric column names, in the order used by the summary table and plots.
METRIC_COLUMNS = (
    "censoring",
    "Actual C",
    "Harrell's C",
    "Uno's C",
    "Baseline AUC",
    "AUC",
    "Brier",
)


def evaluate_model(model, X, survival_train, survival_test,
                   n_time_points, actual_c, baseline_mean_auc):
    """
    Evaluate a fitted survival model on one dataset.

    Parameters
    ----------
    model : fitted scikit-survival estimator
        Already fit; this function does prediction and scoring only.
    X : ndarray
        Covariate matrix to score on.
    survival_train : structured array
        Reference labels (uncensored) used by the IPCW metrics and the
        integrated Brier score as the censoring/time reference.
    survival_test : structured array
        Evaluation labels, dtype ``[('event', bool), ('time', float)]``.
    n_time_points : int
        Number of evenly spaced evaluation times for AUC and Brier score.
    actual_c : float
        Oracle concordance from the data-generating risk scores.
    baseline_mean_auc : float
        Oracle mean time-dependent AUC from the data-generating risk scores.

    Returns
    -------
    dict
        One metric per key in :data:`METRIC_COLUMNS`.
    """
    risk_scores = model.predict(X)

    # Evaluation times, kept just inside the observed range to avoid
    # boundary issues in the time-dependent metrics.
    times = np.linspace(
        survival_test["time"].min() + 0.001,
        survival_test["time"].max() - 0.001,
        n_time_points,
    )

    # Time-dependent AUC (mean over the evaluation grid).
    aucs, _ = cumulative_dynamic_auc(survival_train, survival_test, risk_scores, times)
    mean_auc = np.nanmean(aucs)

    # Integrated Brier score from the predicted survival functions.
    survival_funcs = model.predict_survival_function(X)
    surv_probs = np.asarray([[fn(t) for t in times] for fn in survival_funcs])
    brier = integrated_brier_score(survival_train, survival_test, surv_probs, times)

    # Concordance indices.
    harrell_c = concordance_index_censored(
        survival_test["event"], survival_test["time"], risk_scores
    )[0]
    uno_c = concordance_index_ipcw(survival_train, survival_test, risk_scores)[0]

    # Observed censoring rate (%) on the evaluation set.
    censoring = 100.0 - survival_test["event"].sum() * 100.0 / survival_test.shape[0]

    return {
        "censoring": censoring,
        "Actual C": actual_c,
        "Harrell's C": harrell_c,
        "Uno's C": uno_c,
        "Baseline AUC": baseline_mean_auc,
        "AUC": mean_auc,
        "Brier": brier,
    }