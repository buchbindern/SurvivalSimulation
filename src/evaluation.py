"""
Survival model evaluation on a held-out test set.

Metrics are computed on the test split, using the training split as the
reference distribution for the IPCW-based metrics (Uno's C, time-dependent AUC,
integrated Brier score) -- which is exactly the train/test signature
scikit-survival's metrics are designed for. The oracle quantities
(``Actual C`` and ``Baseline AUC``) are computed from the ground-truth risk
scores on the same test set, giving a fair upper reference.
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


def _evaluation_times(survival_train, survival_test, n_time_points):
    """
    Time grid for the time-dependent metrics.

    Uses a percentile window of the test event times (10th-80th) rather than the
    raw min/max: the tails of the time distribution are sparse, where the IPCW
    censoring estimator becomes unstable or undefined. The upper bound is also
    kept strictly inside the training follow-up, since the censoring estimator is
    fit on the training split.
    """
    test_event_times = survival_test["time"][survival_test["event"]]
    lo = np.percentile(test_event_times, 10)
    hi = np.percentile(test_event_times, 80)
    hi = min(hi, survival_train["time"].max() - 1e-3)
    return np.linspace(lo, hi, n_time_points)


def evaluate_model(model, X_test, survival_train, survival_test,
                   true_risk_test, n_time_points):
    """
    Evaluate a fitted survival model on the test split.

    Parameters
    ----------
    model : fitted scikit-survival estimator
        Fit on the training split; this function only predicts and scores.
    X_test : ndarray
        Test covariates.
    survival_train : structured array
        Training labels -- the censoring/time reference for the IPCW metrics.
    survival_test : structured array
        Test labels being scored.
    true_risk_test : ndarray
        Ground-truth log-risk for the test rows (for the oracle metrics).
    n_time_points : int
        Number of evaluation time points for the time-dependent metrics.

    Returns
    -------
    dict
        One value per key in :data:`METRIC_COLUMNS`.
    """
    risk_scores = model.predict(X_test)
    times = _evaluation_times(survival_train, survival_test, n_time_points)
    tau = times[-1]  # bound IPCW weighting to the shared follow-up

    # Model discrimination.
    harrell_c = concordance_index_censored(
        survival_test["event"], survival_test["time"], risk_scores
    )[0]
    uno_c = concordance_index_ipcw(
        survival_train, survival_test, risk_scores, tau=tau
    )[0]
    aucs, _ = cumulative_dynamic_auc(survival_train, survival_test, risk_scores, times)
    mean_auc = np.nanmean(aucs)

    # Model calibration: integrated Brier score from predicted survival curves.
    survival_funcs = model.predict_survival_function(X_test)
    surv_probs = np.asarray([[fn(t) for t in times] for fn in survival_funcs])
    brier = integrated_brier_score(survival_train, survival_test, surv_probs, times)

    # Oracle metrics on the same test set, from the ground-truth risk scores.
    actual_c = concordance_index_censored(
        survival_test["event"], survival_test["time"], true_risk_test
    )[0]
    base_aucs, _ = cumulative_dynamic_auc(
        survival_train, survival_test, true_risk_test, times
    )
    baseline_mean_auc = np.nanmean(base_aucs)

    # Observed censoring rate (%) on the test split.
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