"""
Breslow baseline hazard estimator.

DeepSurv (like classical CoxPH) outputs a *relative* risk score, not an
absolute survival probability -- there's no notion of "probability of
surviving past time t" without combining it with a baseline hazard. The
ensemble baselines (RSF, GBSA) sidestep this because scikit-survival
gives them `.predict_survival_function()` natively. For DeepSurv, this
module implements the classical nonparametric Breslow (1972) estimator to
recover that baseline hazard from the training data, so its risk score
can be converted into an actual survival curve -- which is what the Brier
score comparison in evaluate.py requires.

S(t | x) = exp( -H0(t) * exp(risk(x)) )

where H0(t) is the Breslow estimate of the cumulative baseline hazard.
"""

from __future__ import annotations

import numpy as np


def fit_breslow_baseline_hazard(
    durations_train: np.ndarray, events_train: np.ndarray, risk_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the cumulative baseline hazard H0(t) at each observed event time.

    Returns:
        event_times: sorted unique times at which an event occurred.
        cum_hazard: H0(t) evaluated at each of those times.
    """
    order = np.argsort(durations_train)
    durations_sorted = durations_train[order]
    events_sorted = events_train[order]
    exp_risk_sorted = np.exp(risk_train[order])

    event_times = np.unique(durations_sorted[events_sorted == 1])
    cum_hazard = np.zeros_like(event_times, dtype=np.float64)

    running_hazard = 0.0
    for i, t in enumerate(event_times):
        at_risk_mask = durations_sorted >= t
        denom = exp_risk_sorted[at_risk_mask].sum()
        n_events_at_t = events_sorted[durations_sorted == t].sum()
        running_hazard += n_events_at_t / max(denom, 1e-12)
        cum_hazard[i] = running_hazard

    return event_times, cum_hazard


def step_function_eval(query_times: np.ndarray, step_times: np.ndarray, step_values: np.ndarray) -> np.ndarray:
    """Evaluate a right-continuous step function (e.g. H0(t)) at arbitrary times.

    Value is 0 before the first observed event time, then holds the most
    recent step value (standard Kaplan-Meier / Breslow convention).
    """
    idx = np.searchsorted(step_times, query_times, side="right") - 1
    values = np.where(idx >= 0, step_values[np.clip(idx, 0, len(step_values) - 1)], 0.0)
    return values


def deepsurv_survival_function(
    risk_scores: np.ndarray, event_times: np.ndarray, cum_hazard: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Survival probability matrix for DeepSurv, shape (n_subjects, len(query_times)).

    Combines each subject's relative risk score with the Breslow baseline
    hazard evaluated at query_times.
    """
    H0_t = step_function_eval(query_times, event_times, cum_hazard)  # shape (n_times,)
    exp_risk = np.exp(risk_scores)  # shape (n_subjects,)
    return np.exp(-np.outer(exp_risk, H0_t))