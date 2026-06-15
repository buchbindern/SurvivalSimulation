"""
Synthetic survival data generation.

``generate_marker`` draws covariates and exponential time-to-event values from a
known hazard model (Bender et al., 2005). ``generate_survival_data`` adds
right-censoring calibrated to a target rate. The full (untruncated) dataset is
returned so the caller can split it into train and test sets; evaluation-time
bookkeeping (choosing time points, computing oracle metrics) is handled
downstream in :mod:`evaluation`.
"""

import numpy as np
from sksurv.util import Surv
import scipy.optimize as opt


def generate_marker(n_samples, m, baseline_hazard, rnd):
    """
    Draw covariates and true (uncensored) event times.

    Returns
    -------
    X : ndarray (n_samples, m)
        Covariates.
    time_event : ndarray (n_samples,)
        True event times.
    hazard_ratio : ndarray (1, m)
        Ground-truth hazard ratios for the covariates.
    true_risk_scores : ndarray (n_samples,)
        Linear predictor (log-risk) for each sample.
    """
    X = rnd.uniform(low=-1.0, high=1.0, size=(n_samples, m))
    hazard_ratio = np.expand_dims(rnd.uniform(low=0.0, high=0.1, size=m), axis=0)
    coef = np.log(hazard_ratio)

    true_risk_scores = np.squeeze(X @ coef.T)
    u = rnd.uniform(size=n_samples)
    time_event = -np.log(u) / (baseline_hazard * np.exp(true_risk_scores))

    return X, time_event, hazard_ratio, true_risk_scores


def generate_survival_data(n_samples, m, baseline_hazard, percentage_cens,
                           rnd, retry=True):
    """
    Generate a censored survival dataset at a target censoring rate.

    Right-censoring times are drawn uniformly on ``[0, x]``, with ``x`` chosen by
    a 1-D optimization so the realized censoring rate matches ``percentage_cens``.
    If the optimization does not converge, one retry is attempted before the
    dataset is flagged as non-converged (and skipped by the caller).

    Returns
    -------
    X : ndarray (n_samples, m)
        Covariates.
    y : structured array
        Censored survival labels, dtype ``[('event', bool), ('time', float)]``.
    true_risk_scores : ndarray (n_samples,)
        Ground-truth log-risk per sample (used to compute oracle metrics).
    converged : bool
        Whether the target censoring rate was reached.
    hazard_ratio : ndarray (1, m)
        Ground-truth hazard ratios.
    """
    X, time_event, hazard_ratio, true_risk_scores = generate_marker(
        n_samples, m, baseline_hazard, rnd
    )

    def get_observed_time(x):
        time_censor = rnd.uniform(high=x, size=n_samples)
        event = time_event < time_censor
        time = np.where(event, time_event, time_censor)
        return event, time

    def censoring_amount(x):
        event, _ = get_observed_time(x)
        cens = 1.0 - event.sum() / event.shape[0]
        return (cens - percentage_cens) ** 2

    res = opt.minimize_scalar(
        censoring_amount, method="bounded", bounds=(0, time_event.max())
    )

    if np.abs(res.fun) > 2.0 / n_samples and retry:
        return generate_survival_data(
            n_samples, m, baseline_hazard, percentage_cens, rnd=rnd, retry=False
        )
    converged = not (np.abs(res.fun) > 2.0 / n_samples and not retry)

    event, time = get_observed_time(res.x)
    y = Surv.from_arrays(event=event, time=time)

    return X, y, true_risk_scores, converged, hazard_ratio