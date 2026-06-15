import numpy as np 
from sksurv.util import Surv 
import scipy.optimize as opt 
from sksurv.metrics import ( 
    concordance_index_censored,
    cumulative_dynamic_auc,
)


def generate_marker(n_samples, m,  baseline_hazard, rnd, time_points=None):
    """
    Generate synthetic survival data using exponential time to event distributions.

    Parameters:
    - n_samples: int, number of samples
    - m: int, number of covariates
    - baseline_hazard: float, baseline hazard rate
    - rnd: np.random.RandomState, random number generator
    - time_points: int, number of time points for evaluation 

    Returns:
    - X: Covariates 
    - time_event: True event times
    - actual_concordance: Concordance index with true risk scores
    - hazard_ratio: Ground truth hazard ratios for covariates
    - risk_scores: Linear predictor scores (log hazard)
    - baseline_auc: Time-dependent AUC scores
    - baseline_mean_auc: Mean of AUC scores
    - time_points: Evaluation time points
    """
    
    # create synthetic risk scores with uniform distribution
    # refer to Bender et al. (2005), https://doi.org/10.1002/sim.2059
    X = np.array(rnd.uniform(low=-1.0, high = 1.0, size=(n_samples, m)))
    hazard_ratio = np.expand_dims(np.array(rnd.uniform(low = 0, high = 0.1, size=m)), axis=0)
    coef = np.log(hazard_ratio)

    logits = np.dot(X, coef.T)

    u = rnd.uniform(size=n_samples)
    risk_scores = np.squeeze(logits)
    time_event = -np.log(u) / (baseline_hazard * np.exp(risk_scores))

    actual = concordance_index_censored(np.ones(n_samples, dtype=bool), time_event, risk_scores)

    y_uncensored = np.array([(True, t) for t in time_event], 
                  dtype=[('event', bool), ('time', float)])
    
    baseline_auc, baseline_mean_auc = cumulative_dynamic_auc(
        y_uncensored, y_uncensored, risk_scores, time_points
    )

    return X, time_event, actual[0], hazard_ratio, risk_scores, baseline_auc, baseline_mean_auc, time_points

def generate_survival_data(n_samples, m, baseline_hazard, percentage_cens, rnd, time_points=None, retry=True):
    """
    Adds right censoring to the generated survival data. 
    
    Parameters:
    - n_samples: int, Number of samples to generate.
    - m : int, Number of covariates (features) per sample.
    - baseline_hazard: float, Baseline hazard rate used to generate exponential survival times.
    - percentage_cens: float, Desired proportion of censored data (between 0.0 and 1.0).
    - rnd: numpy.random.RandomState, Random number generator instance for reproducibility.
    - time_points: int, Time points at which metrics like AUC will be evaluated.
    - retry: bool, Whether to retry censoring optimization if convergence fails.

    Returns:
    - X_test: Covariate matrix after censoring adjustment and truncation to max observed event time.
    - y_test: Survival labels for test set, with dtype [('event', bool), ('time', float)].
    - y_train:  Full uncensored survival labels (used as training reference for metric evaluation).
    - actual_c: Concordance index computed using ground-truth risk scores.
    - converged: Whether the censoring optimization successfully converged to desired censoring level.
    - hazard_ratio: True hazard ratios used to generate the survival times.
    - risk_scores: Linear predictor (log-risk) scores for the test set.
    - baseline_mean_auc: Mean time-dependent AUC using ground-truth scores.
    - eval_time_points: Time points used for cumulative dynamic AUC evaluation.
    
    """
                 
    X, time_event, actual_c, hazard_ratio, risk_scores, baseline_auc, baseline_mean_auc, eval_time_points = generate_marker(
        n_samples, m, baseline_hazard, rnd, time_points)


    def get_observed_time(x):
        rnd_cens = rnd
        time_censor = rnd_cens.uniform(high=x, size=n_samples)
        event = time_event < time_censor
        time = np.where(event, time_event, time_censor)
        return event, time 
    
    def censoring_amount(x): 
        event, _ = get_observed_time(x)
        cens = 1.0 - event.sum() / event.shape[0]
        return (cens - percentage_cens) ** 2

    res = opt.minimize_scalar(censoring_amount, method="bounded", bounds=(0, time_event.max()))

    if ( np.abs(res.fun) > 2.0/n_samples and retry): 
        return generate_survival_data(n_samples, m, baseline_hazard, percentage_cens, rnd=rnd, retry=False, time_points=time_points)
    elif (np.abs(res.fun) > 2.0/n_samples and not retry):
        converged = False
    else:
        converged = True

    event, time = get_observed_time(res.x) 
    
    tau = time[event].max()
    y = Surv.from_arrays(event=event, time=time)
    mask = time < tau
    X_test = X[mask]
    y_test = y[mask]

    return X_test, y_test, y, actual_c, converged, hazard_ratio, risk_scores[mask], baseline_mean_auc, eval_time_points