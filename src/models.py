"""
Survival model definitions.

Exposes a single factory, :func:`build_model`, that returns a configured
(unfitted) survival estimator for a given model type. Keeping the model
choices and their hyperparameters in one place makes it easy to see exactly
what is being compared and to tune a single model without touching the
experiment loop.
"""

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis


# Model types compared in the simulation, in display order.
MODEL_TYPES = ("cph", "rsf", "gbsa")

# Human-readable names, handy for plot titles and tables.
MODEL_NAMES = {
    "cph": "Cox Proportional Hazards",
    "rsf": "Random Survival Forest",
    "gbsa": "Gradient Boosting Survival Analysis",
}


def build_model(model_type, random_state=None):
    """
    Build a configured (unfitted) survival estimator.

    Parameters
    ----------
    model_type : str
        One of ``"cph"``, ``"rsf"``, or ``"gbsa"``.
    random_state : int, numpy.random.RandomState, or None
        Seed / random state passed to the stochastic models (RSF, GBSA).
        Cox PH is deterministic and ignores it.

    Returns
    -------
    estimator
        An unfitted scikit-survival estimator.

    Raises
    ------
    ValueError
        If ``model_type`` is not a recognized key.
    """
    if model_type == "cph":
        return CoxPHSurvivalAnalysis()

    if model_type == "rsf":
        return RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_type == "gbsa":
        return GradientBoostingSurvivalAnalysis(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=3,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown model_type {model_type!r}. Expected one of {MODEL_TYPES}."
    )