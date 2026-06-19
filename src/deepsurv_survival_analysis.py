"""
A scikit-survival-compatible wrapper around the DeepSurv network.

The rest of this codebase (evaluation.evaluate_model, simulation.run_simulation,
plotting.*) is written against the scikit-survival estimator interface:
``model.fit(X, y)``, ``model.predict(X)`` (a risk score), and
``model.predict_survival_function(X)`` (one callable per subject, mapping a
time to a survival probability). CoxPHSurvivalAnalysis / RandomSurvivalForest
/ GradientBoostingSurvivalAnalysis all expose exactly that.

DeepSurv (src.deepsurv.DeepSurv) does not -- it is a raw PyTorch nn.Module
that only knows how to produce a log relative risk score, the way Cox models
do, but without scikit-learn's fit/predict ergonomics or a built-in route to
absolute survival probabilities. DeepSurvSurvivalAnalysis here is the
adapter: it owns training (full-batch Cox partial likelihood, with an
internal validation split for early stopping), and it owns the Breslow
estimator (src.breslow) needed to convert the network's risk score into a
survival curve for predict_survival_function.

This is what lets "deepsurv" become a fourth entry in models.MODEL_TYPES
with no changes anywhere else in the pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.optim import Adam

from src.breslow import fit_breslow_baseline_hazard, step_function_eval
from src.deepsurv import DeepSurv, cox_ph_loss


class _DeepSurvSurvivalFunction:
    """Per-subject callable matching the interface sksurv's StepFunction
    exposes: call it with a time (or array of times) to get survival
    probability. This is what lets evaluation.py's
    ``[fn(t) for t in times]`` pattern work unchanged for DeepSurv.
    """

    def __init__(self, risk_score: float, event_times: np.ndarray, cum_hazard: np.ndarray):
        self.risk_score = risk_score
        self.event_times = event_times
        self.cum_hazard = cum_hazard

    def __call__(self, t):
        scalar_input = np.ndim(t) == 0
        t_arr = np.atleast_1d(t)
        H0 = step_function_eval(t_arr, self.event_times, self.cum_hazard)
        s = np.exp(-H0 * np.exp(self.risk_score))
        return float(s[0]) if scalar_input else s


def _resolve_seed(random_state) -> int:
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, 2**31 - 1))
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    return int(np.random.randint(0, 2**31 - 1))


class DeepSurvSurvivalAnalysis:
    """scikit-survival-style estimator wrapping the DeepSurv network.

    Parameters mirror models.build_model's other entries: a `random_state`
    that accepts a plain int or a numpy RandomState (simulation.py passes
    the latter), so it's a drop-in alongside CoxPHSurvivalAnalysis etc.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 32),
        dropout: float = 0.4,
        lr: float = 1e-3,
        epochs: int = 200,
        weight_decay: float = 1e-4,
        patience: int = 20,
        val_fraction: float = 0.2,
        random_state=None,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def _split_for_early_stopping(self, n: int, events: np.ndarray, rng: np.random.RandomState):
        """Stratify on event status when there's enough data to do so;
        fall back to a plain split (or no split at all) for very small or
        heavily censored datasets, which the censoring-level sweep in
        simulation.py will produce at the high end of CENSORING_LEVELS.
        """
        idx_event = np.where(events == 1)[0]
        idx_censored = np.where(events == 0)[0]

        def split(idx):
            idx = rng.permutation(idx)
            n_val = int(len(idx) * self.val_fraction)
            if len(idx) > 1:
                n_val = max(1, n_val)
            return idx[n_val:], idx[:n_val]

        if len(idx_event) >= 4 and len(idx_censored) >= 4:
            tr_e, va_e = split(idx_event)
            tr_c, va_c = split(idx_censored)
            train_idx = np.concatenate([tr_e, tr_c])
            val_idx = np.concatenate([va_e, va_c])
        else:
            # Not enough samples in one class to stratify meaningfully --
            # use a plain random split, or skip validation entirely for
            # very small datasets (train on everything, no early stopping).
            all_idx = rng.permutation(n)
            n_val = max(1, int(n * self.val_fraction)) if n >= 10 else 0
            val_idx = all_idx[:n_val]
            train_idx = all_idx[n_val:]

        return train_idx, val_idx

    def fit(self, X: np.ndarray, y) -> "DeepSurvSurvivalAnalysis":
        seed = _resolve_seed(self.random_state)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        X = np.asarray(X, dtype=np.float32)
        durations_full = np.asarray(y["time"], dtype=np.float32)
        events_full = np.asarray(y["event"], dtype=np.float32)

        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X_std = (X - mean) / std
        self.feature_mean_ = mean
        self.feature_std_ = std

        train_idx, val_idx = self._split_for_early_stopping(len(X), events_full, rng)
        has_val = len(val_idx) > 0 and events_full[val_idx].sum() > 0

        model = DeepSurv(input_dim=X.shape[1], hidden_dims=self.hidden_dims, dropout=self.dropout)
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_train_t = torch.tensor(X_std[train_idx], dtype=torch.float32)
        durations_train_t = torch.tensor(durations_full[train_idx], dtype=torch.float32)
        events_train_t = torch.tensor(events_full[train_idx], dtype=torch.float32)

        if has_val:
            X_val_t = torch.tensor(X_std[val_idx], dtype=torch.float32)
            durations_val_t = torch.tensor(durations_full[val_idx], dtype=torch.float32)
            events_val_t = torch.tensor(events_full[val_idx], dtype=torch.float32)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            log_risk_train = model(X_train_t)
            loss = cox_ph_loss(log_risk_train, durations_train_t, events_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if not has_val:
                continue  # train for the full epoch budget, no early stopping signal available

            model.eval()
            with torch.no_grad():
                val_loss = cox_ph_loss(model(X_val_t), durations_val_t, events_val_t)

            if val_loss.item() < best_val_loss - 1e-4:
                best_val_loss = val_loss.item()
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                break

        if has_val and best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model
        self.model_.eval()

        # Refit the Breslow baseline hazard on the *full* training data
        # (not just the inner train subset used for early stopping), using
        # the final trained network's risk scores.
        with torch.no_grad():
            risk_full = model(torch.tensor(X_std, dtype=torch.float32)).numpy()
        self.breslow_times_, self.breslow_cumhazard_ = fit_breslow_baseline_hazard(
            durations_full, events_full, risk_full
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Risk score, higher = higher risk. Same sign convention as
        CoxPHSurvivalAnalysis.predict, so it's directly comparable in
        concordance_index_ipcw / cumulative_dynamic_auc.
        """
        X_std = (np.asarray(X, dtype=np.float32) - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            return self.model_(torch.tensor(X_std, dtype=torch.float32)).numpy()

    def predict_survival_function(self, X: np.ndarray):
        """One callable per subject -- see _DeepSurvSurvivalFunction."""
        risk_scores = self.predict(X)
        return np.array(
            [
                _DeepSurvSurvivalFunction(r, self.breslow_times_, self.breslow_cumhazard_)
                for r in risk_scores
            ]
        )
