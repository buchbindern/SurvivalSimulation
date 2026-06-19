"""
DeepSurv: a neural network extension of the Cox proportional hazards model.

Architecture follows Faraggi & Simon (1995), as formalized into a modern
deep learning model by Katzman et al., "DeepSurv: personalized treatment
recommender system using a Cox proportional hazards deep neural network",
BMC Medical Research Methodology, 2018.

The network outputs a single scalar per subject -- a log relative risk
score, analogous to the linear predictor (X @ beta) in classical Cox
regression. It is trained with the Cox partial likelihood loss instead of
MSE/MAE, since the target is relative risk ordering under censoring, not a
direct regression target.

Loss implementation follows the standard Breslow-tie-handling formulation
used in Kvamme et al.'s `pycox` library: sort by duration descending, then
use a cumulative log-sum-exp to compute, for every event, the log of the
total risk in its risk set.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSurv(nn.Module):
    """Feedforward network producing a single log relative risk score.

    Each hidden block is Linear -> BatchNorm -> ReLU -> Dropout. BatchNorm
    requires batch_size > 1; this is not an issue here since Cox loss is
    trained with large/full batches anyway (see train.py).
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (32, 32), dropout: float = 0.4):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log relative risk, shape (batch,)."""
        return self.net(x).squeeze(-1)


def cox_ph_loss(log_risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Negative log Cox partial likelihood (Breslow approximation for ties).

    Args:
        log_risk: model output, shape (n,), higher = higher risk of event.
        durations: time-to-event-or-censoring, shape (n,).
        events: 1 if event observed, 0 if censored, shape (n,).
        eps: numerical stability constant.

    Returns:
        Scalar loss. Differentiable w.r.t. log_risk.

    Note: this loss is only valid if `log_risk`/`durations`/`events` cover
    a risk set that's representative of the full population -- i.e. don't
    call this on tiny random minibatches. Use full-batch or large-batch
    training (see train.py).
    """
    if events.sum() == 0:
        # No events in this batch -- gradient is undefined under Cox
        # partial likelihood. Should not happen with full-batch training
        # on a real dataset; guard against it anyway.
        return torch.tensor(0.0, requires_grad=True, device=log_risk.device)

    order = torch.argsort(durations, descending=True)
    log_risk = log_risk[order]
    events = events[order].float()

    # log_cumsum_exp[i] = log(sum_{j<=i} exp(log_risk[j]))
    # Since subjects are sorted by duration descending, the risk set for
    # subject i is exactly {0, ..., i} (everyone with duration >= duration_i).
    log_cumsum_exp = torch.logcumsumexp(log_risk, dim=0)

    log_lik = (log_risk - log_cumsum_exp) * events
    return -log_lik.sum() / (events.sum() + eps)


@torch.no_grad()
def predict_risk(model: DeepSurv, x: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for inference-mode risk prediction."""
    model.eval()
    return model(x)