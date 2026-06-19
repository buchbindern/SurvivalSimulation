"""
Train DeepSurv on simulated survival data.

Usage:
    python -m src.train --epochs 200 --lr 1e-3

Cox partial likelihood is only valid over a risk set that's representative
of the population, so -- unlike the MSE-trained BatteryCNN in the sibling
battery-rul-cnn repo -- this trains full-batch (entire training set per
gradient step) by default rather than with small shuffled minibatches.
This matches standard DeepSurv practice (Katzman et al. 2018) and is
computationally fine here since simulated survival datasets are modest in
size (low thousands of rows).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from src.data_simulation import simulate_survival_data, train_test_split_survival
from src.deepsurv import DeepSurv, cox_ph_loss

RESULTS_DIR = "results/figures"
CHECKPOINT_PATH = "results/deepsurv_checkpoint.pt"


def standardize(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using train-set statistics only (no leakage from val)."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def train(
    epochs: int = 200,
    lr: float = 1e-3,
    hidden_dims: tuple[int, ...] = (32, 32),
    dropout: float = 0.4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    patience: int = 20,
    seed: int = 42,
):
    torch.manual_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # NOTE: replace this with the actual simulator from
    # SurvivalSimulation.ipynb if its interface differs from the reference
    # version in src/data_simulation.py.
    X, durations, events = simulate_survival_data(n_samples=2000, n_features=8, random_state=seed)
    X_train_full, X_test, durations_train_full, durations_test, events_train_full, events_test = (
        train_test_split_survival(X, durations, events, test_size=0.3, random_state=seed)
    )

    # Further split the training portion into train/val for early stopping.
    # Test set stays fully held out for evaluate.py.
    X_train, X_val, durations_train, durations_val, events_train, events_val = train_test_split_survival(
        X_train_full, durations_train_full, events_train_full, test_size=val_fraction, random_state=seed
    )

    X_train, X_val, mean, std = standardize(X_train, X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    durations_train_t = torch.tensor(durations_train, dtype=torch.float32)
    events_train_t = torch.tensor(events_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    durations_val_t = torch.tensor(durations_val, dtype=torch.float32)
    events_val_t = torch.tensor(events_val, dtype=torch.float32)

    model = DeepSurv(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        log_risk_train = model(X_train_t)
        loss = cox_ph_loss(log_risk_train, durations_train_t, events_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            log_risk_val = model(X_val_t)
            val_loss = cox_ph_loss(log_risk_val, durations_val_t, events_val_t)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss - 1e-4:
            best_val_loss = val_loss.item()
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train_loss {loss.item():.4f}  val_loss {val_loss.item():.4f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "feature_mean": mean,
            "feature_std": std,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Cox partial likelihood loss")
    ax.set_title("DeepSurv training curve")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "deepsurv_training_curve.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved training curve to {fig_path}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSurv on simulated survival data.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )