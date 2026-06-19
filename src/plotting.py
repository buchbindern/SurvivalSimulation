"""
Plotting utilities for the survival-model comparison.

Three functions:

- :func:`plot_model_comparison` -- the full scorecard: every metric, all three
  models side by side at each censoring level. Answers "which model is best?"
- :func:`plot_censoring_performance` -- the robustness story: headline metrics
  plotted against censoring level, one line per model. Answers "how does
  censoring degrade performance?"
- :func:`plot_results` -- per-model detail (a 3-panel breakdown for a single
  model), useful for drilling in.

All functions accept an optional ``save_path`` and return the Matplotlib
figure so callers can display or further customize it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models import MODEL_TYPES, MODEL_NAMES


# Consistent color per model across every figure.
MODEL_COLORS = {
    "cph": "#4C72B0",     # blue
    "rsf": "#DD8452",     # orange
    "gbsa": "#55A868",    # green
    "deepsurv": "#C44E52",  # red
}

# Metrics where a higher value is better vs. lower is better.
LOWER_IS_BETTER = {"Brier"}


def _assemble(results, model_type):
    """Concatenate the per-censoring-level rows into mean / std frames."""
    mean_df = pd.concat(results[model_type]["mean"], ignore_index=True)
    std_df = pd.concat(results[model_type]["std"], ignore_index=True)
    return mean_df, std_df


def _censoring_labels(results):
    """Observed mean censoring (%) per level, as tick labels."""
    mean_df, _ = _assemble(results, MODEL_TYPES[0])
    return [f"{c:.0f}%" for c in mean_df["censoring"].values]


def _finalize(fig, save_path):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(results, save_path=None):
    """
    Grouped-bar scorecard: each metric in its own panel, three models per
    censoring level, with standard-deviation error bars.
    """
    metrics = ["Harrell's C", "Uno's C", "AUC", "Brier"]
    labels = _censoring_labels(results)
    x = np.arange(len(labels))
    n_models = len(MODEL_TYPES)
    width = 0.8 / n_models

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        for i, mt in enumerate(MODEL_TYPES):
            mean_df, std_df = _assemble(results, mt)
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(
                x + offset, mean_df[metric].values, width,
                yerr=std_df[metric].values, capsize=3,
                color=MODEL_COLORS[mt], label=MODEL_NAMES[mt],
                edgecolor="white", linewidth=0.5,
            )

        direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        ax.set_title(f"{metric}  ({direction})")
        ax.set_xlabel("Censoring rate")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric not in LOWER_IS_BETTER:
            ax.set_ylim(0, 1)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Survival Model Comparison Across Censoring Levels",
                 y=1.06, fontsize=14, fontweight="bold")
    return _finalize(fig, save_path)


def plot_censoring_performance(results, save_path=None):
    """
    Line plot of headline metrics vs. censoring level, one line per model with
    a shaded +/- 1 std band. Shows how performance degrades as censoring rises.
    """
    metrics = ["Uno's C", "Brier"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric in zip(axes, metrics):
        for mt in MODEL_TYPES:
            mean_df, std_df = _assemble(results, mt)
            cens = mean_df["censoring"].values
            mean = mean_df[metric].values
            std = std_df[metric].values
            ax.plot(cens, mean, marker="o", color=MODEL_COLORS[mt],
                    label=MODEL_NAMES[mt], linewidth=2)
            ax.fill_between(cens, mean - std, mean + std,
                            color=MODEL_COLORS[mt], alpha=0.15)

        direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        ax.set_title(f"{metric}  ({direction})")
        ax.set_xlabel("Mean censoring rate (%)")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False)
    fig.suptitle("Performance vs. Censoring", y=1.02,
                 fontsize=14, fontweight="bold")
    return _finalize(fig, save_path)


def plot_results(results, model_type="rsf", save_path=None, **kwargs):
    """
    Per-model detail: 3-panel breakdown (concordance / AUC / Brier) for a single
    model across censoring levels.
    """
    data_mean = pd.concat(results[model_type]["mean"])
    data_std = pd.concat(results[model_type]["std"])

    index = pd.Index(data_mean["censoring"].round(3), name="mean percentage censoring")
    for df in (data_mean, data_std):
        df.drop("censoring", axis=1, inplace=True)
        df.index = index

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

    cindex_columns = ["Actual C", "Harrell's C", "Uno's C"]
    data_mean[cindex_columns].plot.bar(
        yerr=data_std[cindex_columns], ax=axes[0],
        width=0.7, linewidth=0.5, capsize=4, **kwargs,
    )
    axes[0].set_ylabel("Concordance")
    axes[0].set_title("Concordance Index")
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.0, color="gray", linewidth=0.8)

    auc_columns = ["Baseline AUC", "AUC"]
    data_mean[auc_columns].plot.bar(
        yerr=data_std[auc_columns], ax=axes[1],
        width=0.7, linewidth=0.5, capsize=4, **kwargs,
    )
    axes[1].set_ylabel("AUC Score")
    axes[1].set_title("Time-dependent AUC")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[1].set_ylim(0, 1.0)

    data_mean[["Brier"]].plot.bar(
        yerr=data_std[["Brier"]], ax=axes[2],
        width=0.7, linewidth=0.5, capsize=4, **kwargs,
    )
    axes[2].set_ylabel("Brier Score")
    axes[2].set_title("Integrated Brier Score")
    axes[2].yaxis.grid(True, linestyle="--", alpha=0.7)
    axes[2].set_ylim(0, 0.3)

    for ax in axes:
        ax.set_xlabel("Mean Percentage Censoring")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelrotation=90)

    return _finalize(fig, save_path)