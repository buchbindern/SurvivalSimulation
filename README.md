# Survival Model Simulation

A reproducible simulation study comparing four survival models — **Cox Proportional Hazards (CoxPH)**, **Random Survival Forest (RSF)**, **Gradient Boosting Survival Analysis (GBSA)**, and **DeepSurv**, a PyTorch neural network trained with the Cox partial likelihood loss — under increasing levels of right-censoring.

The project generates synthetic survival data with controllable censoring, fits each model across repeated trials, and evaluates them with concordance, time-dependent AUC, and the integrated Brier score. It is inspired by the [scikit-survival guide on evaluating survival models](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html), extended to compare four models including a custom neural network, add IPCW-corrected metrics, control the censoring rate directly, and visualize how performance degrades as censoring grows.

## Methods

**Data.** Survival times are drawn from an exponential model whose hazard depends on uniformly sampled covariates (following Bender et al., 2005). Right-censoring is introduced by numerically solving for the censoring distribution that yields a target rate (0%, 25%, 50%), with a convergence check so failed trials are skipped.

**Models.**
- **CoxPH** — linear baseline.
- **RSF** — non-linear, ensemble of survival trees.
- **GBSA** — gradient-boosted survival.
- **DeepSurv** — a feedforward PyTorch network (Faraggi & Simon, 1995; Katzman et al., 2018) trained with the Cox partial likelihood loss instead of a standard regression loss, since the target is relative risk ordering under censoring, not a direct regression target. Implemented as a scikit-survival-compatible estimator (`src/deepsurv_survival_analysis.py`) — it exposes the same `fit` / `predict` / `predict_survival_function` interface as the three classical models, so it drops into the same comparison pipeline unchanged. Because the network only outputs a relative risk score natively, a Breslow (1972) baseline hazard estimator (`src/breslow.py`) is fit on the training risk scores to convert it into a survival curve for the Brier score.

**Evaluation.** Each dataset is split 70/30 into train and test (stratified on the event indicator), models are fit on the training split, and all metrics are computed on the held-out test split — using the training split as the censoring reference for the IPCW-based metrics. Results are averaged over repeated trials per censoring level.

**Metrics.**

| Metric | Measures | Notes |
|---|---|---|
| Harrell's C | Rank concordance | Ignores the censoring distribution |
| Uno's C | Rank concordance | IPCW-corrected; robust under heavy censoring |
| Time-dependent AUC | Discrimination over time | Mean across the evaluation horizon |
| Integrated Brier Score | Calibration + discrimination | **Lower is better** |

`Actual C` and `Baseline AUC` are oracle values from the ground-truth risk scores, included as an upper reference.

## Results

100 repeated trials per censoring level, n=1000 samples, 8 covariates.

### Model comparison

![Model comparison](results/figures/model_comparison.png)

**CoxPH leads on every metric at every censoring level, with DeepSurv a consistent close second — both clearly ahead of the tree-based ensembles.** This is the expected result rather than a disappointing one for DeepSurv: the data is generated from an exactly linear Cox proportional hazards process, so the correctly-specified linear model (CoxPH) has a structural advantage that no amount of model flexibility can overcome on this particular data-generating process. The more informative comparison is DeepSurv against RSF and GBSA, none of which assume linearity either — DeepSurv outperforms both on every metric at every censoring level, despite all three having comparable capacity to find non-linear structure that, here, doesn't exist.

### Performance vs. censoring

![Performance vs. censoring](results/figures/censoring_performance.png)

The Brier score makes the separation clearest: prediction error rises with censoring for every model, **CoxPH and DeepSurv stay closely matched and well ahead of GBSA and RSF throughout**, and RSF degrades fastest of all four.

Per-model breakdowns (concordance / AUC / Brier, oracle vs. fitted, by censoring level) are in `results/figures/{model}_detail.png` for `cph`, `rsf`, `gbsa`, and `deepsurv`.

**Takeaway.** Because evaluation is fully out of sample, the comparison rewards the model whose functional form matches the data-generating process rather than the most flexible one — a concrete illustration of why held-out evaluation matters when comparing survival models, and why a more flexible model (DeepSurv, RSF, GBSA) isn't automatically better than a correctly-specified simpler one (CoxPH).

## Repository structure

```
SurvivalSimulation/
├── README.md
├── requirements.txt
├── run_experiment.py              # CLI entry point: runs the full sweep, saves results + figures
├── src/
│   ├── data_generation.py         # simulate survival data with controllable censoring
│   ├── models.py                  # define CoxPH, RSF, GBSA, DeepSurv
│   ├── deepsurv.py                # DeepSurv network + Cox partial likelihood loss (PyTorch)
│   ├── deepsurv_survival_analysis.py  # scikit-survival-compatible wrapper around DeepSurv
│   ├── breslow.py                 # Breslow baseline hazard estimator (for DeepSurv's survival function)
│   ├── evaluation.py              # compute the evaluation metrics
│   ├── simulation.py              # run the experiment loop and aggregate results
│   └── plotting.py                # produce the comparison figures
├── notebooks/
│   └── survival_simulation_demo.ipynb  # optional: interactive exploration, same src/ pipeline
└── results/
    ├── model_comparison.csv
    └── figures/
        ├── model_comparison.png
        ├── censoring_performance.png
        └── {cph,rsf,gbsa,deepsurv}_detail.png
```

## Environment setup

A dedicated environment is recommended (the compiled dependencies — scikit-survival, NumPy, matplotlib, PyTorch — need to be built against a single, consistent NumPy version).

```bash
conda create -n survival python=3.11
conda activate survival
pip install -r requirements.txt
```

If `pip` fails to build scikit-survival, install that one through conda (which ships prebuilt binaries) and the rest through pip:

```bash
conda install -c conda-forge scikit-survival
pip install -r requirements.txt
```

Tested with Python 3.11, scikit-survival 0.22+, and PyTorch 2.x (CPU or GPU).

## Running it

```bash
python run_experiment.py --n_samples 1000 --m 8 --n_repeats 100
```

This runs the full sweep (CoxPH, RSF, GBSA, DeepSurv × three censoring levels × `n_repeats` trials each), and writes `results/model_comparison.csv` and `results/figures/*.png`. Lower `--n_repeats` for a quick pass (DeepSurv is the slow step — training a network per trial, vs. an instant fit for the three classical models) or raise it for smoother error bars. Full options:

```bash
python run_experiment.py --help
```

Or call the pipeline directly:

```python
from src.simulation import run_simulation
from src.plotting import plot_model_comparison, plot_censoring_performance

results = run_simulation(n_samples=1000, m=8, n_repeats=100, random_state=42)
plot_model_comparison(results)
plot_censoring_performance(results)
```

**To explore interactively** instead of from the command line, `notebooks/survival_simulation_demo.ipynb` calls the same `src/` pipeline shown above — register the environment as a kernel first:

```bash
python -m ipykernel install --user --name survival --display-name "Python (survival)"
jupyter lab notebooks/survival_simulation_demo.ipynb
```

## Reference

Bender, R., Augustin, T., & Blettner, M. (2005). Generating survival times to simulate Cox proportional hazards models. *Statistics in Medicine*, 24(11), 1713–1723.

Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18(1), 24.