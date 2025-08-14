This notebook is inspired by the official scikit-survival documentation on evaluating survival models found [here](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html), but has been signigicantly extended to:
- Simulate datasets with multiple features and customizable hazard ratios to reflect real world covariate effects
- Add support for censoring control and convergence checks
- Compare multiple survival models (CoxPH, RSF, GBSA)
- Compute additional metrics including Uno’s C, Brier Score, and Cumulative/Dynamic AUC
- Visualize performance under different censoring rates