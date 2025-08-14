import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from HelperFunctions.GenerateData import generate_survival_data



def simulation(n_samples, m, n_repeats=100, time_points=10):
    """
    Simulate survival analysis experiments to evaluate model performance over varying levels of censorship.

    This function repeatedly generates synthetic survival data with different right censoring proportions, 
    and fits three survival models: Cox Proportional Hazards (CPH), Random Survival Forest (RSF), 
    and Gradient Boosting Survival Analysis (GBSA). It evaluates them using concordance indexes, 
    time dependent AUC, and integrated Brier score.

    Parameters:
    - n_samples: int, number of synthetic samples to generate per repeat.
    - m: int, number of features (covariates) in the synthetic data.
    - n_repeats: int, number of repeated simulations per censoring level.
    - time_points: int, number of evenly spaced evaluation time points for AUC and Brier score.

    Returns:
    - results: Dictionary of results with model types as keys (cph, rsf, gbsa). 
        Each model key maps to another dict with:
            - mean: List of DataFrames containing mean metric values across repeats for each censoring level.
            - std: List of DataFrames containing standard deviations of metrics across repeats.
            - censoring: List of observed censoring rates for each repeat.

        Each DataFrame includes the following evaluation metrics:
            - censoring: Observed censoring level
            - Actual C: Concordance with ground-truth risk scores
            - Harrel's C: Harrell’s Concordance index
            - Uno's C: Uno’s Concordance index
            - Baseline AUC: AUC using ground-truth risk scores
            - AUC: Model-predicted AUC
            - Brier: Integrated Brier Score 
    """

    rnd = np.random.RandomState(42) 
    
    measures = (
        "censoring",
        "Actual C",
        "Harrel's C",
        "Uno's C",
        "Baseline AUC",
        "AUC",
        "Brier",
    )
    results = {
        "cph": {"mean": [], "std": [], "censoring": []},
        "rsf": {"mean": [], "std": [], "censoring": []},
        "gbsa": {"mean": [], "std": [], "censoring": []}
    }

    # iterate over different amount of censoring
    for cens in [0, 0.25, 0.5]:
        count = 0
        data = {
            "cph": {measure: [] for measure in measures},
            "rsf": {measure: [] for measure in measures},
            "gbsa": {measure: [] for measure in measures}
        }

        # repeaditly perform simulation (put this as mid loop censor -> repeat -> model)
        for _ in range(n_repeats):
            
            X_test, y_test, y_train, actual_c, converged, hazard_ratio, true_risk_scores, baseline_mean_auc, eval_time_points = generate_survival_data(
                n_samples, m, baseline_hazard=0.1, percentage_cens=cens, rnd=rnd, time_points=time_points
            )
        
            if not converged:
                continue  # Skip this repeat if convergence failed both times

            count += 1

            for model_type in ["cph", "rsf", "gbsa"]:
                data_mean = {}
                data_std = {}
                for measure in measures:
                    data_mean[measure] = []
                    data_std[measure] = []

                if model_type == "cph":
                    model = CoxPHSurvivalAnalysis()
                elif model_type == "rsf":
                    model = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, 
                            max_features="sqrt", n_jobs=-1, random_state=rnd)
                else:
                    model = GradientBoostingSurvivalAnalysis(learning_rate=0.1, n_estimators=100,max_depth=3,random_state=rnd,
                    )
                  
                model.fit(X_test, y_test)

             
                risk_scores = model.predict(X_test) 

                times = np.linspace(y_test["time"].min() + 0.001, y_test["time"].max() - 0.001, time_points)

                aucs, _ = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
                mean_auc = np.nanmean(aucs) 

                pred_func = model.predict_survival_function(X_test) # the predict survival fundtion  
                preds = np.asarray([[fn(t) for t in times] for fn in pred_func]) # the actual points on the function
                brier = integrated_brier_score(y_train, y_test, preds, times)

                c_actual = actual_c
                c_harrell = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)
                c_uno = concordance_index_ipcw(y_train, y_test, risk_scores)

                data[model_type]["censoring"].append(100.0 - y_test["event"].sum() * 100.0 / y_test.shape[0])
                data[model_type]["Actual C"].append(c_actual)
                data[model_type]["Harrel's C"].append(c_harrell[0])
                data[model_type]["Uno's C"].append(c_uno[0])
                data[model_type]["Baseline AUC"].append(baseline_mean_auc)
                data[model_type]["AUC"].append(mean_auc)
                data[model_type]["Brier"].append(brier)

        for model_type in ["cph", "rsf", "gbsa"]:
            data_mean = {key: [np.mean(value)] for key, value in data[model_type].items()}
            data_std = {key: [np.std(value, ddof=1)] for key, value in data[model_type].items()}

            results[model_type]["mean"].append(pd.DataFrame(data_mean))
            results[model_type]["std"].append(pd.DataFrame(data_std))


    return results