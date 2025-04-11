import numpy as np
import pandas as pd
import scipy.optimize as opt
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def generate_marker(n_samples, hazard_ratio, baseline_hazard, rnd):
    X = rnd.randn(n_samples, 1)
    hazard_ratio = np.array([hazard_ratio])
    logits = np.dot(X, np.log(hazard_ratio))
    u = rnd.uniform(size=n_samples)
    time_event = -np.log(u) / (baseline_hazard * np.exp(logits))
    X = np.squeeze(X)
    actual = concordance_index_censored(np.ones(n_samples, dtype=bool), time_event, X)
    return X, time_event, actual[0]

def generate_survival_data(n_samples, hazard_ratio, baseline_hazard, percentage_cens, rnd):
    X, time_event, actual_c = generate_marker(n_samples, hazard_ratio, baseline_hazard, rnd)

    def get_observed_time(x):
        rnd_cens = np.random.RandomState(0)
        time_censor = rnd_cens.uniform(high=x, size=n_samples)
        event = time_event < time_censor
        time = np.where(event, time_event, time_censor)
        return event, time

    def censoring_amount(x):
        event, _ = get_observed_time(x)
        cens = 1.0 - event.sum() / event.shape[0]
        return (cens - percentage_cens) ** 2

    res = opt.minimize_scalar(censoring_amount, method="bounded", bounds=(0, time_event.max()))
    event, time = get_observed_time(res.x)
    y = Surv.from_arrays(event=event, time=time)
    tau = y["time"].max()
    mask = time < tau
    X_masked = X[mask]
    y_masked = y[mask]

    return X_masked, y_masked, y, actual_c

def simulation(n_samples, hazard_ratio, n_repeats=3):
    measures = (
        "censoring",
        "Harrel's C",
        "Uno's C",
        "RSF C",
        "RSF C (test)",
        "Best Params"
    )
    data_mean = {}
    data_std = {}
    for measure in measures:
        data_mean[measure] = []
        data_std[measure] = []

    rnd = np.random.RandomState(seed=987)
    for cens in (0.1, 0.25, 0.4, 0.5, 0.6, 0.7):
        data = {
            "censoring": [],
            "Harrel's C": [],
            "Uno's C": [],
            "RSF C": [],
            "RSF C (test)": [],
            "Best Params": []
        }

        for _ in range(n_repeats):
            X_test, y_test, y, actual_c = generate_survival_data(
                n_samples, hazard_ratio, baseline_hazard=0.1, percentage_cens=cens, rnd=rnd
            )

            X_train, X_test_split, y_train, y_test_split = train_test_split(X_test.reshape(-1, 1), y_test, test_size=0.25, random_state=0)

            c_harrell = concordance_index_censored(y_test_split["event"], y_test_split["time"], X_test_split.flatten())
            c_uno = concordance_index_ipcw(y, y_test_split, X_test_split.flatten())

            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, 15],
                "min_samples_split": [5, 10, 15]
            }
            grid_rsf = GridSearchCV(RandomSurvivalForest(random_state=0), param_grid, cv=3, scoring="neg_mean_absolute_error")
            grid_rsf.fit(X_train, y_train)

            best_rsf = grid_rsf.best_estimator_

            risk_scores_train = best_rsf.predict(X_train)
            risk_scores_test = best_rsf.predict(X_test_split)
            c_rsf_train = concordance_index_censored(y_train["event"], y_train["time"], risk_scores_train)
            c_rsf_test = concordance_index_censored(y_test_split["event"], y_test_split["time"], risk_scores_test)

            data["censoring"].append(100.0 - y_test_split["event"].sum() * 100.0 / y_test_split.shape[0])
            data["Harrel's C"].append(actual_c - c_harrell[0])
            data["Uno's C"].append(actual_c - c_uno[0])
            data["RSF C"].append(actual_c - c_rsf_train[0])
            data["RSF C (test)"].append(actual_c - c_rsf_test[0])
            data["Best Params"].append(grid_rsf.best_params_)

        for key, values in data.items():
            if key != "Best Params":
                data_mean[key].append(np.mean(values))
                data_std[key].append(np.std(values, ddof=1))
            else:
                data_mean[key].append(values)

    data_mean = pd.DataFrame.from_dict(data_mean)
    data_std = pd.DataFrame.from_dict({k: v for k, v in data_std.items() if k != "Best Params"})
    return data_mean, data_std

def plot_results(data_mean, data_std, **kwargs):
    index = pd.Index(data_mean["censoring"].round(3), name="mean percentage censoring")
    for df in (data_mean, data_std):
        df.drop("censoring", axis=1, inplace=True)
        df.index = index

    ax = data_mean.plot.bar(yerr=data_std, **kwargs)
    ax.set_ylabel("Actual C - Estimated C")
    ax.yaxis.grid(True)
    ax.axhline(0.0, color="gray")
    return ax