"""Train on past survey years, choose on 2023, test once on 2025 (a year the model never saw).

Only odd years are used: blood pressure and cholesterol are asked in odd years only.
  1. Fit candidates on 2011-2021, compare them on 2023 (PR-AUC): gradient boosting settings, logistic
     regression, and gradient boosting WITHOUT blood pressure and cholesterol (to measure what they add).
  2. Choose the decision thresholds on the 2023 predictions.
  3. Refit the chosen models on 2011-2023 and save them. evaluate.py then tests them on 2025.

Run after build.py: python -m heart_risk.brfss.train   (about 15-20 minutes on a laptop)
"""
import json
import platform
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from ..paths import PROJECT
from ..metrics import threshold_for_recall, threshold_max_f1
from .build import PROCESSED
from .model import FEATURES, NOMINAL_FEATURES, ORDERED_FEATURES, WITHOUT_BP_CHOL, gradient_boosting, logistic_regression

TRAIN_YEARS = [2011, 2013, 2015, 2017, 2019, 2021]
VALID_YEAR, TEST_YEAR = 2023, 2025
MODELS_DIR = PROJECT / "models"
HGB_SETTINGS = {  # a small, deliberate search; each fit sees 2.7 million answers
    "lr0.1_leaves31": dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200),
    "lr0.1_leaves63": dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200),
    "lr0.05_leaves63": dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=500),
    "lr0.05_leaves127": dict(learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=1000),
}


def load(years, columns=FEATURES):
    df = pd.read_parquet(PROCESSED, columns=["year", "weight", "heart_disease", *columns],
                         filters=[("year", "in", list(years))])
    df = df[df["heart_disease"].notna()]
    X = df[columns].copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype(object)  # plain text answers, exactly what the app sends
    return X, (df["heart_disease"] == "Yes").astype(int).to_numpy(), df


def scores(y, p):
    return {"pr_auc": round(average_precision_score(y, p), 4), "roc_auc": round(roc_auc_score(y, p), 4)}


def main():
    t0 = time.time()
    X_tr, y_tr, _ = load(TRAIN_YEARS)
    X_va, y_va, _ = load([VALID_YEAR])
    print(f"train {len(y_tr):,} rows ({y_tr.mean():.2%} heart disease), validation {len(y_va):,}", flush=True)

    valid, preds = {}, {}
    for name, params in HGB_SETTINGS.items():
        m = gradient_boosting(**params).fit(X_tr, y_tr)
        preds[name] = m.predict_proba(X_va)[:, 1]
        valid[name] = {**scores(y_va, preds[name]), "trees": int(m.named_steps["model"].n_iter_)}
        print(f"  gb {name:18s} {valid[name]}  {time.time() - t0:.0f}s", flush=True)
    best = max(HGB_SETTINGS, key=lambda n: valid[n]["pr_auc"])

    m = logistic_regression().fit(X_tr, y_tr)
    preds["logistic_regression"] = m.predict_proba(X_va)[:, 1]
    valid["logistic_regression"] = scores(y_va, preds["logistic_regression"])
    print(f"  logistic regression {valid['logistic_regression']}  {time.time() - t0:.0f}s", flush=True)

    m = gradient_boosting(WITHOUT_BP_CHOL, **HGB_SETTINGS[best]).fit(X_tr[WITHOUT_BP_CHOL], y_tr)
    preds["gb_without_bp_chol"] = m.predict_proba(X_va[WITHOUT_BP_CHOL])[:, 1]
    valid["gb_without_bp_chol"] = scores(y_va, preds["gb_without_bp_chol"])
    print(f"  gb without BP/chol  {valid['gb_without_bp_chol']}  {time.time() - t0:.0f}s", flush=True)

    candidates = {"gradient_boosting": ("gb", best), "logistic_regression": ("lr", "logistic_regression"),
                  "gb_without_bp_chol": ("gb_nobp", "gb_without_bp_chol")}
    thresholds = {k: {"best_f1": round(threshold_max_f1(y_va, preds[v]), 4),
                      "screening_recall_80": round(threshold_for_recall(y_va, preds[v], 0.80), 4)}
                  for k, (_, v) in candidates.items()}

    # Refit on everything before the test year.
    X_all, y_all, raw_all = load([*TRAIN_YEARS, VALID_YEAR])
    MODELS_DIR.mkdir(exist_ok=True)
    final = {"gradient_boosting": gradient_boosting(**HGB_SETTINGS[best]).fit(X_all, y_all),
             "logistic_regression": logistic_regression().fit(X_all, y_all),
             "gb_without_bp_chol": gradient_boosting(WITHOUT_BP_CHOL, **HGB_SETTINGS[best]).fit(X_all[WITHOUT_BP_CHOL], y_all)}
    for name, model in final.items():
        joblib.dump(model, MODELS_DIR / f"brfss_{name}.joblib", compress=3)
    print(f"refit on {len(y_all):,} rows, saved  {time.time() - t0:.0f}s", flush=True)

    # Context for the app: weighted heart disease rate by age group and sex, 2021-2025 (all recent years).
    recent = pd.read_parquet(PROCESSED, columns=["year", "weight", "age_group", "sex", "heart_disease"],
                             filters=[("year", ">=", 2021)])
    recent = recent[recent["heart_disease"].notna() & recent["age_group"].notna() & recent["sex"].notna()]
    recent["hd"] = (recent["heart_disease"] == "Yes").astype(float)
    rates = recent.groupby(["age_group", "sex"], observed=True).apply(
        lambda g: np.average(g["hd"], weights=g["weight"]), include_groups=False).unstack()
    overall = float(np.average(recent["hd"], weights=recent["weight"]))

    meta = {
        "trained_on": str(date.today()), "app_model": "gradient_boosting", "gb_settings": {best: HGB_SETTINGS[best]},
        "train_years": TRAIN_YEARS, "valid_year": VALID_YEAR, "test_year": TEST_YEAR,
        "final_train_rows": int(len(y_all)), "prevalence_final_train": round(float(y_all.mean()), 4),
        "features": FEATURES, "features_without_bp_chol": WITHOUT_BP_CHOL,
        "answers": {**ORDERED_FEATURES, **NOMINAL_FEATURES},
        "validation_2023": valid, "thresholds": thresholds,
        "weighted_rate_2021_2025": round(overall, 4),
        "weighted_rates_by_age_sex_2021_2025": {str(a): {s: round(float(v), 4) for s, v in row.items()}
                                                for a, row in rates.iterrows()},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (MODELS_DIR / "brfss_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"best gb settings: {best}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
