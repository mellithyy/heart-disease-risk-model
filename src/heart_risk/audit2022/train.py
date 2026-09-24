"""Train both candidate models on the training set and save them.

1. 5-fold stratified cross-validation on the TRAINING set only, giving out-of-fold probabilities.
2. Decision thresholds are chosen on those out-of-fold probabilities (never on the test set).
3. The model with the best out-of-fold PR-AUC is the one the app uses.
4. Both models are refitted on the full training set and saved to models/.

Run: python -m heart_risk.train   (from the src/ folder, or with src/ on PYTHONPATH)
"""
import json
import platform
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import PROJECT, RANDOM_STATE, load, split
from .features import AGE_ORDER, FEATURES
from ..metrics import summary, threshold_for_recall, threshold_max_f1
from .models import CANDIDATES

MODELS_DIR = PROJECT / "models" / "audit2022"


def main():
    df = load()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train {len(X_train):,} rows, test {len(X_test):,} rows, prevalence {y_train.mean():.2%}")

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, make in CANDIDATES.items():
        oof = cross_val_predict(make(), X_train, y_train, cv=folds, method="predict_proba", n_jobs=-1)[:, 1]
        t_f1 = threshold_max_f1(y_train, oof)
        t_screen = threshold_for_recall(y_train, oof, 0.80)
        results[name] = {"cv_best_f1": summary(y_train, oof, t_f1),
                         "cv_screening": summary(y_train, oof, t_screen)}
        print(f"{name:20s} CV PR-AUC {results[name]['cv_best_f1']['pr_auc']:.4f}  "
              f"ROC-AUC {results[name]['cv_best_f1']['roc_auc']:.4f}  "
              f"F1 {results[name]['cv_best_f1']['f1']:.4f} at threshold {t_f1:.3f}")

    best = max(results, key=lambda n: results[n]["cv_best_f1"]["pr_auc"])
    print("Chosen for the app:", best)

    MODELS_DIR.mkdir(exist_ok=True)
    for name, make in CANDIDATES.items():
        model = make().fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)

    # Context shown in the app: how common heart disease is for each age group and sex (training data).
    train = X_train.assign(y=y_train.values)
    rates = (train.groupby(["AgeCategory", "Sex"], observed=True)["y"].mean()
             .round(4).unstack().reindex(AGE_ORDER))
    metadata = {
        "trained_on": str(date.today()),
        "app_model": best,
        "features": FEATURES,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "prevalence_train": round(float(y_train.mean()), 4),
        "thresholds": {name: {"best_f1": r["cv_best_f1"]["threshold"],
                              "screening_recall_80": r["cv_screening"]["threshold"]}
                       for name, r in results.items()},
        "cross_validation": results,
        "rates_by_age_sex": {age: row.to_dict() for age, row in rates.iterrows()},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved models and metadata to", MODELS_DIR)


if __name__ == "__main__":
    main()
