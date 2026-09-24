"""Audit of the 2022 version of this project: how much did the evaluation overstate the model?

In 2022 the pipeline was:
  clean -> undersample the WHOLE dataset with Edited Nearest Neighbours (ENN, k=51)
        -> pick 9 features using the undersampled data -> split 80/20 -> tune and test.
ENN deletes the "No" answers that sit close to "Yes" answers, i.e. the hardest cases. Doing it before
the split removed them from the test set too, and raised the test prevalence from 9% to 25%.

This script runs:
  A. the 2022 pipeline as it was (reproduces the reported F1 of about 0.91),
  B. the same method in the correct order (split first, ENN on the training rows only),
     tested on untouched data with the real prevalence.
Run: python -m heart_risk.leakage_audit   (takes a few minutes: ENN needs 51 neighbours per row)
"""
import json

import numpy as np
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..charts import INK_2, MUTED, SERIES, plt, save
from .data import PROJECT, load
from .features import AGE_ORDER, GEN_HEALTH_ORDER
from ..metrics import summary

# The 9 features the 2022 version kept, and its final logistic regression settings.
FEATURES_2022 = ["AgeCategory", "DiffWalking", "PhysicalHealth", "Diabetic", "Stroke",
                 "Smoking", "KidneyDisease", "SkinCancer", "GenHealth"]
SEED_2022 = 101


def clean_2022(df: pd.DataFrame) -> pd.DataFrame:
    """The 2022 cleaning steps, reproduced as they were."""
    d = df.drop_duplicates(ignore_index=True)
    d = d.replace({"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                   "No, borderline diabetes": 0, "Yes (during pregnancy)": 1})
    d["GenHealth"] = d["GenHealth"].map({g: i for i, g in enumerate(GEN_HEALTH_ORDER)})
    d["AgeCategory"] = d["AgeCategory"].map({a: i for i, a in enumerate(AGE_ORDER)})
    d = pd.get_dummies(d, columns=["Race"], dtype=int)
    return d.astype(float)


def fit_and_score(X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver="liblinear", C=0.1)  # 2022 also set penalty="l2", the default
    model.fit(X_train[FEATURES_2022], y_train)
    proba = model.predict_proba(X_test[FEATURES_2022])[:, 1]
    return summary(y_test, proba, 0.5)  # 2022 used model.predict(), i.e. a 0.5 threshold


def main():
    d = clean_2022(load())
    X, y = d.drop(columns="HeartDisease"), d["HeartDisease"].astype(int)
    enn = EditedNearestNeighbours(n_neighbors=51, n_jobs=-1)

    # A. As in 2022: resample everything, then split.
    X_res, y_res = enn.fit_resample(X, y)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED_2022)
    a = fit_and_score(Xa_tr, ya_tr, Xa_te, ya_te)
    a["test_prevalence"] = round(float(ya_te.mean()), 4)
    a["rows_after_enn"] = int(len(y_res))

    # B. Same method, correct order: split first, resample the training rows only.
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED_2022)
    Xb_tr_res, yb_tr_res = enn.fit_resample(Xb_tr, yb_tr)
    b = fit_and_score(Xb_tr_res, yb_tr_res, Xb_te, yb_te)
    b["test_prevalence"] = round(float(yb_te.mean()), 4)

    metrics = json.loads((PROJECT / "reports" / "audit2022" / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((PROJECT / "models" / "audit2022" / "metadata.json").read_text(encoding="utf-8"))
    c = metrics["results"][meta["app_model"]]["best_f1"]

    audit = {"A_2022_as_reported": a, "B_2022_method_fair_test": b, "C_2026_model": c}
    (PROJECT / "reports" / "audit2022" / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for k, v in audit.items():
        print(f"{k:28s} F1 {v['f1']:.3f}  precision {v['precision']:.3f}  recall {v['recall']:.3f}  "
              f"accuracy {v['accuracy']:.3f}  ROC-AUC {v['roc_auc']:.3f}")

    labels = ["2022, as reported\n(test set after undersampling)", "2022 method,\nfair test", "2026 model,\nfair test"]
    values = [a["f1"], b["f1"], c["f1"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.barh(labels[::-1], values[::-1], height=0.55, color=[SERIES[0], SERIES[0], MUTED])
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", color=INK_2)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1-score for heart disease (higher is better)")
    ax.set_title("The 2022 F1 of 0.91 came from a leaky evaluation")
    save(fig, PROJECT / "reports" / "audit2022" / "figures" / "leakage_audit.png")


if __name__ == "__main__":
    np.random.seed(0)
    main()
