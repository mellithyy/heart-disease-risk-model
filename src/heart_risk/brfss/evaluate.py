"""Test the saved models once on 2025, a survey year none of them saw, and draw the report charts.

Run after train.py: python -m heart_risk.brfss.evaluate
Writes reports/brfss_model.json and reports/figures/brfss_*.png
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from ..charts import INK_2, MUTED, SERIES, plt, save
from ..paths import PROJECT
from ..metrics import summary
from .train import MODELS_DIR, TEST_YEAR, load

FIGURES = PROJECT / "reports" / "figures"
LABELS = {"gradient_boosting": "Gradient boosting", "logistic_regression": "Logistic regression",
          "gb_without_bp_chol": "Gradient boosting, no BP/cholesterol"}
NICE = {"age_group": "Age group", "gen_health": "General health", "income": "Income", "education": "Education",
        "sex": "Sex", "race": "Race", "smoked_100": "Ever smoked", "heavy_drinker": "Heavy drinking",
        "exercise": "Exercise", "diff_walking": "Difficulty walking", "stroke": "Stroke",
        "high_blood_pressure": "High blood pressure", "high_cholesterol": "High cholesterol",
        "diabetes": "Diabetes", "asthma": "Asthma", "kidney_disease": "Kidney disease", "copd": "COPD",
        "depression": "Depression", "arthritis": "Arthritis", "bmi": "BMI", "phys_days": "Poor physical-health days",
        "ment_days": "Poor mental-health days"}


def main():
    meta = json.loads((MODELS_DIR / "brfss_metadata.json").read_text(encoding="utf-8"))
    X, y, raw = load([TEST_YEAR])
    w = raw["weight"].to_numpy()
    models = {n: joblib.load(MODELS_DIR / f"brfss_{n}.joblib") for n in LABELS}
    cols = {n: meta["features_without_bp_chol"] if n == "gb_without_bp_chol" else meta["features"] for n in LABELS}
    probas = {n: m.predict_proba(X[cols[n]])[:, 1] for n, m in models.items()}

    results = {}
    for n, p in probas.items():
        t = meta["thresholds"][n]
        results[n] = {"best_f1": summary(y, p, t["best_f1"]), "screening_recall_80": summary(y, p, t["screening_recall_80"]),
                      "weighted_roc_auc": round(roc_auc_score(y, p, sample_weight=w), 4),
                      "weighted_pr_auc": round(average_precision_score(y, p, sample_weight=w), 4),
                      "mean_predicted": round(float(p.mean()), 4)}
    results["baseline_always_no"] = summary(y, np.zeros(len(y)), 0.5)

    gb = probas["gradient_boosting"]
    t_gb = meta["thresholds"]["gradient_boosting"]["best_f1"]
    age_idx = X["age_group"].map({a: i for i, a in enumerate(meta["answers"]["age_group"])})
    groups = {"Sex": X["sex"], "Age band": pd.cut(age_idx, [-1, 4, 8, 12], labels=["18-44", "45-64", "65+"]),
              "Race": X["race"].where(X["race"].isin(["White", "Black", "Hispanic", "Asian"]))}
    subgroups = {}
    for gname, g in groups.items():
        for value in [v for v in pd.Series(g).dropna().unique()]:
            m = (pd.Series(g).to_numpy() == value)
            r = summary(y[m], gb[m], t_gb)
            subgroups[f"{gname}: {value}"] = {"rows": int(m.sum()), "prevalence": round(float(y[m].mean()), 4),
                                              **{k: r[k] for k in ("roc_auc", "pr_auc", "recall", "precision")}}

    sample = X.sample(30_000, random_state=0)
    imp = permutation_importance(models["gradient_boosting"], sample, y[X.index.get_indexer(sample.index)],
                                 scoring="average_precision", n_repeats=3, random_state=0, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=sample.columns).sort_values(ascending=False)

    calibration = {}
    for n in ["gradient_boosting", "logistic_regression"]:
        obs, pred = calibration_curve(y, probas[n], n_bins=15, strategy="quantile")
        calibration[n] = {"predicted": np.round(pred, 4).tolist(), "observed": np.round(obs, 4).tolist()}
    report = {"test_year": TEST_YEAR, "test_rows": int(len(y)), "test_prevalence": round(float(y.mean()), 4),
              "results": results, "subgroups_gradient_boosting": subgroups, "calibration": calibration,
              "permutation_importance": importance.round(4).to_dict()}
    (PROJECT / "reports" / "brfss_model.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Precision-recall
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    for i, (n, p) in enumerate(probas.items()):
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=SERIES[i], lw=2 if i < 2 else 1.6, ls="-" if i < 2 else "--",
                label=f"{LABELS[n]} (PR-AUC {results[n]['best_f1']['pr_auc']:.3f})")
    r = results["gradient_boosting"]["best_f1"]
    ax.plot(r["recall"], r["precision"], "o", ms=8, color=SERIES[0], mec="#fcfcfb", mew=2)
    ax.axhline(y.mean(), color=MUTED, lw=1.2, ls=":")
    ax.text(0.02, y.mean() + 0.015, f"No skill: prevalence {y.mean():.1%}", color=MUTED, fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall (share of real cases found)", ylabel="Precision (share of flags that are right)")
    ax.set_title(f"Tested on {TEST_YEAR}, a year the models never saw")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, FIGURES / "brfss_precision_recall.png")

    # Calibration
    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    ax.plot([0, 0.6], [0, 0.6], color=MUTED, lw=1.2, ls="--")
    for i, n in enumerate(["gradient_boosting", "logistic_regression"]):
        obs, pred = calibration_curve(y, probas[n], n_bins=15, strategy="quantile")
        ax.plot(pred, obs, "-o", ms=4.5, color=SERIES[i], label=LABELS[n], mec="#fcfcfb", mew=1.2)
    ax.set(xlim=(0, 0.6), ylim=(0, 0.6), xlabel="Predicted probability (15 equal-size groups)", ylabel=f"Observed rate in {TEST_YEAR}")
    ax.set_title("Calibration on the unseen year")
    ax.legend(loc="upper left")
    save(fig, FIGURES / "brfss_calibration.png")

    # Importance
    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.barh([NICE.get(c, c) for c in top.index], top.values, height=0.6, color=SERIES[0])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in PR-AUC when the answer is shuffled")
    ax.set_title("What the model relies on (top 12)")
    save(fig, FIGURES / "brfss_importance.png")

    for n in LABELS:
        rr = results[n]["best_f1"]
        print(f"{n:22s} ROC-AUC {rr['roc_auc']:.4f} PR-AUC {rr['pr_auc']:.4f} F1 {rr['f1']:.4f} "
              f"P {rr['precision']:.3f} R {rr['recall']:.3f} | weighted ROC {results[n]['weighted_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
