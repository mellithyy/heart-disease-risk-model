"""Final evaluation on the untouched test set, plus the report figures.

Run after train.py: python -m heart_risk.evaluate
Writes reports/metrics.json and reports/figures/*.png
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from ..charts import INK_2, MUTED, NICE, SERIES, plt, save
from .data import PROJECT, RANDOM_STATE, features_and_target, load, split
from .features import AGE_ORDER, GEN_HEALTH_ORDER
from ..metrics import summary
from .models import CANDIDATES, LABELS

MODELS_DIR, REPORTS = PROJECT / "models" / "audit2022", PROJECT / "reports" / "audit2022"
FIGURES = REPORTS / "figures"


def eda_figure(df):
    """Heart disease rate by age group and by self-rated general health."""
    _, y = features_and_target(df)
    d = df.assign(y=y)
    by_age = d.groupby("AgeCategory")["y"].mean().reindex(AGE_ORDER) * 100
    by_health = d.groupby("GenHealth")["y"].mean().reindex(GEN_HEALTH_ORDER) * 100

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, s, title in [(a1, by_age, "By age group"), (a2, by_health, "By self-rated general health")]:
        ax.bar(range(len(s)), s.values, width=0.62, color=SERIES[0])
        ax.set_xticks(range(len(s)), s.index, rotation=45 if len(s) > 6 else 0, ha="right" if len(s) > 6 else "center")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, s.max() * 1.15)
        for i in (int(np.argmin(s.values)), int(np.argmax(s.values))):
            ax.annotate(f"{s.values[i]:.1f}%", (i, s.values[i]), xytext=(0, 4), textcoords="offset points",
                        ha="center", color=INK_2, fontsize=9)
    a1.set_ylabel("Reported heart disease (%)")
    fig.suptitle(f"Heart disease is rare overall ({d['y'].mean():.1%}) but rises steeply with age and poor health",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "eda_rates.png")


def pr_figure(y, probas, results, prevalence):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, (name, p) in enumerate(probas.items()):
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=SERIES[i], label=f"{LABELS[name]} (PR-AUC {results[name]['pr_auc']:.3f})")
        r = results[name]
        ax.plot(r["recall"], r["precision"], "o", ms=8, color=SERIES[i], mec="#fcfcfb", mew=2)
    ax.axhline(prevalence, color=MUTED, lw=1.2, ls="--")
    ax.text(0.02, prevalence + 0.015, f"No skill: prevalence {prevalence:.1%}", ha="left", color=MUTED, fontsize=9)
    ax.text(0.02, 0.03, "Dots: the chosen decision thresholds", ha="left", color=INK_2, fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall (share of real cases found)", ylabel="Precision (share of flags that are right)")
    ax.set_title("Precision vs recall on the untouched test set")
    ax.legend(loc="upper right")
    save(fig, FIGURES / "precision_recall.png")


def calibration_figure(y, probas):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 0.5], [0, 0.5], color=MUTED, lw=1.2, ls="--")
    ax.text(0.47, 0.44, "Perfect", color=MUTED, fontsize=9, ha="right", rotation=45)
    for i, (name, p) in enumerate(probas.items()):
        obs, pred = calibration_curve(y, p, n_bins=12, strategy="quantile")
        ax.plot(pred, obs, "-o", ms=5, color=SERIES[i], label=LABELS[name], mec="#fcfcfb", mew=1.5)
    ax.set(xlim=(0, 0.5), ylim=(0, 0.5), xlabel="Predicted probability (12 equal-size groups)", ylabel="Observed heart disease rate")
    ax.set_title("Calibration: predicted risk vs what happened")
    ax.legend(loc="upper left")
    save(fig, FIGURES / "calibration.png")


def importance_figure(model, X, y):
    """Permutation importance: how much PR-AUC drops when one answer is shuffled."""
    sample = X.sample(20_000, random_state=RANDOM_STATE)
    imp = permutation_importance(model, sample, y.loc[sample.index], scoring="average_precision",
                                 n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    s = pd.Series(imp.importances_mean, index=X.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh([NICE.get(c, c) for c in s.index], s.values, height=0.6, color=SERIES[0])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    ax.set_title("What the gradient boosting model relies on (top 10)")
    save(fig, FIGURES / "feature_importance.png")
    return s.sort_values(ascending=False).round(4).to_dict()


def odds_ratios(model):
    """Logistic regression coefficients as odds ratios (Yes vs No, one step up the age or health scale,
    or one standard deviation for the numeric answers)."""
    names = model.named_steps["prep"].get_feature_names_out()
    coef = model.named_steps["model"].coef_[0]
    s = pd.Series(np.exp(coef), index=names).round(3)
    return s.reindex(s.sub(1).abs().sort_values(ascending=False).index).to_dict()


def subgroups(X, y, p, threshold):
    bands = pd.cut(X["AgeCategory"].map({a: i for i, a in enumerate(AGE_ORDER)}),
                   [-1, 4, 8, 12], labels=["18-44", "45-64", "65+"])
    out = {}
    for col, groups in [("Sex", X["Sex"]), ("Age band", bands)]:
        for g in groups.dropna().unique():
            m = (groups == g).values
            r = summary(y[m], p[m], threshold)
            out[f"{col}: {g}"] = {"rows": int(m.sum()), "prevalence": round(float(y[m].mean()), 4),
                                  **{k: r[k] for k in ("roc_auc", "pr_auc", "recall", "precision")}}
    return out


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    meta = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    df = load()
    X_train, X_test, y_train, y_test = split(df)
    y_test = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    models = {n: joblib.load(MODELS_DIR / f"{n}.joblib") for n in CANDIDATES}
    probas = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}

    results = {}
    for n, p in probas.items():
        t = meta["thresholds"][n]
        results[n] = {"best_f1": summary(y_test, p, t["best_f1"]),
                      "screening_recall_80": summary(y_test, p, t["screening_recall_80"])}
    always_no = np.zeros(len(y_test))
    results["baseline_always_no"] = summary(y_test, always_no, 0.5)

    best = meta["app_model"]
    report = {
        "test_rows": len(y_test),
        "test_prevalence": round(float(y_test.mean()), 4),
        "results": results,
        "subgroups_app_model": subgroups(X_test, y_test.values, probas[best], meta["thresholds"][best]["best_f1"]),
        "odds_ratios_logistic_regression": odds_ratios(models["logistic_regression"]),
        "permutation_importance_app_model": importance_figure(models[best], X_test, y_test),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    eda_figure(df)
    pr_figure(y_test, probas, {n: results[n]["best_f1"] for n in probas}, y_test.mean())
    calibration_figure(y_test, probas)

    for n in probas:
        r = results[n]["best_f1"]
        print(f"{n:20s} test ROC-AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  F1 {r['f1']:.4f}  "
              f"precision {r['precision']:.4f}  recall {r['recall']:.4f}  accuracy {r['accuracy']:.4f}")
    print("baseline always-No accuracy", results["baseline_always_no"]["accuracy"])


if __name__ == "__main__":
    main()
