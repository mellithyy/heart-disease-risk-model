"""Test the model without blood pressure and cholesterol on every even survey year (2012-2024).

Why: the main model trains on odd years only, because blood pressure and cholesterol are asked in odd
years. So the even years are never used for training. The model without those two questions can be
scored on them, which gives 7 more unseen years (about 3 million people) to check that the 2025 result
was not luck.

Run after train.py: python -m heart_risk.brfss.robustness
Writes reports/brfss_unseen_years.json
"""
import json

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

from ..paths import PROJECT
from .train import MODELS_DIR, load

EVEN_YEARS = [2012, 2014, 2016, 2018, 2020, 2022, 2024]


def main():
    meta = json.loads((MODELS_DIR / "brfss_metadata.json").read_text(encoding="utf-8"))
    features = meta["features_without_bp_chol"]
    model = joblib.load(MODELS_DIR / "brfss_gb_without_bp_chol.joblib")

    years = {}
    for year in EVEN_YEARS:  # one year at a time keeps memory low
        X, y, raw = load([year], features)
        p = model.predict_proba(X)[:, 1]
        years[year] = {"rows": int(len(y)), "prevalence": round(float(y.mean()), 4),
                       # below the prevalence = the model under-predicts that year (see the README)
                       "mean_predicted": round(float(p.mean()), 4),
                       "roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
                       "weighted_roc_auc": round(roc_auc_score(y, p, sample_weight=raw["weight"].to_numpy()), 4)}
        print(year, years[year], flush=True)

    report = {"model": "gb_without_bp_chol", "trained_on": "odd years 2011-2023", "years": years}
    (PROJECT / "reports" / "brfss_unseen_years.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
