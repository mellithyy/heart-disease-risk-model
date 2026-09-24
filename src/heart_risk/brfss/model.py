"""Features and model pipelines for the CDC 2011-2025 data.

The pipelines take the harmonised answers as text ("Yes", "55-59", "Very good", or missing) and output a
probability, so the saved model is all the app needs. Missing answers are allowed everywhere: people skip
questions (income is missing for about 1 in 5), and the model should use what it gets.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .variables import ORDERED

YN = ["No", "Yes"]
ORDERED_FEATURES = {k: ORDERED[k] for k in ["age_group", "gen_health", "income", "education"]}
NOMINAL_FEATURES = {
    "sex": ["Female", "Male"],
    "race": ["White", "Black", "Asian", "American Indian/Alaska Native", "Hispanic", "Other"],
    "smoked_100": YN, "heavy_drinker": YN, "exercise": YN, "diff_walking": YN, "stroke": YN,
    "high_blood_pressure": ["No", "Borderline", "Pregnancy only", "Yes"],
    "high_cholesterol": ["No", "Never checked", "Yes"],
    "diabetes": ["No", "Borderline", "Pregnancy only", "Yes"],
    "asthma": YN, "kidney_disease": YN, "copd": YN, "depression": YN, "arthritis": YN,
}
NUMERIC_FEATURES = ["bmi", "phys_days", "ment_days"]
FEATURES = [*ORDERED_FEATURES, *NOMINAL_FEATURES, *NUMERIC_FEATURES]
# Not used as inputs: sleep (missing in most odd years), last check-up (people with heart disease see
# doctors more, so it reflects the diagnosis rather than risk), medication questions (same reason).
WITHOUT_BP_CHOL = [f for f in FEATURES if f not in ("high_blood_pressure", "high_cholesterol")]


def _encoder(groups: dict) -> OrdinalEncoder:
    return OrdinalEncoder(categories=list(groups.values()), handle_unknown="use_encoded_value",
                          unknown_value=np.nan, encoded_missing_value=np.nan)


def gradient_boosting(features=FEATURES, **params) -> Pipeline:
    ordered = [f for f in ORDERED_FEATURES if f in features]
    nominal = [f for f in NOMINAL_FEATURES if f in features]
    numeric = [f for f in NUMERIC_FEATURES if f in features]
    prep = ColumnTransformer([
        ("ordered", _encoder({f: ORDERED_FEATURES[f] for f in ordered}), ordered),
        ("nominal", _encoder({f: NOMINAL_FEATURES[f] for f in nominal}), nominal),
        ("numeric", "passthrough", numeric),
    ], verbose_feature_names_out=False)
    # Nominal answers are treated as categories (no false order); missing values go their own way in each split.
    is_cat = [False] * len(ordered) + [True] * len(nominal) + [False] * len(numeric)
    defaults = dict(learning_rate=0.1, max_iter=1000, max_leaf_nodes=63, min_samples_leaf=200,
                    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=30, random_state=42)
    return Pipeline([("prep", prep),
                     ("model", HistGradientBoostingClassifier(categorical_features=is_cat, **{**defaults, **params}))])


def logistic_regression(features=FEATURES) -> Pipeline:
    ordered = [f for f in ORDERED_FEATURES if f in features]
    nominal = [f for f in NOMINAL_FEATURES if f in features]
    numeric = [f for f in NUMERIC_FEATURES if f in features]
    prep = ColumnTransformer([
        ("ordered", Pipeline([("encode", _encoder({f: ORDERED_FEATURES[f] for f in ordered})),
                              ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), ordered),
        ("nominal", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))]), nominal),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), numeric),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=1.0, max_iter=3000))])
