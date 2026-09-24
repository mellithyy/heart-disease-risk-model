"""Part 2 (CDC 2011-2025): the harmonisation rules and the saved model.

The rule tests use a few hand-made raw rows, so they run anywhere (the 1.3 GB of CDC files are not needed).
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from heart_risk.brfss.variables import harmonise, raw_columns_needed
from heart_risk.paths import PROJECT


def run(raw: dict, year: int) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    return harmonise(df, year, raw_columns_needed(list(df.columns)))


def test_heart_disease_follows_cdc_definition():
    out = run({"CVDINFR4": [1, 2, 2, 7, 2], "CVDCRHD4": [2, 1, 2, 2, 9]}, 2025)
    # yes if either is yes; no only if both are no; unknown otherwise
    assert out["heart_disease"].astype(object).tolist()[:3] == ["Yes", "Yes", "No"]
    assert out["heart_disease"].isna().tolist()[3:] == [True, True]


def test_days_income_bmi_and_dont_know_codes():
    out = run({"PHYSHLTH": [88, 5, 77, 99], "INCOME3": [11, 8, 3, 77], "_BMI5": [2512, 1500, np.nan, 9999],
               "SMOKE100": [1, 2, 7, 9]}, 2023)
    assert out["phys_days"].tolist()[:2] == [0, 5] and out["phys_days"].isna().tolist()[2:] == [True, True]
    assert out["income"].astype(object).tolist()[:3] == ["75k+", "75k+", "15-20k"]  # 2021+ top brackets merged
    assert out["bmi"].tolist()[:2] == [25.12, 15.0]  # stored with 2 implied decimals
    assert out["smoked_100"].astype(object).tolist()[:2] == ["Yes", "No"] and out["smoked_100"].isna().sum() == 2


def test_renamed_questions_are_found_for_each_year():
    old = raw_columns_needed(["SEX", "DIABETE3", "INCOME2", "CHCKIDNY", "_IMPRACE"])
    new = raw_columns_needed(["SEXVAR", "DIABETE4", "INCOME3", "CHCKDNY2", "_IMPRACE"])
    assert old == {"sex": "SEX", "diabetes": "DIABETE3", "income": "INCOME2", "kidney_disease": "CHCKIDNY", "race": "_IMPRACE"}
    assert new == {"sex": "SEXVAR", "diabetes": "DIABETE4", "income": "INCOME3", "kidney_disease": "CHCKDNY2", "race": "_IMPRACE"}


def test_race_fallback_and_never_checked_cholesterol():
    out = run({"_RACE": [1, 3, 4, 7, 8, 9]}, 2015)  # 2015/2016 have no _IMPRACE
    assert out["race"].astype(object).tolist()[:5] == ["White", "American Indian/Alaska Native", "Asian", "Other", "Hispanic"]
    assert pd.isna(out["race"].iloc[5])
    chol = run({"TOLDHI3": [1, np.nan, np.nan], "CHOLCHK3": [2, 1, 7]}, 2025)
    assert chol["high_cholesterol"].astype(object).tolist()[:2] == ["Yes", "Never checked"]
    assert pd.isna(chol["high_cholesterol"].iloc[2])  # "don't know" stays missing


@pytest.fixture(scope="module")
def saved():
    meta = json.loads((PROJECT / "models" / "brfss_metadata.json").read_text(encoding="utf-8"))
    return joblib.load(PROJECT / "models" / "brfss_gradient_boosting.joblib"), meta


def person(**changes):
    base = {"age_group": "40-44", "gen_health": "Very good", "income": None, "education": "College graduate",
            "sex": "Female", "race": None, "smoked_100": "No", "heavy_drinker": "No", "exercise": "Yes",
            "diff_walking": "No", "stroke": "No", "high_blood_pressure": "No", "high_cholesterol": "No",
            "diabetes": "No", "asthma": "No", "kidney_disease": "No", "copd": "No", "depression": "No",
            "arthritis": "No", "bmi": 24.0, "phys_days": 0.0, "ment_days": 0.0}
    return pd.DataFrame([{**base, **changes}]).astype(object)


def test_saved_model_gives_sensible_probabilities(saved):
    model, meta = saved
    low = model.predict_proba(person()[meta["features"]])[0, 1]
    high = model.predict_proba(person(age_group="75-79", sex="Male", stroke="Yes", high_blood_pressure="Yes",
                                      high_cholesterol="Yes", diabetes="Yes", gen_health="Poor")[meta["features"]])[0, 1]
    assert 0 <= low < 0.05 and 0.3 < high <= 1
    everything_skipped = person(**{f: None for f in meta["features"]})[meta["features"]]
    assert 0 <= model.predict_proba(everything_skipped)[0, 1] <= 1  # missing answers are allowed
