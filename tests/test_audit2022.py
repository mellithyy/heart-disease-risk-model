"""Part 1 (2022 audit, Kaggle 2020 data): the data file, the split and the saved fair model."""
import json

import joblib
import pandas as pd
import pytest

from heart_risk.audit2022.data import load, split
from heart_risk.audit2022.features import FEATURES, preprocessor
from heart_risk.paths import PROJECT


@pytest.fixture(scope="module")
def df():
    return load()


def test_data_is_the_published_file(df):
    assert df.shape == (319_795, 18)
    assert df.isna().sum().sum() == 0
    assert set(FEATURES) | {"HeartDisease"} == set(df.columns)
    assert 0.08 < (df["HeartDisease"] == "Yes").mean() < 0.09


def test_split_is_disjoint_and_keeps_prevalence(df):
    X_train, X_test, y_train, y_test = split(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_test) == pytest.approx(0.2 * len(df), abs=1)
    assert abs(y_train.mean() - y_test.mean()) < 0.001  # stratified: real prevalence in both


def test_preprocessor_handles_every_answer(df):
    out = preprocessor().fit_transform(df[FEATURES].sample(5_000, random_state=0))
    assert not pd.isna(out).any()
    assert out.shape[1] == 8 + 1 + 2 + 6 + 4 + 4  # yes/no, sex, age+health, race, diabetic, numeric


def test_saved_model_gives_probabilities(df):
    meta = json.loads((PROJECT / "models" / "audit2022" / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(PROJECT / "models" / "audit2022" / f"{meta['app_model']}.joblib")
    proba = model.predict_proba(df[meta["features"]].head(200))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    young_healthy = df[meta["features"]].iloc[[0]].assign(AgeCategory="18-24", GenHealth="Excellent",
                                                           Stroke="No", DiffWalking="No")
    old_unwell = young_healthy.assign(AgeCategory="80 or older", GenHealth="Poor", Stroke="Yes", DiffWalking="Yes")
    assert model.predict_proba(old_unwell)[0, 1] > model.predict_proba(young_healthy)[0, 1]
