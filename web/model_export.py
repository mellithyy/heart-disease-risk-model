"""The saved model (models/brfss_gradient_boosting.joblib) as plain numbers a browser can run.

The pipeline is: ordinal codes for the ordered answers (age, general health, income, education), category codes for the
yes/no and multiple-choice answers, BMI and the two day counts as they are, then scikit-learn's HistGradientBoostingClassifier
(301 trees). In the browser the same steps run in web/app.js (predictRaw): each answer becomes its code (a skipped answer is
missing), every tree is walked from its root to a leaf, the leaves are added to the starting value, and the sum becomes a
probability with the logistic function, as predict_proba does.

export() returns that model as a JSON-ready dict; parity_cases() returns random people with scikit-learn's own probability for
each, so the browser's numbers can be checked against them (web/parity.cjs).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "brfss_gradient_boosting.joblib"


def load():
    return joblib.load(MODEL)


def export(model):
    prep, hgb = model.named_steps["prep"], model.named_steps["model"]
    assert hgb.n_trees_per_iteration_ == 1 and hgb.loss == "log_loss"
    # step 1, the pipeline's own encoders: each answer becomes a code (its place in the answer list; skipped = missing)
    step1, order1 = {}, []
    for name, enc, cols in prep.transformers_:
        order1 += list(cols)
        if hasattr(enc, "categories_"):
            # (an answer outside the list, or a skipped one, is a missing value for the trees, as the encoder's settings say)
            assert enc.handle_unknown == "use_encoded_value" and np.isnan(enc.unknown_value) and np.isnan(enc.encoded_missing_value)
            for c, cat in zip(cols, enc.categories_):
                step1[c] = {str(x): float(i) for i, x in enumerate(cat)}
        else:
            assert enc.func is None, "the numeric step is expected to pass values through"
    # step 2, inside the classifier: its own encoder puts the categorical columns first and gives their codes new numbers
    # (the trees' feature numbers follow that new order); the other columns follow as they are
    pre = hgb._preprocessor
    (_, enc2, cat_mask), (_, num_tf, num_mask) = pre.transformers_[0], pre.transformers_[1]
    assert np.isnan(enc2.unknown_value) and np.isnan(enc2.encoded_missing_value)
    cat_cols = [c for c, m in zip(order1, cat_mask) if m]
    num_cols = [c for c, m in zip(order1, num_mask) if m]
    order = cat_cols + num_cols
    assert len(order) == len(order1) and all(hgb._is_categorical_remapped[:len(cat_cols)]) and not any(hgb._is_categorical_remapped[len(cat_cols):])
    cats = {}
    for j, c in enumerate(cat_cols):
        known = [float(x) for x in enc2.categories_[j] if not np.isnan(x)]
        cats[c] = {a: float(known.index(code)) for a, code in step1[c].items() if code in known}
    for c in num_cols:
        if c in step1:
            cats[c] = step1[c]
    trees = []
    for (pred,) in hgb._predictors:
        n = pred.nodes
        # one flat list per tree: for each node [leaf?, feature, threshold, missing goes left?, left, right, value, categorical?, bitset]
        rows = []
        for r in n:
            if r["is_leaf"]:
                rows.append([1, float(r["value"])])
            else:
                t = float(r["num_threshold"])
                # (a split that sends every known value left and only the missing ones right has the threshold +inf; JSON has
                # no infinity, and 1e300 sends every real answer the same way)
                assert not np.isnan(t) and t != -np.inf
                rows.append([0, int(r["feature_idx"]), 1e300 if t == np.inf else t, int(r["missing_go_to_left"]),
                             int(r["left"]), int(r["right"]), int(r["is_categorical"]), int(r["bitset_idx"])])
        trees.append({"n": rows, "b": pred.raw_left_cat_bitsets.astype(np.int64).tolist()})
    return {"base": float(hgb._baseline_prediction.ravel()[0]), "order": order, "codes": cats, "trees": trees}


def parity_cases(model, n=20000, seed=7):
    """random answers (every option, skipped ones too, and BMI and day counts across their range) with scikit-learn's probability"""
    prep = model.named_steps["prep"]
    rng = np.random.default_rng(seed)
    rows = []
    cat_cols = {c: list(cat) for _, enc, cols in prep.transformers_ if hasattr(enc, "categories_") for c, cat in zip(cols, enc.categories_)}
    skippable = {"income", "education", "race"}
    for _ in range(n):
        r = {}
        for c, cat in cat_cols.items():
            opts = cat + ([None] if c in skippable else [])
            r[c] = opts[rng.integers(len(opts))]
        r["bmi"] = round(float(rng.uniform(12, 70)), 2)
        r["phys_days"] = float(rng.integers(0, 31))
        r["ment_days"] = float(rng.integers(0, 31))
        rows.append(r)
    feats = [c for _, _, cols in prep.transformers_ for c in cols]
    X = pd.DataFrame(rows)[model.feature_names_in_ if hasattr(model, "feature_names_in_") else feats].astype(object)
    p = model.predict_proba(X)[:, 1]
    return [{"x": {k: (v if v is None or isinstance(v, str) else float(v)) for k, v in r.items()}, "p": float(q)} for r, q in zip(rows, p)]
