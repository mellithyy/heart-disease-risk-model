"""The two candidate models, each a full pipeline from raw survey answers to a probability."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import RANDOM_STATE
from .features import preprocessor

# No resampling and no class weights: both would distort the predicted probabilities.
# The class imbalance is handled where it belongs, in the choice of decision threshold.
CANDIDATES = {
    # Simple and explainable: every coefficient is an odds ratio.
    "logistic_regression": lambda: Pipeline([
        ("prep", preprocessor()),
        ("model", LogisticRegression(C=1.0, max_iter=2000)),
    ]),
    # Captures interactions (for example age x general health) that a linear model can't.
    "gradient_boosting": lambda: Pipeline([
        ("prep", preprocessor()),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=RANDOM_STATE)),
    ]),
}

LABELS = {"logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting"}
