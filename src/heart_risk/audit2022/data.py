"""Load the survey data and make the one train/test split used everywhere."""
import pandas as pd
from sklearn.model_selection import train_test_split

from ..paths import PROJECT

DATA_FILE = PROJECT / "data" / "heart_2020_cleaned.csv.gz"

TARGET = "HeartDisease"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load() -> pd.DataFrame:
    """The full dataset as published: 319,795 rows, 18 columns, no missing values.

    Duplicate rows (18,078) are kept on purpose: in a survey with mostly Yes/No answers,
    different people often give identical answers, and dropping them would change the population.
    """
    return pd.read_csv(DATA_FILE)


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == "Yes").astype(int).rename("heart_disease")
    return X, y


def split(df: pd.DataFrame):
    """80/20 stratified split, made once, BEFORE any resampling, feature selection or tuning.

    The test set keeps the real prevalence (about 8.6%) and is only used for the final evaluation.
    """
    X, y = features_and_target(df)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
