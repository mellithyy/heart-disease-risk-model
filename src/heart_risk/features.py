"""Column groups and the preprocessing step shared by every model.

Everything is done with scikit-learn encoders inside the model pipeline, so the saved model takes
the raw survey answers (for example "Yes", "55-59", "Very good") and the app needs no extra code.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

YES_NO = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity",
          "Asthma", "KidneyDisease", "SkinCancer"]
AGE_ORDER = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
             "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
GEN_HEALTH_ORDER = ["Poor", "Fair", "Good", "Very good", "Excellent"]
# Race and Diabetic have no natural order. Diabetic keeps all 4 answers: the 2022 version
# merged "borderline" into No and "during pregnancy" into Yes, which loses information.
NOMINAL = ["Race", "Diabetic"]
NUMERIC = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

FEATURES = YES_NO + ["Sex", "AgeCategory", "GenHealth"] + NOMINAL + NUMERIC


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("yes_no", OrdinalEncoder(categories=[["No", "Yes"]] * len(YES_NO)), YES_NO),
            ("sex", OrdinalEncoder(categories=[["Female", "Male"]]), ["Sex"]),
            ("ordered", OrdinalEncoder(categories=[AGE_ORDER, GEN_HEALTH_ORDER]), ["AgeCategory", "GenHealth"]),
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
            # Scaling puts numeric coefficients "per standard deviation" in the logistic regression.
            # It changes nothing for tree models.
            ("numeric", StandardScaler(), NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
