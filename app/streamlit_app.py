"""Heart disease risk estimator: an educational demo of the model in this repo.

Run locally: streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
AGE_ORDER = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
             "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
GEN_HEALTH = ["Poor", "Fair", "Good", "Very good", "Excellent"]
DIABETES = {"No": "No", "Borderline diabetes": "No, borderline diabetes",
            "Yes": "Yes", "Only during pregnancy": "Yes (during pregnancy)"}
RACES = ["White", "Hispanic", "Black", "Asian", "American Indian/Alaskan Native", "Other"]
REPO = "https://github.com/mellithyy/heart-disease-diagnosis"


@st.cache_resource
def load_model():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "models" / f"{meta['app_model']}.joblib")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    return model, meta, metrics


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


st.set_page_config(page_title="Heart Disease Risk Estimator", layout="wide")
model, meta, metrics = load_model()
test = metrics["results"][meta["app_model"]]["best_f1"]

st.title("Heart disease risk estimator")
st.caption(
    "An educational demo built on the CDC's 2020 BRFSS survey of 319,795 US adults. "
    "It estimates how often people with answers like yours reported coronary heart disease or a heart attack. "
    "**It is not a diagnosis and not medical advice.** "
    f"Code, method and full evaluation: [GitHub]({REPO})."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("About you")
    age = st.selectbox("Age group", AGE_ORDER, index=7)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    race = st.selectbox("Race / ethnicity (as asked in the survey)", RACES)
    height = st.number_input("Height (cm)", 120, 220, 170)
    weight = st.number_input("Weight (kg)", 30, 250, 75)
    bmi = weight / (height / 100) ** 2
    st.caption(f"BMI: **{bmi:.1f}**")
with col2:
    st.subheader("Health history")
    stroke = st.checkbox("Ever had a stroke")
    diabetes = st.selectbox("Diabetes", list(DIABETES))
    kidney = st.checkbox("Kidney disease")
    asthma = st.checkbox("Asthma")
    skin_cancer = st.checkbox("Skin cancer")
    diff_walking = st.checkbox("Difficulty walking", help="Serious difficulty walking or climbing stairs")
with col3:
    st.subheader("Lifestyle and wellbeing")
    smoking = st.checkbox("Smoker", help="Smoked at least 100 cigarettes (5 packs) in your life")
    alcohol = st.checkbox("Heavy drinker", help="Men: more than 14 drinks a week. Women: more than 7.")
    active = st.checkbox("Physically active", value=True,
                         help="Any physical activity or exercise in the past 30 days, other than your regular job")
    gen_health = st.select_slider("General health, in your own words", GEN_HEALTH, value="Very good")
    phys_days = st.slider("Days of poor physical health (past 30)", 0, 30, 0)
    ment_days = st.slider("Days of poor mental health (past 30)", 0, 30, 0)
    sleep = st.slider("Hours of sleep in 24 hours", 1, 24, 7)

person = pd.DataFrame([{
    "BMI": round(bmi, 2), "Smoking": yes_no(smoking), "AlcoholDrinking": yes_no(alcohol),
    "Stroke": yes_no(stroke), "PhysicalHealth": phys_days, "MentalHealth": ment_days,
    "DiffWalking": yes_no(diff_walking), "Sex": sex, "AgeCategory": age, "Race": race,
    "Diabetic": DIABETES[diabetes], "PhysicalActivity": yes_no(active), "GenHealth": gen_health,
    "SleepTime": sleep, "Asthma": yes_no(asthma), "KidneyDisease": yes_no(kidney), "SkinCancer": yes_no(skin_cancer),
}])[meta["features"]]

risk = float(model.predict_proba(person)[0, 1])
peer = meta["rates_by_age_sex"][age][sex]
overall = meta["prevalence_train"]
screen = meta["thresholds"][meta["app_model"]]["screening_recall_80"]

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Estimated probability", f"{risk:.1%}")
m2.metric(f"Average for {sex.lower()}s aged {age}", f"{peer:.1%}")
m3.metric("Average across the whole survey", f"{overall:.1%}")
if risk >= screen:
    shown = metrics["results"][meta["app_model"]]["screening_recall_80"]
    st.warning(f"This is above the screening cut-off of {screen:.0%}. On the test data, that cut-off flagged "
               f"{shown['recall']:.0%} of the people who reported heart disease, but only {shown['precision']:.0%} "
               "of the people it flagged had it. A flag means \"worth a check-up\", not \"has heart disease\". "
               "If you have concerns, talk to a doctor.")

# What-if: re-score the same person with one answer changed. This shows model sensitivity,
# not cause and effect: the survey can't tell whether a factor causes heart disease.
changes = []
if smoking:
    changes.append(("Had not smoked", {"Smoking": "No"}))
if not active:
    changes.append(("Were physically active", {"PhysicalActivity": "Yes"}))
if gen_health != "Excellent":
    better = GEN_HEALTH[GEN_HEALTH.index(gen_health) + 1]
    changes.append((f"Rated general health one step higher ({better})", {"GenHealth": better}))
if bmi >= 25:
    changes.append(("Had a BMI of 24.9", {"BMI": 24.9}))
if phys_days > 0:
    changes.append(("Had no poor physical-health days", {"PhysicalHealth": 0}))
if changes:
    rows = []
    for label, change in changes:
        new = float(model.predict_proba(person.assign(**change))[0, 1])
        rows.append({"If you...": label, "Estimate": f"{new:.1%}", "Change (points)": f"{(new - risk) * 100:+.1f}"})
    st.subheader("How the estimate responds to changeable answers")
    st.caption("Model sensitivity only. These are associations in survey data, not proven causes.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with st.expander("About the model and its limits"):
    st.markdown(f"""
- **Model:** {meta['app_model'].replace('_', ' ')} trained on {meta['train_rows']:,} survey answers, with the
  decision threshold chosen by cross-validation on the training data only.
- **Tested on {metrics['test_rows']:,} untouched answers** (real prevalence {metrics['test_prevalence']:.1%}):
  ROC-AUC {test['roc_auc']:.2f}, PR-AUC {test['pr_auc']:.2f}, F1 {test['f1']:.2f}.
  The probabilities are well calibrated: among people given about 20%, about 20% reported heart disease.
- **Self-reported and cross-sectional:** the survey asks whether a doctor *ever* told the person they had
  heart disease. Some answers (stroke, difficulty walking, poor health days) may be consequences of heart
  disease, not causes.
- **Population:** US adults in 2020. Rates elsewhere, including the Gulf, differ.
""")
