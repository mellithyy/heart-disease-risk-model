"""Heart disease risk estimator, built on 15 years of CDC survey data (BRFSS 2011-2025).

Run locally: streamlit run app/streamlit_app.py
Only needs the saved model and the small JSON reports; the 6.7-million-row dataset is not required.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/mellithyy/heart-disease-risk-model"
LINKEDIN = "https://www.linkedin.com/in/mohamed-el-lithy/"
BLUE, ORANGE, RED, MUTED, INK2 = "#2a78d6", "#eb6834", "#e34948", "#898781", "#52514e"
SKIP = "Prefer not to say"


@st.cache_resource
def load():
    meta = json.loads((ROOT / "models" / "brfss_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "models" / "brfss_gradient_boosting.joblib")
    report = json.loads((ROOT / "reports" / "brfss_model.json").read_text(encoding="utf-8"))
    trends = json.loads((ROOT / "reports" / "brfss_trends.json").read_text(encoding="utf-8"))
    return model, meta, report, trends


st.set_page_config(page_title="Heart Disease Risk Estimator", layout="wide", initial_sidebar_state="expanded")
model, meta, report, trends = load()
gb = report["results"]["gradient_boosting"]
nobp = report["results"]["gb_without_bp_chol"]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #0b1628 0%, #16335a 100%); border-left: 6px solid #3fc1c9;
       border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;}
.hero h1 {color: #fff; font-size: 2rem; margin: 0; padding: 0;}
.hero p {color: #c9d6e8; margin: 6px 0 0 0; font-size: 1rem;}
.chip {display: inline-block; margin: 12px 8px 0 0; padding: 3px 12px; border-radius: 999px; font-size: .8rem;
       color: #c8f3f5; background: rgba(63,193,201,.14); border: 1px solid rgba(63,193,201,.45);}
.card {background: #fff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 20px 22px; height: 100%;}
.label {color: #52514e; font-size: .85rem; text-transform: uppercase; letter-spacing: .06em;}
.big {font-size: 3.2rem; font-weight: 700; color: #0b1628; line-height: 1.1; margin: 4px 0;}
.sub {color: #52514e; font-size: .95rem;}
.scale {position: relative; height: 52px; margin: 34px 6px 6px 6px;}
.track {position: absolute; top: 14px; left: 0; right: 0; height: 10px; border-radius: 6px;
        background: linear-gradient(90deg, #cde2fb, #6da7ec 50%, #184f95);}
.mk {position: absolute; transform: translateX(-50%); text-align: center; white-space: nowrap; font-size: .8rem;}
.mk .dot {width: 20px; height: 20px; border-radius: 50%; background: #0b1628; border: 3px solid #fff;
          box-shadow: 0 0 0 1px #0b1628; margin: 9px auto 0 auto;}
.mk .tick {width: 2px; height: 22px; margin: 8px auto 0 auto;}
.mk.you {top: -26px;} .mk.you b {color: #0b1628; font-size: .9rem;}
.pin {position: absolute; top: 8px; width: 3px; height: 22px; border-radius: 2px; transform: translateX(-50%);}
.legend {display: flex; gap: 18px; flex-wrap: wrap; font-size: .82rem; color: #52514e; margin-top: 2px;}
.legend i {display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;}
.ends {position: absolute; top: 30px; width: 100%; display: flex; justify-content: space-between; color: #898781; font-size: .75rem;}
.foot {color: #898781; font-size: .8rem; margin-top: 28px; border-top: 1px solid #e1e0d9; padding-top: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>Heart disease risk estimator</h1>
  <p>Built on 3.2 million CDC survey answers from 2011 to 2023, and tested on 2025, a year the model never saw.</p>
  <span class="chip">CDC BRFSS, 6.7 million adults, 2011-2025</span>
  <span class="chip">ROC-AUC {gb['best_f1']['roc_auc']:.2f} on 2025</span>
  <span class="chip">Calibrated probabilities</span>
  <span class="chip">Not medical advice</span>
</div>
""", unsafe_allow_html=True)

answers = meta["answers"]
with st.sidebar:
    st.header("Your answers")
    st.caption("Nothing you enter is saved. Skip anything with \"Prefer not to say\".")
    with st.expander("About you", expanded=True):
        age = st.selectbox("Age group", answers["age_group"], index=answers["age_group"].index("55-59"))
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        c1, c2 = st.columns(2)
        height = c1.number_input("Height (cm)", 120, 220, 170)
        weight = c2.number_input("Weight (kg)", 30, 250, 78)
        bmi = weight / (height / 100) ** 2
        st.caption(f"BMI {bmi:.1f}")
        race = st.selectbox("Race / ethnicity", [SKIP, *answers["race"]])
        education = st.selectbox("Education", [SKIP, *answers["education"]])
        income = st.selectbox("Household income (USD)", [SKIP, *answers["income"]])
    with st.expander("Health conditions (ever told by a doctor)", expanded=True):
        bp = st.selectbox("High blood pressure", ["No", "Yes", "Borderline", "Only during pregnancy"])
        chol = st.selectbox("High cholesterol", ["No", "Yes", "Never checked"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes", "Borderline (prediabetes)", "Only during pregnancy"])
        stroke = st.checkbox("Stroke")
        kidney = st.checkbox("Kidney disease")
        copd = st.checkbox("COPD, emphysema or chronic bronchitis")
        asthma = st.checkbox("Asthma")
        arthritis = st.checkbox("Arthritis")
        depression = st.checkbox("Depression")
    with st.expander("Lifestyle", expanded=True):
        smoked = st.checkbox("Smoked at least 100 cigarettes in your life")
        heavy = st.checkbox("Heavy drinker", help="Men: more than 14 drinks a week. Women: more than 7.")
        exercise = st.checkbox("Exercised in the past month", value=True, help="Any physical activity outside your job.")
    with st.expander("Wellbeing", expanded=True):
        gen_health = st.select_slider("General health", ["Poor", "Fair", "Good", "Very good", "Excellent"], value="Very good")
        phys = st.slider("Days of poor physical health (past 30)", 0, 30, 0)
        ment = st.slider("Days of poor mental health (past 30)", 0, 30, 0)
        walk = st.checkbox("Serious difficulty walking or climbing stairs")

yn = lambda b: "Yes" if b else "No"  # noqa: E731
skip = lambda v: None if v == SKIP else v  # noqa: E731
person = pd.DataFrame([{
    "age_group": age, "gen_health": gen_health, "income": skip(income), "education": skip(education),
    "sex": sex, "race": skip(race), "smoked_100": yn(smoked), "heavy_drinker": yn(heavy), "exercise": yn(exercise),
    "diff_walking": yn(walk), "stroke": yn(stroke),
    "high_blood_pressure": {"Only during pregnancy": "Pregnancy only"}.get(bp, bp),
    "high_cholesterol": chol,
    "diabetes": {"Borderline (prediabetes)": "Borderline", "Only during pregnancy": "Pregnancy only"}.get(diabetes, diabetes),
    "asthma": yn(asthma), "kidney_disease": yn(kidney), "copd": yn(copd), "depression": yn(depression),
    "arthritis": yn(arthritis), "bmi": round(bmi, 2), "phys_days": float(phys), "ment_days": float(ment),
}])[meta["features"]].astype(object)

risk = float(model.predict_proba(person)[0, 1])
peer = meta["weighted_rates_by_age_sex_2021_2025"][age][sex]
overall = meta["weighted_rate_2021_2025"]
screen = meta["thresholds"]["gradient_boosting"]["screening_recall_80"]
screen_res = gb["screening_recall_80"]

tab1, tab2, tab3 = st.tabs(["Your estimate", "15 years of data", "How the model works"])

with tab1:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        top = min(1.0, max(0.25, risk, peer) * 1.3)
        pos = lambda v: 100 * min(v / top, 1)  # noqa: E731
        ratio = risk / peer if peer else float("nan")
        if ratio > 1.15:
            compare = f"<b>{ratio:.1f} times</b> the average"
        elif ratio >= 0.85:
            compare = "<b>about the same as</b> the average"
        else:
            compare = "<b>below</b> the average"
        shown = f"{risk:.1%}" if risk < 0.1 else f"{risk:.0%}"
        st.markdown(f"""
<div class="card">
  <div class="label">Estimated probability</div>
  <div class="big">{shown}</div>
  <div class="sub">of people with answers like yours reported coronary heart disease or a heart attack.
  That is {compare} for {sex.lower()}s aged {age} ({peer:.1%}).</div>
  <div class="scale">
    <div class="track"></div>
    <div class="ends"><span>0%</span><span>{top:.0%}</span></div>
    <div class="mk you" style="left:{pos(risk)}%"><b>You</b><div class="dot"></div></div>
    <div class="pin" style="left:{pos(peer)}%; background:{ORANGE}"></div>
    <div class="pin" style="left:{pos(overall)}%; background:{MUTED}"></div>
  </div>
  <div class="legend"><span><i style="background:{ORANGE}"></i>{sex}s aged {age}: {peer:.1%}</span>
  <span><i style="background:{MUTED}"></i>All US adults: {overall:.1%}</span></div>
</div>
""", unsafe_allow_html=True)
        if risk >= screen:
            st.warning(f"Above the screening cut-off of {screen:.0%}. In the 2025 test, that cut-off found "
                       f"{screen_res['recall']:.0%} of people with heart disease, but only {screen_res['precision']:.0%} "
                       "of the people it flagged had it. A flag means \"worth a check-up\", not \"has heart disease\".")
        else:
            st.info(f"Below the screening cut-off of {screen:.0%} used in the evaluation.")

    with right:
        st.markdown("##### What moves your estimate")
        st.caption("The same answers with one thing changed. This shows how the model responds, "
                   "not what would happen to you: the survey shows associations, not causes.")
        changes = []
        if smoked:
            changes.append(("Had never smoked", {"smoked_100": "No"}))
        if not exercise:
            changes.append(("Exercised in the past month", {"exercise": "Yes"}))
        if bp == "Yes":
            changes.append(("No high blood pressure", {"high_blood_pressure": "No"}))
        if chol == "Yes":
            changes.append(("No high cholesterol", {"high_cholesterol": "No"}))
        if bmi >= 25:
            changes.append(("BMI of 24.9", {"bmi": 24.9}))
        if gen_health != "Excellent":
            better = ["Poor", "Fair", "Good", "Very good", "Excellent"]
            nxt = better[better.index(gen_health) + 1]
            changes.append((f"General health: {nxt}", {"gen_health": nxt}))
        if heavy:
            changes.append(("Not a heavy drinker", {"heavy_drinker": "No"}))
        if phys > 0:
            changes.append(("No poor physical-health days", {"phys_days": 0.0}))
        if changes:
            rows = []
            for label, change in changes:
                new = float(model.predict_proba(person.assign(**change).astype(object))[0, 1])
                rows.append({"change": label, "points": round(100 * (new - risk), 1), "new": f"{new:.1%}"})
            d = pd.DataFrame(rows).sort_values("points")
            d["direction"] = d["points"].map(lambda v: "Lower" if v < 0 else "Higher")
            d["change"] = [f"{c} ({v:+.1f})" for c, v in zip(d["change"], d["points"])]
            base = alt.Chart(d).encode(y=alt.Y("change:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)))
            bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
                x=alt.X("points:Q", title="Change in estimate (percentage points)"),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["Lower", "Higher"], range=[BLUE, RED]), legend=None),
                tooltip=[alt.Tooltip("change:N", title="If"), alt.Tooltip("new:N", title="New estimate"),
                         alt.Tooltip("points:Q", title="Change (points)")])
            st.altair_chart(bars.properties(height=alt.Step(40)), width="stretch")
            if (d["points"] > 0).any():
                st.caption("A change can point the \"wrong\" way. Survey data shows reverse causation: for example, "
                           "people already diagnosed with heart disease are often told to exercise, so exercise "
                           "appears among them. This is why these are associations, not advice.")
        else:
            st.write("Nothing to compare: none of the changeable answers apply.")

with tab2:
    st.markdown("##### Heart disease among US adults, 2011-2025")
    rows = [{"year": int(y), "rate": v, "series": "As reported"} for y, v in trends["crude"].items()]
    rows += [{"year": int(y), "rate": v, "series": "Age-adjusted"} for y, v in trends["adjusted"].items()]
    line = alt.Chart(pd.DataFrame(rows)).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("year:Q", title=None, axis=alt.Axis(format="d", tickCount=8)),
        y=alt.Y("rate:Q", title="Share of adults", axis=alt.Axis(format=".1%"), scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["As reported", "Age-adjusted"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["year:Q", "series:N", alt.Tooltip("rate:Q", format=".2%")])
    st.altair_chart(line.properties(height=280), width="stretch")
    st.caption("As reported, the share stayed near 6.5%. Adjusted for the ageing population it fell from "
               f"{trends['adjusted']['2011']:.1%} to {trends['adjusted']['2025']:.1%}. The odd/even zig-zag comes "
               "from CDC alternating two questionnaires.")

    st.markdown("##### Risk factors over time")
    names = {"obesity": "Obesity (BMI 30+)", "diabetes": "Diabetes", "smoking": "Ever smoked",
             "inactive": "No exercise", "bp": "High blood pressure", "depression": "Depression"}
    d = pd.DataFrame([{"factor": names[k], "year": int(y), "share": v}
                      for k, s_ in trends["factors_share"].items() for y, v in s_.items()])
    ch = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=18), color=BLUE).encode(
        x=alt.X("year:Q", title=None, axis=alt.Axis(format="d", values=[2011, 2015, 2019, 2023])),
        y=alt.Y("share:Q", title=None, axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
        tooltip=["factor:N", "year:Q", alt.Tooltip("share:Q", format=".1%")],
    ).properties(width=250, height=120).facet(
        facet=alt.Facet("factor:N", title=None, sort=list(names.values()), header=alt.Header(labelFontSize=12)),
        columns=3).resolve_scale(y="independent")
    st.altair_chart(ch)
    st.caption("Obesity, diabetes, high blood pressure and depression rose; smoking fell. "
               "The no-exercise zig-zag is the questionnaire again.")

    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        st.markdown("##### Strength of each factor")
        labels = {"stroke": "Stroke", "kidney": "Kidney disease", "diabetes": "Diabetes", "copd": "COPD",
                  "bp": "High blood pressure", "chol": "High cholesterol", "smoking": "Ever smoked", "inactive": "No exercise"}
        d = pd.DataFrame([{"factor": labels[k], "year": int(y), "odds_ratio": v}
                          for k, s_ in trends["odds_ratios"].items() for y, v in s_.items()])
        med = d.groupby("factor")["odds_ratio"].median().rename("odds_ratio").reset_index()
        bars = alt.Chart(med).mark_bar(cornerRadiusEnd=4, color=BLUE, height=18).encode(
            x=alt.X("odds_ratio:Q", title="Odds ratio (1 = no link)"),
            y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            tooltip=["factor:N", alt.Tooltip("odds_ratio:Q", format=".2f", title="Median odds ratio")])
        rule = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=alt.Step(34)), width="stretch")
        st.caption("Median over 2011-2025 of Mantel-Haenszel odds ratios: each year, people of the same "
                   "age group and sex are compared.")
    with c2:
        st.markdown("##### By state, age-adjusted, 2021-2025")
        d = pd.DataFrame(trends["state_tiles"])
        base = alt.Chart(d).encode(x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None))
        tiles = base.mark_rect(cornerRadius=4, stroke="#fcfcfb", strokeWidth=3).encode(
            color=alt.Color("rate:Q", scale=alt.Scale(range=["#cde2fb", "#184f95"]),
                            legend=alt.Legend(title="Share", format=".0%", orient="bottom", gradientLength=220)),
            tooltip=["state:N", alt.Tooltip("rate:Q", format=".1%", title="Age-adjusted rate")])
        text = base.mark_text(fontWeight="bold", fontSize=10).encode(
            text="state:N", color=alt.condition("datum.rate > 0.07", alt.value("#ffffff"), alt.value("#0b1628")))
        st.altair_chart((tiles + text).properties(height=330), width="stretch")
        st.caption("Highest in West Virginia, Kentucky and Arkansas; lowest in DC, Colorado and Hawaii.")

with tab3:
    st.markdown("##### Tested on 2025, a year the model never saw")
    rows = []
    for key, name in [("gradient_boosting", "Gradient boosting (used here)"), ("logistic_regression", "Logistic regression"),
                      ("gb_without_bp_chol", "Gradient boosting without blood pressure and cholesterol"),
                      ("baseline_always_no", "Always answer \"No\"")]:
        r = report["results"][key]
        r = r["best_f1"] if "best_f1" in r else r
        rows.append({"Model": name, "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "F1": r["f1"],
                     "Precision": r["precision"], "Recall": r["recall"], "Accuracy": r["accuracy"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy"]})
    st.caption(f"{report['test_rows']:,} people in 2025, {report['test_prevalence']:.1%} with heart disease. "
               "Always answering \"No\" gets high accuracy and finds nobody, which is why accuracy is not the measure here.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Calibration: does 20% mean 20%?")
        rows = [{"model": "Gradient boosting" if k == "gradient_boosting" else "Logistic regression", "predicted": p, "observed": o}
                for k, v in report["calibration"].items() for p, o in zip(v["predicted"], v["observed"])]
        d = pd.DataFrame(rows)
        diag = alt.Chart(pd.DataFrame({"x": [0, 0.6], "y": [0, 0.6]})).mark_line(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q")
        pts = alt.Chart(d).mark_line(point=True).encode(
            x=alt.X("predicted:Q", title="Predicted", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 0.6])),
            y=alt.Y("observed:Q", title="Observed in 2025", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 0.6])),
            color=alt.Color("model:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["model:N", alt.Tooltip("predicted:Q", format=".1%"), alt.Tooltip("observed:Q", format=".1%")])
        st.altair_chart((diag + pts).properties(height=320), width="stretch")
    with c2:
        st.markdown("##### What the model relies on")
        nice = {"age_group": "Age group", "gen_health": "General health", "income": "Income", "education": "Education",
                "sex": "Sex", "race": "Race", "smoked_100": "Ever smoked", "heavy_drinker": "Heavy drinking",
                "exercise": "Exercise", "diff_walking": "Difficulty walking", "stroke": "Stroke",
                "high_blood_pressure": "High blood pressure", "high_cholesterol": "High cholesterol",
                "diabetes": "Diabetes", "asthma": "Asthma", "kidney_disease": "Kidney disease", "copd": "COPD",
                "depression": "Depression", "arthritis": "Arthritis", "bmi": "BMI",
                "phys_days": "Poor physical-health days", "ment_days": "Poor mental-health days"}
        imp = pd.Series(report["permutation_importance"]).head(12).rename(index=nice).reset_index()
        imp.columns = ["answer", "importance"]
        st.altair_chart(alt.Chart(imp).mark_bar(cornerRadiusEnd=4, color=BLUE, height=16).encode(
            x=alt.X("importance:Q", title="Drop in PR-AUC when shuffled"), y=alt.Y("answer:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            tooltip=["answer:N", alt.Tooltip("importance:Q", format=".3f")]).properties(height=320), width="stretch")

    st.markdown(f"""
##### Method and limits
- **Data:** CDC's Behavioral Risk Factor Surveillance System, 15 yearly files harmonised into one table (questions
  renamed and recoded over the years are mapped to one standard). Blood pressure and cholesterol are asked in odd
  years, so the model uses the odd years: trained on 2011-2021, tuned on 2023, retrained on 2011-2023, tested once on 2025.
- **Blood pressure and cholesterol matter:** without them, PR-AUC on 2025 drops from {gb['best_f1']['pr_auc']:.3f}
  to {nobp['best_f1']['pr_auc']:.3f}.
- **Self-reported and cross-sectional:** people report whether a doctor *ever* told them they had heart disease.
  Some answers (stroke, poor health days, difficulty walking) may be consequences of heart disease, not causes.
- **Population:** US adults. Rates and risk factors differ in other countries, including the Gulf.
""")

st.markdown(f"""<div class="foot">Educational project, not medical advice. Data: CDC BRFSS 2011-2025 (public domain).
Code and full method: <a href="{REPO}">GitHub</a>. Built by Mohamed Ellithy · <a href="{LINKEDIN}">LinkedIn</a></div>""",
            unsafe_allow_html=True)
