"""Harmonisation map: one standard variable, many yearly names and codings.

CDC renames questions when their wording or answers change (DIABETE3 -> DIABETE4, SEX -> SEX1 -> SEXVAR,
INCOME2 -> INCOME3 ...). Each standard variable lists the raw names to look for (the first one found in a
year's file is used) and how to turn the raw codes into readable values.

Common BRFSS codes: 1 = Yes, 2 = No, 7 = Don't know, 9 = Refused, blank = not asked.
"Don't know" and "Refused" become missing values, never a category of their own.
Checked against each year's codebook and the value counts in reports/data_quality/.
"""
import numpy as np
import pandas as pd

YES_NO = {1: "Yes", 2: "No"}
AGE_GROUPS = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
              8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80 or older"}
GEN_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
# BPHIGH and DIABETE share this answer layout.
CONDITION_4 = {1: "Yes", 2: "Pregnancy only", 3: "No", 4: "Borderline"}
RACE = {1: "White", 2: "Black", 3: "Asian", 4: "American Indian/Alaska Native", 5: "Hispanic", 6: "Other"}
# 2015 and 2016 have no _IMPRACE (imputed race), only _RACE (8 groups, 9 = unknown). Mapping checked on
# 2017, which has both: identical for every known answer. Only _RACE's 9 stays missing (about 2%).
RACE_FROM_8 = {1: "White", 2: "Black", 3: "American Indian/Alaska Native", 4: "Asian",
               5: "Other", 6: "Other", 7: "Other", 8: "Hispanic"}
# INCOME3 (2021+) splits the top bracket into 75-100k, 100-150k, 150-200k and 200k+. They are merged back
# into 75k+ so every year uses the same 8 brackets. Brackets are not adjusted for inflation.
INCOME = {1: "<10k", 2: "10-15k", 3: "15-20k", 4: "20-25k", 5: "25-35k", 6: "35-50k", 7: "50-75k", 8: "75k+"}
EDUCATION = {1: "Less than high school", 2: "Less than high school", 3: "Less than high school",
             4: "High school", 5: "Some college", 6: "College graduate"}
CHECKUP = {1: "Within 1 year", 2: "1-2 years", 3: "2-5 years", 4: "5+ years", 8: "Never"}
SEX = {1: "Male", 2: "Female"}
# _RFDRHV* is a calculated "heavy drinker" flag coded 1 = No, 2 = Yes. CDC revised how drinks are
# counted over the years (suffixes 4 to 9), so compare it across years with care.
HEAVY = {1: "No", 2: "Yes"}


def labels(mapping):
    return lambda s, year: s.map(mapping)


def days(s, year):
    """Days in the past 30: 88 = none, 77/99 = don't know/refused."""
    return s.where(s.between(1, 30)).fillna(s.eq(88).map({True: 0.0, False: np.nan}))


def hours(s, year):
    return s.where(s.between(1, 24))


def bmi(s, year):
    return (s / 100).where(s.between(1200, 9999))  # stored with 2 implied decimals


def income(s, year):
    return s.where(s.between(1, 11)).clip(upper=8).map(INCOME)


# standard name: (raw names in priority order, recode function)
VARIABLES = {
    "state": (["_STATE"], lambda s, y: s.astype("Int64")),
    "weight": (["_LLCPWT"], lambda s, y: s),
    "heart_attack": (["CVDINFR4"], labels(YES_NO)),
    "coronary_heart_disease": (["CVDCRHD4"], labels(YES_NO)),
    "stroke": (["CVDSTRK3"], labels(YES_NO)),
    "high_blood_pressure": (["BPHIGH4", "BPHIGH6"], labels(CONDITION_4)),
    "high_cholesterol": (["TOLDHI2", "TOLDHI3"], labels(YES_NO)),
    "diabetes": (["DIABETE3", "DIABETE4"], labels(CONDITION_4)),
    "smoked_100": (["SMOKE100"], labels(YES_NO)),
    "heavy_drinker": (["_RFDRHV4", "_RFDRHV5", "_RFDRHV6", "_RFDRHV7", "_RFDRHV8", "_RFDRHV9"], labels(HEAVY)),
    "exercise": (["EXERANY2"], labels(YES_NO)),
    "bmi": (["_BMI5"], bmi),
    "sleep_hours": (["SLEPTIM1"], hours),
    "age_group": (["_AGEG5YR"], labels(AGE_GROUPS)),
    "sex": (["SEX", "SEX1", "SEXVAR"], labels(SEX)),
    "race": (["_IMPRACE", "_RACE"], labels(RACE)),
    "gen_health": (["GENHLTH"], labels(GEN_HEALTH)),
    "phys_days": (["PHYSHLTH"], days),
    "ment_days": (["MENTHLTH"], days),
    "diff_walking": (["DIFFWALK"], labels(YES_NO)),
    "asthma": (["ASTHMA3"], labels(YES_NO)),
    "kidney_disease": (["CHCKIDNY", "CHCKDNY1", "CHCKDNY2"], labels(YES_NO)),
    "copd": (["CHCCOPD", "CHCCOPD1", "CHCCOPD2", "CHCCOPD3"], labels(YES_NO)),
    "depression": (["ADDEPEV2", "ADDEPEV3"], labels(YES_NO)),
    "arthritis": (["HAVARTH3", "HAVARTH4", "HAVARTH5"], labels(YES_NO)),
    "income": (["INCOME2", "INCOME3"], income),
    "education": (["EDUCA"], labels(EDUCATION)),
    "last_checkup": (["CHECKUP1"], labels(CHECKUP)),
    "cost_barrier": (["MEDCOST", "MEDCOST1"], labels(YES_NO)),
}

# When the fallback column uses a different coding, its recode is given here.
RECODE_BY_RAW = {"_RACE": labels(RACE_FROM_8)}

# Cholesterol questions only go to people whose cholesterol was ever checked: BLOODCHO (2 = never)
# until 2015, then CHOLCHK1/2/3 (1 = never). "Never checked" is kept as its own answer.
CHOL_CHECK = ["BLOODCHO", "CHOLCHK1", "CHOLCHK2", "CHOLCHK3"]

# Everything else is a category.
NUMERIC = {"year", "state", "weight", "bmi", "sleep_hours", "phys_days", "ment_days"}

ORDERED = {
    "age_group": list(AGE_GROUPS.values()),
    "gen_health": ["Poor", "Fair", "Good", "Very good", "Excellent"],
    "income": list(INCOME.values()),
    "education": ["Less than high school", "High school", "Some college", "College graduate"],
    "last_checkup": ["Never", "5+ years", "2-5 years", "1-2 years", "Within 1 year"],
}


def raw_columns_needed(available: list[str]) -> dict[str, str]:
    """For one year's file: standard name -> the raw column to read (only those present)."""
    found = {}
    for std, (names, _) in VARIABLES.items():
        hit = next((n for n in names if n in available), None)
        if hit:
            found[std] = hit
    return found


def harmonise(raw: pd.DataFrame, year: int, found: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["year"] = np.int16(year)
    for std, (_, recode) in VARIABLES.items():
        if std in found:
            out[std] = RECODE_BY_RAW.get(found[std], recode)(raw[found[std]], year)
        else:
            out[std] = np.nan

    check = next((c for c in CHOL_CHECK if c in raw.columns), None)
    if check is not None:
        never = raw[check].eq(2) if check == "BLOODCHO" else raw[check].eq(1)
        out.loc[never & out["high_cholesterol"].isna(), "high_cholesterol"] = "Never checked"

    # Target: ever told heart attack OR coronary heart disease/angina (CDC's _MICHD definition).
    # Yes if either is Yes; No only if both are No; otherwise unknown.
    ha, chd = out["heart_attack"], out["coronary_heart_disease"]
    out["heart_disease"] = np.where((ha == "Yes") | (chd == "Yes"), "Yes",
                                    np.where((ha == "No") & (chd == "No"), "No", None))
    for col in out.columns:
        if col not in NUMERIC:
            out[col] = out[col].astype("object").astype("category")
    out["weight"] = out["weight"].astype("float32")
    return out
