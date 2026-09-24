"""What 15 years of the survey (2011-2025, 6.7 million adults) say about heart disease and its factors.

All rates use the survey weights (_LLCPWT), so they describe US adults, not just the people who answered.
- Age-adjusted rates: the US population got older over these years, which alone raises heart disease.
  Direct standardisation fixes one age mix (all years pooled) so years and states compare fairly.
- Factor strength: Mantel-Haenszel odds ratios within 26 age x sex groups, per year. They show whether
  a factor's link with heart disease is stable over time, beyond the fact that it is more common in older men.

Run after build.py: python -m heart_risk.brfss.trends
"""
import json

import numpy as np
import pandas as pd

from ..charts import INK, INK_2, MUTED, SERIES, plt, save
from ..paths import PROJECT
from .build import PROCESSED

FIGURES = PROJECT / "reports" / "figures"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FIPS = {1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL", 13: "GA",
        15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
        25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ",
        35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC",
        46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY"}
# Tile-grid map: one square per state, roughly where it sits on a real map.
TILES = {"AK": (0, 0), "ME": (11, 0), "VT": (10, 1), "NH": (11, 1), "WA": (1, 2), "ID": (2, 2), "MT": (3, 2),
         "ND": (4, 2), "MN": (5, 2), "IL": (6, 2), "WI": (7, 2), "MI": (8, 2), "NY": (9, 2), "RI": (10, 2),
         "MA": (11, 2), "OR": (1, 3), "NV": (2, 3), "WY": (3, 3), "SD": (4, 3), "IA": (5, 3), "IN": (6, 3),
         "OH": (7, 3), "PA": (8, 3), "NJ": (9, 3), "CT": (10, 3), "CA": (1, 4), "UT": (2, 4), "CO": (3, 4),
         "NE": (4, 4), "MO": (5, 4), "KY": (6, 4), "WV": (7, 4), "VA": (8, 4), "MD": (9, 4), "DE": (10, 4),
         "AZ": (2, 5), "NM": (3, 5), "KS": (4, 5), "AR": (5, 5), "TN": (6, 5), "NC": (7, 5), "SC": (8, 5),
         "DC": (9, 5), "OK": (4, 6), "LA": (5, 6), "MS": (6, 6), "AL": (7, 6), "GA": (8, 6), "HI": (0, 7),
         "TX": (4, 7), "FL": (9, 7)}

FACTORS = {  # (column, exposed answer, reference answer, label)
    "stroke": ("stroke", "Yes", "No", "Ever had a stroke"),
    "kidney": ("kidney_disease", "Yes", "No", "Kidney disease"),
    "diabetes": ("diabetes", "Yes", "No", "Diabetes"),
    "copd": ("copd", "Yes", "No", "COPD"),
    "bp": ("high_blood_pressure", "Yes", "No", "High blood pressure"),
    "chol": ("high_cholesterol", "Yes", "No", "High cholesterol"),
    "smoking": ("smoked_100", "Yes", "No", "Ever smoked (100+ cigarettes)"),
    "inactive": ("exercise", "No", "Yes", "No exercise in past month"),
}


def load() -> pd.DataFrame:
    cols = ["year", "state", "weight", "age_group", "sex", "heart_disease", "bmi", "diabetes", "smoked_100",
            "exercise", "high_blood_pressure", "high_cholesterol", "depression", "stroke", "kidney_disease", "copd"]
    df = pd.read_parquet(PROCESSED, columns=cols)
    df = df[df["heart_disease"].notna()].copy()
    df["hd"] = (df["heart_disease"] == "Yes").astype("int8")
    return df


def weighted_rate(df, flag, by):
    d = df[flag.notna()].assign(_f=flag[flag.notna()].astype(float), _w=df["weight"])
    g = d.groupby(by, observed=True)
    return (g.apply(lambda x: np.average(x["_f"], weights=x["_w"]), include_groups=False))


def age_adjusted(df, by, standard):
    """Direct standardisation: age-specific weighted rates, combined with one fixed age mix."""
    d = df[df["age_group"].notna()]
    rates = weighted_rate(d, d["hd"], [*by, "age_group"]).unstack("age_group")
    return (rates[standard.index] * standard.values).sum(axis=1) / standard.sum()


def mantel_haenszel(df, col, exposed, ref):
    """Odds ratio for exposed vs ref, pooled over the 26 age x sex groups (each group compares like with like)."""
    d = df[df[col].isin([exposed, ref]) & df["age_group"].notna() & df["sex"].notna()]
    x, y = d[col] == exposed, d["hd"] == 1
    cells = pd.DataFrame({"a": x & y, "b": x & ~y, "c": ~x & y, "d": ~x & ~y}).astype(int)
    s = cells.groupby([d["age_group"], d["sex"]], observed=True).sum()
    n = s.sum(axis=1)
    return float((s["a"] * s["d"] / n).sum() / (s["b"] * s["c"] / n).sum())


def charts(res):
    # 1. Heart disease over time: crude vs age-adjusted
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    yrs = list(res["crude"].keys())
    for i, (key, name) in enumerate([("crude", "As reported (crude)"), ("adjusted", "Age-adjusted")]):
        v = [100 * res[key][y] for y in yrs]
        ax.plot([int(y) for y in yrs], v, "-o", ms=4, color=SERIES[i], label=name, mec="#fcfcfb", mew=1)
        ax.annotate(f"{v[-1]:.1f}%", (int(yrs[-1]), v[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_ylabel("Adults ever told CHD or heart attack (%)")
    ax.text(0.99, 0.97, "The odd/even zig-zag comes from CDC alternating two questionnaires",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8.5)
    ax.set_title("Heart disease 2011-2025: flat as reported, falling once ageing is removed")
    ax.set_xticks(range(2011, 2026, 2))
    ax.legend(loc="lower left")
    save(fig, FIGURES / "brfss_trend.png")

    # 2. Risk factors over time (small multiples, one series each)
    panels = [("obesity", "Obesity (BMI 30+)"), ("diabetes", "Diabetes"), ("smoking", "Ever smoked"),
              ("inactive", "No exercise"), ("bp", "High blood pressure"), ("depression", "Depression")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    for ax, (key, name) in zip(axes.flat, panels):
        s = pd.Series(res["factors_share"][key]).dropna()
        ax.plot(s.index.astype(int), 100 * s.values, "-o", ms=3.5, color=SERIES[0])
        ax.set_title(name, fontsize=11)
        first, last = s.index[0], s.index[-1]
        ax.text(0.02, 0.06, f"{int(first)}: {100 * s.iloc[0]:.1f}%   {int(last)}: {100 * s.iloc[-1]:.1f}%",
                transform=ax.transAxes, color=INK_2, fontsize=8.5)
        lo, hi = 100 * s.min(), 100 * s.max()
        ax.set_ylim(lo - (hi - lo) * 0.6 - 1, hi + (hi - lo) * 0.3 + 1)
        ax.set_xticks([2011, 2015, 2019, 2023])
        if key == "inactive":
            ax.text(0.98, 0.94, "zig-zag: odd/even questionnaire", transform=ax.transAxes, ha="right", va="top",
                    color=MUTED, fontsize=8)
    fig.suptitle("Risk factors among US adults, 2011-2025 (weighted %)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "brfss_risk_factors.png")

    # 3. Factor strength per year (Mantel-Haenszel odds ratios, age and sex held constant)
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.9), sharey=True)
    for ax, (key, (_, _, _, name)) in zip(axes.flat, FACTORS.items()):
        s = pd.Series(res["odds_ratios"][key]).dropna()
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        ax.plot(s.index.astype(int), s.values, "-o", ms=3.5, color=SERIES[0])
        ax.set_title(name, fontsize=10.5)
        ax.text(0.97, 0.1, f"median {s.median():.1f}", transform=ax.transAxes, ha="right", color=INK_2, fontsize=8.5)
        ax.set_xticks([2011, 2017, 2023])
    axes[0, 0].set_ylabel("Odds ratio")
    axes[1, 0].set_ylabel("Odds ratio")
    fig.suptitle("How strongly each factor goes with heart disease, year by year (same age and sex compared)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES / "brfss_factor_stability.png")

    # 4. States, age-adjusted, 2021-2025 pooled (tile grid)
    rates = {FIPS[int(k)]: v for k, v in res["states_2021_2025"].items() if int(k) in FIPS}
    vals = np.array(list(rates.values())) * 100
    edges = np.quantile(vals, np.linspace(0, 1, len(BLUES) + 1))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for st, (cx, cy) in TILES.items():
        if st not in rates:
            continue
        v = 100 * rates[st]
        k = min(int(np.searchsorted(edges, v, side="right")) - 1, len(BLUES) - 1)
        ax.add_patch(plt.Rectangle((cx, -cy), 0.92, 0.92, color=BLUES[max(k, 0)], lw=0))
        ink = "#ffffff" if k >= 3 else INK
        ax.text(cx + 0.46, -cy + 0.58, st, ha="center", va="center", fontsize=9, fontweight="bold", color=ink)
        ax.text(cx + 0.46, -cy + 0.28, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color=ink)
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-7.3, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Heart disease by state, age-adjusted, 2021-2025 (% of adults). Darker = higher.")
    save(fig, FIGURES / "brfss_states.png")


def main():
    df = load()
    std = df.groupby("age_group", observed=True)["weight"].sum()  # pooled age mix as the standard
    df["obese"] = np.where(df["bmi"].notna(), (df["bmi"] >= 30).astype(float), np.nan)

    res = {"rows": int(len(df)), "years": sorted(int(y) for y in df["year"].unique())}
    res["crude"] = {str(k): round(float(v), 5) for k, v in weighted_rate(df, df["hd"], ["year"]).items()}
    res["adjusted"] = {str(k): round(float(v), 5) for k, v in age_adjusted(df, ["year"], std).items()}

    def share(col, yes):
        flag = df["obese"] if col == "obese" else (df[col] == yes).astype(float).where(df[col].notna())
        return {str(k): round(float(v), 5) for k, v in weighted_rate(df, flag, ["year"]).items()}

    res["factors_share"] = {"obesity": share("obese", None), "diabetes": share("diabetes", "Yes"),
                            "smoking": share("smoked_100", "Yes"), "inactive": share("exercise", "No"),
                            "bp": share("high_blood_pressure", "Yes"), "depression": share("depression", "Yes")}
    res["odds_ratios"] = {}
    for key, (col, exp, ref, _) in FACTORS.items():
        res["odds_ratios"][key] = {}
        for y, d in df.groupby("year"):
            if d[col].notna().mean() > 0.5:
                res["odds_ratios"][key][str(y)] = round(mantel_haenszel(d, col, exp, ref), 3)
    recent = df[df["year"] >= 2021]
    res["states_2021_2025"] = {str(int(k)): round(float(v), 5)
                               for k, v in age_adjusted(recent, ["state"], std).items()}
    res["state_tiles"] = [{"state": FIPS[int(k)], "col": TILES[FIPS[int(k)]][0], "row": TILES[FIPS[int(k)]][1],
                           "rate": v} for k, v in res["states_2021_2025"].items() if int(k) in FIPS]
    (PROJECT / "reports" / "brfss_trends.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    charts(res)
    for y in ("2011", "2025"):
        print(y, "crude", res["crude"][y], "age-adjusted", res["adjusted"][y])
    print("odds ratios (median):", {k: round(float(np.median(list(v.values()))), 2) for k, v in res["odds_ratios"].items()})


if __name__ == "__main__":
    main()
