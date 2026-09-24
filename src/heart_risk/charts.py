"""Shared chart style for the report figures (static PNGs shown in the README).

Palette: categorical slots in fixed order (blue, orange, aqua), gray for references,
thin marks, recessive grid, text in ink colours rather than series colours.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# Plain-English names for the survey columns, used on chart axes and in the app.
NICE = {"AgeCategory": "Age group", "GenHealth": "General health (self-rated)", "Sex": "Sex",
        "Stroke": "Ever had a stroke", "Smoking": "Smoked 100+ cigarettes", "Diabetic": "Diabetes",
        "Race": "Race", "KidneyDisease": "Kidney disease", "PhysicalHealth": "Bad physical-health days",
        "DiffWalking": "Difficulty walking", "MentalHealth": "Bad mental-health days", "BMI": "BMI",
        "SleepTime": "Sleep hours", "Asthma": "Asthma", "SkinCancer": "Skin cancer",
        "AlcoholDrinking": "Heavy drinking", "PhysicalActivity": "Physically active"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
})


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
