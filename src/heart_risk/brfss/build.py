"""Build the harmonised 2011-2025 dataset from the raw yearly files.

For each year: read the zipped SAS file in chunks (no need to unzip 800 MB), keep only the columns in
the harmonisation map, translate them to the standard names and answers, save one parquet file.
Then combine all years into data/processed/brfss_2011_2025.parquet and write a data-quality report
(reports/data_quality/) showing each variable's answers year by year, so a coding change can't hide.

Run after download.py: python -m heart_risk.brfss.build
"""
import sys
import time
import zipfile

import pandas as pd

from ..paths import PROJECT
from .download import RAW_DIR, YEARS
from .variables import CHOL_CHECK, NUMERIC, ORDERED, VARIABLES, harmonise, raw_columns_needed

INTERIM = PROJECT / "data" / "interim" / "brfss"
PROCESSED = PROJECT / "data" / "processed" / "brfss_2011_2025.parquet"
QUALITY = PROJECT / "reports" / "data_quality"


def read_year(year: int):
    archive = zipfile.ZipFile(RAW_DIR / f"LLCP{year}XPT.zip")
    with archive.open(archive.namelist()[0]) as f:
        parts, found, keep = [], None, None
        for chunk in pd.read_sas(f, format="xport", chunksize=100_000, encoding="latin-1"):
            chunk.columns = [c.upper().strip() for c in chunk.columns]
            if found is None:
                found = raw_columns_needed(list(chunk.columns))
                keep = list(dict.fromkeys([*found.values(), *[c for c in CHOL_CHECK if c in chunk.columns]]))
            parts.append(chunk[keep])
    return pd.concat(parts, ignore_index=True), found


def quality_rows(df: pd.DataFrame, year: int, found: dict) -> list[dict]:
    rows = []
    for std in [*VARIABLES, "heart_disease"]:
        s = df[std]
        row = {"year": year, "variable": std, "raw_column": found.get(std, "derived" if std == "heart_disease" else "not asked"),
               "rows": len(s), "missing_pct": round(100 * s.isna().mean(), 1)}
        if std not in NUMERIC:
            shares = s.value_counts(normalize=True, dropna=True).round(3)
            row["answers"] = "; ".join(f"{k}: {v:.1%}" for k, v in shares.items() if v > 0)
        elif std not in ("state", "weight") and s.notna().any():
            row["answers"] = f"median {s.median():.1f}, min {s.min():.1f}, max {s.max():.1f}"
        rows.append(row)
    return rows


def main(years) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    QUALITY.mkdir(parents=True, exist_ok=True)
    report = []
    for year in years:
        t = time.time()
        raw, found = read_year(year)
        df = harmonise(raw, year, found)
        df.to_parquet(INTERIM / f"brfss_{year}.parquet", index=False)
        report += quality_rows(df, year, found)
        prev = (df["heart_disease"] == "Yes").sum() / df["heart_disease"].notna().sum()
        print(f"{year}: {len(df):,} rows, {len(found)} of {len(VARIABLES)} variables found, "
              f"heart disease {prev:.1%} (unweighted), {time.time() - t:.0f}s")
    pd.DataFrame(report).to_csv(QUALITY / "harmonisation_by_year.csv", index=False)

    # Combine. Categories differ by year (e.g. no "Never checked" in even years), so rebuild them.
    all_years = pd.concat([pd.read_parquet(p) for p in sorted(INTERIM.glob("brfss_*.parquet"))], ignore_index=True)
    for col in all_years.columns:
        if col not in NUMERIC:
            cats = ORDERED.get(col)
            all_years[col] = pd.Categorical(all_years[col].astype("object"), categories=cats, ordered=bool(cats))
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    all_years.to_parquet(PROCESSED, index=False)
    print(f"Combined: {len(all_years):,} rows x {all_years.shape[1]} columns -> {PROCESSED}")


if __name__ == "__main__":
    main([int(y) for y in sys.argv[1:]] or YEARS)
