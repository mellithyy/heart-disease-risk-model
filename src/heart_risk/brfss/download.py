"""Download the yearly BRFSS files (SAS transport format, zipped) from the CDC website.

2011 is the first year of the current survey design (landline + cell phone, raking weights).
Earlier files exist but CDC advises against comparing them with 2011 onwards, so they are not used.

Run: python -m heart_risk.brfss.download            (all years, skips files already downloaded)
     python -m heart_risk.brfss.download 2024 2025  (only these years)
"""
import sys
import zipfile

import requests

from ..paths import PROJECT

YEARS = range(2011, 2026)
RAW_DIR = PROJECT / "data" / "raw" / "brfss"
URL = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"


def download(year: int) -> None:
    target = RAW_DIR / f"LLCP{year}XPT.zip"
    if target.exists() and zipfile.is_zipfile(target):
        print(f"{year}: already downloaded ({target.stat().st_size / 1e6:.0f} MB)")
        return
    partial = target.with_suffix(".part")
    with requests.get(URL.format(year=year), stream=True, timeout=120) as r:
        r.raise_for_status()
        expected = int(r.headers.get("Content-Length", 0))
        with open(partial, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    size = partial.stat().st_size
    # Check before keeping the file: complete, and a readable zip.
    if expected and size != expected:
        raise IOError(f"{year}: got {size:,} bytes, expected {expected:,}")
    if not zipfile.is_zipfile(partial):
        raise IOError(f"{year}: the download is not a valid zip file")
    partial.replace(target)
    print(f"{year}: downloaded {size / 1e6:.0f} MB")


def main(years) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        download(year)


if __name__ == "__main__":
    main([int(y) for y in sys.argv[1:]] or YEARS)
