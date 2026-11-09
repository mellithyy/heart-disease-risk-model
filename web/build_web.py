"""The heart risk app as one static page for mohamedellithy.com/heart-app/ (no server: the model runs in the visitor's browser).

    python web/build_web.py        -> web/dist/ (index.html + fonts), web/build/ (model.json, parity.json for the check)
    node web/parity.cjs            -> the page's model against scikit-learn on 20,000 random answers

The page holds: the model's 301 trees (model_export.py), the small report files the Streamlit app reads (metadata, test results,
15-year trends) and the English and Arabic texts (i18n.json). Fonts are served beside it (Switzer: ITF Free Font License,
Beiruti: SIL OFL; their licences travel with them). The site's own files (icons, sharing picture, visit counter) are linked by
their address on the site. A Content-Security-Policy in the page allows only its own scripts (by their hashes) and the counter.
"""
import base64
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import model_export  # noqa: E402

SITE = "https://mohamedellithy.com"
URL = SITE + "/heart-app/"
FONTS = ROOT.parents[2] / "job-search" / "portfolio-site" / "site_v7" / "assets" / "fonts"     # (the site's own font files)
UMAMI_ID = "2dafadef-1c0b-42d1-9bfd-0def96ebed61"      # (the site's visit counter: same website, counted only on the published address)
TRACK = ('(function(){if(location.hostname!=="mohamedellithy.com")return;'
         'try{var q=new URLSearchParams(location.search),me=q.get("me");'
         'if(me==="1")localStorage.setItem("umami.disabled","1");else if(me==="0")localStorage.removeItem("umami.disabled");'
         'if(localStorage.getItem("umami.disabled"))return;}catch(e){}'
         'var s=document.createElement("script");s.async=true;s.src="/assets/u.js";'
         's.setAttribute("data-website-id","' + UMAMI_ID + '");s.setAttribute("data-domains","mohamedellithy.com");'
         'document.head.appendChild(s);})();')


def data():
    meta = json.loads((ROOT / "models" / "brfss_metadata.json").read_text(encoding="utf-8"))
    rep = json.loads((ROOT / "reports" / "brfss_model.json").read_text(encoding="utf-8"))
    tr = json.loads((ROOT / "reports" / "brfss_trends.json").read_text(encoding="utf-8"))
    return {
        "meta": {k: meta[k] for k in ("answers", "weighted_rate_2021_2025", "weighted_rates_by_age_sex_2021_2025", "thresholds")},
        "report": {k: rep[k] for k in ("test_rows", "test_prevalence", "results", "calibration", "permutation_importance")},
        "trends": {k: tr[k] for k in ("crude", "adjusted", "factors_share", "odds_ratios", "state_tiles")},
        "text": json.loads((HERE / "i18n.json").read_text(encoding="utf-8")),
    }


def page(model, d):
    en = d["text"]["en"]
    blob = json.dumps(dict(d, model=model), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    js = (HERE / "predict.js").read_text(encoding="utf-8").replace('if (typeof module !== "undefined") module.exports = { makePredictor };', "")
    js += "\n" + (HERE / "app.js").read_text(encoding="utf-8")
    scripts = [js, TRACK]
    hashes = " ".join("'sha256-" + base64.b64encode(hashlib.sha256(s.encode("utf-8")).digest()).decode() + "'" for s in scripts)
    csp = ("default-src 'self'; script-src 'self' " + hashes + "; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
           "font-src 'self'; connect-src 'self' https://gateway.umami.is; object-src 'none'; base-uri 'self'; form-action 'none'")
    css = (HERE / "app.css").read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{en['page_title']}</title>
<meta name="description" content="{en['page_desc']}">
<link rel="canonical" href="{URL}">
<meta name="author" content="Mohamed Ellithy">
<meta property="og:type" content="website">
<meta property="og:title" content="{en['title']}">
<meta property="og:description" content="{en['page_desc']}">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="{SITE}/assets/og/heart-en.jpg">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#f2efe8">
<link rel="icon" href="/assets/icons/icon-32.png" sizes="32x32" type="image/png">
<link rel="apple-touch-icon" href="/assets/icons/apple-touch-icon.png">
<link rel="preload" href="fonts/Switzer-Variable.woff2" as="font" type="font/woff2" crossorigin>
<style>{css}</style>
</head>
<body>
<div id="app"><noscript><p style="padding:24px">{en['title']}: this page needs JavaScript; the model runs in your browser.</p></noscript></div>
<script type="application/json" id="app-data">{blob}</script>
<script>{js}</script>
<script>{TRACK}</script>
</body>
</html>
"""


def build():
    m = model_export.load()
    model = model_export.export(m)
    b = HERE / "build"
    b.mkdir(exist_ok=True)
    (b / "model.json").write_text(json.dumps(model, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (b / "parity.json").write_text(json.dumps(model_export.parity_cases(m)), encoding="utf-8")
    dist = HERE / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    (dist / "fonts").mkdir(parents=True)
    html = page(model, data())
    assert not re.search(r"\bNaN\b|Infinity", html.split('id="app-data">')[1].split("</script>")[0])
    (dist / "index.html").write_text(html, encoding="utf-8", newline="\n")
    for src, name in [(FONTS / "switzer" / "Switzer-Variable.woff2", "Switzer-Variable.woff2"), (FONTS / "switzer" / "LICENSE-FFL.txt", "LICENSE-Switzer-FFL.txt"),
                      (FONTS / "Beiruti-arabic.woff2", "Beiruti-arabic.woff2"), (FONTS / "OFL-Beiruti.txt", "OFL-Beiruti.txt")]:
        shutil.copy2(src, dist / "fonts" / name)
    size = (dist / "index.html").stat().st_size
    print(f"web/dist/index.html {size / 1e6:.2f} MB, {len(model['trees'])} trees; fonts copied; run node web/parity.cjs to check the model")


if __name__ == "__main__":
    build()
