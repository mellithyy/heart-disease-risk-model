// The heart risk app: the answers, the estimate and its charts, in English or Arabic. Everything runs here, in the page: the
// model (predict.js, the saved scikit-learn model as numbers) and the report files are written into the page by build_web.py.
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("app-data").textContent);
  const predict = makePredictor(D.model), M = D.meta, R = D.report, TR = D.trends, TX = D.text;
  const SITE = "https://mohamedellithy.com/", REPO = "https://github.com/mellithyy/heart-disease-risk-model";
  const SKIP = "__skip", NS = "http://www.w3.org/2000/svg";
  const REDUCE = matchMedia("(prefers-reduced-motion: reduce)");

  // ---- language
  let L = (() => {
    const q = new URLSearchParams(location.search).get("lang");
    if (q === "ar" || q === "en") return q;
    try { const s = localStorage.getItem("heart.lang"); if (s === "ar" || s === "en") return s; } catch (e) { /* no storage */ }
    return "en";
  })();
  const fill = (s, a) => String(s).replace(/\{(\d)\}/g, (_, i) => a[+i]);
  const t = (k, ...a) => fill(TX[L][k] !== undefined ? TX[L][k] : TX.en[k], a);
  const ans = v => (TX[L].a[v] !== undefined ? TX[L].a[v] : v);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  // (inside an Arabic sentence a number or a range is kept left to right: "60-64", "13.0%", "-1.2", not "64-60" or "1.2-")
  const iso = s => (L === "ar" ? "\u2066" + s + "\u2069" : String(s));
  const pct = (v, d) => iso((100 * v).toFixed(d) + "%");
  const num = v => iso(v.toLocaleString("en-US"));
  const age = v => (TX[L].a[v] !== undefined ? TX[L].a[v] : iso(v));
  const rtl = () => L === "ar";

  // ---- the answers (the Streamlit app's starting answers)
  const START = {
    age_group: "55-59", sex: "Female", height: 170, weight: 78, race: SKIP, education: SKIP, income: SKIP,
    high_blood_pressure: "No", high_cholesterol: "No", diabetes: "No",
    stroke: false, kidney_disease: false, copd: false, asthma: false, arthritis: false, depression: false,
    smoked_100: false, heavy_drinker: false, exercise: true,
    gen_health: "Very good", phys_days: 0, ment_days: 0, diff_walking: false,
  };
  const S = Object.assign({}, START);
  const BOOL = new Set(Object.keys(START).filter(k => typeof START[k] === "boolean"));
  let tab = "you", played = false;
  const bmiOf = () => S.weight / Math.pow(S.height / 100, 2);
  function person(change) {
    const yn = b => (b ? "Yes" : "No"), sk = v => (v === SKIP ? null : v);
    const p = {
      age_group: S.age_group, gen_health: S.gen_health, income: sk(S.income), education: sk(S.education), sex: S.sex, race: sk(S.race),
      smoked_100: yn(S.smoked_100), heavy_drinker: yn(S.heavy_drinker), exercise: yn(S.exercise), diff_walking: yn(S.diff_walking),
      stroke: yn(S.stroke), high_blood_pressure: S.high_blood_pressure, high_cholesterol: S.high_cholesterol, diabetes: S.diabetes,
      asthma: yn(S.asthma), kidney_disease: yn(S.kidney_disease), copd: yn(S.copd), depression: yn(S.depression),
      arthritis: yn(S.arthritis), bmi: Math.round(bmiOf() * 100) / 100, phys_days: S.phys_days, ment_days: S.ment_days,
    };
    return Object.assign(p, change || {});
  }

  // ---- the drawings that do not change: icons, the heart in the head of the page, the body
  const ICONS = `<svg class="sprite" width="0" height="0" aria-hidden="true" focusable="false"><defs>
<symbol id="i-people" viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.2"/><path d="M3.3 19.5c.6-3.3 2.8-5.3 5.7-5.3s5.1 2 5.7 5.3"/><circle cx="16.6" cy="8.8" r="2.6"/><path d="M15.4 13.9c2.7-.3 4.8 1.3 5.4 4.4"/></symbol>
<symbol id="i-cal" viewBox="0 0 24 24"><rect x="3.5" y="5" width="17" height="15.5" rx="2.6"/><path d="M3.5 9.6h17M8 3.2v3.6M16 3.2v3.6"/><path d="M7.4 13.2h1.2M11.4 13.2h1.2M15.4 13.2h1.2M7.4 16.6h1.2M11.4 16.6h1.2"/></symbol>
<symbol id="i-clip" viewBox="0 0 24 24"><rect x="5" y="4.6" width="14" height="16" rx="2.2"/><path d="M9.2 4.6V3.4h5.6v1.2"/><path d="M8.6 10.2l1.2 1.2 2.2-2.4M13.6 10.4h2M8.6 15.2l1.2 1.2 2.2-2.4M13.6 15.4h2"/></symbol>
<symbol id="i-roc" viewBox="0 0 24 24"><path d="M4.5 3.5v16h16"/><path d="M4.5 19.5l15-15" stroke-dasharray="1.6 2.2" opacity=".7"/><path d="M4.5 19.5C5.6 11 9.6 6.2 19.5 4.6" stroke="#d93a3f" stroke-width="1.8"/></symbol>
<symbol id="i-watch" viewBox="0 0 24 24"><circle cx="12" cy="13.6" r="7"/><path d="M12 13.6V9.8M9.8 3.3h4.4M12 3.3v3.3M18.1 7.4l1.4-1.4"/></symbol>
<symbol id="i-person" viewBox="0 0 24 24"><circle cx="12" cy="7.8" r="3.7"/><path d="M4.8 20.6c.8-4 3.6-6.4 7.2-6.4s6.4 2.4 7.2 6.4"/></symbol>
<symbol id="i-steth" viewBox="0 0 24 24"><path d="M5 3.4h2M13.6 3.4h2M6 3.4v5.4a4.3 4.3 0 0 0 8.6 0V3.4"/><path d="M10.3 13.1v2.6a4.2 4.2 0 0 0 8.4 0v-2.4"/><circle cx="18.7" cy="11" r="2.3"/></symbol>
<symbol id="i-shoe" viewBox="0 0 24 24"><path d="M2.8 16.2v-1.6c0-.8.5-1.5 1.3-1.8l2.6-1c.8-.3 1.4-.9 1.8-1.7l1.6-3.4c.3-.6 1-.8 1.6-.5l1.2.7c.5.3.7.9.5 1.4l-.5 1.1 2.4 1.5c1 .6 2.2 1 3.4 1.1l.6.1c1.3.1 2.3 1.2 2.3 2.5v1.6"/><path d="M2.8 16.2h18.8v1.4c0 .7-.6 1.3-1.3 1.3H4.1c-.7 0-1.3-.6-1.3-1.3z"/><path d="M11.3 9.6l1.7 1M10.4 11.7l1.8 1"/></symbol>
<symbol id="i-smile" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.6"/><path d="M8.5 14.2c.9 1.4 2.1 2.1 3.5 2.1s2.6-.7 3.5-2.1"/><path d="M9.2 9.9v.4M14.8 9.9v.4" stroke-width="2"/></symbol>
<symbol id="i-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.2M12 7.9v.2" stroke-width="1.8"/></symbol>
<symbol id="i-lock" viewBox="0 0 24 24"><rect x="5.2" y="10.6" width="13.6" height="9.6" rx="2.2"/><path d="M8.4 10.6V8.1a3.6 3.6 0 0 1 7.2 0v2.5M12 14.4v2"/></symbol>
<symbol id="i-flag" viewBox="0 0 24 24"><path d="M6 21V4.2"/><path d="M6 4.6c2.8-1.6 5.4 1.4 8.2 0 1.7-.8 3-.8 4.3 0v8.2c-1.3-.8-2.6-.8-4.3 0-2.8 1.4-5.4-1.6-8.2 0"/></symbol>
<symbol id="i-down" viewBox="0 0 24 24"><path d="M6.5 9.5l5.5 5.5 5.5-5.5"/></symbol>
<symbol id="i-check" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.6"/><path d="M8.3 12.3l2.5 2.5 4.9-5.2"/></symbol>
<symbol id="i-gauge" viewBox="0 0 24 24"><path d="M3.6 16.5a8.4 8.4 0 1 1 16.8 0"/><path d="M12 16.5l3.6-4.6"/><circle cx="12" cy="16.5" r="1.3"/></symbol>
<symbol id="i-trend" viewBox="0 0 24 24"><path d="M3.5 19.5h17"/><path d="M4 15.5l4.3-4.2 3.6 2.6 4.6-6 3.5 2.6"/></symbol>
<symbol id="i-tree" viewBox="0 0 24 24"><rect x="9" y="3.4" width="6" height="4.2" rx="1.2"/><rect x="3.4" y="16.4" width="6" height="4.2" rx="1.2"/><rect x="14.6" y="16.4" width="6" height="4.2" rx="1.2"/><path d="M12 7.6v4.2M6.4 16.4v-2.2c0-1.3 1-2.4 2.4-2.4h6.4c1.3 0 2.4 1.1 2.4 2.4v2.2"/></symbol>
</defs></svg>`;
  const ic = (id, cls) => `<svg class="i${cls ? " " + cls : ""}" aria-hidden="true" focusable="false"><use href="#i-${id}"/></svg>`;

  // the heart, drawn in lines as in an anatomy book: arteries tinted red, veins blue; the coronary arteries draw themselves once
  const HEART = `<svg class="heart pre" id="heart" viewBox="0 0 360 430" aria-hidden="true" focusable="false">
<path class="o v" d="M121 117 L73 114 L73 146 L121 147 Z"/><ellipse class="cap" cx="73" cy="130" rx="6.2" ry="16"/><ellipse class="lumen v" cx="73" cy="130" rx="3.1" ry="10.5"/><path class="h" d="M84 120 L114 121 M84 125 L110 126"/>
<path class="o v" d="M108 62 L111 178 L148 172 L147 62 Z"/><ellipse class="cap" cx="127.5" cy="62" rx="19.5" ry="6.5"/><ellipse class="lumen v" cx="127.5" cy="62" rx="13" ry="3.6"/><path class="h" d="M114 78 L115.5 154 M119 84 L120 140 M124 90 L124.6 128"/>
<path class="o a" d="M178 20 C180 44 184 66 186 96 L208 80 C205 56 203 38 202 20 Z"/><ellipse class="cap" cx="190" cy="20" rx="12" ry="4.6"/><ellipse class="lumen a" cx="190" cy="20" rx="7.5" ry="2.4"/><path class="h" d="M183 34 C185 50 187 64 189 82"/>
<path class="o a" d="M229 12 L230 70 L245 68 L245 12 Z"/><ellipse class="cap" cx="237" cy="12" rx="8" ry="3.4"/><ellipse class="lumen a" cx="237" cy="12" rx="4.6" ry="1.8"/><path class="h" d="M233 24 L233.6 60"/>
<path class="o a" d="M262 68 C266 48 272 34 278 22 L294 28 C287 40 282 54 279 72 Z"/><ellipse class="cap" cx="286" cy="25" rx="8.8" ry="3.6" transform="rotate(22 286 25)"/><ellipse class="lumen a" cx="286" cy="25" rx="5" ry="1.9" transform="rotate(22 286 25)"/>
<path class="o a" d="M158 196 C152 132 166 80 218 67 C264 56 303 80 307 124 L309 176 L279 176 L277 130 C275 106 254 94 228 99 C204 104 194 138 197 192 Z"/>
<path class="h" d="M165 172 C161 130 171 96 204 79 M171 170 C168 134 177 106 201 91 M177 168 C175 138 182 116 199 103 M291 108 C295 122 296 140 296 162 M298 102 C302 118 303 138 303 160"/>
<path class="o" d="M150 172 C118 172 96 198 92 234 C88 270 100 298 122 320 C156 354 212 393 250 412 C264 419 277 413 284 400 C312 352 337 294 337 246 C337 214 325 194 304 182 C276 168 200 162 150 172 Z"/>
<path class="d" d="M150 176 C127 196 117 228 121 264 C123 280 128 292 134 302"/><path class="d" d="M212 206 C200 232 195 262 198 292"/>
<path class="h" d="M99 226 C97 250 103 272 115 290 M105 214 C102 238 107 260 117 280 M111 206 C108 228 111 248 119 266"/>
<path class="h" d="M128 318 C150 338 176 356 204 374 M140 314 C160 330 182 346 206 360 M152 310 C170 324 188 336 206 346"/>
<path class="h" d="M327 238 C328 280 315 320 297 356 M319 236 C320 272 309 308 293 340 M311 234 C311 264 303 294 289 322 M303 236 C302 258 296 280 286 300"/>
<path class="h" d="M262 400 C270 404 276 400 281 393 M256 392 C264 396 270 394 275 388"/>
<path class="o v" d="M214 202 C210 170 222 144 248 130 C272 118 312 112 347 112 L347 142 C330 142 312 146 292 156 C276 166 268 184 266 206 Z"/><ellipse class="cap" cx="347" cy="127" rx="6.5" ry="15"/><ellipse class="lumen v" cx="347" cy="127" rx="3.2" ry="10"/>
<path class="h" d="M221 196 C219 172 227 152 247 140 M228 192 C227 174 233 160 248 150 M300 127 C318 123 330 122 340 122"/>
<path class="o" d="M288 171 C295 161 310 156 324 159 C336 162 343 172 339 181 C344 187 341 197 331 198 C333 206 323 212 313 208 C305 214 291 211 287 202 C282 193 282 179 288 171 Z"/>
<path class="d" d="M296 182 C306 178 318 180 327 188 M300 194 C307 192 314 194 319 199"/>
<path class="cor vn" d="M249 212 C243 266 240 324 259 394"/><path class="cor vn" d="M246 268 C262 278 276 294 284 312"/><path class="cor vn" d="M272 212 C292 216 312 228 326 248"/>
<path class="cor" d="M240 207 C232 262 227 324 250 398"/><path class="cor thin" d="M237 252 C257 262 278 278 294 302"/><path class="cor thin" d="M233 306 C250 318 263 334 271 356"/><path class="cor thin" d="M238 352 C228 360 222 370 220 382"/>
<path class="cor" d="M266 208 C288 210 312 222 330 246"/><path class="cor thin" d="M312 224 C318 250 318 276 312 300"/>
<path class="cor" d="M178 198 C150 214 130 250 128 290 C127 314 142 334 164 348"/><path class="cor thin" d="M130 268 C150 278 168 296 178 320"/><path class="cor thin" d="M164 348 C180 360 200 370 216 376"/>
</svg>`;

  // the body seen from the front (the person's left on the viewer's right, as in an anatomy book; the same in both languages).
  // Each area carries data-area; update() lights the ones the answers point to.
  const BODY = `<svg class="body" id="bodyfig" viewBox="14 8 212 548" role="img" focusable="false">
<defs><radialGradient id="bglow" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#d93a3f" stop-opacity=".34"/><stop offset=".6" stop-color="#d93a3f" stop-opacity=".1"/><stop offset="1" stop-color="#d93a3f" stop-opacity="0"/></radialGradient>
<g id="bvp"><path pathLength="1" d="M127 160C127 146 115 143 115 155L117 282"/><path pathLength="1" d="M117 282C117 294 133 300 140 322L145 410L142 516"/><path pathLength="1" d="M117 282C117 294 103 300 98 322L95 410L98 516"/><path pathLength="1" d="M122 146L126 98M117 146L113 98"/><path pathLength="1" d="M124 144C140 132 170 126 184 138L193 232L206 318"/><path pathLength="1" d="M116 144C100 132 70 126 56 138L47 232L34 318"/></g></defs>
<path class="b-out" d="M131 72C131.2 75.3 130.2 87.3 132 92C133.8 96.7 137 97.7 142 100C147 102.3 155.7 104 162 106C168.3 108 175.3 109.3 180 112C184.7 114.7 187.7 117.3 190 122C192.3 126.7 193 132 194 140C195 148 195.2 159.2 196 170C196.8 180.8 198 194.7 199 205C200 215.3 200.7 222 202 232C203.3 242 205.3 254 207 265C208.7 276 210.8 289.5 212 298C213.2 306.5 212.8 310.7 214 316C215.2 321.3 217.8 324.7 219 330C220.2 335.3 221.5 342.7 221 348C220.5 353.3 218.2 359.7 216 362C213.8 364.3 210 364 208 362C206 360 204.8 354.7 204 350C203.2 345.3 203.7 339.3 203 334C202.3 328.7 201.5 325.3 200 318C198.5 310.7 196 300 194 290C192 280 189.7 267.3 188 258C186.3 248.7 185.2 242.8 184 234C182.8 225.2 182 214.8 181 205C180 195.2 179.2 184.2 178 175C176.8 165.8 174.7 154.2 174 150C173.5 154.2 172.3 165 171 175C169.7 185 167.5 199.2 166 210C164.5 220.8 162.3 231.3 162 240C161.7 248.7 162.7 253.7 164 262C165.3 270.3 168.7 280.7 170 290C171.3 299.3 172.2 307.7 172 318C171.8 328.3 170.5 340 169 352C167.5 364 164.8 380 163 390C161.2 400 158.7 403.7 158 412C157.3 420.3 158.7 431 159 440C159.3 449 160.7 456.8 160 466C159.3 475.2 156.7 486 155 495C153.3 504 150.2 513.5 150 520C149.8 526.5 153 529.7 154 534C155 538.3 157.7 543.3 156 546C154.3 548.7 148.3 549.7 144 550C139.7 550.3 132.7 550.3 130 548C127.3 545.7 128.2 540.7 128 536C127.8 531.3 128.5 526.8 129 520C129.5 513.2 130.5 504 131 495C131.5 486 132.3 475.5 132 466C131.7 456.5 129.7 447 129 438C128.3 429 128.3 421.7 128 412C127.7 402.3 127.5 391.2 127 380C126.5 368.8 125.8 355.3 125 345C124.2 334.7 122.8 324.5 122 318C121.2 311.5 120.3 308 120 306C119.7 308 118.8 311.5 118 318C117.2 324.5 115.8 334.7 115 345C114.2 355.3 113.5 368.8 113 380C112.5 391.2 112.3 402.3 112 412C111.7 421.7 111.7 429 111 438C110.3 447 108.3 456.5 108 466C107.7 475.5 108.5 486 109 495C109.5 504 110.5 513.2 111 520C111.5 526.8 112.2 531.3 112 536C111.8 540.7 112.7 545.7 110 548C107.3 550.3 100.3 550.3 96 550C91.7 549.7 85.7 548.7 84 546C82.3 543.3 85 538.3 86 534C87 529.7 90.2 526.5 90 520C89.8 513.5 86.7 504 85 495C83.3 486 80.7 475.2 80 466C79.3 456.8 80.7 449 81 440C81.3 431 82.7 420.3 82 412C81.3 403.7 78.8 400 77 390C75.2 380 72.5 364 71 352C69.5 340 68.2 328.3 68 318C67.8 307.7 68.7 299.3 70 290C71.3 280.7 74.7 270.3 76 262C77.3 253.7 78.3 248.7 78 240C77.7 231.3 75.5 220.8 74 210C72.5 199.2 70.3 185 69 175C67.7 165 66.5 154.2 66 150C65.3 154.2 63.2 165.8 62 175C60.8 184.2 60 195.2 59 205C58 214.8 57.2 225.2 56 234C54.8 242.8 53.7 248.7 52 258C50.3 267.3 48 280 46 290C44 300 41.5 310.7 40 318C38.5 325.3 37.7 328.7 37 334C36.3 339.3 36.8 345.3 36 350C35.2 354.7 34 360 32 362C30 364 26.2 364.3 24 362C21.8 359.7 19.5 353.3 19 348C18.5 342.7 19.8 335.3 21 330C22.2 324.7 24.8 321.3 26 316C27.2 310.7 26.8 306.5 28 298C29.2 289.5 31.3 276 33 265C34.7 254 36.7 242 38 232C39.3 222 40 215.3 41 205C42 194.7 43.2 180.8 44 170C44.8 159.2 45 148 46 140C47 132 47.7 126.7 50 122C52.3 117.3 55.3 114.7 60 112C64.7 109.3 71.7 108 78 106C84.3 104 93 102.3 98 100C103 97.7 106.2 96.7 108 92C109.8 87.3 108.8 75.3 109 72Z"/>
<ellipse class="b-out" cx="120" cy="46" rx="27" ry="32"/>
<g class="org ves" data-area="heart"><use class="v-idle" href="#bvp"/><use class="v-lit" href="#bvp"/></g>
<g class="org soft legs" data-area="legs"><ellipse class="glow" cx="120" cy="420" rx="62" ry="110" fill="url(#bglow)"/>
<ellipse class="s" cx="147" cy="362" rx="12" ry="37" transform="rotate(5 147 362)"/><ellipse class="s" cx="93" cy="362" rx="12" ry="37" transform="rotate(-5 93 362)"/>
<ellipse class="s" cx="145" cy="466" rx="8.5" ry="33"/><ellipse class="s" cx="95" cy="466" rx="8.5" ry="33"/></g>
<g class="org" data-area="brain"><ellipse class="glow" cx="120" cy="37" rx="34" ry="28" fill="url(#bglow)"/>
<path class="s" d="M101 42C98 34 103 25 111 25C114 20 126 20 129 25C137 25 142 33 139 41C141 47 136 52 130 50C127 54 122 54 120 51C118 54 113 54 110 50C104 52 99 47 101 42Z"/>
<path class="d" d="M120 23C119 32 121 42 120 51M107 34c3-2 6 0 8 2M125 33c3-2 6-1 8 2M106 44c3 1 6 0 8-2M126 42c3 2 6 2 8 0"/></g>
<g class="org soft" data-area="lungs"><ellipse class="glow" cx="120" cy="164" rx="52" ry="50" fill="url(#bglow)"/>
<path class="d" d="M120 98V121M120 121C118 125 115 127 111 131M120 121C122 125 125 127 129 131"/>
<path class="s" d="M115 123C103 121 91 139 89 163C87 183 89 198 97 202C105 205 113 200 115 192Z"/>
<path class="s" d="M125 123C137 121 149 139 151 163C153 183 151 198 143 202C135 205 128 200 127 195C131 187 129 179 125 175Z"/>
<path class="d" d="M111 131C106 140 101 152 99 166M129 131C134 140 139 152 141 166"/></g>
<g class="org" data-area="heart"><ellipse class="glow" cx="129" cy="172" rx="30" ry="28" fill="url(#bglow)"/>
<path class="s" d="M129 162C126 155 115 155 114.5 164C114 172 122 179 130 186C137 180 145 172 144.5 164C144 156 133 155 129 162Z" transform="rotate(-14 129 170)"/>
<path class="d" d="M124 166c2 4 6 7 9 8" transform="rotate(-14 129 170)"/></g>
<g class="org" data-area="pancreas"><ellipse class="glow" cx="124" cy="229" rx="34" ry="16" fill="url(#bglow)"/>
<path class="s" d="M103 232C108 225 122 226 131 227C138 228 144 225 148 221C152 225 148 233 139 235C128 238 112 241 104 239C100 238 100 235 103 232Z"/></g>
<g class="org" data-area="kidneys"><ellipse class="glow" cx="120" cy="259" rx="44" ry="22" fill="url(#bglow)"/>
<path class="s" d="M98 246C90 246 88 258 90 266C92 274 102 274 104 268C101 264 101 258 104 254C104 249 102 246 98 246Z"/>
<path class="s" d="M142 246C150 246 152 258 150 266C148 274 138 274 136 268C139 264 139 258 136 254C136 249 138 246 142 246Z"/></g>
<g class="org" data-area="joints"><g class="s"><circle cx="180" cy="128" r="5"/><circle cx="60" cy="128" r="5"/><circle cx="193" cy="233" r="4.5"/><circle cx="47" cy="233" r="4.5"/><circle cx="149" cy="300" r="5"/><circle cx="91" cy="300" r="5"/><circle cx="143" cy="412" r="5"/><circle cx="97" cy="412" r="5"/></g></g>
</svg>`;

  // ---- the page
  const $ = (s, el) => (el || document).querySelector(s);
  const home = () => SITE + (rtl() ? "ar/" : "");
  function opts(list, cur, label, withSkip) {
    return (withSkip ? [SKIP] : []).concat(list).map(v => `<option value="${esc(v)}"${v === cur ? " selected" : ""}>${esc(v === SKIP ? t("skip") : (label ? label(v) : ans(v)))}</option>`).join("");
  }
  const sel = (k, lab, list, o) => `<div class="field"><label class="lab" for="f-${k}">${t(lab)}</label><div class="sel${S[k] === SKIP ? " muted" : ""}"><select id="f-${k}" data-k="${k}">${opts(list, S[k], o && o.label, o && o.skip)}</select>${ic("down")}</div></div>`;
  const pressed = (k, v) => (BOOL.has(k) ? (S[k] ? "Yes" : "No") === v : S[k] === v);
  const btns = (k, list, label) => list.map(v => `<button type="button" data-k="${k}" data-v="${esc(v)}" aria-pressed="${pressed(k, v)}">${esc(label ? label(v) : ans(v))}</button>`).join("");
  const seg = (k, lab, list, cls) => `<div class="field${cls ? " " + cls : ""}"><span class="lab" id="l-${k}">${t(lab)}</span><div class="seg${cls === "full" ? " wide" : ""}" role="group" aria-labelledby="l-${k}">${btns(k, list)}</div></div>`;
  const chips = (k, lab, list, label) => `<div class="field full"><span class="lab" id="l-${k}">${t(lab)}</span><div class="chips" role="group" aria-labelledby="l-${k}">${btns(k, list, label)}</div></div>`;
  const yn = (k, lab, hint) => `<div class="yn"><span class="yn-t"><span class="lab" id="l-${k}">${t(lab)}</span>${hint ? `<span class="hint">${t(hint)}</span>` : ""}</span><div class="seg" role="group" aria-labelledby="l-${k}">${btns(k, ["No", "Yes"])}</div></div>`;
  const dayWord = n => {
    if (L !== "ar") return t(n === 1 ? "d_1" : "d_n", n);
    return n === 0 ? t("d_0", n) : n === 1 ? t("d_1") : n === 2 ? t("d_2") : n <= 10 ? t("d_few", n) : t("d_n", n);
  };
  const days = (k, lab) => `<div class="field"><label class="lab" for="f-${k}">${t(lab)}</label><div class="range"><div class="rg"><input type="range" id="f-${k}" data-k="${k}" min="0" max="30" step="1" value="${S[k]}"><div class="scale-ends" aria-hidden="true"><span>0</span><span>30</span></div></div><output id="o-${k}" for="f-${k}">${dayWord(S[k])}</output></div></div>`;
  const diabLabel = v => (v === "Borderline" ? ans("Borderline diabetes") : ans(v));
  const qcard = (id, icon, title, sub, n, body) => `<section class="qcard" aria-labelledby="g-${id}"><div class="qhead"><span class="qico">${ic(icon)}</span><div><h3 id="g-${id}">${t(title)}</h3><p>${t(sub)}</p></div><span class="count">${ic("check")}${t("q_count", n)}</span></div>${body}</section>`;

  function render() {
    const de = document.documentElement;
    de.lang = L; de.dir = rtl() ? "rtl" : "ltr";
    document.title = t("page_title");
    $('meta[name="description"]').setAttribute("content", t("page_desc"));
    const A = M.answers, gb = R.results.gradient_boosting.best_f1;
    const ring = 2 * Math.PI * 7.6, share = ring * R.test_prevalence;
    const kpis = [
      [ic("people"), `6.7<small>${t("k_million")}</small>`, t("k1")],
      [ic("cal"), "15", t("k2")],
      [ic("clip"), `3.1<small>${t("k_million")}</small>`, t("k3")],
      [`<svg class="i" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="7.6" stroke="#c3cfdb" stroke-width="2.6"/><circle cx="12" cy="12" r="7.6" stroke="#d93a3f" stroke-width="2.6" stroke-linecap="butt" stroke-dasharray="${share.toFixed(2)} ${ring.toFixed(2)}" transform="rotate(-90 12 12)"/></svg>`,
        num(R.test_rows), t("k4", pct(R.test_prevalence, 1))],
      [ic("roc"), gb.roc_auc.toFixed(2), t("k5", gb.pr_auc.toFixed(3))],
      [ic("watch"), "22", t("k6")],
    ].map(([i, n, tx]) => `<li class="kpi"><span class="kpi-ico">${i}</span><p class="n">${n}</p><p class="t">${tx}</p></li>`).join("");
    $("#app").innerHTML = `
<a class="skip" href="#main">${t("skip_link")}</a>
${ICONS}
<header class="top"><div class="wrap">
  <a class="mark" href="${home()}" aria-label="${t("home_label")}"><span class="acc">&gt;</span> M<span class="acc">.</span>E<span class="acc cur">_</span></a>
  <nav class="nav" aria-label="${t("nav_label")}">
    <a href="${home()}work/heart">${t("nav_project")}</a>
    <a class="gh" href="${REPO}">${t("nav_code")}</a>
    <button class="lang" type="button" id="lang" lang="${rtl() ? "en" : "ar"}">${t("nav_lang")}</button>
  </nav>
</div></header>
<section class="hero" aria-labelledby="title">
  <div class="ecg" aria-hidden="true"><svg id="ecg" focusable="false"><path class="trace" id="trace"/><circle class="halo" id="halo" r="9" cx="-40" cy="-40"/><circle class="head" id="head" r="3.6" cx="-40" cy="-40"/></svg></div>
  <div class="wrap">
    <div class="hero-text">
      <h1 id="title">${t("title")}</h1>
      <p class="lede">${t("lede")}</p>
      <ul class="facts"><li>${ic("info")}${t("fact_med")}</li><li>${ic("lock")}${t("fact_local")}</li></ul>
    </div>
    <div class="heart-wrap">${HEART}</div>
  </div>
</section>
<main id="main">
  <section class="numbers wrap" aria-labelledby="num-h">
    <div class="sec-h"><h2 id="num-h">${t("num_title")}</h2><p>${t("num_src")}</p></div>
    <ul class="kpis">${kpis}</ul>
  </section>
  <div class="tabs-row wrap"><div class="tabs" role="tablist" aria-label="${t("tabs_label")}">
    ${[["you", "tab_you", "gauge"], ["years", "tab_years", "trend"], ["model", "tab_model", "tree"]].map(([k, l, i]) => `<button class="tab" type="button" role="tab" id="tab-${k}" aria-controls="v-${k}" aria-selected="${tab === k}" tabindex="${tab === k ? 0 : -1}" data-tab="${k}">${ic(i)}${t(l)}</button>`).join("")}
  </div></div>
  <section class="view wrap" id="v-you" role="tabpanel" aria-labelledby="tab-you"${tab === "you" ? "" : " hidden"}>
    <div class="est">
      <form class="form" id="form" aria-labelledby="form-h">
        <div class="form-h"><h2 id="form-h">${t("form_title")}</h2><button class="linkbtn" type="button" id="reset">${t("reset")}</button></div>
        ${qcard("you", "person", "g_you", "g_you_sub", 6, `<div class="fields">
          ${sel("age_group", "f_age", A.age_group)}
          ${seg("sex", "f_sex", ["Female", "Male"], "sexf")}
          <div class="field full"><span class="lab" id="l-hw">${t("f_hw")}</span>
            <div class="hw" role="group" aria-labelledby="l-hw">
              <label class="num"><span class="vh">${t("f_height")}</span><input type="number" id="f-height" data-k="height" min="120" max="220" step="1" value="${S.height}" inputmode="numeric"><span class="u" aria-hidden="true">${t("u_cm")}</span></label>
              <label class="num"><span class="vh">${t("f_weight")}</span><input type="number" id="f-weight" data-k="weight" min="30" max="250" step="1" value="${S.weight}" inputmode="numeric"><span class="u" aria-hidden="true">${t("u_kg")}</span></label>
              <span class="bmi" aria-live="polite">${t("bmi")} <b id="bmi-v"></b><span class="hint">${t("bmi_hint")}</span></span>
            </div></div>
          ${sel("race", "f_race", A.race, {skip: true})}
          ${sel("education", "f_edu", A.education, {skip: true})}
          ${sel("income", "f_income", A.income, {skip: true})}
        </div>`)}
        ${qcard("health", "steth", "g_health", "g_health_sub", 9, `<div class="fields">
          ${chips("high_blood_pressure", "f_bp", ["No", "Yes", "Borderline", "Pregnancy only"])}
          ${chips("high_cholesterol", "f_chol", ["No", "Yes", "Never checked"])}
          ${chips("diabetes", "f_diab", ["No", "Yes", "Borderline", "Pregnancy only"], diabLabel)}
          <p class="sub-lab">${t("other_cond")}</p>
          <div class="yn-list">${yn("stroke", "f_stroke")}${yn("kidney_disease", "f_kidney")}${yn("asthma", "f_asthma")}${yn("arthritis", "f_arth")}${yn("depression", "f_dep")}${yn("copd", "f_copd")}</div>
        </div>`)}
        ${qcard("life", "shoe", "g_life", "g_life_sub", 3, `<div class="yn-list one">${yn("smoked_100", "f_smoked")}${yn("heavy_drinker", "f_heavy", "f_heavy_hint")}${yn("exercise", "f_ex", "f_ex_hint")}</div>`)}
        ${qcard("well", "smile", "g_well", "g_well_sub", 4, `<div class="fields">
          ${seg("gen_health", "f_gen", ["Poor", "Fair", "Good", "Very good", "Excellent"], "full")}
          ${days("phys_days", "f_phys")}${days("ment_days", "f_ment")}
          <div class="yn-list one">${yn("diff_walking", "f_walk")}</div>
        </div>`)}
      </form>
      <div class="aside">
        <section class="card result" id="result" aria-labelledby="res-h">
          <div class="result-h">
            <div class="rh-row"><h2 id="res-h">${t("res_title")}</h2><span class="live"><i></i>${t("live")}</span></div>
            <p id="res-for"></p>
          </div>
          <div class="res-body">
            <div class="res-dial"><div class="gauge-box" id="gauge-box"><svg class="gauge" id="gauge" role="img" focusable="false"></svg></div><ul class="g-legend" id="g-legend" aria-hidden="true"></ul></div>
            <div class="res-text">
              <div class="verdict" id="verdict" aria-live="polite"></div>
              <div class="dots"><div class="dots-grid" id="dots" aria-hidden="true">${"<i></i>".repeat(100)}</div><p id="dots-t"></p></div>
              <div class="screen" id="screen"></div>
            </div>
          </div>
        </section>
        <section class="card bodyc" aria-labelledby="bd-h">
          <div class="card-h"><div><h2 id="bd-h">${t("bd_title")}</h2><p>${t("bd_sub")}</p></div></div>
          <div class="bd-row"><div class="fig">${BODY}</div><div class="bd-side"><span class="pill" id="bd-n"></span><div class="bd-list" id="bd-list"></div></div></div>
        </section>
        <section class="card moves" aria-labelledby="mv-h">
          <h2 id="mv-h">${t("mv_title")}</h2><p class="sub">${t("mv_sub")}</p>
          <div class="mv-key" id="mv-key" aria-hidden="true"><span class="lo"><i></i>${t("mv_lo")}</span><span class="hi"><i></i>${t("mv_hi")}</span></div>
          <ul class="mv" id="moves"></ul>
          <p class="note" id="mv-note"></p>
        </section>
      </div>
    </div>
  </section>
  <section class="view wrap" id="v-years" role="tabpanel" aria-labelledby="tab-years"${tab === "years" ? "" : " hidden"}>${yearsView()}</section>
  <section class="view wrap" id="v-model" role="tabpanel" aria-labelledby="tab-model"${tab === "model" ? "" : " hidden"}>${modelView()}</section>
</main>
<footer class="foot"><div class="wrap"><p>${t("foot")}</p><div class="r"><a href="${REPO}">${t("nav_code")}</a><span>${t("foot_by")}</span></div></div></footer>
<div class="mini" id="mini" aria-hidden="true"><span>${t("sticky")}</span><b id="mini-v"></b></div>`;
    bind();
    update();
    hero();
  }

  // ---- your estimate
  function update() {
    const bmi = bmiOf();
    $("#bmi-v").textContent = isFinite(bmi) ? bmi.toFixed(1) : "-";
    const p = person(), risk = predict(p), peer = M.weighted_rates_by_age_sex_2021_2025[S.age_group][S.sex];
    const all = M.weighted_rate_2021_2025, cut = M.thresholds.gradient_boosting.screening_recall_80, sr = R.results.gradient_boosting.screening_recall_80;
    const top = Math.min(1, Math.max(0.25, risk, peer) * 1.3);
    const ratio = peer ? risk / peer : NaN;
    const cmp = ratio > 1.15 ? t("cmp_times", iso(ratio.toFixed(1))) : ratio >= 0.85 ? t("cmp_same") : t("cmp_below");
    const shown = risk < 0.1 ? pct(risk, 1) : pct(risk, 0), group = t("grp_" + S.sex, age(S.age_group));
    $("#res-for").textContent = t("for_" + S.sex, age(S.age_group));
    $("#verdict").innerHTML = `<p>${t("res_1", `<b>${shown}</b>`)}</p><p>${t("res_2", cmp, group, pct(peer, 1))}</p>`;
    const n = Math.round(100 * risk);
    document.querySelectorAll("#dots i").forEach((d, i) => d.classList.toggle("on", i < n));
    $("#dots-t").innerHTML = n < 1 ? t("dots_lt") : t("dots_n", iso(n));
    const up = risk >= cut;
    $("#screen").className = "screen" + (up ? " up" : "");
    $("#screen").innerHTML = `${ic(up ? "flag" : "check")}<div><h3>${t(up ? "scr_up_t" : "scr_lo_t", pct(cut, 0))}</h3><p>${up ? t("scr_up_p", pct(sr.recall, 0), pct(sr.precision, 0)) : t("scr_lo_p", pct(sr.recall, 0))}</p></div>`;
    $("#mini-v").textContent = shown;
    // (on the dial a line holds words or a number, never both: in Arabic "80 or older" is written 80+ there)
    gauge({risk, peer, all, cut, top, shown, peerLab: t("lab_" + S.sex, L === "ar" && S.age_group === "80 or older" ? iso("80+") : age(S.age_group))});
    moves(p, risk);
    bodyMap();
  }

  // ---- the dial: a half circle from 0 to the scale's top; the estimate fills it. The comparison marks (people of the same
  // age and gender, all US adults) sit outside the arc with their labels; the screening cut-off and the scale numbers sit
  // inside. Every label is measured and placed where it touches no tick, mark or other label; a label that finds no free
  // place sends the outside labels to a list under the dial (always used when the dial is narrow, as on phones).
  const G = {W: 480, CX: 240, CY: 222, R: 150};
  let gaugeArgs = null, gaugeCompact = null;
  function gauge(a) {
    gaugeArgs = a;
    const svg = $("#gauge"), box = $("#gauge-box");
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const {W, CX, CY, R: RR} = G, ar = rtl(), top = a.top;
    const compact = box.clientWidth < 400;
    gaugeCompact = compact;
    const ang = v => Math.PI * (ar ? Math.min(v / top, 1) : 1 - Math.min(v / top, 1));
    const pt = (an, r) => [CX + r * Math.cos(an), CY - r * Math.sin(an)];
    const arc = (v0, v1, r) => { const p0 = pt(ang(v0), r), p1 = pt(ang(v1), r);
      return `M${p0[0].toFixed(2)} ${p0[1].toFixed(2)} A${r} ${r} 0 0 ${ar ? 0 : 1} ${p1[0].toFixed(2)} ${p1[1].toFixed(2)}`; };
    const el = (tag, at, parent) => { const n = document.createElementNS(NS, tag); for (const k in at) n.setAttribute(k, at[k]); (parent || svg).appendChild(n); return n; };
    // side: which edge of the text sits on x ("l" its left edge, "r" its right edge, "m" its middle). Every line of the dial
    // is either words or a number, never both, and is laid out left to right: Safari breaks right-to-left SVG lines that hold
    // numbers, so no browser is left to reorder anything (a line of Arabic words still reads right to left inside itself).
    const AR = /[\u0600-\u06FF]/, mixed = s => AR.test(s) && /\d/.test(s);
    // (one text element per line: Safari spreads one line's Arabic letters over the next line's position otherwise)
    const text = (lines, x, y, side, cls, lh) => {
      const g = el("g", {class: "lbl" + (cls ? " " + cls : "")});
      lines.forEach(([s, c], i) => {
        const te = el("text", {x: x.toFixed(1), y: (y + i * (lh || 17)).toFixed(1), "text-anchor": side === "m" ? "middle" : side === "l" ? "start" : "end", direction: "ltr", class: c}, g);
        te.textContent = AR.test(s) ? s : s.replace(/[\u2066-\u2069]/g, "");
      });
      return g;
    };
    const bb = (e, pad) => { const b = e.getBBox(), q = pad || 0; return {x0: b.x - q, y0: b.y - q, x1: b.x + b.width + q, y1: b.y + b.height + q}; };
    const hit = (p, q) => p.x0 < q.x1 && q.x0 < p.x1 && p.y0 < q.y1 && q.y0 < p.y1;
    const near = b => Math.hypot(Math.max(b.x0 - CX, 0, CX - b.x1), Math.max(b.y0 - CY, 0, CY - b.y1));         // nearest point to the centre
    const far = b => Math.max(...[[b.x0, b.y0], [b.x1, b.y0], [b.x0, b.y1], [b.x1, b.y1]].map(([x, y]) => Math.hypot(x - CX, y - CY)));
    const sideIn = an => { const c = Math.cos(an); return c < -0.35 ? "l" : c > 0.35 ? "r" : "m"; };       // inside: text runs toward the centre
    const segHits = (x0, y0, x1, y1, b) => {                                                                // a leader line through a box
      for (let s = 0; s <= 1.0001; s += 0.05) { const x = x0 + (x1 - x0) * s, y = y0 + (y1 - y0) * s; if (x > b.x0 && x < b.x1 && y > b.y0 && y < b.y1) return true; }
      return false; };

    // the track, the estimate (a fill shorter than the pin is left to the pin: its round end would peek out under it) and the ticks
    el("path", {d: arc(0, top, RR), class: "g-track"});
    if (Math.PI * RR * Math.min(a.risk, top) / top > 36) el("path", {d: arc(0, Math.min(a.risk, top), RR), class: "g-fill"});
    const leadG = el("g", {class: "g-leads"});
    const T = 100 * top, maj = niceTicks(0, T, 4).filter(v => v <= T + 1e-9), step = maj.length > 1 ? maj[1] - maj[0] : 10;
    let mi = step / 10; if (T / mi > 42) mi = step / 5; if (T / mi > 42) mi = step / 2;
    for (let k = 0; k * mi <= T + 1e-9; k++) {
      const v = k * mi, big = Math.abs(v / step - Math.round(v / step)) < 1e-6, an = ang(v / 100), p = pt(an, RR - 20), q = pt(an, RR - (big ? 30 : 25));
      el("line", {x1: p[0].toFixed(2), y1: p[1].toFixed(2), x2: q[0].toFixed(2), y2: q[1].toFixed(2), class: big ? "g-tk big" : "g-tk"});
    }
    // the screening cut-off across the track
    const ac = ang(a.cut), c0 = pt(ac, RR - 14), c1 = pt(ac, RR + 14);
    el("line", {x1: c0[0], y1: c0[1], x2: c1[0], y2: c1[1], class: "g-cutline"});
    // the comparison marks, then the estimate's pin
    const marks = [{v: a.all, cls: "all", lines: [[t("g_all"), "g-lab"], [pct(a.all, 1), "g-val"]], r0: RR + 26},
      {v: a.peer, cls: "peer", lines: [[a.peerLab, "g-lab"], [pct(a.peer, 1), "g-val"]], r0: RR + 26}];
    for (const m of marks) {
      const an = ang(m.v), p = pt(an, RR - 13), q = pt(an, RR + 26);
      el("line", {x1: p[0], y1: p[1], x2: q[0], y2: q[1], class: "g-mk " + m.cls});
      el("circle", {cx: q[0], cy: q[1], r: 3.4, class: "g-dot " + m.cls});
    }
    const ay = ang(a.risk), py = pt(ay, RR);
    el("circle", {cx: py[0], cy: py[1], r: 15, class: "g-pin"});
    el("circle", {cx: py[0], cy: py[1], r: 5.5, class: "g-pin-c"});
    // the big number (digits read left to right in both languages)
    const big = el("text", {x: CX, y: CY - 4, "text-anchor": "middle", direction: "ltr"});
    const digits = (100 * a.risk).toFixed(a.risk < 0.1 ? 1 : 0);
    el("tspan", {class: "g-big"}, big).textContent = digits;
    el("tspan", {class: "g-pct", dx: 2}, big).textContent = "%";
    const measurable = svg.getBoundingClientRect().width > 0;
    // (the box of the digits themselves: a text box also holds the font's line height, far above the digits)
    const bigB = measurable ? big.getBBox() : {x: 0, width: 0};
    const inner = [{x0: bigB.x - 4, x1: bigB.x + bigB.width + 4, y0: CY - 4 - 72 * 0.74 - 5, y1: CY + 2}];
    let bottom = CY + 16, topY = 0;

    // inside: the cut-off label first, then the scale numbers that do not collide with it or the big number
    if (measurable) {
      let placed = false;
      for (const r of [RR - 50, RR - 58, RR - 66, RR - 74]) {
        for (const s of [0, 4, -4, 8, -8, 12, -12]) {
          const an = Math.min(Math.PI - 0.06, Math.max(0.06, ac + s * Math.PI / 180)), p = pt(an, r);
          const te = text([[t("g_cut"), "g-scale"], [pct(a.cut, 0), "g-scale g-cutv"]], p[0], p[1] - 3, sideIn(an), "", ar ? 19 : 15);
          const b = bb(te, 2);
          if (far(b) < RR - 33 && !inner.some(q => hit(b, q)) && b.x0 > 2 && b.x1 < W - 2) { inner.push(b); placed = true; break; }
          te.remove();
        }
        if (placed) break;
      }
      for (const v of maj) {
        const an = ang(v / 100), p = pt(an, RR - 46), te = text([[iso(v + "%"), "g-scale"]], p[0], p[1] + 4, sideIn(an));
        const b = bb(te, 2);
        if (inner.some(q => hit(b, q)) || far(b) > RR - 33) te.remove(); else inner.push(b);
      }
    }

    // outside: You, the people of the same age and gender, all adults. Each label's box is measured once; its possible places
    // (along the arc, pushed outward, or under the arc's end when the marker sits near it) are then plain arithmetic, and the
    // cheapest set where no box or leader line touches another is drawn. Moving You costs most, all adults least.
    let fallback = compact || !measurable;
    if (!fallback) {
      const outs = [{v: a.risk, lines: [[t("you"), "g-you"], [a.shown, "g-val"]], r0: RR + 17, wt: 2.5}].concat(marks.slice().reverse().map((m, i) => Object.assign({wt: i ? 1 : 1.5}, m)));
      const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
      const offArc = l => { for (let s = 0; s <= 1.0001; s += 0.04) if (Math.hypot(l[0] + (l[2] - l[0]) * s - CX, l[1] + (l[3] - l[1]) * s - CY) < RR + 13) return false; return true; };
      // (a name that ends with the age range breaks before it: "Women aged" / "55-59"; any other near its middle)
      const split = s => { const w = s.split(" "); if (w.length < 2) return null; if (/\d/.test(w[w.length - 1])) return [w.slice(0, -1).join(" "), w[w.length - 1]]; let k = 1, d = 1e9;
        for (let i = 1; i < w.length; i++) { const e = Math.abs(w.slice(0, i).join(" ").length - w.slice(i).join(" ").length); if (e < d) { d = e; k = i; } }
        return [w.slice(0, k).join(" "), w.slice(k).join(" ")]; };
      const C = outs.map(m => {
        const list = [], a0 = ang(m.v), d0 = pt(a0, m.r0), sp = split(m.lines[0][0]);
        const forms = [{lines: m.lines, extra: 0}].concat(sp ? [{lines: [[sp[0], m.lines[0][1]], [sp[1], m.lines[0][1]], m.lines[1]], extra: 4}] : [])
          .filter(f => !f.lines.some(l => mixed(l[0])));
        for (const f of forms) {
        const te = text(f.lines, 0, 0, "l", "g-ext"), r = te.getBBox(); te.remove();
        const box = (x, y, side) => { const x0 = side === "l" ? x : side === "r" ? x - r.width : x - r.width / 2;
          return {x0: x0 - 3, x1: x0 + r.width + 3, y0: y + r.y - 3, y1: y + r.y + r.height + 3}; };
        const add = (x, y, side, cost, under, forceLead) => {
          const b = box(x, y, side);
          // (the dial may grow upward a little for a label, at a cost; never sideways)
          if (b.x0 < 1 || b.x1 > W - 1 || b.y0 < -60) return false;
          cost += f.extra + Math.max(0, 2 - b.y0) * 0.3;
          if (under ? b.y0 <= CY + 17 : near(b) <= RR + 30) return false;
          if (inner.some(q => hit(b, q))) return false;
          const lx = clamp(d0[0], b.x0 + 3, b.x1 - 3), ly = clamp(d0[1], b.y0 + 3, b.y1 - 3);
          let l = null;
          if (forceLead || Math.hypot(lx - d0[0], ly - d0[1]) > 14) { l = [d0[0], d0[1], lx, ly]; if (!offArc(l)) return false; }
          list.push({x, y, side, b, lead: l, lines: f.lines, cost: cost * m.wt});
          return true;
        };
        // beside the arc (the box centred on its ray) or above it (the box resting on its ray), at the marker's angle or moved
        // along the arc; at each angle the label goes out from the arc until it is clear, and twice more beyond that
        for (const s of [0, 3, -3, 6, -6, 9, -9, 12, -12, 15, -15, 18, -18, 22, -22, 26, -26, 30, -30, 35, -35, 40, -40, 46, -46]) {
          const an = a0 + s * Math.PI / 180;
          if (an < 0.14 || an > Math.PI - 0.14) continue;
          const c = Math.cos(an), side = c < -0.28 ? "r" : c > 0.28 ? "l" : "m";
          let found = 0;
          for (let dr = 32; dr <= 96 && found < 3; dr += found ? 12 : 3) {
            const p = pt(an, RR + dr), x = side === "r" ? p[0] - 1 : side === "l" ? p[0] + 1 : p[0];
            const y = side === "m" ? p[1] - (r.y + r.height) : p[1] - (r.y + r.height / 2);
            if (add(x, y, side, Math.abs(s) + 0.25 * (dr - 32), false, s !== 0)) found++;
          }
        }
        // under the arc's end, when the marker sits near that end
        const c0 = Math.cos(a0);
        if (Math.abs(c0) > 0.85) for (const k of [0, 1, 2]) add(c0 < 0 ? 4 : W - 4, CY + 40 + k * 40 - r.y, c0 < 0 ? "l" : "r", 45 + 8 * k, true, true);
        }
        list.sort((p, q) => p.cost - q.cost);
        return {m, list};
      });
      const fits = (p, q) => !hit(p.b, q.b) && !(p.lead && segHits(...p.lead, q.b)) && !(q.lead && segHits(...q.lead, p.b));
      let best = null;
      const [L0, L1, L2] = C.map(c => c.list.slice(0, 160));
      for (const p of L0) {
        if (best && p.cost >= best.cost) break;
        for (const q of L1) {
          if (best && p.cost + q.cost >= best.cost) break;
          if (!fits(p, q)) continue;
          for (const r of L2) {
            const c = p.cost + q.cost + r.cost;
            if (best && c >= best.cost) break;
            if (fits(p, r) && fits(q, r)) { best = {cost: c, pick: [p, q, r]}; break; }
          }
        }
      }
      if (!best) fallback = true;
      else best.pick.forEach(p => {
        text(p.lines, p.x, p.y, p.side, "g-ext");
        if (p.lead) el("line", {x1: p.lead[0], y1: p.lead[1], x2: p.lead[2], y2: p.lead[3]}, leadG);
        bottom = Math.max(bottom, p.b.y1); topY = Math.min(topY, p.b.y0);
      });
    }
    const y0 = Math.floor(Math.min(0, topY - 3));
    svg.setAttribute("viewBox", compact ? "70 50 340 196" : `0 ${y0} ${W} ${Math.ceil(Math.max(CY + 22, bottom + 6)) - y0}`);
    svg.setAttribute("aria-label", t("g_aria", a.shown, a.peerLab, pct(a.peer, 1), pct(a.all, 1), pct(a.cut, 0)));
    const lg = $("#g-legend");
    lg.classList.toggle("on", fallback);
    lg.innerHTML = fallback ? `<li><i class="you"></i>${t("you")} <b>${a.shown}</b></li><li><i class="peer"></i>${a.peerLab} <b>${pct(a.peer, 1)}</b></li><li><i class="all"></i>${t("g_all")} <b>${pct(a.all, 1)}</b></li>` : "";
  }

  // ---- what moves the estimate: the same answers with one thing changed (the rules of the Streamlit app)
  function moves(p, risk) {
    const ch = [], gh = ["Poor", "Fair", "Good", "Very good", "Excellent"];
    if (S.smoked_100) ch.push([t("mv_never_smoked"), {smoked_100: "No"}]);
    if (!S.exercise) ch.push([t("mv_exercise"), {exercise: "Yes"}]);
    if (S.high_blood_pressure === "Yes") ch.push([t("mv_no_bp"), {high_blood_pressure: "No"}]);
    if (S.high_cholesterol === "Yes") ch.push([t("mv_no_chol"), {high_cholesterol: "No"}]);
    if (bmiOf() >= 25) ch.push([t("mv_bmi"), {bmi: 24.9}]);
    if (S.gen_health !== "Excellent") { const n = gh[gh.indexOf(S.gen_health) + 1]; ch.push([t("mv_gen", ans(n)), {gen_health: n}]); }
    if (S.heavy_drinker) ch.push([t("mv_not_heavy"), {heavy_drinker: "No"}]);
    if (S.phys_days > 0) ch.push([t("mv_no_phys"), {phys_days: 0}]);
    const el = $("#moves");
    $("#mv-key").hidden = !ch.length;
    if (!ch.length) { el.innerHTML = `<li class="none">${t("mv_none")}</li>`; $("#mv-note").textContent = t("mv_foot"); return; }
    const rows = ch.map(([lab, c]) => { const nw = predict(Object.assign({}, p, c)), pts = Math.round(1000 * (nw - risk)) / 10;
      const s = iso((pts > 0 ? "+" : pts < 0 ? "\u2212" : "") + Math.abs(pts).toFixed(1));
      return {lab, v: pts, s, tip: t("mv_tip", lab, pct(nw, 1), s)}; }).sort((a, b) => a.v - b.v);
    const mx = Math.max(1, ...rows.map(r => Math.abs(r.v)));
    el.innerHTML = rows.map(r => { const w = r.v ? Math.max(2, 100 * Math.abs(r.v) / mx).toFixed(1) : 0, bar = `<i style="inline-size:${w}%"></i>`;
      return `<li class="${r.v < 0 ? "down" : r.v > 0 ? "up" : "same"}" data-tip="${esc(r.tip)}"><div class="mv-row"><span>${esc(r.lab)}</span><b>${r.s}</b></div><div class="track" aria-hidden="true"><span>${r.v < 0 ? bar : ""}</span><span>${r.v > 0 ? bar : ""}</span></div></li>`; }).join("");
    $("#mv-note").textContent = t("mv_foot") + (rows.some(r => r.v > 0) ? " " + t("mv_reverse") : "");
  }

  // ---- the body card: which areas the answers point to
  const AREAS = ["heart", "lungs", "brain", "kidneys", "pancreas", "joints", "legs"];
  function bodyMap() {
    const why = {}; AREAS.forEach(a => { why[a] = []; });
    if (S.high_blood_pressure === "Yes") why.heart.push(t("b_bp"));
    if (S.high_cholesterol === "Yes") why.heart.push(t("b_chol"));
    if (S.smoked_100) why.lungs.push(t("b_smoked"));
    if (S.copd) why.lungs.push(t("b_copd"));
    if (S.asthma) why.lungs.push(t("b_asthma"));
    if (S.kidney_disease) why.kidneys.push(t("b_kidney"));
    if (S.stroke) why.brain.push(t("b_stroke"));
    if (S.diabetes === "Yes") why.pancreas.push(t("b_diab")); else if (S.diabetes === "Borderline") why.pancreas.push(t("b_prediab"));
    if (S.arthritis) why.joints.push(t("b_arth"));
    if (S.diff_walking) why.legs.push(t("b_walk"));
    const lit = AREAS.filter(a => why[a].length), off = AREAS.filter(a => !why[a].length);
    document.querySelectorAll("#bodyfig [data-area]").forEach(g => g.classList.toggle("lit", why[g.dataset.area].length > 0));
    const list = xs => (xs.length < 2 ? xs.join("") : L === "ar" ? xs.join(" و") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1]);
    $("#bd-n").textContent = t("bd_n", iso(lit.length));
    $("#bd-n").classList.toggle("on", lit.length > 0);
    $("#bodyfig").setAttribute("aria-label", lit.length ? t("bd_aria", list(lit.map(a => t("a_" + a)))) : t("bd_aria_none"));
    $("#bd-list").innerHTML = lit.length
      ? `<ul class="lit">${lit.map(a => `<li><span class="dot"></span><span><b>${t("a_" + a)}</b><span>${why[a].join(L === "ar" ? "، " : ", ")}</span></span></li>`).join("")}</ul>`
        + (off.length ? `<p class="off">${t("bd_off", list(off.map(a => L === "ar" ? t("a_" + a) : t("a_" + a).toLowerCase())))}</p>` : "")
      : `<p class="off none">${t("bd_none")}</p>`;
  }

  // ---- the head of the page: an ECG trace runs across once and lights the coronary arteries when it reaches the heart
  let ecgW = 0;
  function ecgPath() {
    const svg = $("#ecg"), trace = $("#trace");
    const sr = svg.getBoundingClientRect(), w = sr.width, h = sr.height, b = h * 0.62, ar = rtl();
    const beat = Math.max(170, Math.min(230, w / 7.5)), k = beat / 200, amp = h / 120;
    const P = (x, y) => `${(ar ? w - x : x).toFixed(1)} ${y.toFixed(1)}`;
    let x = 26 * k, d = `M${P(0, b)} L${P(x, b)}`;
    while (x < w + beat) {
      const x0 = x;
      d += ` L${P(x0 + 34 * k, b)} C${P(x0 + 40 * k, b - 11 * amp)} ${P(x0 + 54 * k, b - 11 * amp)} ${P(x0 + 60 * k, b)} L${P(x0 + 72 * k, b)}`;
      d += ` L${P(x0 + 76 * k, b + 8 * amp)} L${P(x0 + 82 * k, b - 62 * amp)} L${P(x0 + 88 * k, b + 20 * amp)} L${P(x0 + 93 * k, b)} L${P(x0 + 112 * k, b)}`;
      d += ` C${P(x0 + 122 * k, b - 17 * amp)} ${P(x0 + 146 * k, b - 17 * amp)} ${P(x0 + 156 * k, b)} L${P(x0 + 200 * k, b)}`;
      x = x0 + 200 * k;
    }
    trace.setAttribute("d", d);
    ecgW = w;
    return trace.getTotalLength();
  }
  function hero() {
    const trace = $("#trace"), head = $("#head"), halo = $("#halo"), heart = $("#heart"), svg = $("#ecg");
    heart.querySelectorAll(".cor").forEach(p => { const n = p.getTotalLength(); p.style.strokeDasharray = n; p.style.setProperty("--len", n); });
    const len = ecgPath();
    if (played || REDUCE.matches) { heart.classList.remove("pre"); return; }
    played = true;
    trace.style.strokeDasharray = len + " " + len;
    trace.style.strokeDashoffset = len;
    const T = 3200; let t0 = null, lit = false;
    const frame = ts => {
      if (!trace.isConnected) return;
      if (t0 === null) t0 = ts;
      const p = Math.min(1, (ts - t0) / T), at = len * p;            // a steady speed reads like a monitor sweep
      trace.style.strokeDashoffset = len - at;
      const q = trace.getPointAtLength(at);
      head.setAttribute("cx", q.x); head.setAttribute("cy", q.y); halo.setAttribute("cx", q.x); halo.setAttribute("cy", q.y);
      const hb = heart.getBoundingClientRect(), sb = svg.getBoundingClientRect(), x = q.x + sb.left;
      if (!lit && (rtl() ? x < hb.right - hb.width * 0.25 : x > hb.left + hb.width * 0.25)) { lit = true; heart.classList.remove("pre"); heart.classList.add("go"); }
      if (p < 1) requestAnimationFrame(frame);
      else { head.classList.add("gone"); halo.classList.add("gone"); heart.classList.remove("pre"); }
    };
    requestAnimationFrame(frame);
  }

  // ---- 15 years of data
  function yearsView() {
    const years = Object.keys(TR.crude).map(Number);
    const trend = lineChart([
      {name: t("s_crude"), color: "var(--ink)", pts: years.map(y => [y, TR.crude[y]])},
      {name: t("s_adj"), color: "var(--red)", pts: years.map(y => [y, TR.adjusted[y]])}],
      {h: 260, yFmt: v => pct(v, 1), xTicks: [2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025], tipFmt: v => pct(v, 2)});
    const fac = ["obesity", "diabetes", "smoking", "inactive", "bp", "depression"].map(k => {
      const ys = Object.keys(TR.factors_share[k]).map(Number);
      return `<div class="sm"><h3>${t("fac_" + k)}</h3><div class="chart">${lineChart([{name: t("fac_" + k), color: "var(--blue)", pts: ys.map(y => [y, TR.factors_share[k][y]])}],
        {h: 130, w: innerWidth > 980 ? 380 : innerWidth > 640 ? 440 : 900, yFmt: v => pct(v, 0), xTicks: [2011, 2015, 2019, 2023], tipFmt: v => pct(v, 1), dots: 2.4, yTicks: 3})}</div></div>`; }).join("");
    const odds = Object.keys(TR.odds_ratios).map(k => { const v = Object.values(TR.odds_ratios[k]).slice().sort((a, b) => a - b), n = v.length;
      const med = n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
      return {label: t("odd_" + k), v: med, color: "var(--ink)", tip: `${t("odd_" + k)}: ${iso(med.toFixed(2))}`}; }).sort((a, b) => b.v - a.v);
    return `
      <section class="card blk"><div class="blk-h"><span class="qico">${ic("trend")}</span><h2>${t("y_trend")}</h2></div>
        <div class="key"><span><i style="background:var(--ink)"></i>${t("s_crude")}</span><span><i style="background:var(--red)"></i>${t("s_adj")}</span></div>
        <div class="chart">${trend}</div>
        <p class="cap">${t("y_trend_note", pct(TR.adjusted["2011"], 1), pct(TR.adjusted["2025"], 1))}</p></section>
      <section class="card blk"><div class="blk-h"><span class="qico">${ic("people")}</span><h2>${t("y_factors")}</h2></div><div class="mult">${fac}</div><p class="cap">${t("y_factors_note")}</p></section>
      <div class="cols">
        <section class="card blk"><h2>${t("y_odds")}</h2>${hbars(odds, {axis: t("odds_axis"), ref: 1, min: 0})}<p class="cap">${t("y_odds_note")}</p></section>
        <section class="card blk"><h2>${t("y_states")}</h2><div class="chart tiles">${tiles(TR.state_tiles)}</div><p class="cap">${t("y_states_note")}</p></section>
      </div>`;
  }

  // ---- how the model works
  function modelView() {
    const cols = [["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["f1", "F1"], ["precision", t("col_prec")], ["recall", t("col_rec")], ["accuracy", t("col_acc")]];
    const rows = [["gradient_boosting", "mod_gb"], ["logistic_regression", "mod_lr"], ["gb_without_bp_chol", "mod_nobp"], ["baseline_always_no", "mod_no"]].map(([k, l]) => {
      const r = R.results[k].best_f1 || R.results[k];
      return `<tr${k === "gradient_boosting" ? ' class="us"' : ""}><th scope="row">${t(l)}</th>${cols.map(([c, h]) => `<td data-l="${esc(h)}">${r[c].toFixed(3)}</td>`).join("")}</tr>`; }).join("");
    const cal = R.calibration, calS = [["gradient_boosting", "cal_gb", "var(--red)"], ["logistic_regression", "cal_lr", "var(--blue)"]].map(([k, l, c]) =>
      ({name: t(l), color: c, pts: cal[k].predicted.map((p, i) => [p, cal[k].observed[i]])}));
    const imp = Object.entries(R.permutation_importance).slice(0, 12).map(([k, v]) => ({label: TX[L].feat[k] || k, v, color: "var(--ink)", tip: `${TX[L].feat[k] || k}: ${iso(v.toFixed(3))}`}));
    const gb = R.results.gradient_boosting.best_f1, nb = R.results.gb_without_bp_chol.best_f1;
    const lims = TX[L].lim.map((s, i) => `<li>${i === 1 ? fill(s, [gb.pr_auc.toFixed(3), nb.pr_auc.toFixed(3)]) : s}</li>`).join("");
    return `
      <section class="card blk"><div class="blk-h"><span class="qico">${ic("clip")}</span><h2>${t("m_test")}</h2></div>
        <table class="tbl"><thead><tr><th scope="col">${t("col_model")}</th>${cols.map(([, h]) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>
        <p class="cap">${t("m_test_note", num(R.test_rows), pct(R.test_prevalence, 1))}</p></section>
      <div class="cols">
        <section class="card blk"><h2>${t("m_cal")}</h2>
          <div class="key"><span><i style="background:var(--red)"></i>${t("cal_gb")}</span><span><i style="background:var(--blue)"></i>${t("cal_lr")}</span></div>
          <div class="chart">${lineChart(calS, {h: 330, w: 520, x0: 0, x1: 0.6, y0: 0, y1: 0.6, xTicks: [0, .1, .2, .3, .4, .5, .6], yTicks: 6, xFmt: v => pct(v, 0), yFmt: v => pct(v, 0), diag: true, xLabel: t("cal_x"), yLabel: t("cal_y"), tipFmt: v => pct(v, 1), tipX: v => pct(v, 1)})}</div></section>
        <section class="card blk"><h2>${t("m_imp")}</h2>${hbars(imp, {axis: t("imp_axis"), min: 0})}</section>
      </div>
      <section class="card blk"><div class="blk-h"><span class="qico">${ic("info")}</span><h2>${t("m_method")}</h2></div><ul class="lims">${lims}</ul></section>`;
  }

  // ---- charts (SVG; in Arabic they read from the right, as the page does. Their text takes the page's direction, so in Arabic
  // 'end' puts a label's left edge on its point and 'start' its right edge: the same anchors serve both languages)
  // bars as plain HTML (readable at any width; in Arabic the page's direction turns them round): a label, then the bar on its track
  function hbars(rows, o) {
    const vals = rows.map(r => r.v).concat(o.ref !== undefined ? [o.ref] : []);
    let lo = Math.min(o.min !== undefined ? o.min : 0, ...vals), hi = Math.max(0, ...vals);
    const pad = (hi - lo) * 0.06 || 1; if (lo < 0) lo -= pad; hi += pad;
    const ticks = niceTicks(lo, hi, 4), step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
    const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9)), F = v => (o.sign && v > 0 ? "+" : "") + (Math.abs(v) < 1e-9 ? 0 : v).toFixed(dec);
    const P = v => (100 * (v - lo) / (hi - lo)).toFixed(2) + "%";
    const lines = (lo < 0 ? `<b class="hb-z" style="inset-inline-start:${P(0)}"></b>` : "") + (o.ref !== undefined ? `<b class="hb-z" style="inset-inline-start:${P(o.ref)}"></b>` : "");
    const grid = ticks.map(v => `<b class="hb-g" style="inset-inline-start:${P(v)}"></b>`).join("");
    const body = rows.map(r => {
      const a = Math.min(Math.max(0, lo), r.v), b = Math.max(Math.max(0, lo), r.v), w = Math.max(0.4, 100 * (b - a) / (hi - lo));
      return `<div class="hb-r"><span class="hb-l">${esc(r.label)}</span><span class="hb-t">${grid}${lines}<i style="inset-inline-start:${P(a)};width:${w.toFixed(2)}%;background:${r.color}" data-tip="${esc(r.tip || r.label)}"></i></span></div>`;
    }).join("");
    const axis = `<div class="hb-r hb-ax"><span class="hb-l"></span><span class="hb-t">${ticks.map(v => `<em style="inset-inline-start:${P(v)}">${F(v)}</em>`).join("")}</span></div>`;
    return `<div class="hb">${body}${axis}<div class="hb-r hb-cr"><span class="hb-l"></span><span class="hb-cap">${esc(o.axis)}</span></div></div>`;
  }

  function lineChart(series, o) {
    // (about as wide as its card, so the text keeps its size on a phone)
    const W = Math.min(o.w || 900, Math.max(290, innerWidth - 72)), H = o.h || 260, ml = o.yLabel ? 74 : 56, mr = 18, mt = 12, mb = o.xLabel ? 46 : 28, ar = rtl();
    const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1]));
    const xa = o.x0 !== undefined ? o.x0 : Math.min(...xs), xb = o.x1 !== undefined ? o.x1 : Math.max(...xs);
    let ya = o.y0 !== undefined ? o.y0 : Math.min(...ys), yb = o.y1 !== undefined ? o.y1 : Math.max(...ys);
    if (o.y0 === undefined) { const p = (yb - ya) * 0.12 || 0.01; ya -= p; yb += p; }
    const X = v => { const x = ml + (v - xa) / (xb - xa) * (W - ml - mr); return ar ? W - x : x; }, Y = v => mt + (1 - (v - ya) / (yb - ya)) * (H - mt - mb);
    const yt = niceTicks(ya, yb, o.yTicks || 5).filter(v => v >= ya - 1e-9 && v <= yb + 1e-9);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" focusable="false">`;
    s += yt.map(v => `<line class="grid" x1="${X(xa)}" x2="${X(xb)}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${ar ? W - ml + 10 : ml - 10}" y="${Y(v) + 4}" text-anchor="${ar ? "start" : "end"}" direction="ltr">${esc(o.yFmt(v))}</text>`).join("");
    (o.xTicks || []).forEach(v => { s += `<text x="${X(v)}" y="${H - mb + 20}" text-anchor="middle" direction="ltr">${esc(o.xFmt ? o.xFmt(v) : v)}</text>`; });
    if (o.diag) s += `<line class="zero" x1="${X(xa)}" y1="${Y(xa)}" x2="${X(Math.min(xb, yb))}" y2="${Y(Math.min(xb, yb))}"/>`;
    for (const se of series) {
      s += `<polyline fill="none" stroke="${se.color}" stroke-width="2.2" stroke-linejoin="round" points="${se.pts.map(p => X(p[0]) + "," + Y(p[1])).join(" ")}"/>`;
      s += se.pts.map(p => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${o.dots || 3.4}" fill="${se.color}" data-tip="${esc(se.name + ": " + (o.tipX ? o.tipX(p[0]) : p[0]) + ", " + o.tipFmt(p[1]))}"/>`).join("");
    }
    if (o.xLabel) s += `<text x="${(X(xa) + X(xb)) / 2}" y="${H - 4}" text-anchor="middle">${esc(o.xLabel)}</text>`;
    if (o.yLabel) s += `<text transform="translate(${ar ? W - 14 : 14} ${(Y(ya) + Y(yb)) / 2}) rotate(${ar ? 90 : -90})" text-anchor="middle">${esc(o.yLabel)}</text>`;
    return s + "</svg>";
  }

  function tiles(list) {
    // (a map keeps its own left and right, in Arabic too)
    const cell = 46, cols = 1 + Math.max(...list.map(d => d.col)), rows = 1 + Math.max(...list.map(d => d.row));
    const lo = Math.min(...list.map(d => d.rate)), hi = Math.max(...list.map(d => d.rate));
    const mix = f => { const a = [237, 243, 250], b = [184, 44, 52]; return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * f)).join(",")})`; };
    let s = `<svg viewBox="0 0 ${cols * cell} ${rows * cell + 40}" role="img" direction="ltr" focusable="false">`;
    for (const d of list) {
      const f = (d.rate - lo) / (hi - lo), x = d.col * cell, y = d.row * cell;
      s += `<rect x="${x + 2}" y="${y + 2}" width="${cell - 4}" height="${cell - 4}" rx="7" fill="${mix(f)}" data-tip="${d.state}: ${pct(d.rate, 1)}"/>`;
      s += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle" style="fill:${f > 0.5 ? "#ffffff" : "#0f2742"};font-weight:600;pointer-events:none">${d.state}</text>`;
    }
    const kw = 220, ky = rows * cell + 14;
    for (let i = 0; i < 20; i++) s += `<rect x="${i * kw / 20}" y="${ky}" width="${kw / 20 + .5}" height="8" fill="${mix(i / 19)}"/>`;
    s += `<text x="0" y="${ky + 24}">${pct(lo, 1)}</text><text x="${kw}" y="${ky + 24}" text-anchor="end">${pct(hi, 1)}</text>`;
    return s + "</svg>";
  }

  function niceTicks(a, b, n) {
    const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
    return out;
  }

  // ---- what the controls do
  function showTab(k, focus) {
    tab = k;
    document.querySelectorAll("[data-tab]").forEach(x => { const on = x.dataset.tab === k; x.setAttribute("aria-selected", on); x.tabIndex = on ? 0 : -1; if (on && focus) x.focus(); });
    document.querySelectorAll(".view").forEach(v => { v.hidden = v.id !== "v-" + k; });
    if (k === "you") update();          // (the dial measures its labels, which needs it on screen)
    mini();
  }
  function bind() {
    $("#lang").addEventListener("click", () => { L = rtl() ? "en" : "ar"; try { localStorage.setItem("heart.lang", L); } catch (e) { /* none */ } render(); });
    const tabs = [...document.querySelectorAll("[data-tab]")];
    tabs.forEach((b, i) => {
      b.addEventListener("click", () => showTab(b.dataset.tab));
      b.addEventListener("keydown", e => {
        const fwd = rtl() ? "ArrowLeft" : "ArrowRight", back = rtl() ? "ArrowRight" : "ArrowLeft";
        const j = e.key === fwd ? i + 1 : e.key === back ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : null;
        if (j === null) return;
        e.preventDefault(); showTab(tabs[(j + tabs.length) % tabs.length].dataset.tab, true);
      });
    });
    $("#reset").addEventListener("click", () => { Object.assign(S, START); render(); $("#reset").focus(); });
    const form = $("#form");
    form.addEventListener("submit", e => e.preventDefault());
    form.addEventListener("change", e => { const el = e.target, k = el.dataset.k; if (!k || el.tagName !== "SELECT") return;
      S[k] = el.value; el.parentNode.classList.toggle("muted", el.value === SKIP); update(); });
    form.addEventListener("input", e => { const el = e.target, k = el.dataset.k; if (!k) return;
      if (el.type === "range") { S[k] = +el.value; $("#o-" + k).textContent = dayWord(S[k]); update(); }
      if (el.type === "number") { const v = +el.value, lo = +el.min, hi = +el.max; if (el.value !== "" && v >= lo && v <= hi) { S[k] = v; update(); } } });
    form.addEventListener("click", e => { const b = e.target.closest("button[data-v]"); if (!b) return;
      const k = b.dataset.k, v = b.dataset.v;
      S[k] = BOOL.has(k) ? v === "Yes" : v;
      b.parentNode.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); update(); });
    form.addEventListener("focusout", e => { const el = e.target; if (el.type === "number") { el.value = S[el.dataset.k]; } });
    if ("ResizeObserver" in window) {
      if (ro) ro.disconnect();
      ro = new ResizeObserver(() => { const bx = $("#gauge-box"); if (bx && gaugeArgs && (bx.clientWidth < 400) !== gaugeCompact) gauge(gaugeArgs); });
      ro.observe($("#gauge-box"));
    }
    mini();
  }
  let ro = null;

  // the estimate in a line at the foot of the screen, while its card is out of sight on screens where it is not beside the answers
  let io = null;
  function mini() {
    if (io) io.disconnect();
    const m = $("#mini"), card = $("#result");
    if (!("IntersectionObserver" in window) || !card) return;
    io = new IntersectionObserver(es => { m.classList.toggle("on", tab === "you" && innerWidth <= 1080 && !es[0].isIntersecting); });
    io.observe(card);
  }

  // the line charts follow the screen's width: drawn again when it changes much (the answers stay as they are)
  let lastW = innerWidth, rz = 0;
  addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => {
    if ($("#ecg") && Math.abs($("#ecg").getBoundingClientRect().width - ecgW) > 1) { const len = ecgPath(), tr = $("#trace"); tr.style.strokeDasharray = len + " " + len; tr.style.strokeDashoffset = 0; }
    if (Math.abs(innerWidth - lastW) < 60) return; lastW = innerWidth;
    $("#v-years").innerHTML = yearsView(); $("#v-model").innerHTML = modelView(); mini(); }, 200); });
  // the dial measures its words: measured again once the fonts have arrived
  if (document.fonts && document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", () => { if (gaugeArgs) gauge(gaugeArgs); });

  // one tip for every chart mark: under the pointer, or where a finger taps
  const tip = document.createElement("div"); tip.className = "tip"; tip.setAttribute("role", "status"); document.body.appendChild(tip);
  function showTip(e) { const el = e.target.closest && e.target.closest("[data-tip]"); if (!el) { tip.classList.remove("on"); return; }
    tip.textContent = el.getAttribute("data-tip"); tip.classList.add("on");
    const w = tip.offsetWidth, h = tip.offsetHeight, x = Math.min(innerWidth - w - 8, Math.max(8, e.clientX - w / 2)), y = e.clientY - h - 14;
    tip.style.left = x + "px"; tip.style.top = (y < 8 ? e.clientY + 18 : y) + "px"; }
  document.addEventListener("pointermove", showTip);
  document.addEventListener("pointerdown", showTip);
  document.addEventListener("scroll", () => tip.classList.remove("on"), {passive: true});

  render();
})();
