// The browser's model against scikit-learn: every random case in web/build/parity.json must give the same probability.
// usage: node web/parity.cjs   (after python web/build_web.py, which writes web/build/model.json and parity.json)
const fs = require("fs"), path = require("path");
const { makePredictor } = require("./predict.js");
const B = path.join(__dirname, "build");
const M = JSON.parse(fs.readFileSync(path.join(B, "model.json"), "utf8"));
const cases = JSON.parse(fs.readFileSync(path.join(B, "parity.json"), "utf8"));
const predict = makePredictor(M);
let worst = 0, at = null;
for (const c of cases) { const d = Math.abs(predict(c.x) - c.p); if (d > worst) { worst = d; at = c; } }
console.log(`${cases.length} cases, largest difference ${worst.toExponential(2)}${at ? " (scikit-learn " + at.p + ")" : ""}`);
process.exit(worst < 1e-12 ? 0 : 1);
