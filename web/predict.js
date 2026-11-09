// The model in the browser: the same steps as the saved scikit-learn pipeline (web/model_export.py says which), so the
// probability here equals predict_proba's (web/parity.cjs checks it on 20,000 random answers).
// makePredictor(model) -> function(answers) -> probability; answers are the app's words ("Yes", "55-59", ...) or null when skipped.
function makePredictor(M) {
  const order = M.order, trees = M.trees, base = M.base;
  const codes = {};
  for (const f in M.codes) codes[f] = new Map(Object.entries(M.codes[f]));
  function encode(a) {
    return order.map(f => {
      if (codes[f]) { const v = a[f]; return v == null || !codes[f].has(v) ? NaN : codes[f].get(v); }
      return Number(a[f]);
    });
  }
  function raw(x) {
    let s = base;
    for (const t of trees) {
      let n = t.n[0];
      while (n[0] === 0) {
        const v = x[n[1]];
        let left;
        if (Number.isNaN(v)) left = n[3] === 1;
        else if (n[6] === 1) {
          // a category goes left when its bit is set in the node's list of left-going categories
          const w = t.b[n[7]];
          left = v < 0 ? n[3] === 1 : ((w[v >>> 5] >>> (v & 31)) & 1) === 1;
        } else left = v <= n[2];
        n = t.n[left ? n[4] : n[5]];
      }
      s += n[1];
    }
    return s;
  }
  return a => 1 / (1 + Math.exp(-raw(encode(a))));
}
if (typeof module !== "undefined") module.exports = { makePredictor };
