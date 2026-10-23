// Run the browser nonlinear FE engine on [model, options] pairs from a JSON file and print the results.
// Usage: node web/parity_fe_nl.js cases.json > out.json
const fs = require("fs");
const T = require("./fe2d.js");
require("./fe2d_nl.js");
const items = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = items.map(([model, opts]) => {
  try {
    const m = T.FE.parseFE(model), r = T.FENL.runNonlinear(m, opts), last = r.steps[r.steps.length - 1];
    const ex = {};
    for (const [k, v] of Object.entries(last.extra)) ex[k] = Array.from(v, Number);
    return { ok: true, status: r.status, lam_peak: r.lam_peak, monitor_dof: r.monitor_dof, lam: r.steps.map((s) => s.lam), monitor: r.steps.map((s) => s.monitor),
      iterations: r.steps.map((s) => s.iterations), u: Array.from(last.u), gp: Array.from(last.gp_stress), reaction: last.reaction, extra: ex,
      nodal: T.FENL.nodalStress(r, last), crack: Array.from(T.FENL.crackWidths(r, last)), summary: T.FENL.summary(r) };
  } catch (e) {
    return { ok: false, error: e.message, stack: e.stack };
  }
});
process.stdout.write(JSON.stringify(out));
