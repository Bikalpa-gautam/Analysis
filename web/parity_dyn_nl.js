// Run the browser time-history engine on tasks from a JSON file and print the results.
// Usage: node web/parity_dyn_nl.js tasks.json > out.json
const fs = require("fs");
const T = require("./fe2d.js");
require("./fe2d_nl.js");
require("./dynamics_nl.js");
const D = T.DYN;
const tasks = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const A = (x) => (x ? Array.from(x) : x);
const out = tasks.map((tk) => {
  try {
    const rec = tk.record ? D.parseRecord(tk.record) : null;
    if (tk.kind === "record") return { ok: true, a: A(rec.a), match: rec.match ? { T: rec.match.T, target: rec.match.target, achieved: rec.match.achieved } : null,
      sa: A(D.responseSpectrum(rec.a, rec.dt, tk.periods || [0.1, 0.5, 1.0, 2.0]).Sa) };
    if (tk.kind === "bw") { const r = D.boucWenSdof(tk.params, rec); return { ok: true, u: A(r.u), fs: A(r.fs), z: A(r.z), E_in: A(r.E_in), iterations: r.iterations, energy_error: r.energy_error }; }
    if (tk.kind === "fe") {
      const m = T.FE.parseFE(tk.model), r = D.feDynamic(m, rec, tk.options || {});
      return { ok: true, status: r.status, message: r.message, periods: r.periods, rayleigh: r.rayleigh, monitor_dof: r.monitor_dof, u_mon: A(r.u_mon), base_shear: A(r.base_shear),
        iterations: r.iterations, substeps: r.substeps, energy_error: r.energy.error, E_in: A(r.energy.E_in), mass_total: r.mass_total, frames: r.frames.length,
        u_last: A(r.frames[r.frames.length - 1].u) };
    }
    return { ok: false, error: "unknown task" };
  } catch (e) {
    return { ok: false, error: e.message, stack: e.stack };
  }
});
process.stdout.write(JSON.stringify(out));
