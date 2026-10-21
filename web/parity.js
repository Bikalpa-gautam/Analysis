// Run the browser engine on models from a JSON file and print condensed results.
// Usage: node web/parity.js models.json > out.json
const fs = require("fs");
const T = require("./engine.js");
const models = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const fin = (v) => (v == null || !isFinite(v) ? null : v);
const out = models.map((data) => {
  try {
    const r = T.runProject(data);
    return {
      ok: true,
      summary: { status: r.summary.status, max_utilisation: fin(r.summary.max_utilisation) },
      combos: r.combos.combos.map((c) => ({ id: c.id, kind: c.kind, factors: c.factors, label: c.label })),
      cases: Object.fromEntries(Object.entries(r.cases).map(([cid, cr]) => [cid, {
        displacements: cr.displacements, reactions: cr.reactions,
        members: Object.fromEntries(Object.entries(cr.members).map(([mid, m]) => [mid, { x: m.x, N: m.N, V: m.V, M: m.M, v: m.v, v_rel: m.v_rel }])),
      }])),
      envelopes: Object.fromEntries(Object.entries(r.envelopes).map(([mid, e]) => [mid, { M_max: e.max.M, M_min: e.min.M, V_max: e.max.V, N_min: e.min.N }])),
      checks: Object.fromEntries(Object.entries(r.checks).map(([mid, lst]) => [mid, lst.map((c) => ({
        id: c.id, status: c.status, utilisation: fin(c.utilisation), demand: fin(c.demand), capacity: fin(c.capacity), x: c.x, combination: c.combination,
      }))])),
    };
  } catch (e) {
    return { ok: false, error: `${e.constructor.name}: ${e.message}` };
  }
});
process.stdout.write(JSON.stringify(out));
