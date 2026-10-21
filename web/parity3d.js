// Run the browser 3D engine on models (or building specs) from a JSON file and print condensed results.
// Usage: node web/parity3d.js models.json > out.json
// Each entry is either a tragwerk3d model or {"building_spec": {...}} for the generator.
const fs = require("fs");
const T = require("./engine3d.js");
const items = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const fin = (v) => (v == null || !isFinite(v) ? null : v);
const QS = ["N", "Vy", "Vz", "T", "My", "Mz"];
const out = items.map((item) => {
  try {
    if (item.building_spec !== undefined) return { ok: true, model: T.generateBuilding(item.building_spec) };
    const r = T.runProject3d(item);
    return {
      ok: true,
      summary: { status: r.summary.status, max_utilisation: fin(r.summary.max_utilisation) },
      combos: r.combos.combos.map((c) => ({ id: c.id, kind: c.kind, factors: c.factors, label: c.label })),
      cases: Object.fromEntries(Object.entries(r.cases).map(([cid, cr]) => [cid, {
        displacements: cr.displacements, reactions: cr.reactions, equilibrium: cr.equilibrium,
        members: Object.fromEntries(Object.entries(cr.members).map(([mid, m]) => [mid,
          Object.assign({ x: m.x, u: m.u, v: m.v, w: m.w, v_rel: m.v_rel, w_rel: m.w_rel }, Object.fromEntries(QS.map((q) => [q, m[q]])))])),
      }])),
      envelopes: Object.fromEntries(Object.entries(r.envelopes).map(([mid, e]) => [mid,
        Object.fromEntries(QS.flatMap((q) => [[q + "_max", e.max[q]], [q + "_min", e.min[q]]]))])),
      reactions: Object.fromEntries(Object.entries(r.reactions).map(([nid, rr]) => [nid, { uls_max: rr.uls_max, uls_min: rr.uls_min }])),
      checks: Object.fromEntries(Object.entries(r.checks).map(([mid, lst]) => [mid, lst.map((c) => ({
        id: c.id, status: c.status, utilisation: fin(c.utilisation), demand: fin(c.demand), capacity: fin(c.capacity), x: c.x, combination: c.combination,
        steps: c.steps.map((s) => [s.symbol, fin(s.value)]),
      }))])),
    };
  } catch (e) {
    return { ok: false, error: `${e.constructor.name}: ${e.message}` };
  }
});
process.stdout.write(JSON.stringify(out));
