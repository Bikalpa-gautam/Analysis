// Run the browser FE engine on tragwerk-fe models from a JSON file and print the results.
// Usage: node web/parity_fe.js models.json > out.json
const fs = require("fs");
const T = require("./fe2d.js");
const items = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = items.map((item) => {
  try {
    const r = T.FE.run(item);
    const det = T.FE.elementDetails(r.model, 0, r.u);
    return { ok: true, nodes: r.model.nodes, elements: r.model.elements, u: r.u, reactions: r.reactions, nodal: r.nodal, elem_nodal: r.elemNodal,
      gp: r.gp, energy: r.energy, f: Array.from(r.f), k0: det.k, m0: det.m, detJ0: det.gauss.map((g) => g.detJ),
      mises: r.field("mises"), s1: r.field("s1"), s2: r.field("s2"), edges: Object.fromEntries(Object.entries(r.model.edges).map(([k, v]) => [k, v.length])) };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});
process.stdout.write(JSON.stringify(out));
