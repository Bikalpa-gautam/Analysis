// Node checks of the modelling operations (run by tests/test_modeller.py).
const assert = require("assert");
const path = require("path");
const T = require(path.join(__dirname, "..", "..", "web", "modeller.js"));
const M = T.Modeller;
const clone = (o) => JSON.parse(JSON.stringify(o));
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)), `${msg}: ${a} vs ${b}`);

function solve(d, cid) {
  const m = T.parseModel3d(d), fa = new T.FrameAnalysis3D(m);
  return { m, fa, r: fa.solveCase(m.load_cases[cid]) };
}

// 1. floor load = generator's slab loads: finishes only (facade 0), same tributary rule, both orientations
for (const dist of ["two_way", "one_way_x", "one_way_y"]) {
  const g = T.generateBuilding({ spans_x: [6, 4.5], spans_y: [5, 7], storeys: [3.5, 3], facade: 0, distribution: dist, wind: 0 });
  const d = clone(g);
  M.applyEdit(d, { op: "addCase", id: "F", type: "permanent", name: "floor" });
  for (const z of g.meta.grid.levels.slice(1)) assert.ok(M.applyEdit(d, { op: "floor", z, q: g.meta.spec.finishes, caseId: "F", dist }));
  // reverse some beams so that member direction does not matter
  const flipped = clone(d);
  for (const m of flipped.members.filter((_, k) => k % 3 === 0)) {
    if (m.design && m.design.type === "column") continue;
    [m.i, m.j] = [m.j, m.i];
  }
  const a = solve(d, "G2"), b = solve(d, "F"), c = solve(flipped, "F");
  for (const nid of Object.keys(a.m.supports)) for (let q = 0; q < 6; q++) {
    close(b.r.reactions[nid][q], a.r.reactions[nid][q], 1e-9, `floor load reaction ${dist} ${nid} ${q}`);
    close(c.r.reactions[nid][q], a.r.reactions[nid][q], 1e-9, `floor load (flipped beams) ${dist} ${nid} ${q}`);
  }
}

// 2. split: exact elements, so nodal results do not change; loads follow the pieces
{
  const g = T.generateBuilding({ spans_x: [6], spans_y: [5], storeys: [3] });
  const d = clone(g);
  M.applyEdit(d, { op: "point", member: "BX-A1-L1", p: -40, a: 2.2, dir: "global_z", caseId: "Q" });
  const ref = clone(d);
  assert.ok(M.applyEdit(d, { op: "split", member: "BX-A1-L1", n: 3 }));
  assert.ok(M.applyEdit(d, { op: "split", member: "C-A1-L1", n: 2 }));
  for (const cid of ["G1", "G2", "Q"]) {
    const a = solve(ref, cid), b = solve(d, cid);
    for (const nid of Object.keys(a.m.nodes)) for (let q = 0; q < 6; q++)
      close(b.r.displacements[b.fa.nidx[nid]][q], a.r.displacements[a.fa.nidx[nid]][q], 1e-7, `split displacement ${cid} ${nid}`);
  }
  assert.strictEqual(d.members.length, ref.members.length + 3);
}

// 3. translational repeat builds the same frame as the generator
{
  const one = T.generateBuilding({ spans_x: [6, 6], spans_y: [5], storeys: [3.2], wind: 0, snow: 0 });
  const four = T.generateBuilding({ spans_x: [6, 6], spans_y: [5], storeys: [3.2, 3.2, 3.2, 3.2], wind: 0, snow: 0 });
  const d = clone(one);
  const floor = d.members.filter((m) => !m.id.startsWith("C-")).map((m) => m.id);
  const top = d.nodes.filter((n) => n.z > 0).map((n) => n.id);
  assert.ok(M.applyEdit(d, { op: "repeat", members: floor, nodes: top, d: [0, 0, 3.2], n: 3, link: true, linkSection: "Column" }));
  assert.strictEqual(d.nodes.length, four.nodes.length);
  assert.strictEqual(d.members.length, four.members.length);
  // no duplicate member between the same two nodes
  const pairs = new Set(d.members.map((m) => [m.i, m.j].sort().join("|")));
  assert.strictEqual(pairs.size, d.members.length);
  // repeating onto existing nodes reuses them
  const again = clone(d);
  M.applyEdit(again, { op: "repeat", members: floor, nodes: top, d: [0, 0, 3.2], n: 1, link: true, linkSection: "Column" });
  assert.strictEqual(again.nodes.length, d.nodes.length);
}

// 4. merge coincident nodes, delete node, supports, nodal loads, load cases
{
  const d = M.blankModel();
  const ops = [
    { op: "addNode", id: "N1", x: 0, y: 0, z: 0 }, { op: "addNode", id: "N2", x: 0, y: 0, z: 3 }, { op: "addNode", id: "N3", x: 5, y: 0, z: 3 },
    { op: "addNode", id: "N4", x: 5, y: 0, z: 0 }, { op: "addMember", id: "M1", i: "N1", j: "N2", section: "Column 400×400" },
    { op: "addMember", id: "M2", i: "N2", j: "N3", section: "Beam 300×500" }, { op: "addMember", id: "M3", i: "N3", j: "N4", section: "Column 400×400" },
    { op: "support", node: "N1", kind: "fixed" }, { op: "support", node: "N4", kind: "fixed" },
    { op: "load", member: "M2", w: -20, dir: "global_z", caseId: "Q" }, { op: "nodal", node: "N2", caseId: "Q", f: [10, 0, 0, 0, 0, 0] },
  ];
  for (const e of ops) assert.ok(M.applyEdit(d, e), JSON.stringify(e));
  assert.ok(!M.applyEdit(d, { op: "addMember", id: "M9", i: "N2", j: "N1", section: "x" }), "duplicate member refused");
  assert.ok(!M.applyEdit(d, { op: "addNode", id: "N9", x: 5, y: 0, z: 3 }), "coincident node refused");
  const r = T.runProject3d(d);
  assert.ok(r.summary.status === "pass" || r.summary.status === "fail");
  // portal frame horizontal equilibrium
  const R = r.cases.Q.reactions;
  close(R.N1[0] + R.N4[0], -10, 1e-9, "horizontal reaction");
  close(R.N1[2] + R.N4[2], 100, 1e-9, "vertical reaction");
  // duplicate node then merge
  d.nodes.push({ id: "D", x: 5, y: 0, z: 3 });
  d.members.push({ id: "M4", i: "N2", j: "D", section: "Beam 300×500" });
  assert.ok(M.applyEdit(d, { op: "merge" }));
  assert.ok(!d.nodes.some((n) => n.id === "D") && !d.members.some((m) => m.id === "M4"));
  // delete node removes its members, and orphaned nodes with them
  const e = clone(d);
  assert.ok(M.applyEdit(e, { op: "delNode", node: "N3" }));
  assert.ok(!e.members.some((m) => m.i === "N3" || m.j === "N3") && !e.nodes.some((n) => n.id === "N4"));
  // load cases
  assert.ok(M.applyEdit(d, { op: "addCase", id: "W", type: "variable", category: "wind", group: "wind", name: "Wind" }));
  assert.ok(M.applyEdit(d, { op: "editCase", caseId: "W", patch: { name: "Wind +X" } }));
  assert.ok(M.applyEdit(d, { op: "delCase", caseId: "W" }));
  assert.strictEqual(M.supportKind(d.supports[0]), "fixed");
}
console.log("modeller ok");
