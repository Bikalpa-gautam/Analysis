/* Tragwerk bridge generator: a continuous RC girder road bridge (EN 1991-2 LM1).
 * A port of tragwerk/bridge.py; the test suite checks that both give the same model.
 */
(function (root) {
  "use strict";
  const T = typeof module !== "undefined" && module.exports ? require("./dynamics.js") : root.Tragwerk;
  const CONCRETE_NAMES = { 20: "C20/25", 25: "C25/30", 30: "C30/37", 35: "C35/45", 40: "C40/50", 45: "C45/55", 50: "C50/60", 55: "C55/67", 60: "C60/75", 70: "C70/85", 80: "C80/95", 90: "C90/105" };
  const TS_AXLES = [300, 200, 100], UDL_LANE = [9, 2.5, 2.5], UDL_REST = 2.5;
  const r6 = (v) => Math.round(v * 1e6) / 1e6;
  const sec0 = (b, h, top, bottom, side, links, cover) => {
    const s = { type: "rc_rect", b_mm: b, h_mm: h, concrete: "concrete", rebar: "B500B", cover_mm: cover, top: { n: top[0], dia_mm: top[1] }, bottom: { n: bottom[0], dia_mm: bottom[1] },
      links: { dia_mm: links[0], legs: links[1], spacing_mm: links[2] } };
    if (side) s.side = { n: side[0], dia_mm: side[1] };
    return s;
  };
  const cracked = (s) => Object.assign(s, { torsion_factor: 0.1 });
  const BRIDGE_DEFAULTS = () => ({
    name: "Road bridge, 3 spans, 2 girders", spans: [16.0, 20.0, 16.0], width: 10.0, carriageway: 8.0, girders: 2, pier_heights: [6.0, 6.0],
    pier_columns: "under_girders", divisions: 8, fck: 35.0,
    girder: cracked(sec0(1000, 1600, [10, 40], [10, 40], [4, 20], [14, 4, 200], 50)),
    crossbeam: cracked(sec0(500, 1200, [4, 25], [4, 25], [2, 16], [12, 2, 200], 45)),
    pier: sec0(1000, 1000, [5, 32], [5, 32], [3, 32], [12, 2, 200], 50),
    pier_cap: cracked(sec0(1600, 1800, [18, 40], [8, 32], [4, 25], [16, 4, 150], 50)),
    slab_mm: 250.0, surfacing: 2.4, parapet: 1.5, traffic: true, ts_stations: [0.05, 0.45, 0.95], pier_l0_factor: 1.0, combination_rule: "6.10", modes: 0, seismic: null,
  });

  function check(spec) {
    if (!(spec.spans.length >= 1 && spec.spans.length <= 8) || spec.spans.some((s) => !(Number(s) >= 5 && Number(s) <= 200))) throw new Error("spans: 1 to 8 spans of 5 to 200 m");
    if (!(spec.girders >= 1 && spec.girders <= 6)) throw new Error("girders: 1 to 6");
    if (!(spec.width >= 3 && spec.width <= 40) || !(spec.carriageway >= 2.5 && spec.carriageway <= spec.width)) throw new Error("width 3 to 40 m and a carriageway of 2.5 m up to the deck width");
    if (spec.spans.length > 1 && (!spec.pier_heights.length || spec.pier_heights.some((h) => !(Number(h) >= 1 && Number(h) <= 100)))) throw new Error("pier_heights: 1 to 100 m");
    if (!["under_girders", "single"].includes(spec.pier_columns)) throw new Error("pier_columns: under_girders or single");
    if (!(spec.divisions >= 2 && spec.divisions <= 40) || spec.divisions % 2) throw new Error("divisions: an even number from 2 to 40");
    if (spec.ts_stations.some((f) => !(Number(f) >= 0 && Number(f) <= 1)) || spec.ts_stations.length > 8) throw new Error("ts_stations: up to 8 fractions between 0 and 1");
    if (spec.spans.length * spec.ts_stations.length > 60) throw new Error("too many tandem positions: reduce spans × stations to 60");
  }
  function notionalLanes(w) {
    if (w < 5.4) return [1, 3.0];
    if (w < 6.0) return [2, w / 2];
    return [Math.floor(w / 3.0 + 1e-9), 3.0];
  }
  function lever(y, ys) {
    const n = ys.length, w = new Array(n).fill(0);
    if (n === 1 || y <= ys[0]) { w[0] = 1; return w; }
    if (y >= ys[n - 1]) { w[n - 1] = 1; return w; }
    for (let g = 0; g < n - 1; g++) if (ys[g] <= y && y <= ys[g + 1]) { const t = (y - ys[g]) / (ys[g + 1] - ys[g]); w[g] = 1 - t; w[g + 1] = t; return w; }
    return w;
  }
  function leverStrip(y0, y1, ys) {
    const pts = [...new Set([y0, y1, ...ys.filter((y) => y0 < y && y < y1)])].sort((a, b) => a - b), out = new Array(ys.length).fill(0);
    for (let k = 0; k + 1 < pts.length; k++) {
      const a = pts[k], b = pts[k + 1], wa = lever(a, ys), wb = lever(b, ys);
      for (let g = 0; g < ys.length; g++) out[g] += (b - a) * (wa[g] + wb[g]) / 2;
    }
    return out;
  }

  function generateBridge(specIn) {
    const spec = Object.assign(BRIDGE_DEFAULTS(), JSON.parse(JSON.stringify(specIn || {})));
    check(spec);
    const nsp = spec.spans.length, ng = Math.trunc(spec.girders), nd = Math.trunc(spec.divisions);
    const xsSup = [0];
    for (const s of spec.spans) xsSup.push(r6(xsSup[xsSup.length - 1] + Number(s)));
    const heights = nsp > 1 ? [...Array(nsp - 1).keys()].map((k) => Number(spec.pier_heights[Math.min(k, spec.pier_heights.length - 1)])) : [];
    const zd = heights.length ? Math.max(...heights) : 0;
    const sg = spec.width / ng, ys = [...Array(ng).keys()].map((g) => r6(-spec.width / 2 + sg / 2 + g * sg));
    const st = [];
    for (let s = 0; s < nsp; s++) for (let e = 0; e < nd + (s === nsp - 1 ? 1 : 0); e++) st.push(r6(xsSup[s] + Number(spec.spans[s]) * e / nd));
    const supIdx = [...Array(nsp + 1).keys()].map((s) => s * nd), midIdx = [...Array(nsp).keys()].map((s) => s * nd + nd / 2);
    const gname = (g, i) => `G${g + 1}-${i}`;
    const supNames = ["A0"].concat([...Array(nsp - 1).keys()].map((k) => `P${k + 1}`), [`A${nsp}`]);
    const nodes = [], members = [], supports = [];
    for (let g = 0; g < ng; g++) st.forEach((x, i) => nodes.push({ id: gname(g, i), x, y: ys[g], z: zd }));
    const system = (s) => (nsp === 1 ? "simply_supported" : s === 0 || s === nsp - 1 ? "end_span" : "interior_span");
    for (let g = 0; g < ng; g++) for (let s = 0; s < nsp; s++) for (let e = 0; e < nd; e++) {
      const i = s * nd + e;
      members.push({ id: `G${g + 1}-S${s + 1}-${e + 1}`, i: gname(g, i), j: gname(g, i + 1), section: "Girder", design: { type: "beam", deflection_system: system(s) } });
    }
    const pierTop = new Map();
    for (let k = 1; k < nsp; k++) {
      const i = supIdx[k], zb = r6(zd - heights[k - 1]);
      if (spec.pier_columns === "under_girders" && ng > 1) {
        for (let g = 0; g < ng; g++) {
          const bot = `P${k}-${g + 1}-B`;
          nodes.push({ id: bot, x: st[i], y: ys[g], z: zb });
          members.push({ id: `P${k}-${g + 1}`, i: bot, j: gname(g, i), section: "Pier", design: { type: "column", l0_factor: spec.pier_l0_factor } });
          supports.push({ node: bot, ux: true, uy: true, uz: true, rx: true, ry: true, rz: true });
        }
        continue;
      }
      const mid = [...Array(ng).keys()].filter((g) => Math.abs(ys[g]) < 1e-9);
      let top;
      if (mid.length) top = gname(mid[0], i);
      else { top = `P${k}-T`; nodes.push({ id: top, x: st[i], y: 0, z: zd }); }
      const bot = `P${k}-B`;
      nodes.push({ id: bot, x: st[i], y: 0, z: zb });
      pierTop.set(i, top);
      members.push({ id: `P${k}`, i: bot, j: top, section: "Pier", design: { type: "column", l0_factor: spec.pier_l0_factor } });
      supports.push({ node: bot, ux: true, uy: true, uz: true, rx: true, ry: true, rz: true });
    }
    if (ng > 1 || pierTop.size) {
      for (const i of [...new Set(supIdx.concat(midIdx))].sort((a, b) => a - b)) {
        const row = [...Array(ng).keys()].map((g) => [ys[g], gname(g, i)]);
        if (pierTop.has(i) && pierTop.get(i).endsWith("-T")) row.push([0, pierTop.get(i)]);
        row.sort((p, q) => p[0] - q[0] || (p[1] < q[1] ? -1 : p[1] > q[1] ? 1 : 0));
        const label = supIdx.includes(i) ? supNames[supIdx.indexOf(i)] : `M${midIdx.indexOf(i) + 1}`;
        const sec = pierTop.has(i) ? "Pier cap" : "Crossbeam";
        for (let q = 0; q + 1 < row.length; q++) members.push({ id: `X-${label}-${q + 1}`, i: row[q][1], j: row[q + 1][1], section: sec, design: { type: "beam", deflection_system: "simply_supported" } });
      }
    }
    for (const [end, i] of [["A0", 0], ["An", st.length - 1]]) for (let g = 0; g < ng; g++) supports.push({ node: gname(g, i), ux: end === "A0", uy: g === 0, uz: true, rx: ng === 1, ry: false, rz: false });

    const girderMembers = (g) => { const out = []; for (let s = 0; s < nsp; s++) for (let e = 0; e < nd; e++) out.push(`G${g + 1}-S${s + 1}-${e + 1}`); return out; };
    const spanMembers = (g, s) => [...Array(nd).keys()].map((e) => `G${g + 1}-S${s + 1}-${e + 1}`);
    const lineLoads = (per, mids) => {
      const out = [];
      per.forEach((w, g) => { if (Math.abs(w) > 1e-12) for (const m of mids(g)) out.push({ type: "udl", member: m, w: r6(-w), direction: "global_z" }); });
      return out;
    };
    const slab = leverStrip(-spec.width / 2, spec.width / 2, ys);
    const g1 = lineLoads(slab.map((v) => 25.0 * spec.slab_mm / 1000.0 * v), girderMembers);
    const c = spec.carriageway, surf = leverStrip(-c / 2, c / 2, ys), parL = lever(-spec.width / 2, ys), parR = lever(spec.width / 2, ys);
    const g2 = lineLoads([...Array(ng).keys()].map((g) => spec.surfacing * surf[g] + spec.parapet * (parL[g] + parR[g])), girderMembers);
    const cases = [{ id: "G1", name: "Self weight of the structure and deck slab", type: "permanent", self_weight: true, loads: g1 },
      { id: "G2", name: "Surfacing and parapets", type: "permanent", loads: g2 }];
    if (spec.traffic) {
      const [nl, wl] = notionalLanes(c), y0 = -c / 2;
      const lanes = [...Array(nl).keys()].map((k) => [y0 + k * wl, y0 + (k + 1) * wl]);
      let udl = new Array(ng).fill(0);
      lanes.forEach(([a, b], k) => { const q = k < 3 ? UDL_LANE[k] : UDL_REST, sh = leverStrip(a, b, ys); udl = udl.map((u, g) => u + q * sh[g]); });
      if (y0 + nl * wl < c / 2 - 1e-9) { const sh = leverStrip(y0 + nl * wl, c / 2, ys); udl = udl.map((u, g) => u + UDL_REST * sh[g]); }
      for (let s = 0; s < nsp; s++) cases.push({ id: `UDL${s + 1}`, name: `LM1 UDL on span ${s + 1}`, type: "variable", category: "traffic_UDL", loads: lineLoads(udl, (g) => spanMembers(g, s)) });
      let axle = new Array(ng).fill(0);
      lanes.slice(0, 3).forEach(([a, b], k) => {
        const yc = (a + b) / 2;
        for (const yw of [yc - 1.0, yc + 1.0]) { const sh = lever(yw, ys); axle = axle.map((p, g) => p + TS_AXLES[k] / 2 * sh[g]); }
      });
      const Ltot = xsSup[xsSup.length - 1];
      for (let s = 0; s < nsp; s++) spec.ts_stations.forEach((f, q) => {
        const x0 = xsSup[s] + Number(spec.spans[s]) * Number(f);
        const xa = x0 + 1.2 <= Ltot + 1e-9 ? [x0, x0 + 1.2] : [x0 - 1.2, x0];
        const loads = [];
        for (let g = 0; g < ng; g++) {
          if (Math.abs(axle[g]) < 1e-12) continue;
          for (let x of xa) {
            x = Math.min(Math.max(x, 0), Ltot);
            let kmax = 0;
            for (let k = 0; k < st.length - 1; k++) if (st[k] <= x + 1e-9) kmax = k;
            const i = Math.max(0, Math.min(st.length - 2, kmax));
            let si = Math.floor(i / nd), e = i % nd;
            if (si >= nsp) { si = nsp - 1; e = nd - 1; }
            loads.push({ type: "point", member: `G${g + 1}-S${si + 1}-${e + 1}`, p: r6(-axle[g]), a: r6(x - st[si * nd + e]), direction: "global_z" });
          }
        }
        const pct = Math.floor(100 * Number(f) + 0.5);
        cases.push({ id: `TS${s + 1}.${q + 1}`, name: `LM1 tandem, span ${s + 1} at ${pct} %`, type: "variable", category: "traffic_TS", group: "TS", loads });
      });
    }
    const cname = CONCRETE_NAMES[Math.trunc(spec.fck)] || `C${spec.fck}`;
    const sections = { Girder: JSON.parse(JSON.stringify(spec.girder)), Crossbeam: JSON.parse(JSON.stringify(spec.crossbeam)), Pier: JSON.parse(JSON.stringify(spec.pier)) };
    if (pierTop.size && ng > 1) sections["Pier cap"] = JSON.parse(JSON.stringify(spec.pier_cap));
    for (const s of Object.values(sections)) s.concrete = cname;
    const zs = [...new Set(heights.map((h) => r6(zd - h)).concat([zd]))].sort((a, b) => a - b);
    const out = { schema: T.SCHEMA3D, name: spec.name, design: { design_code: "EC2", combination_rule: spec.combination_rule },
      materials: { [cname]: { type: "concrete", fck: spec.fck }, B500B: { type: "rebar", fyk: 500 } }, sections, nodes, supports, members, load_cases: cases };
    if (spec.modes || spec.seismic) out.dynamics = { modes: Math.trunc(spec.modes || 12) };
    if (spec.seismic) out.seismic = JSON.parse(JSON.stringify(spec.seismic));
    out.meta = { generator: "tragwerk.bridge", spec, grid: { x: Object.fromEntries(supNames.map((n, k) => [n, xsSup[k]])), y: Object.fromEntries(ys.map((y, g) => [String(g + 1), y])), levels: zs },
      slabs: [...Array(nsp).keys()].map((s) => ({ level: 1, x0: xsSup[s], x1: xsSup[s + 1], y0: -spec.width / 2, y1: spec.width / 2, z: zd, t_mm: spec.slab_mm })), deck_z: zd };
    return out;
  }
  Object.assign(T, { generateBridge, BRIDGE_DEFAULTS, bridgeLever: lever, bridgeLeverStrip: leverStrip, notionalLanes });
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
