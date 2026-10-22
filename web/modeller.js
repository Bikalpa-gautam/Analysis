/*
 * Tragwerk modeller: editing operations on a tragwerk3d/0.1 model (a plain JSON object),
 * in the spirit of STAAD.Pro's geometry and load commands. Every operation is a small
 * record ("edit") that is replayed on the base model, so edits survive regeneration of
 * a parametric building and give undo/redo for free.
 *
 *   applyEdit(model, edit) -> true if applied, false if it no longer applies
 *
 * Operations: addNode, moveNode, delNode, addMember, setMember, remove (member), section,
 * release, support, nodal, load (member UDL), point (member point load), floor (area load
 * to beams by tributary areas), addCase, editCase, delCase, repeat (translational repeat),
 * move, split, merge, batch.
 */
(function (root) {
  "use strict";
  const T = typeof module !== "undefined" && module.exports ? require("./engine3d.js") : root.Tragwerk;
  const TOL = 1e-6;

  const zOf = (n) => n.z ?? 0;
  const nodeMap = (d) => new Map(d.nodes.map((n) => [n.id, n]));
  const ids = (arr) => new Set(arr.map((x) => x.id));
  function uniqueId(taken, base) {
    let id = base;
    while (taken.has(id)) id += "'";
    taken.add(id);
    return id;
  }
  function nextId(taken, prefix) {
    let k = 1;
    while (taken.has(`${prefix}${k}`)) k++;
    return `${prefix}${k}`;
  }
  function findNodeAt(d, x, y, z, tol = 1e-4) {
    return d.nodes.find((n) => Math.abs(n.x - x) < tol && Math.abs(n.y - y) < tol && Math.abs(zOf(n) - z) < tol) || null;
  }
  const samePair = (m, a, b) => (m.i === a && m.j === b) || (m.i === b && m.j === a);
  function dropMembers(d, gone) {
    d.members = d.members.filter((m) => !gone.has(m.id));
    for (const lc of d.load_cases) lc.loads = (lc.loads || []).filter((l) => !(l.member && gone.has(l.member)));
  }
  function dropNodes(d, gone) {
    d.nodes = d.nodes.filter((n) => !gone.has(n.id));
    d.supports = (d.supports || []).filter((s) => !gone.has(s.node));
    for (const lc of d.load_cases) lc.loads = (lc.loads || []).filter((l) => !(l.node && gone.has(l.node)));
    if (d.dynamics && d.dynamics.masses) d.dynamics.masses = d.dynamics.masses.filter((m) => !gone.has(m.node));
  }
  function pruneOrphans(d, candidates) {
    const used = new Set();
    for (const m of d.members) { used.add(m.i); used.add(m.j); }
    const gone = new Set([...candidates].filter((id) => !used.has(id)));
    if (gone.size) dropNodes(d, gone);
  }
  function memberLength(nm, m) {
    const a = nm.get(m.i), b = nm.get(m.j);
    return Math.hypot(b.x - a.x, b.y - a.y, zOf(b) - zOf(a));
  }

  // ---------------------------------------------------------------- floor load
  // panels at level z bounded by beams along X and Y; returns {panels, skipped}
  function floorPanels(d, z, bounds) {
    const nm = nodeMap(d), at = (n) => Math.abs(zOf(n) - z) < TOL;
    const lev = d.nodes.filter(at);
    const inside = (v, lo, hi) => (lo == null || v >= lo - TOL) && (hi == null || v <= hi + TOL);
    const b = bounds || {};
    const uniq = (vals) => { const s = [...vals].sort((p, q) => p - q), o = []; for (const v of s) if (!o.length || v - o[o.length - 1] > TOL) o.push(v); return o; };
    const xs = uniq(lev.filter((n) => inside(n.x, b.x0, b.x1)).map((n) => n.x)), ys = uniq(lev.filter((n) => inside(n.y, b.y0, b.y1)).map((n) => n.y));
    const beamsX = [], beamsY = [];
    for (const m of d.members) {
      const a = nm.get(m.i), c = nm.get(m.j);
      if (!a || !c || !at(a) || !at(c)) continue;
      if (Math.abs(a.y - c.y) < TOL && Math.abs(a.x - c.x) > TOL) beamsX.push({ m, y: a.y, x0: Math.min(a.x, c.x), x1: Math.max(a.x, c.x), xi: a.x, sgn: c.x > a.x ? 1 : -1 });
      else if (Math.abs(a.x - c.x) < TOL && Math.abs(a.y - c.y) > TOL) beamsY.push({ m, x: a.x, y0: Math.min(a.y, c.y), y1: Math.max(a.y, c.y), yi: a.y, sgn: c.y > a.y ? 1 : -1 });
    }
    const coverX = (y, x0, x1) => beamsX.find((bm) => Math.abs(bm.y - y) < TOL && bm.x0 <= x0 + TOL && bm.x1 >= x1 - TOL);
    const coverY = (x, y0, y1) => beamsY.find((bm) => Math.abs(bm.x - x) < TOL && bm.y0 <= y0 + TOL && bm.y1 >= y1 - TOL);
    const panels = [];
    let skipped = 0;
    for (let j = 0; j + 1 < ys.length; j++) for (let i = 0; i + 1 < xs.length; i++) {
      const x0 = xs[i], x1 = xs[i + 1], y0 = ys[j], y1 = ys[j + 1];
      const e = { s: coverX(y0, x0, x1), n: coverX(y1, x0, x1), w: coverY(x0, y0, y1), e: coverY(x1, y0, y1) };
      if (e.s && e.n && e.w && e.e) panels.push({ x0, x1, y0, y1, edges: e });
      else skipped++;
    }
    return { panels, skipped };
  }
  function tributary(L, S, q, mode) {
    if (mode === "none" || q === 0) return [];
    if (mode === "full") return [[0, L, q * S / 2, q * S / 2]];
    const c = Math.min(L, S) / 2, p = q * c, segs = [[0, c, 0, p]];
    if (L - 2 * c > 1e-9) segs.push([c, L - c, p, p]);
    segs.push([L - c, L, p, 0]);
    return segs;
  }
  function applyFloor(d, e) {
    const lc = d.load_cases.find((c) => c.id === e.caseId);
    if (!lc) return false;
    const { panels } = floorPanels(d, e.z, e.bounds);
    if (!panels.length) return false;
    const mx = { two_way: "two_way", one_way_x: "none", one_way_y: "full" }[e.dist || "two_way"];
    const my = { two_way: "two_way", one_way_x: "full", one_way_y: "none" }[e.dist || "two_way"];
    const r6 = (v) => Math.round(v * 1e6) / 1e6;
    const add = (bm, s0, sgn, segs) => {
      for (const [a, b, w1, w2] of segs) {
        let pa = sgn * (s0 + a), pb = sgn * (s0 + b), wa = w1, wb = w2;      // positions from the beam's start node
        if (pa > pb) { [pa, pb] = [pb, pa]; [wa, wb] = [wb, wa]; }
        lc.loads = (lc.loads || []).concat([{ type: "udl", member: bm.m.id, w1: -wa, w2: -wb, a: r6(pa), b: r6(pb), direction: "global_z" }]);
      }
    };
    for (const p of panels) {
      const Lx = p.x1 - p.x0, Ly = p.y1 - p.y0;
      for (const bm of [p.edges.s, p.edges.n]) add(bm, bm.sgn > 0 ? p.x0 - bm.xi : bm.xi - p.x1, 1, tributary(Lx, Ly, e.q, mx).map(([a, b, w1, w2]) => (bm.sgn > 0 ? [a, b, w1, w2] : [Lx - b, Lx - a, w2, w1])));
      for (const bm of [p.edges.w, p.edges.e]) add(bm, bm.sgn > 0 ? p.y0 - bm.yi : bm.yi - p.y1, 1, tributary(Ly, Lx, e.q, my).map(([a, b, w1, w2]) => (bm.sgn > 0 ? [a, b, w1, w2] : [Ly - b, Ly - a, w2, w1])));
    }
    return true;
  }

  // ---------------------------------------------------------------- split one member into n
  function splitMember(d, mid, n) {
    const m = d.members.find((x) => x.id === mid);
    if (!m || !(n >= 2)) return false;
    const nm = nodeMap(d), a = nm.get(m.i), b = nm.get(m.j);
    if (!a || !b) return false;
    const L = memberLength(nm, m), takenN = ids(d.nodes), takenM = ids(d.members);
    const nodeIds = [m.i];
    for (let k = 1; k < n; k++) {
      const t = k / n, x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t, z = zOf(a) + (zOf(b) - zOf(a)) * t;
      const ex = findNodeAt(d, x, y, z);
      if (ex) nodeIds.push(ex.id);
      else { const id = uniqueId(takenN, `${mid}.${k}`); d.nodes.push({ id, x, y, z }); nodeIds.push(id); }
    }
    nodeIds.push(m.j);
    const parts = [];
    const rel = m.releases || {};
    for (let k = 0; k < n; k++) {
      const id = k === 0 ? m.id : uniqueId(takenM, `${mid}.${k + 1}`);
      const part = JSON.parse(JSON.stringify(m));
      part.id = id; part.i = nodeIds[k]; part.j = nodeIds[k + 1];
      part.releases = {};
      if (k === 0 && rel.i !== undefined) part.releases.i = rel.i;
      if (k === n - 1 && rel.j !== undefined) part.releases.j = rel.j;
      parts.push([part, L * k / n, L * (k + 1) / n]);
    }
    d.members = d.members.filter((x) => x !== m).concat(parts.map((p) => p[0]));
    for (const lc of d.load_cases) {
      const out = [];
      for (const l of lc.loads || []) {
        if (l.member !== mid) { out.push(l); continue; }
        if (l.type === "udl") {
          const a0 = l.a ?? 0, b0 = l.b ?? L, w1 = l.w ?? l.w1, w2 = l.w ?? l.w2;
          const w = (x) => w1 + (w2 - w1) * (x - a0) / (b0 - a0);
          for (const [p, x0, x1] of parts) {
            const lo = Math.max(a0, x0), hi = Math.min(b0, x1);
            if (hi - lo > 1e-9) out.push({ type: "udl", member: p.id, w1: w(lo), w2: w(hi), a: lo - x0, b: Math.min(hi - x0, x1 - x0), direction: l.direction || "global_z" });
          }
        } else if (l.type === "point") {
          const pp = parts.find(([, x0, x1]) => l.a >= x0 - 1e-12 && l.a <= x1 + 1e-12) || parts[parts.length - 1];
          out.push(Object.assign({}, l, { member: pp[0].id, a: Math.min(Math.max(l.a - pp[1], 0), pp[2] - pp[1]) }));
        } else out.push(l);
      }
      lc.loads = out;
    }
    return true;
  }

  // ---------------------------------------------------------------- merge coincident nodes
  function mergeNodes(d, tol = 1e-4) {
    const map = new Map(), keep = [];
    for (const n of d.nodes) {
      const k = keep.find((q) => Math.abs(q.x - n.x) < tol && Math.abs(q.y - n.y) < tol && Math.abs(zOf(q) - zOf(n)) < tol);
      if (k) map.set(n.id, k.id); else keep.push(n);
    }
    if (!map.size) return false;
    const to = (id) => map.get(id) || id;
    d.nodes = keep;
    const seenPairs = [], gone = new Set();
    for (const m of d.members) {
      m.i = to(m.i); m.j = to(m.j);
      if (m.i === m.j || seenPairs.some(([a, b]) => (a === m.i && b === m.j) || (a === m.j && b === m.i))) gone.add(m.id);
      else seenPairs.push([m.i, m.j]);
    }
    dropMembers(d, gone);
    const sup = new Map();
    for (const s of d.supports || []) { s.node = to(s.node); if (!sup.has(s.node)) sup.set(s.node, s); }
    d.supports = [...sup.values()];
    for (const lc of d.load_cases) for (const l of lc.loads || []) if (l.node) l.node = to(l.node);
    return true;
  }

  // ---------------------------------------------------------------- translational repeat
  function repeat(d, e) {
    const nm = nodeMap(d), takenN = ids(d.nodes), takenM = ids(d.members);
    const mems = (e.members || []).map((id) => d.members.find((m) => m.id === id)).filter(Boolean);
    const set = new Set((e.nodes || []).filter((id) => nm.has(id)));
    for (const m of mems) { set.add(m.i); set.add(m.j); }
    if (!set.size || !(e.n >= 1)) return false;
    const order = d.nodes.filter((n) => set.has(n.id)).map((n) => n.id);
    const [dx, dy, dz] = e.d;
    const linkSec = e.linkSection && d.sections[e.linkSection] ? e.linkSection : (mems[0] ? mems[0].section : Object.keys(d.sections)[0]);
    let prev = new Map(order.map((id) => [id, id]));
    for (let k = 1; k <= e.n; k++) {
      const cur = new Map();
      for (const id of order) {
        const n0 = nm.get(id), x = n0.x + k * dx, y = n0.y + k * dy, z = zOf(n0) + k * dz;
        const ex = findNodeAt(d, x, y, z);
        if (ex) cur.set(id, ex.id);
        else { const nid = uniqueId(takenN, `${id}+${k}`); d.nodes.push({ id: nid, x, y, z }); cur.set(id, nid); }
        // supports travel with the copy if asked (horizontal repeats of a frame standing on its supports)
        const s0 = e.supports && (d.supports || []).find((q) => q.node === id), to = cur.get(id);
        if (s0 && !(d.supports || []).some((q) => q.node === to)) d.supports.push(Object.assign({}, s0, { node: to }));
      }
      for (const m of mems) {
        const a = cur.get(m.i), b = cur.get(m.j);
        if (a === b || d.members.some((x) => samePair(x, a, b))) continue;
        const c = JSON.parse(JSON.stringify(m));
        c.id = uniqueId(takenM, `${m.id}+${k}`); c.i = a; c.j = b;
        d.members.push(c);
      }
      const supported = new Set((d.supports || []).map((q) => q.node));
      if (e.link) for (const id of order) {
        const a = prev.get(id), b = cur.get(id);
        if (a === b || d.members.some((x) => samePair(x, a, b))) continue;
        if (e.supports && supported.has(a) && supported.has(b)) continue;   // no member between two supports
        d.members.push({ id: uniqueId(takenM, `L${id}+${k}`), i: a, j: b, section: linkSec });
      }
      prev = cur;
    }
    return true;
  }

  // ---------------------------------------------------------------- dispatcher
  const SUPPORTS = {
    fixed: { ux: true, uy: true, uz: true, rx: true, ry: true, rz: true },
    pinned: { ux: true, uy: true, uz: true, rx: false, ry: false, rz: true },
    roller: { ux: false, uy: false, uz: true, rx: false, ry: false, rz: false },
  };
  function supportKind(s) {
    if (!s) return "none";
    for (const [k, v] of Object.entries(SUPPORTS)) if (Object.keys(v).every((q) => !!s[q] === v[q])) return k;
    return "custom";
  }

  function applyEdit(d, e) {
    const nm = () => nodeMap(d);
    const mem = (id) => d.members.find((x) => x.id === id);
    switch (e.op) {
      case "batch": {
        let ok = true;
        for (const x of e.ops) ok = applyEdit(d, x) && ok;
        return ok;
      }
      case "addNode": {
        if (nm().has(e.id) || findNodeAt(d, e.x, e.y, e.z)) return false;
        d.nodes.push({ id: e.id, x: e.x, y: e.y, z: e.z });
        return true;
      }
      case "moveNode": {
        const n = nm().get(e.node);
        if (!n) return false;
        n.x = e.x; n.y = e.y; n.z = e.z;
        return true;
      }
      case "delNode": {
        if (!nm().has(e.node)) return false;
        const touched = new Set(), gone = new Set();
        for (const m of d.members) if (m.i === e.node || m.j === e.node) { gone.add(m.id); touched.add(m.i); touched.add(m.j); }
        dropMembers(d, gone);
        dropNodes(d, new Set([e.node]));
        touched.delete(e.node);
        pruneOrphans(d, touched);
        return true;
      }
      case "addMember": {
        const n = nm();
        if (mem(e.id) || !n.has(e.i) || !n.has(e.j) || e.i === e.j || d.members.some((m) => samePair(m, e.i, e.j))) return false;
        const section = e.section in d.sections ? e.section : Object.keys(d.sections)[0];
        const m = { id: e.id, i: e.i, j: e.j, section };
        if (e.type && e.type !== "auto") m.design = { type: e.type };
        if (e.roll) m.roll_deg = e.roll;
        d.members.push(m);
        return true;
      }
      case "setMember": {
        const m = mem(e.member);
        if (!m) return false;
        const p = e.patch || {};
        if (p.section !== undefined) { if (!(p.section in d.sections)) return false; m.section = p.section; }
        if (p.roll_deg !== undefined) m.roll_deg = Number(p.roll_deg) || 0;
        if (p.type !== undefined) { m.design = Object.assign({}, m.design || {}); if (p.type === "auto") delete m.design.type; else m.design.type = p.type; }
        if (p.l0_factor !== undefined) { m.design = Object.assign({}, m.design || {}, { l0_factor: Number(p.l0_factor) }); delete m.design.l0_factor_y; delete m.design.l0_factor_z; }
        if (p.deflection_system !== undefined) m.design = Object.assign({}, m.design || {}, { deflection_system: p.deflection_system });
        if (p.releases !== undefined) m.releases = p.releases;
        return true;
      }
      case "remove": {
        const m = mem(e.member);
        if (!m) return false;
        dropMembers(d, new Set([m.id]));
        pruneOrphans(d, [m.i, m.j]);
        return true;
      }
      case "section": { const m = mem(e.member); if (!m || !(e.section in d.sections)) return false; m.section = e.section; return true; }
      case "release": { const m = mem(e.member); if (!m) return false; m.releases = { i: true, j: true }; return true; }
      case "support": {
        if (!nm().has(e.node)) return false;
        d.supports = (d.supports || []).filter((s) => s.node !== e.node);
        if (e.kind !== "none") d.supports.push(Object.assign({ node: e.node }, e.kind === "custom" ? e.fix : SUPPORTS[e.kind]));
        return true;
      }
      case "nodal": {
        const lc = d.load_cases.find((c) => c.id === e.caseId);
        if (!lc || !nm().has(e.node)) return false;
        const [fx, fy, fz, mx, my, mz] = e.f;
        lc.loads = (lc.loads || []).concat([{ type: "nodal", node: e.node, fx, fy, fz, mx, my, mz }]);
        return true;
      }
      case "load": {
        const lc = d.load_cases.find((c) => c.id === e.caseId), m = mem(e.member);
        if (!lc || !m) return false;
        lc.loads = (lc.loads || []).concat([{ type: "udl", member: m.id, w: e.w, direction: e.dir }]);
        return true;
      }
      case "point": {
        const lc = d.load_cases.find((c) => c.id === e.caseId), m = mem(e.member);
        if (!lc || !m) return false;
        const L = memberLength(nm(), m);
        lc.loads = (lc.loads || []).concat([{ type: "point", member: m.id, p: e.p, a: Math.min(Math.max(e.a, 0), L), direction: e.dir }]);
        return true;
      }
      case "floor": return applyFloor(d, e);
      case "addCase": {
        if (d.load_cases.some((c) => c.id === e.id)) return false;
        const c = { id: e.id, name: e.name || e.id, type: e.type, loads: [] };
        if (e.type === "variable") c.category = e.category || "B";
        if (e.group) c.group = e.group;
        if (e.self_weight) c.self_weight = true;
        d.load_cases.push(c);
        return true;
      }
      case "editCase": {
        const c = d.load_cases.find((x) => x.id === e.caseId);
        if (!c) return false;
        for (const [k, v] of Object.entries(e.patch || {})) { if (v === null || v === "") delete c[k]; else c[k] = v; }
        if (c.type === "permanent") delete c.category; else if (!c.category) c.category = "B";
        return true;
      }
      case "delCase": {
        const before = d.load_cases.length;
        d.load_cases = d.load_cases.filter((c) => c.id !== e.caseId);
        if (d.dynamics && d.dynamics.mass_cases) delete d.dynamics.mass_cases[e.caseId];
        return d.load_cases.length < before;
      }
      case "repeat": return repeat(d, e);
      case "move": {
        const n = nm(), sel = (e.nodes || []).filter((id) => n.has(id));
        if (!sel.length) return false;
        for (const id of sel) { const q = n.get(id); q.x += e.d[0]; q.y += e.d[1]; q.z = zOf(q) + e.d[2]; }
        return true;
      }
      case "split": return splitMember(d, e.member, e.n);
      case "merge": return mergeNodes(d, e.tol ?? 1e-4);
      default: return false;
    }
  }

  const f2 = (v) => (Math.round(v * 1000) / 1000).toString();
  function editLabel(e) {
    switch (e.op) {
      case "batch": return e.label || `${e.ops.length} changes`;
      case "addNode": return `Node ${e.id} at (${f2(e.x)}, ${f2(e.y)}, ${f2(e.z)})`;
      case "moveNode": return `Node ${e.node} moved to (${f2(e.x)}, ${f2(e.y)}, ${f2(e.z)})`;
      case "delNode": return `Deleted node ${e.node}`;
      case "addMember": return `Member ${e.id}: ${e.i} → ${e.j}`;
      case "setMember": return `${e.member}: ${Object.entries(e.patch || {}).map(([k, v]) => `${k} ${typeof v === "object" ? "set" : v}`).join(", ")}`;
      case "remove": return `Removed ${e.member}`;
      case "section": return `${e.member}: section ${e.section}`;
      case "release": return `${e.member}: pinned at both ends`;
      case "support": return `Support at ${e.node}: ${e.kind}`;
      case "nodal": return `Nodal load at ${e.node} in ${e.caseId}`;
      case "load": return `${e.member}: ${f2(e.w)} kN/m ${e.dir.replace("global_", "along ").replace("local_", "local ")} in ${e.caseId}`;
      case "point": return `${e.member}: ${f2(e.p)} kN at ${f2(e.a)} m in ${e.caseId}`;
      case "floor": return `Floor load ${f2(e.q)} kN/m² at z = ${f2(e.z)} in ${e.caseId}`;
      case "addCase": return `Load case ${e.id} (${e.type})`;
      case "editCase": return `Load case ${e.caseId} changed`;
      case "delCase": return `Deleted load case ${e.caseId}`;
      case "repeat": return `Repeat ${(e.members || []).length} members, ${(e.nodes || []).length} nodes ${e.n}× by (${e.d.map(f2).join(", ")})${e.link ? ", linked" : ""}${e.supports ? ", with supports" : ""}`;
      case "move": return `Moved ${(e.nodes || []).length} nodes by (${e.d.map(f2).join(", ")})`;
      case "split": return `Split ${e.member} into ${e.n}`;
      case "merge": return "Merged coincident nodes";
      default: return e.op;
    }
  }

  const CONCRETE = "C30/37";
  const sec = (b, h, top, bottom, side, links, cover) => {
    const s = { type: "rc_rect", b_mm: b, h_mm: h, concrete: CONCRETE, rebar: "B500B", cover_mm: cover, top: { n: top[0], dia_mm: top[1] }, bottom: { n: bottom[0], dia_mm: bottom[1] },
      links: { dia_mm: links[0], legs: links[1], spacing_mm: links[2] } };
    if (side) s.side = { n: side[0], dia_mm: side[1] };
    return s;
  };
  function blankModel() {
    return { schema: T.SCHEMA3D, name: "New model", design: { design_code: "EC2", combination_rule: "6.10" },
      materials: { [CONCRETE]: { type: "concrete", fck: 30 }, B500B: { type: "rebar", fyk: 500 } },
      sections: { "Column 400×400": sec(400, 400, [3, 20], [3, 20], [1, 20], [8, 2, 240], 35), "Beam 300×500": sec(300, 500, [4, 20], [3, 20], null, [8, 2, 200], 30) },
      nodes: [], supports: [], members: [],
      load_cases: [{ id: "G", name: "Self weight and finishes", type: "permanent", self_weight: true, loads: [] }, { id: "Q", name: "Imposed load", type: "variable", category: "B", loads: [] }] };
  }

  T.Modeller = { applyEdit, editLabel, nextId, uniqueId, findNodeAt, floorPanels, tributary, blankModel, supportKind, SUPPORTS };
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
