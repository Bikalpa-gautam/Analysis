/*
 * Tragwerk engine for the browser: a line-by-line port of the Python core
 * (tragwerk.solver, tragwerk.combinations, tragwerk.codes.ec2, tragwerk.project).
 * The test suite runs both on the same models and requires agreement to 1e-9
 * (web/parity.js, tests/test_js_parity.py). Change both or neither.
 */
(function (root) {
  "use strict";
  const VERSION = "0.1.0";

  // ---------------------------------------------------------------- model
  const LOAD_DIRECTIONS = ["global_x", "global_y", "local_x", "local_y"];
  const VARIABLE_CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "snow", "snow_high", "wind", "temperature"];
  const DESIGN_TYPES = ["beam", "column", "none"];
  const DEFLECTION_SYSTEMS = ["simply_supported", "end_span", "interior_span", "cantilever"];

  class ModelError extends Error {}
  class StructureUnstable extends Error {}

  function req(d, key, where) {
    if (d == null || !(key in d)) throw new ModelError(`${where}: missing required field '${key}'`);
    return d[key];
  }
  function num(v, where, opt = {}) {
    if (typeof v !== "number" || !isFinite(v)) throw new ModelError(`${where}: expected a number, got ${JSON.stringify(v)}`);
    if (opt.positive && v <= 0) throw new ModelError(`${where}: must be > 0, got ${v}`);
    if (opt.nonneg && v < 0) throw new ModelError(`${where}: must be >= 0, got ${v}`);
    return v;
  }
  const barArea = (n, dia) => n * Math.PI * dia * dia / 4;

  function layer(d, where) {
    if (d == null) return null;
    const n = d.n == null ? 0 : d.n;
    if (!Number.isInteger(n) || n < 0) throw new ModelError(`${where}.n: expected a whole number of bars >= 0`);
    return { n, dia_mm: num(req(d, "dia_mm", where), where + ".dia_mm", { positive: true }),
      depth_mm: d.depth_mm == null ? null : num(d.depth_mm, where + ".depth_mm", { positive: true }) };
  }

  function sectionLayers(sec) {
    const linkDia = sec.links ? sec.links.dia_mm : 0;
    const depth = (lay, face) => {
      if (lay.depth_mm != null) return lay.depth_mm;
      const off = sec.cover_mm + linkDia + lay.dia_mm / 2;
      return face === "top" ? off : sec.h_mm - off;
    };
    const out = [];
    if (sec.top && sec.top.n > 0) out.push([barArea(sec.top.n, sec.top.dia_mm), depth(sec.top, "top"), "top"]);
    if (sec.bottom && sec.bottom.n > 0) out.push([barArea(sec.bottom.n, sec.bottom.dia_mm), depth(sec.bottom, "bottom"), "bottom"]);
    sec.extra_layers.forEach((lay, k) => {
      if (lay.n > 0) {
        if (lay.depth_mm == null) throw new ModelError(`section ${sec.name}: extra layer ${k} needs depth_mm`);
        out.push([barArea(lay.n, lay.dia_mm), lay.depth_mm, `extra ${k + 1}`]);
      }
    });
    return out;
  }
  function layerDepth(sec, lay, face) {
    if (lay.depth_mm != null) return lay.depth_mm;
    const off = sec.cover_mm + (sec.links ? sec.links.dia_mm : 0) + lay.dia_mm / 2;
    return face === "top" ? off : sec.h_mm - off;
  }

  function parseModel(data) {
    if (data == null || typeof data !== "object") throw new ModelError("model must be a JSON object");
    const schema = data.schema || "tragwerk/0.1";
    if (schema !== "tragwerk/0.1") throw new ModelError(`unsupported schema '${schema}', expected 'tragwerk/0.1'`);
    const materials = {};
    for (const [name, md] of Object.entries(req(data, "materials", "model"))) {
      const where = `materials.${name}`;
      const kind = req(md, "type", where);
      if (kind === "concrete") {
        const fck = num(req(md, "fck", where), where + ".fck", { positive: true });
        if (fck < 12 || fck > 90) throw new ModelError(`${where}.fck: EN 1992-1-1 covers 12 to 90 MPa, got ${fck}`);
        materials[name] = { kind, name, fck, density: num(md.density ?? 25, where + ".density", { positive: true }),
          dmax_mm: num(md.dmax_mm ?? 16, where + ".dmax_mm", { positive: true }) };
      } else if (kind === "rebar") {
        const fyk = num(md.fyk ?? 500, where + ".fyk", { positive: true });
        if (fyk < 400 || fyk > 600) throw new ModelError(`${where}.fyk: EN 1992-1-1 covers 400 to 600 MPa, got ${fyk}`);
        materials[name] = { kind, name, fyk, Es: num(md.Es ?? 200000, where + ".Es", { positive: true }),
          eps_uk: num(md.eps_uk ?? 0.05, where + ".eps_uk", { positive: true }) };
      } else throw new ModelError(`${where}.type: expected 'concrete' or 'rebar', got ${JSON.stringify(kind)}`);
    }
    const sections = {};
    for (const [name, sd] of Object.entries(req(data, "sections", "model"))) {
      const where = `sections.${name}`;
      if ((sd.type || "rc_rect") !== "rc_rect") throw new ModelError(`${where}.type: only 'rc_rect' is supported in this version`);
      const conc = materials[req(sd, "concrete", where)], reb = materials[req(sd, "rebar", where)];
      if (!conc || conc.kind !== "concrete") throw new ModelError(`${where}.concrete: '${sd.concrete}' is not a concrete material`);
      if (!reb || reb.kind !== "rebar") throw new ModelError(`${where}.rebar: '${sd.rebar}' is not a rebar material`);
      let links = null;
      if (sd.links) {
        const ld = sd.links;
        links = { dia_mm: num(req(ld, "dia_mm", where + ".links"), where + ".links.dia_mm", { positive: true }),
          legs: Math.trunc(num(ld.legs ?? 2, where + ".links.legs", { positive: true })),
          spacing_mm: num(req(ld, "spacing_mm", where + ".links"), where + ".links.spacing_mm", { positive: true }),
          fyk: ld.fyk == null ? null : num(ld.fyk, where + ".links.fyk", { positive: true }) };
        links.area_mm2 = barArea(links.legs, links.dia_mm);
      }
      const sec = {
        name, b_mm: num(req(sd, "b_mm", where), where + ".b_mm", { positive: true }),
        h_mm: num(req(sd, "h_mm", where), where + ".h_mm", { positive: true }),
        concrete: conc, rebar: reb, cover_mm: num(req(sd, "cover_mm", where), where + ".cover_mm", { nonneg: true }),
        bottom: layer(sd.bottom, where + ".bottom"), top: layer(sd.top, where + ".top"), links,
        extra_layers: (sd.extra_layers || []).map((x, k) => layer(x, `${where}.extra_layers[${k}]`)),
        stiffness_factor: num(sd.stiffness_factor ?? 1.0, where + ".stiffness_factor", { positive: true }),
      };
      sec.area_m2 = sec.b_mm * sec.h_mm * 1e-6;
      sec.inertia_m4 = sec.b_mm * Math.pow(sec.h_mm, 3) / 12 * 1e-12;
      for (const [, depth, label] of sectionLayers(sec)) {
        if (!(depth > 0 && depth < sec.h_mm)) throw new ModelError(`${where}: ${label} bars at depth ${depth.toFixed(1)} mm lie outside the section`);
      }
      sections[name] = sec;
    }
    const nodes = {};
    req(data, "nodes", "model").forEach((nd, k) => {
      const where = `nodes[${k}]`;
      const id = String(req(nd, "id", where));
      if (id in nodes) throw new ModelError(`${where}: duplicate node id '${id}'`);
      nodes[id] = { id, x: num(req(nd, "x", where), where + ".x"), y: num(req(nd, "y", where), where + ".y") };
    });
    if (Object.keys(nodes).length < 2) throw new ModelError("model needs at least two nodes");
    const members = {};
    req(data, "members", "model").forEach((md, k) => {
      const where = `members[${k}]`;
      const id = String(req(md, "id", where));
      if (id in members) throw new ModelError(`${where}: duplicate member id '${id}'`);
      const i = String(req(md, "i", where)), j = String(req(md, "j", where));
      for (const [end, nid] of [["i", i], ["j", j]]) if (!(nid in nodes)) throw new ModelError(`${where}.${end}: unknown node '${nid}'`);
      if (i === j) throw new ModelError(`${where}: start and end node are the same`);
      if (Math.hypot(nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y) < 1e-6) throw new ModelError(`${where}: member has zero length`);
      const sec = String(req(md, "section", where));
      if (!(sec in sections)) throw new ModelError(`${where}.section: unknown section '${sec}'`);
      const rel = md.releases || {}, dd = md.design || {};
      let type = dd.type;
      if (type == null) {
        const dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
        type = Math.abs(dy) <= Math.abs(dx) * Math.tan(30 * Math.PI / 180) ? "beam" : "column";
      }
      if (!DESIGN_TYPES.includes(type)) throw new ModelError(`${where}.design.type: expected one of ${DESIGN_TYPES.join(", ")}`);
      const system = dd.deflection_system || "simply_supported";
      if (!DEFLECTION_SYSTEMS.includes(system)) throw new ModelError(`${where}.design.deflection_system: expected one of ${DEFLECTION_SYSTEMS.join(", ")}`);
      members[id] = { id, i, j, section: sec, release_i: !!rel.i, release_j: !!rel.j,
        design: { type, deflection_system: system,
          l0_factor: num(dd.l0_factor ?? 1.0, where + ".design.l0_factor", { positive: true }),
          creep_coefficient: num(dd.creep_coefficient ?? 2.0, where + ".design.creep_coefficient", { nonneg: true }) } };
    });
    if (!Object.keys(members).length) throw new ModelError("model needs at least one member");
    const supports = {};
    (data.supports || []).forEach((sd, k) => {
      const where = `supports[${k}]`;
      const nid = String(req(sd, "node", where));
      if (!(nid in nodes)) throw new ModelError(`${where}.node: unknown node '${nid}'`);
      if (nid in supports) throw new ModelError(`${where}: node '${nid}' already has a support`);
      supports[nid] = { node: nid, ux: !!sd.ux, uy: !!sd.uy, rz: !!sd.rz,
        kx: num(sd.kx ?? 0, where + ".kx", { nonneg: true }), ky: num(sd.ky ?? 0, where + ".ky", { nonneg: true }),
        kr: num(sd.kr ?? 0, where + ".kr", { nonneg: true }) };
    });
    if (!Object.keys(supports).length) throw new ModelError("model has no supports");
    const memberLength = (mid) => { const m = members[mid]; return Math.hypot(nodes[m.j].x - nodes[m.i].x, nodes[m.j].y - nodes[m.i].y); };
    const load_cases = {};
    req(data, "load_cases", "model").forEach((cd, k) => {
      const where = `load_cases[${k}]`;
      const id = String(req(cd, "id", where));
      if (id in load_cases) throw new ModelError(`${where}: duplicate load case id '${id}'`);
      const type = req(cd, "type", where);
      if (type !== "permanent" && type !== "variable") throw new ModelError(`${where}.type: expected 'permanent' or 'variable'`);
      const cat = cd.category ?? null;
      if (type === "variable" && !VARIABLE_CATEGORIES.includes(cat)) throw new ModelError(`${where}.category: variable actions need one of ${VARIABLE_CATEGORIES.join(", ")}`);
      const nodal = [], dist = [], point = [];
      (cd.loads || []).forEach((ld, q) => {
        const lw = `${where}.loads[${q}]`;
        const lt = req(ld, "type", lw);
        if (lt === "nodal") {
          const nid = String(req(ld, "node", lw));
          if (!(nid in nodes)) throw new ModelError(`${lw}.node: unknown node '${nid}'`);
          nodal.push({ node: nid, fx: num(ld.fx ?? 0, lw + ".fx"), fy: num(ld.fy ?? 0, lw + ".fy"), mz: num(ld.mz ?? 0, lw + ".mz") });
          return;
        }
        const mid = String(req(ld, "member", lw));
        if (!(mid in members)) throw new ModelError(`${lw}.member: unknown member '${mid}'`);
        const L = memberLength(mid);
        const direction = ld.direction || "global_y";
        if (!LOAD_DIRECTIONS.includes(direction)) throw new ModelError(`${lw}.direction: expected one of ${LOAD_DIRECTIONS.join(", ")}`);
        if (lt === "udl") {
          let w1, w2;
          if ("w" in ld) w1 = w2 = num(ld.w, lw + ".w");
          else { w1 = num(req(ld, "w1", lw), lw + ".w1"); w2 = num(req(ld, "w2", lw), lw + ".w2"); }
          const a = num(ld.a ?? 0, lw + ".a", { nonneg: true }), b = num(ld.b ?? L, lw + ".b", { nonneg: true });
          if (!(0 <= a && a < b && b <= L + 1e-9)) throw new ModelError(`${lw}: need 0 <= a < b <= member length (${L.toFixed(3)} m)`);
          dist.push({ member: mid, w1, w2, a, b: Math.min(b, L), direction });
        } else if (lt === "point") {
          const a = num(req(ld, "a", lw), lw + ".a", { nonneg: true });
          if (a > L + 1e-9) throw new ModelError(`${lw}.a: position ${a} m is beyond the member length (${L.toFixed(3)} m)`);
          point.push({ member: mid, p: num(req(ld, "p", lw), lw + ".p"), a: Math.min(a, L), direction });
        } else throw new ModelError(`${lw}.type: expected 'nodal', 'udl' or 'point', got ${JSON.stringify(lt)}`);
      });
      load_cases[id] = { id, type, name: String(cd.name ?? id), category: cat, group: cd.group ?? null,
        self_weight: !!cd.self_weight, nodal, distributed: dist, point };
    });
    if (!Object.keys(load_cases).length) throw new ModelError("model has no load cases");
    if (!Object.values(load_cases).some((lc) => lc.type === "permanent")) throw new ModelError("model needs at least one permanent load case (self weight counts)");
    const sd = data.design || {};
    const rule = sd.combination_rule || "6.10";
    if (rule !== "6.10" && rule !== "6.10ab") throw new ModelError("design.combination_rule: expected '6.10' or '6.10ab'");
    const code = sd.design_code || "EC2";
    if (code !== "EC2" && code !== "SIA262") throw new ModelError("design.design_code: expected 'EC2' or 'SIA262'");
    const settings = { design_code: code, combination_rule: rule, params: Object.assign({}, sd.params || {}),
      stations_per_member: Math.trunc(num(sd.stations_per_member ?? 40, "design.stations_per_member", { positive: true })) };
    const model = { name: String(data.name ?? "Untitled model"), nodes, members, sections, supports, load_cases, settings, raw: data };
    model.geometry = (mid) => {
      const m = members[mid], ni = nodes[m.i], nj = nodes[m.j];
      const dx = nj.x - ni.x, dy = nj.y - ni.y, L = Math.hypot(dx, dy);
      return [L, dx / L, dy / L];
    };
    return model;
  }

  // ---------------------------------------------------------------- linear algebra
  const zeros = (n) => new Array(n).fill(0);
  const mat = (r, c) => Array.from({ length: r }, () => zeros(c));
  function matvec(A, x) { return A.map((row) => row.reduce((s, a, k) => s + a * x[k], 0)); }
  function solveSmall(A, b) { // Gaussian elimination with partial pivoting (2x2 at most here)
    const n = b.length, M = A.map((r, i) => r.slice().concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = zeros(n);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }

  // ---------------------------------------------------------------- solver
  const GP = [0.5 - Math.sqrt(15) / 10, 0.5, 0.5 + Math.sqrt(15) / 10];
  const GW = [5 / 18, 8 / 18, 5 / 18];
  const analysisModulus = (sec) => 22000 * Math.pow((sec.concrete.fck + 8) / 10, 0.3) * 1000;

  function beamK(EA, EI, L) {
    const a = EA / L, b = EI / Math.pow(L, 3);
    return [
      [a, 0, 0, -a, 0, 0],
      [0, 12 * b, 6 * b * L, 0, -12 * b, 6 * b * L],
      [0, 6 * b * L, 4 * b * L * L, 0, -6 * b * L, 2 * b * L * L],
      [-a, 0, 0, a, 0, 0],
      [0, -12 * b, -6 * b * L, 0, 12 * b, -6 * b * L],
      [0, 6 * b * L, 2 * b * L * L, 0, -6 * b * L, 4 * b * L * L],
    ];
  }
  function rotation(c, s) {
    const T = mat(6, 6);
    for (const o of [0, 3]) { T[o][o] = c; T[o][o + 1] = s; T[o + 1][o] = -s; T[o + 1][o + 1] = c; T[o + 2][o + 2] = 1; }
    return T;
  }
  function shape(xi, L) {
    return [1 - xi, 1 - 3 * xi * xi + 2 * xi ** 3, L * (xi - 2 * xi * xi + xi ** 3), xi, 3 * xi * xi - 2 * xi ** 3, L * (-xi * xi + xi ** 3)];
  }
  function localComponents(direction, w, c, s) {
    if (direction === "local_x") return [w, 0];
    if (direction === "local_y") return [0, w];
    if (direction === "global_x") return [c * w, -s * w];
    return [s * w, c * w];
  }
  function equivalentNodalLoads(L, loads) {
    const p = zeros(6);
    for (const [a, b, qx1, qx2, qy1, qy2] of loads.dist) {
      for (let g = 0; g < 3; g++) {
        const s = a + (b - a) * GP[g], w = (b - a) * GW[g], t = (s - a) / (b - a);
        const qx = qx1 + (qx2 - qx1) * t, qy = qy1 + (qy2 - qy1) * t;
        const N = shape(s / L, L);
        p[0] += N[0] * w * qx; p[3] += N[3] * w * qx;
        p[1] += N[1] * w * qy; p[2] += N[2] * w * qy; p[4] += N[4] * w * qy; p[5] += N[5] * w * qy;
      }
    }
    for (const [a, Px, Py] of loads.point) {
      const N = shape(a / L, L);
      p[0] += N[0] * Px; p[3] += N[3] * Px;
      p[1] += N[1] * Py; p[2] += N[2] * Py; p[4] += N[4] * Py; p[5] += N[5] * Py;
    }
    return p;
  }
  function condense(k, cidx) {
    if (!cidx.length) return k;
    const aidx = [0, 1, 2, 3, 4, 5].filter((i) => !cidx.includes(i));
    const kcc = cidx.map((r) => cidx.map((c) => k[r][c]));
    const kk = mat(6, 6);
    for (const r of aidx) {
      const kac_r = cidx.map((c) => k[r][c]);
      for (const c of aidx) {
        const kca_c = cidx.map((q) => k[q][c]);
        const y = solveSmall(kcc, kca_c);
        kk[r][c] = k[r][c] - kac_r.reduce((s, v, q) => s + v * y[q], 0);
      }
    }
    return kk;
  }
  function condenseLoad(k, cidx, p) {
    if (!cidx.length) return p.slice();
    const aidx = [0, 1, 2, 3, 4, 5].filter((i) => !cidx.includes(i));
    const kcc = cidx.map((r) => cidx.map((c) => k[r][c]));
    const y = solveSmall(kcc, cidx.map((c) => p[c]));
    const out = zeros(6);
    for (const r of aidx) out[r] = p[r] - cidx.reduce((s, c, q) => s + k[r][c] * y[q], 0);
    return out;
  }
  function cumDist(x, loads, side) {
    let Sx = 0, Sy = 0, My = 0;
    for (const [a, b, qx1, qx2, qy1, qy2] of loads.dist) {
      if (x <= a) continue;
      const t = Math.min(x, b);
      const bx = (qx2 - qx1) / (b - a), ax = qx1 - bx * a, by = (qy2 - qy1) / (b - a), ay = qy1 - by * a;
      Sx += ax * (t - a) + bx * (t * t - a * a) / 2;
      Sy += ay * (t - a) + by * (t * t - a * a) / 2;
      const F = (s) => ay * x * s + (by * x - ay) * s * s / 2 - by * s ** 3 / 3;
      My += F(t) - F(a);
    }
    for (const [a, Px, Py] of loads.point) {
      if (a < x || (side > 0 && a <= x)) { Sx += Px; Sy += Py; My += (x - a) * Py; }
    }
    return [Sx, Sy, My];
  }

  class FrameAnalysis {
    constructor(model) {
      this.model = model;
      this.nodeIds = Object.keys(model.nodes);
      this.nidx = Object.fromEntries(this.nodeIds.map((n, k) => [n, k]));
      const n = this.nodeIds.length;
      this.ndof = 3 * n;
      const K = mat(this.ndof, this.ndof);
      const connected = new Array(n).fill(false);
      this.members = {};
      for (const [mid, m] of Object.entries(model.members)) {
        const sec = model.sections[m.section];
        const [L, c, s] = model.geometry(mid);
        const E = analysisModulus(sec) * sec.stiffness_factor;
        const md = { id: mid, ni: this.nidx[m.i], nj: this.nidx[m.j], L, c, s, EA: E * sec.area_m2, EI: E * sec.inertia_m4, rel: [m.release_i, m.release_j] };
        md.k_full = beamK(md.EA, md.EI, L);
        md.cidx = [[2, md.rel[0]], [5, md.rel[1]]].filter(([, r]) => r).map(([i]) => i);
        md.k_local = condense(md.k_full, md.cidx);
        md.T = rotation(c, s);
        const dofs = this.dofs(md);
        const kg = this.toGlobal(md.T, md.k_local);
        for (let r = 0; r < 6; r++) for (let q = 0; q < 6; q++) K[dofs[r]][dofs[q]] += kg[r][q];
        connected[md.ni] = connected[md.nj] = true;
        this.members[mid] = md;
      }
      connected.forEach((ok, k) => { if (!ok) throw new StructureUnstable(`node '${this.nodeIds[k]}' is not connected to any member`); });
      const fixed = new Array(this.ndof).fill(false);
      this.springs = {};
      for (const [nid, sp] of Object.entries(model.supports)) {
        const b = 3 * this.nidx[nid];
        for (const [off, flag, kk] of [[0, sp.ux, sp.kx], [1, sp.uy, sp.ky], [2, sp.rz, sp.kr]]) {
          if (flag) fixed[b + off] = true;
          else if (kk > 0) { K[b + off][b + off] += kk; this.springs[b + off] = kk; }
        }
      }
      let scale = 1;
      for (let d = 0; d < this.ndof; d++) scale = Math.max(scale, Math.abs(K[d][d]));
      this.autoFixed = [];
      for (let k = 0; k < n; k++) {
        const d = 3 * k + 2;
        if (!fixed[d] && Math.abs(K[d][d]) < 1e-12 * scale) { fixed[d] = true; this.autoFixed.push(d); }
      }
      this.K = K; this.fixed = fixed;
      this.free = []; for (let d = 0; d < this.ndof; d++) if (!fixed[d]) this.free.push(d);
      this.factorise();
      this.stations = this.buildStations();
    }
    dofs(md) { return [3 * md.ni, 3 * md.ni + 1, 3 * md.ni + 2, 3 * md.nj, 3 * md.nj + 1, 3 * md.nj + 2]; }
    toGlobal(T, k) { // T^T k T
      const kT = mat(6, 6), out = mat(6, 6);
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) { let s = 0; for (let q = 0; q < 6; q++) s += k[r][q] * T[q][c]; kT[r][c] = s; }
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) { let s = 0; for (let q = 0; q < 6; q++) s += T[q][r] * kT[q][c]; out[r][c] = s; }
      return out;
    }
    factorise() {
      const f = this.free, m = f.length;
      const L = mat(m, m);
      let dmax = 0, dmin = Infinity;
      for (let j = 0; j < m; j++) {
        let s = this.K[f[j]][f[j]];
        for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
        if (!(s > 0)) throw new StructureUnstable(this.mechanismMessage(f[j]));
        L[j][j] = Math.sqrt(s);
        dmax = Math.max(dmax, s); dmin = Math.min(dmin, s);
        for (let i = j + 1; i < m; i++) {
          let t = this.K[f[i]][f[j]];
          for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
          L[i][j] = t / L[j][j];
        }
      }
      if (m && dmin < 1e-13 * dmax) {
        // locate the weakest pivot for the message
        let jmin = 0;
        for (let j = 0; j < m; j++) if (L[j][j] * L[j][j] === dmin) jmin = j;
        throw new StructureUnstable(this.mechanismMessage(f[jmin]));
      }
      this.chol = L;
    }
    mechanismMessage(dof) {
      const names = ["move in X", "move in Y", "rotate"];
      return `structure is unstable (mechanism): node '${this.nodeIds[Math.floor(dof / 3)]}' is free to ${names[dof % 3]}. Check supports and member releases.`;
    }
    solve(rhs) {
      const L = this.chol, m = rhs.length, y = zeros(m), x = zeros(m);
      for (let i = 0; i < m; i++) { let s = rhs[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
      for (let i = m - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < m; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
      return x;
    }
    localLoads(lc) {
      const out = {};
      for (const [mid, md] of Object.entries(this.members)) {
        out[mid] = { dist: [], point: [] };
        if (lc.self_weight) {
          const sec = this.model.sections[this.model.members[mid].section];
          const w = -sec.concrete.density * sec.area_m2;
          const [qx, qy] = localComponents("global_y", w, md.c, md.s);
          out[mid].dist.push([0, md.L, qx, qx, qy, qy]);
        }
      }
      for (const ld of lc.distributed) {
        const md = this.members[ld.member];
        const [qx1, qy1] = localComponents(ld.direction, ld.w1, md.c, md.s);
        const [qx2, qy2] = localComponents(ld.direction, ld.w2, md.c, md.s);
        out[ld.member].dist.push([ld.a, ld.b, qx1, qx2, qy1, qy2]);
      }
      for (const ld of lc.point) {
        const md = this.members[ld.member];
        const [px, py] = localComponents(ld.direction, ld.p, md.c, md.s);
        out[ld.member].point.push([ld.a, px, py]);
      }
      return out;
    }
    buildStations() {
      const nseg = this.model.settings.stations_per_member;
      const breaks = Object.fromEntries(Object.keys(this.members).map((m) => [m, []]));
      for (const lc of Object.values(this.model.load_cases)) {
        for (const ld of lc.distributed) breaks[ld.member].push(ld.a, ld.b);
        for (const ld of lc.point) breaks[ld.member].push(ld.a);
      }
      const out = {};
      for (const [mid, md] of Object.entries(this.members)) {
        const inner = [];
        for (const b of Array.from(new Set(breaks[mid])).sort((p, q) => p - q)) {
          if (b > 1e-9 && b < md.L - 1e-9 && !inner.some((c) => Math.abs(b - c) < 1e-9)) inner.push(b);
        }
        const grid = [];
        for (let i = 0; i <= nseg; i++) {
          const x = i === nseg ? md.L : (md.L / nseg) * i;
          if (!inner.some((b) => Math.abs(x - b) < 1e-9)) grid.push(x);
        }
        const pts = grid.map((x) => [x, 1]).concat(inner.map((b) => [b, -1]), inner.map((b) => [b, 1]));
        pts.sort((p, q) => (p[0] - q[0]) || (p[1] - q[1]));
        const x = pts.map((p) => p[0]), side = pts.map((p) => p[1]);
        x[0] = 0; x[x.length - 1] = md.L; side[side.length - 1] = -1;
        out[mid] = { x, side };
      }
      return out;
    }
    solveCase(lc) {
      const n = this.nodeIds.length, P = zeros(this.ndof);
      for (const ld of lc.nodal) { const b = 3 * this.nidx[ld.node]; P[b] += ld.fx; P[b + 1] += ld.fy; P[b + 2] += ld.mz; }
      for (const d of this.autoFixed) if (Math.abs(P[d]) > 1e-12) throw new StructureUnstable(`a moment is applied at node '${this.nodeIds[Math.floor(d / 3)]}', where every member is hinged`);
      const loads = this.localLoads(lc), peq = {};
      for (const [mid, md] of Object.entries(this.members)) {
        const p = equivalentNodalLoads(md.L, loads[mid]);
        peq[mid] = p;
        const pc = condenseLoad(md.k_full, md.cidx, p);
        const dofs = this.dofs(md);
        for (let r = 0; r < 6; r++) { let s = 0; for (let q = 0; q < 6; q++) s += md.T[q][r] * pc[q]; P[dofs[r]] += s; }
      }
      const U = zeros(this.ndof);
      if (this.free.length) { const uf = this.solve(this.free.map((d) => P[d])); this.free.forEach((d, k) => { U[d] = uf[k]; }); }
      const R = matvec(this.K, U).map((v, k) => v - P[k]);
      const reactions = {}, springs = {};
      for (const nid of Object.keys(this.model.supports)) {
        const b = 3 * this.nidx[nid];
        reactions[nid] = [0, 1, 2].map((k) => (this.fixed[b + k] ? R[b + k] : 0));
        const sf = [0, 1, 2].map((k) => -(this.springs[b + k] || 0) * U[b + k]);
        if (sf.some((v) => v !== 0)) springs[nid] = sf;
      }
      const members = {};
      for (const [mid, md] of Object.entries(this.members)) members[mid] = this.memberResult(md, U, loads[mid], peq[mid]);
      const [eq, applied] = this.equilibrium(lc, loads, reactions, springs);
      const disp = []; for (let k = 0; k < n; k++) disp.push([U[3 * k], U[3 * k + 1], U[3 * k + 2]]);
      return { caseId: lc.id, displacements: disp, reactions, springs, members, equilibrium: eq, applied };
    }
    memberResult(md, U, loads, peq) {
      const dofs = this.dofs(md);
      const ug = dofs.map((d) => U[d]);
      const ul = matvec(md.T, ug);
      const k = md.k_full;
      if (md.cidx.length) {
        const aidx = [0, 1, 2, 3, 4, 5].filter((i) => !md.cidx.includes(i));
        const kcc = md.cidx.map((r) => md.cidx.map((c) => k[r][c]));
        const rhs = md.cidx.map((r) => peq[r] - aidx.reduce((s, c) => s + k[r][c] * ul[c], 0));
        const uc = solveSmall(kcc, rhs);
        md.cidx.forEach((c, q) => { ul[c] = uc[q]; });
      }
      const f = matvec(k, ul).map((v, q) => v - peq[q]);
      const { x: xs, side } = this.stations[md.id];
      const N = [], V = [], M = [];
      xs.forEach((x, q) => {
        const [Sx, Sy, My] = cumDist(x, loads, side[q]);
        N.push(-f[0] - Sx); V.push(f[1] + Sy); M.push(-f[2] + x * f[1] + My);
      });
      // unique positions, exact integration of curvature and axial strain
      const ux = []; xs.forEach((x) => { if (!ux.length || x !== ux[ux.length - 1]) ux.push(x); });
      const F1 = zeros(ux.length), G = zeros(ux.length), E = zeros(ux.length);
      for (let q = 1; q < ux.length; q++) {
        const x0 = ux[q - 1], x1 = ux[q];
        let a1 = 0, a2 = 0, a3 = 0;
        for (let g = 0; g < 3; g++) {
          const sg = x0 + (x1 - x0) * GP[g], w = (x1 - x0) * GW[g];
          const [Sx, , My] = cumDist(sg, loads, 1);
          const kap = (-f[2] + sg * f[1] + My) / md.EI;
          const eps = (-f[0] - Sx) / md.EA;
          a1 += w * kap; a2 += w * sg * kap; a3 += w * eps;
        }
        F1[q] = F1[q - 1] + a1; G[q] = G[q - 1] + a2; E[q] = E[q - 1] + a3;
      }
      const L = md.L, last = ux.length - 1;
      const theta = (ul[4] - ul[1] - (L * F1[last] - G[last])) / L;
      const vU = ux.map((x, q) => ul[1] + theta * x + x * F1[q] - G[q]);
      const uU = ux.map((x, q) => ul[0] + E[q]);
      const pos = new Map(ux.map((x, q) => [x, q]));
      const v = xs.map((x) => vU[pos.get(x)]), u = xs.map((x) => uU[pos.get(x)]);
      const vrel = xs.map((x, q) => v[q] - (ul[1] + (ul[4] - ul[1]) * x / L));
      return { x: xs, side, N, V, M, u, v, v_rel: vrel, end_forces: f };
    }
    equilibrium(lc, loads, reactions, springs) {
      let Fx = 0, Fy = 0, Mo = 0;
      for (const ld of lc.nodal) {
        const nd = this.model.nodes[ld.node];
        Fx += ld.fx; Fy += ld.fy; Mo += ld.mz + nd.x * ld.fy - nd.y * ld.fx;
      }
      for (const [mid, md] of Object.entries(this.members)) {
        const ni = this.model.nodes[this.model.members[mid].i];
        const add = (px, py, s) => {
          const gx = md.c * px - md.s * py, gy = md.s * px + md.c * py;
          const X = ni.x + md.c * s, Y = ni.y + md.s * s;
          return [gx, gy, X * gy - Y * gx];
        };
        for (const [a, b, qx1, qx2, qy1, qy2] of loads[mid].dist) {
          for (let g = 0; g < 3; g++) {
            const w = (b - a) * GW[g], s = a + (b - a) * GP[g], t = (s - a) / (b - a);
            const [gx, gy, mo] = add(qx1 + (qx2 - qx1) * t, qy1 + (qy2 - qy1) * t, s);
            Fx += w * gx; Fy += w * gy; Mo += w * mo;
          }
        }
        for (const [a, px, py] of loads[mid].point) { const [gx, gy, mo] = add(px, py, a); Fx += gx; Fy += gy; Mo += mo; }
      }
      const applied = { Fx, Fy, Mz_origin: Mo };
      let rx = 0, ry = 0, rm = 0;
      for (const [nid, r] of Object.entries(reactions).concat(Object.entries(springs))) {
        const nd = this.model.nodes[nid];
        rx += r[0]; ry += r[1]; rm += r[2] + nd.x * r[1] - nd.y * r[0];
      }
      return [{ Fx: Fx + rx, Fy: Fy + ry, Mz_origin: Mo + rm }, applied];
    }
  }

  // ---------------------------------------------------------------- combinations
  const PSI = { A: [0.7, 0.5, 0.3], B: [0.7, 0.5, 0.3], C: [0.7, 0.7, 0.6], D: [0.7, 0.7, 0.6], E: [1.0, 0.9, 0.8],
    F: [0.7, 0.7, 0.6], G: [0.7, 0.5, 0.3], H: [0, 0, 0], snow: [0.5, 0.2, 0], snow_high: [0.7, 0.5, 0.2], wind: [0.6, 0.2, 0], temperature: [0.6, 0.5, 0] };
  const CATEGORY_LABEL = { A: "Cat. A, residential", B: "Cat. B, office", C: "Cat. C, congregation", D: "Cat. D, shopping",
    E: "Cat. E, storage", F: "Cat. F, light traffic", G: "Cat. G, medium traffic", H: "Cat. H, roofs", snow: "Snow, H ≤ 1000 m",
    snow_high: "Snow, H > 1000 m", wind: "Wind", temperature: "Temperature" };

  function product(lists) {
    let out = [[]];
    for (const l of lists) { const next = []; for (const pre of out) for (const v of l) next.push(pre.concat([v])); out = next; }
    return out;
  }
  function generateCombinations(model) {
    const params = model.settings.params;
    const f = { sup: Number(params.gamma_G_sup ?? 1.35), inf: Number(params.gamma_G_inf ?? 1.0), Q: Number(params.gamma_Q ?? 1.5), xi: Number(params.xi ?? 0.85) };
    const psi = Object.assign({}, PSI);
    for (const [cat, vals] of Object.entries(params.psi || {})) psi[cat] = vals.map(Number);
    const vary = !!params.vary_permanent_individually;
    const rule = model.settings.combination_rule;
    const caseIds = Object.keys(model.load_cases);
    const idx = Object.fromEntries(caseIds.map((c, k) => [c, k]));
    const perm = caseIds.filter((c) => model.load_cases[c].type === "permanent");
    const vars = caseIds.filter((c) => model.load_cases[c].type === "variable");
    if (vars.length > 12) throw new ModelError(`at most 12 variable load cases are supported, got ${vars.length}`);
    const ps = Object.fromEntries(vars.map((c) => [c, psi[model.load_cases[c].category]]));
    const gSets = vary ? product(perm.map(() => [f.sup, f.inf])) : [perm.map(() => f.sup), perm.map(() => f.inf)];
    const grouped = new Map(), free = [];
    for (const c of vars) {
      const g = model.load_cases[c].group;
      if (g) { if (!grouped.has(g)) grouped.set(g, []); grouped.get(g).push(c); } else free.push(c);
    }
    const choices = [...grouped.values()].map((m) => [null].concat(m)).concat(free.map((c) => [null, c]));
    const subsets = product(choices).map((pick) => pick.filter((c) => c !== null));
    const seen = new Set(), combos = [], counts = {};
    const prefix = { ULS: "ULS", SLS_char: "CHAR", SLS_freq: "FREQ", SLS_qp: "QP" };
    const label = (fs) => { const parts = []; fs.forEach((v, k) => { if (Math.abs(v) > 1e-12) parts.push(`${v.toFixed(2)}·${caseIds[k]}`); }); return parts.length ? parts.join(" + ") : "0"; };
    const add = (kind, fs, leading) => {
      const key = kind + "|" + fs.map((v) => Math.round(v * 1e9) / 1e9).join(",");
      if (seen.has(key)) return;
      seen.add(key);
      counts[kind] = (counts[kind] || 0) + 1;
      combos.push({ id: `${prefix[kind]}${String(counts[kind]).padStart(3, "0")}`, kind, factors: fs, leading, label: label(fs) });
    };
    for (const g of gSets) for (const S of subsets) for (const lead of (S.length ? S : [null])) {
      if (rule === "6.10") {
        const v = zeros(caseIds.length);
        perm.forEach((c, k) => { v[idx[c]] = g[k]; });
        for (const c of S) v[idx[c]] = f.Q * (c === lead ? 1.0 : ps[c][0]);
        add("ULS", v, lead);
      } else {
        const a = zeros(caseIds.length), b = zeros(caseIds.length);
        perm.forEach((c, k) => { a[idx[c]] = g[k]; b[idx[c]] = g[k] > 1.0 ? f.xi * g[k] : g[k]; });
        for (const c of S) { a[idx[c]] = f.Q * ps[c][0]; b[idx[c]] = f.Q * (c === lead ? 1.0 : ps[c][0]); }
        add("ULS", a, null); add("ULS", b, lead);
      }
    }
    for (const S of subsets) {
      for (const lead of (S.length ? S : [null])) {
        const ch = zeros(caseIds.length), fr = zeros(caseIds.length);
        for (const c of perm) ch[idx[c]] = fr[idx[c]] = 1.0;
        for (const c of S) { ch[idx[c]] = c === lead ? 1.0 : ps[c][0]; fr[idx[c]] = c === lead ? ps[c][1] : ps[c][2]; }
        add("SLS_char", ch, lead); add("SLS_freq", fr, lead);
      }
      const qp = zeros(caseIds.length);
      for (const c of perm) qp[idx[c]] = 1.0;
      for (const c of S) qp[idx[c]] = ps[c][2];
      add("SLS_qp", qp, null);
    }
    return { caseIds, combos };
  }

  // ---------------------------------------------------------------- EC2 materials and section
  const EC2_DEFAULTS = { gamma_c: 1.5, gamma_s: 1.15, alpha_cc: 1.0, alpha_ct: 1.0, eps_ud_factor: 0.9, C_Rd_c_numerator: 0.18,
    k1_shear: 0.15, vmin_coefficient: 0.035, cot_theta_min: 1.0, cot_theta_max: 2.5, e0_min_mm: 20.0, e0_h_divisor: 30.0,
    deflection_limit_divisor: 250.0, beta_sustained: 0.5, k1_spacing: 1.0, k2_spacing_mm: 5.0, As_min_fctm_factor: 0.26,
    As_min_absolute: 0.0013, As_max_ratio: 0.04, rho_w_min_factor: 0.08, sl_max_factor: 0.75, column_As_min_N_factor: 0.10,
    column_As_min_Ac_ratio: 0.002, column_phi_min_mm: 8.0, column_link_spacing_phi_factor: 20.0, column_link_spacing_max_mm: 400.0 };
  const COMBO_PARAMS = ["gamma_G_sup", "gamma_G_inf", "gamma_Q", "xi", "psi", "vary_permanent_individually"];
  function makeParams(over) {
    const p = Object.assign({}, EC2_DEFAULTS);
    for (const [k, v] of Object.entries(over || {})) {
      if (COMBO_PARAMS.includes(k)) continue;
      if (!(k in EC2_DEFAULTS)) throw new ModelError(`unknown design parameter(s): ${k}`);
      p[k] = Number(v);
    }
    p.C_Rd_c = p.C_Rd_c_numerator / p.gamma_c;
    return p;
  }
  function concreteProps(c, p) {
    const fck = c.fck, fcm = fck + 8;
    let fctm, eps_c2, eps_cu2, n, lam, eta;
    if (fck <= 50) { fctm = 0.30 * Math.pow(fck, 2 / 3); eps_c2 = 2.0e-3; eps_cu2 = 3.5e-3; n = 2.0; lam = 0.8; eta = 1.0; }
    else {
      fctm = 2.12 * Math.log(1 + fcm / 10);
      eps_c2 = (2.0 + 0.085 * Math.pow(fck - 50, 0.53)) * 1e-3;
      eps_cu2 = (2.6 + 35 * Math.pow((90 - fck) / 100, 4)) * 1e-3;
      n = 1.4 + 23.4 * Math.pow((90 - fck) / 100, 4);
      lam = 0.8 - (fck - 50) / 400; eta = 1.0 - (fck - 50) / 200;
    }
    const fcd = p.alpha_cc * fck / p.gamma_c;
    return { fck, fcm, fctm, Ecm: 22000 * Math.pow(fcm / 10, 0.3), fcd, eps_c2, eps_cu2, n, lam, eta, nu: 0.6 * (1 - fck / 250),
      sigma(e) { if (e <= 0) return 0; if (e < eps_c2) return fcd * (1 - Math.pow(1 - e / eps_c2, n)); return fcd; } };
  }
  function steelProps(r, p) {
    const fyd = r.fyk / p.gamma_s;
    return { fyk: r.fyk, fyd, Es: r.Es, eps_yd: fyd / r.Es, eps_ud: p.eps_ud_factor * r.eps_uk,
      sigma(e) { return Math.min(Math.max(r.Es * e, -fyd), fyd); } };
  }
  const GL_X = [-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664];
  const GL_W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];

  function interp(x, xp, fp) {
    const n = xp.length;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
    const dx = xp[lo + 1] - xp[lo];
    return dx > 0 ? fp[lo] + (fp[lo + 1] - fp[lo]) * (x - xp[lo]) / dx : fp[lo];
  }

  class RCSection {
    constructor(sec, p) {
      this.sec = sec; this.p = p;
      this.c = concreteProps(sec.concrete, p); this.st = steelProps(sec.rebar, p);
      this.b = sec.b_mm; this.h = sec.h_mm;
      const lay = sectionLayers(sec);
      this.As = lay.map((l) => l[0]); this.ys = lay.map((l) => l[1]); this.labels = lay.map((l) => l[2]);
      this.curves = {};
    }
    get Ac() { return this.b * this.h; }
    resultants(et, eb) {
      const h = this.h, c = this.c;
      const eps = (y) => et + (eb - et) * y / h;
      const cuts = [0, h];
      const de = eb - et;
      if (Math.abs(de) > 1e-15) for (const target of [0, c.eps_c2]) { const y = (target - et) * h / de; if (y > 0 && y < h) cuts.push(y); }
      const cs = Array.from(new Set(cuts)).sort((a, b) => a - b);
      let Nc = 0, Mc = 0;
      for (let k = 0; k + 1 < cs.length; k++) {
        const y0 = cs[k], y1 = cs[k + 1];
        let sN = 0, sM = 0;
        for (let g = 0; g < 5; g++) {
          const yg = 0.5 * (y1 - y0) * GL_X[g] + 0.5 * (y1 + y0), wg = 0.5 * (y1 - y0) * GL_W[g];
          const sg = c.sigma(eps(yg));
          sN += wg * sg; sM += wg * sg * (h / 2 - yg);
        }
        Nc += this.b * sN; Mc += this.b * sM;
      }
      let Ns = 0, Ms = 0;
      this.ys.forEach((y, k) => {
        const e = eps(y);
        const s = this.st.sigma(e) - c.sigma(e);
        Ns += this.As[k] * s; Ms += this.As[k] * s * (h / 2 - y);
      });
      return [(Nc + Ns) * 1e-3, (Mc + Ms) * 1e-6];
    }
    dMax(flip) { if (!this.ys.length) return this.h; return Math.max(...this.ys.map((y) => (flip ? this.h - y : y))); }
    plane(s, flip) {
      const c = this.c, h = this.h, eud = this.st.eps_ud, dm = this.dMax(flip);
      s = Math.min(Math.max(s, 0), 4);
      let et, eb;
      if (s <= 2) { et = s <= 1 ? -eud * (1 - s) : (s - 1) * c.eps_cu2; eb = et + (-eud - et) * h / dm; }
      else if (s <= 3) { const xab = c.eps_cu2 / (c.eps_cu2 + eud) * dm; const x = xab + (s - 2) * (h - xab); et = c.eps_cu2; eb = c.eps_cu2 * (1 - h / x); }
      else { const yc = (1 - c.eps_c2 / c.eps_cu2) * h; eb = (s - 3) * c.eps_c2; const m = (eb - c.eps_c2) / (h - yc); et = c.eps_c2 - m * yc; }
      return flip ? [eb, et] : [et, eb];
    }
    boundaryPoint(s, flip) { const [et, eb] = this.plane(s, flip); return this.resultants(et, eb); }
    get N_min() { return this.boundaryPoint(0, false)[0]; }
    get N_max() { return this.boundaryPoint(4, false)[0]; }
    curve(flip, n = 400) {
      const key = flip ? 1 : 0;
      if (!this.curves[key]) {
        const N = [], M = [];
        for (let k = 0; k < 4; k++) {
          const cnt = n, end = k === 3;
          for (let i = 0; i < cnt; i++) {
            const s = end ? k + i / (cnt - 1) : k + i / cnt;
            const [Ni, Mi] = this.boundaryPoint(s, flip);
            N.push(Ni); M.push(Mi);
          }
        }
        this.curves[key] = [N, M];
      }
      return this.curves[key];
    }
    mBounds(N) { const [Np, Mp] = this.curve(false), [Nn, Mn] = this.curve(true); return [interp(N, Np, Mp), interp(N, Nn, Mn)]; }
    solveState(N, flip = false, tol = 1e-9) {
      let lo = 0, hi = 4;
      const Nlo = this.boundaryPoint(lo, flip)[0], Nhi = this.boundaryPoint(hi, flip)[0];
      if (!(Nlo - 1e-9 <= N && N <= Nhi + 1e-9)) throw new Error("axial force outside the resistance range");
      for (let it = 0; it < 100; it++) {
        const mid = 0.5 * (lo + hi);
        if (this.boundaryPoint(mid, flip)[0] < N) lo = mid; else hi = mid;
        if (hi - lo < tol) break;
      }
      return this.state(0.5 * (lo + hi), flip);
    }
    state(s, flip) {
      const [et, eb] = this.plane(s, flip);
      const [N, M] = this.resultants(et, eb);
      const [ec, ex] = flip ? [eb, et] : [et, eb];
      let x = null;
      if (ec > 0 && ex < 0) x = ec / (ec - ex) * this.h; else if (ec > 0 && ex >= 0) x = Infinity;
      const steel = this.ys.map((y, k) => { const e = et + (eb - et) * y / this.h; return [this.labels[k], y, e, this.st.sigma(e)]; });
      return { s, eps_top: et, eps_bot: eb, N, M, x, steel };
    }
    utilisation(N, M) {
      const [Np] = this.curve(false);
      const Nmin = Np[0], Nmax = Np[Np.length - 1];
      const [hi, lo] = this.mBounds(N);
      let u;
      if (M >= 0) {
        u = hi > 0 ? M / hi : (M > hi ? Infinity : 0);
        if (lo > 0 && M > 0) u = Math.max(u, lo / M);
        if (lo > 0 && M <= 0) u = Infinity;
      } else {
        u = lo < 0 ? M / lo : (M < lo ? Infinity : 0);
        if (hi < 0 && M < 0) u = Math.max(u, hi / M);
        if (hi < 0 && M >= 0) u = Infinity;
      }
      if (N > Nmax) u = Math.max(u, N / Nmax);
      if (N < Nmin) u = Math.max(u, Nmin < 0 ? N / Nmin : Infinity);
      return Number.isNaN(u) ? 0 : u;
    }
    transformed(Eeff, sagging = true) {
      const ae = this.st.Es / Eeff, b = this.b, h = this.h;
      const ys = this.ys.map((y) => (sagging ? y : h - y)), As = this.As;
      const sumA = As.reduce((s, a) => s + a, 0);
      const A1 = b * h + (ae - 1) * sumA;
      const y1 = (b * h * h / 2 + (ae - 1) * As.reduce((s, a, k) => s + a * ys[k], 0)) / A1;
      const I1 = b * h ** 3 / 12 + b * h * (h / 2 - y1) ** 2 + (ae - 1) * As.reduce((s, a, k) => s + a * (ys[k] - y1) ** 2, 0);
      const f = (x) => b * x * x / 2 + As.reduce((s, a, k) => s + (ys[k] < x ? ae - 1 : ae) * a * (x - ys[k]), 0);
      let x;
      if (f(h) <= 0) x = h;
      else { let lo = 1e-9, hi = h; for (let it = 0; it < 200; it++) { const mid = 0.5 * (lo + hi); if (f(mid) < 0) lo = mid; else hi = mid; } x = 0.5 * (lo + hi); }
      const I2 = b * x ** 3 / 3 + As.reduce((s, a, k) => s + (ys[k] < x ? ae - 1 : ae) * a * (ys[k] - x) ** 2, 0);
      return { alpha_e: ae, A_I: A1, y_I: y1, I_I: I1, x_II: x, I_II: I2 };
    }
  }

  // ---------------------------------------------------------------- EC2 checks
  const PASS = "pass", FAIL = "fail", INFO = "info", NOT_COVERED = "not_covered";
  const K_SYSTEM = { simply_supported: 1.0, end_span: 1.3, interior_span: 1.5, cantilever: 0.4 };
  const statusU = (u) => (u <= 1 + 1e-9 ? PASS : FAIL);
  function fmt(v, digits = 3) {
    if (v === 0) return "0";
    if (!isFinite(v)) return "∞";
    const a = Math.abs(v);
    if (a >= 1e5 && a < 1e12) return Math.round(v).toLocaleString("en-US").replace(/,/g, " ");
    if (a >= 1e12 || a < 1e-3) return v.toExponential(digits);
    if (a >= 100) return v.toFixed(1);
    if (a >= 10) return v.toFixed(2);
    return a < 1 ? String(Number(v.toPrecision(digits))) : v.toFixed(3);
  }
  const step = (symbol, expression, substituted, value, unit = "", clause = "", note = "") => ({ symbol, expression, substituted, value, unit, clause, note });
  const check = (o) => Object.assign({ utilisation: null, demand: null, capacity: null, unit: "", x: null, combination: null, combination_label: null, steps: [], notes: [] }, o);
  function argmax2(rows, fn) {
    let best = -Infinity, bi = 0, bj = 0;
    rows.forEach((row, i) => row.forEach((_, j) => { const u = fn(i, j); if (u > best) { best = u; bi = i; bj = j; } }));
    return [bi, bj, best];
  }

  function checkBendingAxial(member, rc, f, p) {
    const isColumn = member.design.type === "column";
    const e0 = Math.max(rc.h / p.e0_h_divisor, p.e0_min_mm);
    // 6.1(4): columns carry at least N_Ed * e0, applied in the more unfavourable direction
    const U = f.uls_M.map((row) => row.map(() => 0));
    const Md = f.uls_M.map((row, i) => row.map((M, j) => {
      const Nc = -f.uls_N[i][j];
      if (!isColumn) { U[i][j] = rc.utilisation(Nc, M); return M; }
      const Mmin = Nc > 0 ? Nc * e0 * 1e-3 : 0;
      const small = Math.abs(M) < Mmin;
      const Mp = small ? Mmin : M, Mn = small ? -Mmin : M;
      const up = rc.utilisation(Nc, Mp), un = rc.utilisation(Nc, Mn);
      U[i][j] = Math.max(up, un);
      return un > up ? Mn : Mp;
    }));
    const [ci, si] = argmax2(Md, (i, j) => U[i][j]);
    const NEd = -f.uls_N[ci][si], MEd = Md[ci][si], MEdRaw = f.uls_M[ci][si];
    const c = rc.c, st = rc.st, steps = [], notes = [];
    steps.push(step("f_{cd}", "α_{cc}·f_{ck}/γ_{c}", `${fmt(p.alpha_cc)}·${fmt(c.fck)}/${fmt(p.gamma_c)}`, c.fcd, "MPa", "3.1.6(1)"));
    steps.push(step("f_{yd}", "f_{yk}/γ_{s}", `${fmt(st.fyk)}/${fmt(p.gamma_s)}`, st.fyd, "MPa", "3.2.7"));
    steps.push(step("ε_{c2}, ε_{cu2}", "Table 3.1", `${(c.eps_c2 * 1e3).toFixed(2)} ‰, ${(c.eps_cu2 * 1e3).toFixed(2)} ‰`, c.eps_cu2 * 1e3, "‰", "3.1.7"));
    steps.push(step("N_{Ed}", "axial force (compression +)", fmt(NEd), NEd, "kN"));
    if (isColumn) {
      steps.push(step("e_{0}", "max(h/30, 20 mm)", `max(${fmt(rc.h)}/30, 20)`, e0, "mm", "6.1(4)"));
      steps.push(step("M_{Ed}", "max(|M_{Ed,analysis}|, N_{Ed}·e_{0}) with the sign of M", `max(${fmt(Math.abs(MEdRaw))}, ${fmt(Math.max(NEd, 0) * e0 * 1e-3)})`, MEd, "kNm", "6.1(4)"));
    } else steps.push(step("M_{Ed}", "bending moment (sagging +)", fmt(MEd), MEd, "kNm"));
    const Nmin = rc.N_min, Nmax = rc.N_max;
    let status, capacity, util;
    if (NEd > Nmax || NEd < Nmin) {
      capacity = NEd > Nmax ? Nmax : Nmin;
      util = capacity !== 0 ? NEd / capacity : Infinity;
      steps.push(step("N_{Rd}", "resistance of the section to pure axial force", "", capacity, "kN", "6.1"));
      notes.push("The axial force alone exceeds the section resistance.");
      status = FAIL;
    } else {
      const flip = MEd < 0;
      const state = rc.solveState(NEd, flip);
      const MRd = state.M;
      capacity = MRd;
      if (state.x !== null && isFinite(state.x)) steps.push(step("x", "neutral axis depth from the compressed face (strain compatibility)", "", state.x, "mm", "6.1(2)"));
      const face = flip ? "bottom" : "top";
      steps.push(step(`ε_{c,${face}}`, "strain at the compressed face", "", (flip ? state.eps_bot : state.eps_top) * 1e3, "‰", "Fig. 6.1"));
      for (const [lab, y, e, s] of state.steel) steps.push(step(`σ_{s,${lab}}`, `E_{s}·ε_{s} ≤ f_{yd} with ε_{s} = ${(e * 1e3).toFixed(2)} ‰ (bars at depth ${y.toFixed(0)} mm)`, "", s, "MPa", "3.2.7"));
      steps.push(step("M_{Rd}", "resistance moment at N_{Ed} (parabola-rectangle, Fig. 6.1)", "", MRd, "kNm", "6.1"));
      if (MRd * MEd > 0) util = MEd / MRd; else if (Math.abs(MEd) < 1e-12) util = 0; else util = Infinity;
      status = statusU(util);
      if (!isColumn && state.x !== null && isFinite(state.x) && !flip) {
        const d = rc.ys.length ? Math.max(...rc.ys) : rc.h;
        const xd = state.x / d;
        if (xd > 0.45 && c.fck <= 50) notes.push(`x/d = ${xd.toFixed(2)} > 0.45: low ductility; avoid moment redistribution (5.6.3).`);
      }
    }
    steps.push(step("η", "M_{Ed}/M_{Rd}", capacity ? `${fmt(MEd)}/${fmt(capacity)}` : "", util, "", "6.1"));
    if (isColumn) notes.push("Second-order effects are checked separately (slenderness, 5.8.3.1).");
    return check({ id: "ec2.bending_axial", title: "Bending with axial force", clause: "EN 1992-1-1 6.1", member: member.id, status,
      utilisation: util, demand: MEd, capacity, unit: "kNm", x: f.x[si], combination: f.uls_ids[ci], combination_label: f.uls_labels[ci], steps, notes });
  }

  function tensionSteel(rc) {
    const h = rc.h;
    let Ab = 0, Ayb = 0, At = 0, Ayt = 0;
    rc.ys.forEach((y, k) => { if (y > h / 2) { Ab += rc.As[k]; Ayb += rc.As[k] * y; } if (y < h / 2) { At += rc.As[k]; Ayt += rc.As[k] * y; } });
    return { Ab, At, db: Ab > 0 ? Ayb / Ab : 0.9 * h, dt: At > 0 ? h - Ayt / At : 0.9 * h };
  }
  function shearResistance(rc, sec, p, Nc, M) {
    const c = rc.c, b = rc.b, ts = tensionSteel(rc);
    const sag = M >= -1e-9; // |M| ~ 0: treat as sagging (deterministic)
    const Asl = sag ? ts.Ab : ts.At, d = sag ? ts.db : ts.dt;
    const k = Math.min(1 + Math.sqrt(200 / d), 2.0);
    const rho = Math.min(Asl / (b * d), 0.02);
    const sig = Math.min(Nc * 1e3 / rc.Ac, 0.2 * c.fcd);
    const vmin = p.vmin_coefficient * Math.pow(k, 1.5) * Math.sqrt(c.fck);
    const v1 = p.C_Rd_c * k * Math.cbrt(100 * rho * c.fck) + p.k1_shear * sig;
    const v2 = vmin + p.k1_shear * sig;
    const VRdc = Math.max(Math.max(v1, v2), 0) * b * d * 1e-3;
    const Vlim = 0.5 * b * d * c.nu * c.fcd * 1e-3;
    const out = { k, rho, sigma_cp: sig, vmin, VRdc, d, Asl, Vlim };
    if (sec.links) {
      const z = 0.9 * d, fywd = (sec.links.fyk || sec.rebar.fyk) / p.gamma_s;
      const a = sec.links.area_mm2 / sec.links.spacing_mm * z * fywd;
      const cc = 1.0 * b * z * c.nu * c.fcd;
      let cot = Math.sqrt(Math.max(cc / a - 1, 0));
      cot = Math.min(Math.max(cot, p.cot_theta_min), p.cot_theta_max);
      const VRds = a * cot * 1e-3, VRdmax = cc / (cot + 1 / cot) * 1e-3, VRdLinks = Math.min(VRds, VRdmax);
      Object.assign(out, { z, fywd, cot, VRds, VRdmax, VRd_links: VRdLinks, VRd: Math.max(VRdc, VRdLinks) });
    } else out.VRd = Math.min(VRdc, Vlim);
    return out;
  }
  function checkShear(member, rc, sec, f, p) {
    const R = f.uls_V.map((row, i) => row.map((_, j) => shearResistance(rc, sec, p, -f.uls_N[i][j], f.uls_M[i][j])));
    const U = (i, j) => { const V = Math.abs(f.uls_V[i][j]), VRd = R[i][j].VRd; return VRd > 0 ? V / VRd : (V > 0 ? Infinity : 0); };
    const [ci, si, util] = argmax2(f.uls_V, U);
    const g = R[ci][si], c = rc.c, VEd = Math.abs(f.uls_V[ci][si]);
    const face = f.uls_M[ci][si] >= -1e-9 ? "bottom" : "top";
    const Nc = -f.uls_N[ci][si];
    const steps = [
      step("V_{Ed}", "design shear force", fmt(VEd), VEd, "kN"),
      step("d", `effective depth to the ${face} (tension) reinforcement`, "", g.d, "mm"),
      step("k", "1 + √(200/d) ≤ 2.0", `1 + √(200/${fmt(g.d)})`, g.k, "", "6.2.2(1)"),
      step("ρ_{l}", "A_{sl}/(b_{w}·d) ≤ 0.02", `${fmt(g.Asl)}/(${fmt(rc.b)}·${fmt(g.d)})`, g.rho, "", "6.2.2(1)"),
      step("σ_{cp}", "N_{Ed}/A_{c} < 0.2·f_{cd}", `${fmt(Nc * 1e3)}/${fmt(rc.Ac)}`, g.sigma_cp, "MPa", "6.2.2(1)"),
      step("v_{min}", "0.035·k^{3/2}·f_{ck}^{1/2}", `0.035·${fmt(g.k)}^{1.5}·${fmt(c.fck)}^{0.5}`, g.vmin, "MPa", "6.2.2(1) Eq. 6.3N"),
      step("V_{Rd,c}", "[C_{Rd,c}·k·(100·ρ_{l}·f_{ck})^{1/3} + k_{1}·σ_{cp}]·b_{w}·d ≥ (v_{min} + k_{1}·σ_{cp})·b_{w}·d",
        `[${fmt(p.C_Rd_c)}·${fmt(g.k)}·(100·${fmt(g.rho)}·${fmt(c.fck)})^{1/3} + ${fmt(p.k1_shear)}·${fmt(g.sigma_cp)}]·${fmt(rc.b)}·${fmt(g.d)}`, g.VRdc, "kN", "6.2.2(1) Eq. 6.2"),
    ];
    const notes = [];
    if (sec.links) {
      const L = sec.links;
      steps.push(
        step("A_{sw}/s", `${L.legs} legs Ø${L.dia_mm} at ${L.spacing_mm} mm`, `${fmt(L.area_mm2)}/${fmt(L.spacing_mm)}`, L.area_mm2 / L.spacing_mm, "mm²/mm"),
        step("z", "0.9·d", `0.9·${fmt(g.d)}`, g.z, "mm", "6.2.3(1)"),
        step("cot θ", "chosen in [1, 2.5] to maximise min(V_{Rd,s}, V_{Rd,max})", "", g.cot, "", "6.2.3(2) Eq. 6.7N"),
        step("V_{Rd,s}", "(A_{sw}/s)·z·f_{ywd}·cot θ", `${fmt(L.area_mm2 / L.spacing_mm)}·${fmt(g.z)}·${fmt(g.fywd)}·${fmt(g.cot)}`, g.VRds, "kN", "6.2.3(3) Eq. 6.8"),
        step("V_{Rd,max}", "α_{cw}·b_{w}·z·ν_{1}·f_{cd}/(cot θ + tan θ)", `1.0·${fmt(rc.b)}·${fmt(g.z)}·${fmt(c.nu)}·${fmt(c.fcd)}/(${fmt(g.cot)} + ${fmt(1 / g.cot)})`, g.VRdmax, "kN", "6.2.3(3) Eq. 6.9"),
        step("V_{Rd}", "max(V_{Rd,c}, min(V_{Rd,s}, V_{Rd,max}))", "", g.VRd, "kN", "6.2.1"));
    } else {
      steps.push(step("V_{Rd}", "min(V_{Rd,c}, 0.5·b_{w}·d·ν·f_{cd})", `min(${fmt(g.VRdc)}, ${fmt(g.Vlim)})`, g.VRd, "kN", "6.2.2(6)"));
      notes.push("No shear links defined: resistance from concrete alone (6.2.2).");
    }
    steps.push(step("η", "V_{Ed}/V_{Rd}", `${fmt(VEd)}/${fmt(g.VRd)}`, util, "", "6.2.1"));
    notes.push("Checked at every station, including at the support axis (conservative; 6.2.1(8) allows d from the face).");
    return check({ id: "ec2.shear", title: "Shear", clause: "EN 1992-1-1 6.2", member: member.id, status: statusU(util), utilisation: util,
      demand: VEd, capacity: g.VRd, unit: "kN", x: f.x[si], combination: f.uls_ids[ci], combination_label: f.uls_labels[ci], steps, notes });
  }

  function maxOf(A) { let m = -Infinity; for (const r of A) for (const v of r) if (v > m) m = v; return m; }
  function minOf(A) { let m = Infinity; for (const r of A) for (const v of r) if (v < m) m = v; return m; }

  function checkDetailing(member, rc, sec, f, p) {
    const out = [], c = rc.c, st = rc.st, b = rc.b, h = rc.h, Ac = rc.Ac;
    const AsTot = rc.As.reduce((s, a) => s + a, 0);
    const isColumn = member.design.type === "column";
    const AsMax = p.As_max_ratio * Ac;
    let u = AsTot / AsMax;
    out.push(check({ id: "ec2.as_max", title: "Maximum longitudinal reinforcement", clause: "EN 1992-1-1 9.2.1.1(3), 9.5.2(3)", member: member.id,
      status: statusU(u), utilisation: u, demand: AsTot, capacity: AsMax, unit: "mm²",
      steps: [step("A_{s,max}", "0.04·A_{c}", `0.04·${fmt(Ac)}`, AsMax, "mm²"), step("A_{s,prov}", "all longitudinal bars", "", AsTot, "mm²")],
      notes: ["Limit applies outside lap locations."] }));
    if (!isColumn) {
      const faces = [];
      if (maxOf(f.uls_M) > 1e-9) faces.push(["bottom", sec.bottom]);
      if (minOf(f.uls_M) < -1e-9) faces.push(["top", sec.top]);
      for (const [face, lay] of faces) {
        const prov = lay && lay.n > 0 ? barArea(lay.n, lay.dia_mm) : 0;
        const d = prov > 0 ? (face === "bottom" ? layerDepth(sec, lay, face) : h - layerDepth(sec, lay, face)) : 0.9 * h;
        const req2 = Math.max(p.As_min_fctm_factor * c.fctm / st.fyk, p.As_min_absolute) * b * d;
        const uu = prov > 0 ? req2 / prov : Infinity;
        out.push(check({ id: `ec2.as_min.${face}`, title: `Minimum tension reinforcement, ${face}`, clause: "EN 1992-1-1 9.2.1.1(1)", member: member.id,
          status: statusU(uu), utilisation: uu, demand: req2, capacity: prov, unit: "mm²",
          steps: [step("f_{ctm}", "Table 3.1", "", c.fctm, "MPa", "3.1.2"),
            step("A_{s,min}", "max(0.26·f_{ctm}/f_{yk}, 0.0013)·b_{t}·d", `max(0.26·${fmt(c.fctm)}/${fmt(st.fyk)}, 0.0013)·${fmt(b)}·${fmt(d)}`, req2, "mm²", "Eq. 9.1N"),
            step("A_{s,prov}", `${face} bars`, "", prov, "mm²")], notes: ["Ratio shown is A_{s,min}/A_{s,prov}."] }));
      }
      if (sec.links) {
        const L = sec.links, rhoW = L.area_mm2 / (L.spacing_mm * b), fykW = L.fyk || st.fyk;
        const rhoMin = p.rho_w_min_factor * Math.sqrt(c.fck) / fykW;
        let uu = rhoMin / rhoW;
        out.push(check({ id: "ec2.links_min", title: "Minimum shear reinforcement", clause: "EN 1992-1-1 9.2.2(5)", member: member.id,
          status: statusU(uu), utilisation: uu, demand: rhoMin, capacity: rhoW,
          steps: [step("ρ_{w}", "A_{sw}/(s·b_{w})", `${fmt(L.area_mm2)}/(${fmt(L.spacing_mm)}·${fmt(b)})`, rhoW, "", "Eq. 9.4"),
            step("ρ_{w,min}", "0.08·√f_{ck}/f_{yk}", `0.08·√${fmt(c.fck)}/${fmt(fykW)}`, rhoMin, "", "Eq. 9.5N")] }));
        const d = rc.ys.length ? Math.max(...rc.ys) : 0.9 * h;
        const sMax = p.sl_max_factor * d;
        uu = L.spacing_mm / sMax;
        out.push(check({ id: "ec2.links_spacing", title: "Maximum link spacing", clause: "EN 1992-1-1 9.2.2(6)", member: member.id,
          status: statusU(uu), utilisation: uu, demand: L.spacing_mm, capacity: sMax, unit: "mm",
          steps: [step("s_{l,max}", "0.75·d·(1 + cot α), α = 90°", `0.75·${fmt(d)}`, sMax, "mm", "Eq. 9.6N")] }));
      } else out.push(check({ id: "ec2.links_min", title: "Minimum shear reinforcement", clause: "EN 1992-1-1 9.2.2(5)", member: member.id,
        status: FAIL, notes: ["Beams need minimum shear reinforcement; no links are defined."] }));
    } else {
      const NEd = maxOf(f.uls_N.map((r) => r.map((v) => -v)));
      const req2 = Math.max(p.column_As_min_N_factor * Math.max(NEd, 0) * 1e3 / st.fyd, p.column_As_min_Ac_ratio * Ac);
      const uu = AsTot > 0 ? req2 / AsTot : Infinity;
      out.push(check({ id: "ec2.column_as_min", title: "Minimum column reinforcement", clause: "EN 1992-1-1 9.5.2(2)", member: member.id,
        status: statusU(uu), utilisation: uu, demand: req2, capacity: AsTot, unit: "mm²",
        steps: [step("N_{Ed}", "maximum compression", "", NEd, "kN"),
          step("A_{s,min}", "max(0.10·N_{Ed}/f_{yd}, 0.002·A_{c})", `max(0.10·${fmt(NEd * 1e3)}/${fmt(st.fyd)}, 0.002·${fmt(Ac)})`, req2, "mm²", "Eq. 9.12N")] }));
      const dias = [sec.top, sec.bottom, ...sec.extra_layers].filter((l) => l && l.n > 0).map((l) => l.dia_mm);
      if (dias.length) {
        const phi = Math.min(...dias);
        let uu2 = p.column_phi_min_mm / phi;
        out.push(check({ id: "ec2.column_phi_min", title: "Minimum bar diameter in columns", clause: "EN 1992-1-1 9.5.2(1)", member: member.id,
          status: statusU(uu2), utilisation: uu2, demand: p.column_phi_min_mm, capacity: phi, unit: "mm",
          steps: [step("φ_{min}", "recommended value", "", p.column_phi_min_mm, "mm")] }));
        if (sec.links) {
          const sMax = Math.min(p.column_link_spacing_phi_factor * phi, Math.min(b, h), p.column_link_spacing_max_mm);
          uu2 = sec.links.spacing_mm / sMax;
          out.push(check({ id: "ec2.column_links", title: "Maximum link spacing in columns", clause: "EN 1992-1-1 9.5.3(3)", member: member.id,
            status: statusU(uu2), utilisation: uu2, demand: sec.links.spacing_mm, capacity: sMax, unit: "mm",
            steps: [step("s_{cl,tmax}", "min(20·φ_{l,min}, b, 400 mm)", `min(20·${fmt(phi)}, ${fmt(Math.min(b, h))}, 400)`, sMax, "mm")] }));
        } else out.push(check({ id: "ec2.column_links", title: "Column links", clause: "EN 1992-1-1 9.5.3", member: member.id, status: FAIL,
          notes: ["Columns need transverse reinforcement; no links are defined."] }));
      }
    }
    const dg = sec.concrete.dmax_mm, link = sec.links ? sec.links.dia_mm : 0;
    for (const [face, lay] of [["bottom", sec.bottom], ["top", sec.top]]) {
      if (!lay || lay.n < 2) continue;
      const clear = (b - 2 * sec.cover_mm - 2 * link - lay.n * lay.dia_mm) / (lay.n - 1);
      const req2 = Math.max(p.k1_spacing * lay.dia_mm, dg + p.k2_spacing_mm, 20);
      const uu = clear > 0 ? req2 / clear : Infinity;
      out.push(check({ id: `ec2.bar_spacing.${face}`, title: `Clear bar spacing, ${face}`, clause: "EN 1992-1-1 8.2(2)", member: member.id,
        status: statusU(uu), utilisation: uu, demand: req2, capacity: clear, unit: "mm",
        steps: [step("a", "(b − 2·c − 2·φ_{w} − n·φ)/(n − 1)", `(${fmt(b)} − 2·${fmt(sec.cover_mm)} − 2·${fmt(link)} − ${lay.n}·${fmt(lay.dia_mm)})/${lay.n - 1}`, clear, "mm"),
          step("a_{min}", "max(k_{1}·φ, d_{g} + k_{2}, 20 mm)", `max(${fmt(lay.dia_mm)}, ${fmt(dg)} + 5, 20)`, req2, "mm")],
        notes: ["Assumes one layer of bars across the width."] }));
    }
    return out;
  }

  function checkSlenderness(member, rc, f, p) {
    const l0 = member.design.l0_factor * f.L * 1000, i = rc.h / Math.sqrt(12), lam = l0 / i;
    const NEd = maxOf(f.uls_N.map((r) => r.map((v) => -v)));
    const n = Math.max(NEd, 1e-9) * 1e3 / (rc.Ac * rc.c.fcd);
    const lamLim = 20 * 0.7 * 1.1 * 0.7 / Math.sqrt(n);
    const steps = [
      step("l_{0}", "effective length = factor·L", `${fmt(member.design.l0_factor)}·${fmt(f.L * 1000)}`, l0, "mm", "5.8.3.2"),
      step("i", "h/√12 (bending in the frame plane)", `${fmt(rc.h)}/√12`, i, "mm"),
      step("λ", "l_{0}/i", `${fmt(l0)}/${fmt(i)}`, lam, "", "5.8.3.2(1)"),
      step("n", "N_{Ed}/(A_{c}·f_{cd})", `${fmt(NEd * 1e3)}/(${fmt(rc.Ac)}·${fmt(rc.c.fcd)})`, n, "", "5.8.3.1(1)"),
      step("λ_{lim}", "20·A·B·C/√n with A = 0.7, B = 1.1, C = 0.7", `20·0.7·1.1·0.7/√${fmt(n)}`, lamLim, "", "Eq. 5.13N"),
    ];
    if (lam <= lamLim) return check({ id: "ec2.slenderness", title: "Slenderness (second-order effects)", clause: "EN 1992-1-1 5.8.3.1", member: member.id,
      status: PASS, utilisation: lam / lamLim, demand: lam, capacity: lamLim, steps, notes: ["λ ≤ λ_lim: second-order effects may be ignored."] });
    return check({ id: "ec2.slenderness", title: "Slenderness (second-order effects)", clause: "EN 1992-1-1 5.8.3.1", member: member.id,
      status: NOT_COVERED, utilisation: lam / lamLim, demand: lam, capacity: lamLim, steps,
      notes: ["λ > λ_lim: second-order effects must be considered (5.8.5). This version does not compute them; verify this column by another method."] });
  }

  function checkSpanDepth(member, rc, sec, f, p) {
    const system = member.design.deflection_system, K = K_SYSTEM[system], fck = rc.c.fck;
    const tf = system === "cantilever" ? "top" : "bottom", cf = tf === "top" ? "bottom" : "top";
    const t = sec[tf], cmp = sec[cf];
    if (!t || t.n === 0) return check({ id: "ec2.span_depth", title: "Span/depth ratio", clause: "EN 1992-1-1 7.4.2", member: member.id, status: FAIL,
      notes: ["No tension reinforcement on the relevant face."] });
    const d = tf === "bottom" ? layerDepth(sec, t, tf) : rc.h - layerDepth(sec, t, tf);
    const rho = barArea(t.n, t.dia_mm) / (rc.b * d);
    let rhoC = (cmp ? barArea(cmp.n, cmp.dia_mm) : 0) / (rc.b * d);
    const rho0 = Math.sqrt(fck) * 1e-3, sf = Math.sqrt(fck);
    let lim, expr, eq;
    if (rho <= rho0) {
      lim = K * (11 + 1.5 * sf * rho0 / rho + 3.2 * sf * Math.pow(rho0 / rho - 1, 1.5));
      expr = "K·[11 + 1.5·√f_{ck}·ρ_{0}/ρ + 3.2·√f_{ck}·(ρ_{0}/ρ − 1)^{3/2}]"; eq = "Eq. 7.16a";
    } else {
      const used = rhoC < 0.5 * rho ? rhoC : 0;
      lim = K * (11 + 1.5 * sf * rho0 / (rho - used) + sf / 12 * Math.sqrt(used / rho0));
      expr = "K·[11 + 1.5·√f_{ck}·ρ_{0}/(ρ − ρ') + (1/12)·√f_{ck}·√(ρ'/ρ_{0})]"; eq = "Eq. 7.16b";
      if (used !== rhoC) { rhoC = 0; expr += ", ρ' taken as 0 (conservative; the equation is singular as ρ' → ρ)"; }
    }
    const actual = f.L * 1000 / d, u = actual / lim;
    const steps = [
      step("K", `structural system: ${system.replace(/_/g, " ")}`, "", K, "", "Table 7.4N"),
      step("ρ_{0}", "√f_{ck}·10^{-3}", `√${fmt(fck)}·10^{-3}`, rho0, "", "7.4.2(2)"),
      step("ρ", `A_{s}/(b·d), ${tf} bars as provided (conservative)`, "", rho, ""),
      step("ρ'", "compression reinforcement ratio", "", rhoC, ""),
      step("(l/d)_{lim}", expr, "", lim, "", eq),
      step("l/d", "span/effective depth", `${fmt(f.L * 1000)}/${fmt(d)}`, actual, ""),
    ];
    return check({ id: "ec2.span_depth", title: "Span/depth ratio (deemed to satisfy)", clause: "EN 1992-1-1 7.4.2", member: member.id,
      status: statusU(u), utilisation: u, demand: actual, capacity: lim, steps,
      notes: ["Uses the provided reinforcement and omits the 310/σ_s factor, both conservative."] });
  }

  function deflectionCurves(rc, sec, f, phi, p, cantilever) {
    const c = rc.c, Eeff = c.Ecm / (1 + phi);
    const sag = rc.transformed(Eeff, true), hog = rc.transformed(Eeff, false);
    const McrS = c.fctm * sag.I_I / (rc.h - sag.y_I) * 1e-6, McrH = c.fctm * hog.I_I / (rc.h - hog.y_I) * 1e-6;
    const n = f.x.length;
    const xr = f.x.map((x) => (cantilever && f.tip_at_i ? f.L - x : x));
    const order = xr.map((_, k) => k).sort((a, b) => xr[a] - xr[b]);
    const xs = order.map((k) => xr[k]);
    const zeta = [], w = [];
    for (const row of f.qp_M) {
      const z = zeros(n), kap = zeros(n);
      row.forEach((M, k) => {
        const pos = M >= 0;
        const I1 = pos ? sag.I_I : hog.I_I, I2 = pos ? sag.I_II : hog.I_II, Mcr = pos ? McrS : McrH;
        const aM = Math.abs(M);
        z[k] = aM > Mcr ? 1 - p.beta_sustained * (Mcr / aM) ** 2 : 0;
        const k1 = M * 1e6 / (Eeff * I1) * 1e3, k2 = M * 1e6 / (Eeff * I2) * 1e3;
        kap[k] = z[k] * k2 + (1 - z[k]) * k1;
      });
      const kk = order.map((k) => kap[k]);
      const F1 = zeros(n), G = zeros(n);
      for (let q = 1; q < n; q++) {
        const dx = xs[q] - xs[q - 1];
        F1[q] = F1[q - 1] + 0.5 * dx * (kk[q] + kk[q - 1]);
        G[q] = G[q - 1] + 0.5 * dx * (kk[q] * xs[q] + kk[q - 1] * xs[q - 1]);
      }
      let ws;
      if (cantilever) ws = xs.map((x, q) => x * F1[q] - G[q]);
      else { const theta = -(xs[n - 1] * F1[n - 1] - G[n - 1]) / xs[n - 1]; ws = xs.map((x, q) => theta * x + x * F1[q] - G[q]); }
      const back = zeros(n); order.forEach((k, q) => { back[k] = ws[q]; });
      w.push(back); zeta.push(z);
    }
    return { w, Eeff, sag, hog, Mcr_sag: McrS, Mcr_hog: McrH, zeta };
  }
  function checkDeflection(member, rc, sec, f, p) {
    const phi = member.design.creep_coefficient, cant = member.design.deflection_system === "cantilever";
    const r = deflectionCurves(rc, sec, f, phi, p, cant);
    const W = r.w.map((row) => row.map((v) => Math.abs(v) * 1000));
    const [ci, si, wmax] = argmax2(W, (i, j) => W[i][j]);
    const lim = f.L * 1000 / p.deflection_limit_divisor, u = wmax / lim, sag = r.sag;
    const steps = [
      step("E_{c,eff}", "E_{cm}/(1 + φ(∞,t_{0}))", `${fmt(rc.c.Ecm)}/(1 + ${fmt(phi)})`, r.Eeff, "MPa", "Eq. 7.20"),
      step("α_{e}", "E_{s}/E_{c,eff}", "", sag.alpha_e, ""),
      step("I_{I}", "uncracked transformed section, sagging", "", sag.I_I * 1e-6, "×10⁶ mm⁴"),
      step("I_{II}", "fully cracked transformed section, sagging", "", sag.I_II * 1e-6, "×10⁶ mm⁴"),
      step("M_{cr}", "f_{ctm}·I_{I}/(h − y_{I}), sagging", "", r.Mcr_sag, "kNm", "7.4.3(3)"),
      step("ζ", "1 − β·(M_{cr}/M_{qp})^{2}, β = 0.5, at the governing station", "", r.zeta[ci][si], "", "Eq. 7.19"),
      step("w_{max}", "∫ κ·m̄ dx with κ = ζ·κ_{II} + (1 − ζ)·κ_{I}" + (cant ? " (tip, relative to the root tangent)" : " (relative to the member chord)"), "", wmax, "mm", "Eq. 7.18"),
      step("w_{lim}", "span/250", `${fmt(f.L * 1000)}/250`, lim, "mm", "7.4.1(4)"),
      step("η", "w_{max}/w_{lim}", `${fmt(wmax)}/${fmt(lim)}`, u, ""),
    ];
    return check({ id: "ec2.deflection", title: "Long-term deflection (quasi-permanent)", clause: "EN 1992-1-1 7.4.3", member: member.id,
      status: statusU(u), utilisation: u, demand: wmax, capacity: lim, unit: "mm", x: f.x[si], combination: f.qp_ids[ci], combination_label: f.qp_labels[ci], steps,
      notes: ["Moments from the linear-elastic analysis (no redistribution from cracking).", "Shrinkage curvature (7.4.3(6)) is not included; add a margin or check it separately."] });
  }
  function checkMemberEC2(member, sec, rc, f, p) {
    if (member.design.type === "none") return [];
    const out = [checkBendingAxial(member, rc, f, p), checkShear(member, rc, sec, f, p)].concat(checkDetailing(member, rc, sec, f, p));
    if (member.design.type === "column") out.push(checkSlenderness(member, rc, f, p));
    else {
      const ld = checkSpanDepth(member, rc, sec, f, p), defl = checkDeflection(member, rc, sec, f, p);
      if (ld.status === FAIL && defl.status === PASS) { ld.status = INFO; ld.notes.push("Limit exceeded, so the calculated deflection (7.4.3) governs; it passes."); }
      out.push(ld, defl);
    }
    return out;
  }
  function checkMemberSIA(member) {
    if (member.design.type === "none") return [];
    return [check({ id: "sia262.pending", title: "SIA 262 checks", clause: "SIA 262:2013", member: member.id, status: NOT_COVERED,
      notes: ["SIA 262 rules are not implemented in this version. Run the model with design_code 'EC2' or verify the member by another method."] })];
  }

  // ---------------------------------------------------------------- project
  function combine(F, A) { // F: combos x cases, A: cases x stations
    return F.map((fr) => A[0].map((_, j) => fr.reduce((s, fv, c) => s + fv * A[c][j], 0)));
  }
  function runProject(data) {
    const t0 = (typeof performance !== "undefined" ? performance : Date).now();
    const model = parseModel(data);
    const fa = new FrameAnalysis(model);
    const cases = {};
    for (const [cid, lc] of Object.entries(model.load_cases)) cases[cid] = fa.solveCase(lc);
    const cs = generateCombinations(model);
    const params = makeParams(model.settings.params);
    const warnings = [];
    for (const d of fa.autoFixed) warnings.push(`Node '${fa.nodeIds[Math.floor(d / 3)]}' is hinged on every member; its rotation was restrained automatically.`);
    const sel = (kind) => cs.combos.filter((c) => c.kind === kind);
    const uls = sel("ULS"), qp = sel("SLS_qp");
    const F = uls.map((c) => c.factors), Fq = qp.map((c) => c.factors);
    const envelopes = {}, member_forces = {};
    for (const mid of Object.keys(model.members)) {
      const st = (q) => cs.caseIds.map((c) => cases[c].members[mid][q]);
      const N = combine(F, st("N")), V = combine(F, st("V")), M = combine(F, st("M"));
      member_forces[mid] = { N, V, M };
      const env = { x: fa.stations[mid].x, side: fa.stations[mid].side, max: {}, min: {}, max_combo: {}, min_combo: {} };
      for (const [q, C] of [["N", N], ["V", V], ["M", M]]) {
        env.max[q] = []; env.min[q] = []; env.max_combo[q] = []; env.min_combo[q] = [];
        C[0].forEach((_, j) => {
          let mx = -Infinity, mn = Infinity, ix = 0, im = 0;
          C.forEach((row, i) => { if (row[j] > mx) { mx = row[j]; ix = i; } if (row[j] < mn) { mn = row[j]; im = i; } });
          env.max[q].push(mx); env.min[q].push(mn); env.max_combo[q].push(uls[ix].id); env.min_combo[q].push(uls[im].id);
        });
      }
      envelopes[mid] = env;
    }
    const reactions = {};
    for (const nid of Object.keys(model.supports)) {
      const R = cs.caseIds.map((c) => cases[c].reactions[nid]);
      const RU = F.map((fr) => [0, 1, 2].map((q) => fr.reduce((s, fv, c) => s + fv * R[c][q], 0)));
      const per = Object.fromEntries(cs.caseIds.map((c, k) => [c, R[k]]));
      const mx = [0, 1, 2].map((q) => Math.max(...RU.map((r) => r[q]))), mn = [0, 1, 2].map((q) => Math.min(...RU.map((r) => r[q])));
      reactions[nid] = { per_case: per, uls_max: mx, uls_min: mn,
        uls_max_combo: [0, 1, 2].map((q) => uls[RU.findIndex((r) => r[q] === mx[q])].id),
        uls_min_combo: [0, 1, 2].map((q) => uls[RU.findIndex((r) => r[q] === mn[q])].id) };
    }
    const rcCache = {}, checks = {};
    for (const [mid, m] of Object.entries(model.members)) {
      const sec = model.sections[m.section];
      if (!rcCache[m.section]) rcCache[m.section] = new RCSection(sec, params);
      const md = fa.members[mid];
      let tipAtI = false;
      if (m.design.deflection_system === "cantilever") {
        const isFree = (nid) => !(nid in model.supports) && !Object.values(model.members).some((o) => o.id !== mid && (o.i === nid || o.j === nid));
        tipAtI = isFree(m.i) && !isFree(m.j);
      }
      const mf = member_forces[mid];
      const qM = combine(Fq, cs.caseIds.map((c) => cases[c].members[mid].M));
      const f = { x: fa.stations[mid].x, L: md.L, uls_N: mf.N, uls_V: mf.V, uls_M: mf.M, uls_ids: uls.map((c) => c.id), uls_labels: uls.map((c) => c.label),
        qp_M: qM, qp_ids: qp.map((c) => c.id), qp_labels: qp.map((c) => c.label), tip_at_i: tipAtI };
      checks[mid] = model.settings.design_code === "EC2" ? checkMemberEC2(m, sec, rcCache[m.section], f, params) : checkMemberSIA(m);
    }
    const all = Object.values(checks).flat();
    let gov = null, bu = -1;
    for (const c of all) {
      const u = c.utilisation != null ? c.utilisation : (c.status === FAIL ? Infinity : -1);
      if ((c.status === PASS || c.status === FAIL) && u > bu) { gov = c; bu = u; }
    }
    const sts = new Set(all.map((c) => c.status));
    const status = sts.has(FAIL) ? FAIL : (sts.has(NOT_COVERED) || sts.has("warn") ? NOT_COVERED : PASS);
    const t1 = (typeof performance !== "undefined" ? performance : Date).now();
    return { version: VERSION, model, analysis: fa, cases, combos: cs, params, envelopes, reactions, checks, warnings,
      summary: { status, max_utilisation: gov && gov.utilisation != null && isFinite(gov.utilisation) ? gov.utilisation : null,
        governing: gov ? { member: gov.member, check: gov.title, clause: gov.clause } : null, runtime_ms: t1 - t0 } };
  }

  const api = { VERSION, parseModel, runProject, FrameAnalysis, generateCombinations, RCSection, makeParams, concreteProps, steelProps,
    shearResistance, sectionLayers, PSI, CATEGORY_LABEL, ModelError, StructureUnstable, fmt, combine };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Tragwerk = api;
})(typeof window !== "undefined" ? window : globalThis);
