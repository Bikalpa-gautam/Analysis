/*
 * Tragwerk 3D engine for the browser: a port of tragwerk.model3d, solver3d,
 * codes/ec2/checks3d, project3d and building (Python). The test suite runs both
 * on the same models and requires agreement (tests/test_js_parity3d.py).
 * Load after engine.js; extends window.Tragwerk / module.exports.
 */
(function (root) {
  "use strict";
  const T = typeof module !== "undefined" && module.exports ? require("./engine.js") : root.Tragwerk;
  const I = T._internal;
  const { ModelError, StructureUnstable, req, num, layer, GP, GW, shape, solveSmall, zeros, fmt, step, check, statusU,
    PASS, FAIL, INFO, NOT_COVERED } = I;
  const SCHEMA3D = "tragwerk3d/0.1";
  const DIRS = ["global_x", "global_y", "global_z", "local_x", "local_y", "local_z"];

  // ---------------------------------------------------------------- vectors
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  function localAxes(ex, rollDeg = 0) {
    const Z = [0, 0, 1];
    const c = cross(Z, ex);
    let ey, ez;
    if (norm(c) < 1e-6) { ez = [1, 0, 0]; ey = cross(ez, ex); }
    else { ey = scale(c, 1 / norm(c)); ez = cross(ex, ey); }
    if (rollDeg) {
      const r = rollDeg * Math.PI / 180, cs = Math.cos(r), sn = Math.sin(r);
      const ny = [0, 1, 2].map((k) => cs * ey[k] + sn * ez[k]), nz = [0, 1, 2].map((k) => -sn * ey[k] + cs * ez[k]);
      ey = ny; ez = nz;
    }
    return [ex, scale(ey, 1 / norm(ey)), scale(ez, 1 / norm(ez))];
  }

  // ---------------------------------------------------------------- sections
  function sectionBars(sec) {
    const wl = sec.links ? sec.links.dia_mm : 0, out = [];
    const ylim = (d) => sec.b_mm / 2 - sec.cover_mm - wl - d / 2, zlim = (d) => sec.h_mm / 2 - sec.cover_mm - wl - d / 2;
    const row = (lay, z, label) => {
      const a = Math.PI * lay.dia_mm ** 2 / 4, yl = ylim(lay.dia_mm);
      for (let k = 0; k < lay.n; k++) out.push([a, lay.n === 1 ? 0 : -yl + 2 * yl * k / (lay.n - 1), z, label]);
    };
    let zt = null, zb = null;
    if (sec.top && sec.top.n > 0) { zt = zlim(sec.top.dia_mm); row(sec.top, zt, "top"); }
    if (sec.bottom && sec.bottom.n > 0) { zb = -zlim(sec.bottom.dia_mm); row(sec.bottom, zb, "bottom"); }
    if (sec.side && sec.side.n > 0) {
      const a = Math.PI * sec.side.dia_mm ** 2 / 4;
      const z1 = zt !== null ? zt : zlim(sec.side.dia_mm), z0 = zb !== null ? zb : -zlim(sec.side.dia_mm), ys = ylim(sec.side.dia_mm);
      for (let k = 1; k <= sec.side.n; k++) { const z = z0 + (z1 - z0) * k / (sec.side.n + 1); out.push([a, ys, z, "side"], [a, -ys, z, "side"]); }
    }
    return out;
  }
  function sectionView(sec, axis) {
    const groups = new Map();
    for (const [a, y, z] of sec.bars) {
      const depth = axis === "y" ? sec.h_mm / 2 - z : sec.b_mm / 2 - y;
      const key = Math.round(depth * 1e6) / 1e6;
      groups.set(key, (groups.get(key) || 0) + a);
    }
    const depths = [...groups.keys()].sort((p, q) => p - q);
    const layersList = depths.map((d, k) => [groups.get(d), d, `row ${k + 1}`]);
    let links = null;
    if (sec.links) {
      const legs = axis === "y" ? sec.links.legs : (sec.links_legs_y || sec.links.legs);
      links = Object.assign({}, sec.links, { legs, area_mm2: legs * Math.PI * sec.links.dia_mm ** 2 / 4 });
    }
    return axis === "y"
      ? { name: sec.name, b_mm: sec.b_mm, h_mm: sec.h_mm, concrete: sec.concrete, rebar: sec.rebar, layersList, links, cover_mm: sec.cover_mm }
      : { name: sec.name, b_mm: sec.h_mm, h_mm: sec.b_mm, concrete: sec.concrete, rebar: sec.rebar, layersList, links, cover_mm: sec.cover_mm };
  }
  function sectionAs2d(sec) {
    const extra = [];
    for (const [, y, z, lab] of sec.bars) if (lab === "side" && y > 0) extra.push({ n: 2, dia_mm: sec.side.dia_mm, depth_mm: sec.h_mm / 2 - z });
    return { name: sec.name, b_mm: sec.b_mm, h_mm: sec.h_mm, concrete: sec.concrete, rebar: sec.rebar, cover_mm: sec.cover_mm,
      bottom: sec.bottom, top: sec.top, links: sec.links, extra_layers: extra, stiffness_factor: sec.stiffness_factor };
  }

  // ---------------------------------------------------------------- parsing
  function parseModel3d(data) {
    if (!data || typeof data !== "object") throw new ModelError("model must be a JSON object");
    if (data.schema !== SCHEMA3D) throw new ModelError(`expected schema '${SCHEMA3D}'`);
    const materials = {};
    for (const [name, md] of Object.entries(req(data, "materials", "model"))) {
      const where = `materials.${name}`, kind = req(md, "type", where);
      if (kind === "concrete") {
        const fck = num(req(md, "fck", where), where + ".fck", { positive: true });
        if (fck < 12 || fck > 90) throw new ModelError(`${where}.fck: EN 1992-1-1 covers 12 to 90 MPa, got ${fck}`);
        materials[name] = { kind, name, fck, density: num(md.density ?? 25, where + ".density", { positive: true }), dmax_mm: num(md.dmax_mm ?? 16, where + ".dmax_mm", { positive: true }) };
      } else if (kind === "rebar") {
        const fyk = num(md.fyk ?? 500, where + ".fyk", { positive: true });
        if (fyk < 400 || fyk > 600) throw new ModelError(`${where}.fyk: EN 1992-1-1 covers 400 to 600 MPa, got ${fyk}`);
        materials[name] = { kind, name, fyk, Es: num(md.Es ?? 200000, where + ".Es", { positive: true }), eps_uk: num(md.eps_uk ?? 0.05, where + ".eps_uk", { positive: true }) };
      } else throw new ModelError(`${where}.type: expected 'concrete' or 'rebar', got ${JSON.stringify(kind)}`);
    }
    const sections = {};
    for (const [name, sd] of Object.entries(req(data, "sections", "model"))) {
      const where = `sections.${name}`;
      if ((sd.type || "rc_rect") !== "rc_rect") throw new ModelError(`${where}.type: only 'rc_rect' is supported in this version`);
      const conc = materials[sd.concrete], reb = materials[sd.rebar];
      if (!conc || conc.kind !== "concrete") throw new ModelError(`${where}.concrete: '${sd.concrete}' is not a concrete material`);
      if (!reb || reb.kind !== "rebar") throw new ModelError(`${where}.rebar: '${sd.rebar}' is not a rebar material`);
      let links = null, legsY = null;
      if (sd.links) {
        const ld = sd.links;
        links = { dia_mm: num(req(ld, "dia_mm", where + ".links"), where + ".links.dia_mm", { positive: true }),
          legs: Math.trunc(num(ld.legs ?? 2, where + ".links.legs", { positive: true })),
          spacing_mm: num(req(ld, "spacing_mm", where + ".links"), where + ".links.spacing_mm", { positive: true }),
          fyk: ld.fyk == null ? null : num(ld.fyk, where + ".links.fyk", { positive: true }) };
        links.area_mm2 = links.legs * Math.PI * links.dia_mm ** 2 / 4;
        if (ld.legs_y != null) legsY = Math.trunc(num(ld.legs_y, where + ".links.legs_y", { positive: true }));
      }
      const sec = { name, b_mm: num(req(sd, "b_mm", where), where + ".b_mm", { positive: true }), h_mm: num(req(sd, "h_mm", where), where + ".h_mm", { positive: true }),
        concrete: conc, rebar: reb, cover_mm: num(req(sd, "cover_mm", where), where + ".cover_mm", { nonneg: true }),
        top: layer(sd.top, where + ".top"), bottom: layer(sd.bottom, where + ".bottom"), side: layer(sd.side, where + ".side"), links, links_legs_y: legsY,
        stiffness_factor: num(sd.stiffness_factor ?? 1, where + ".stiffness_factor", { positive: true }),
        torsion_factor: num(sd.torsion_factor ?? 1, where + ".torsion_factor", { positive: true }) };
      const wl = links ? links.dia_mm : 0;
      for (const [lab, lay] of [["top", sec.top], ["bottom", sec.bottom], ["side", sec.side]]) {
        if (lay && lay.n > 0) {
          const zl = sec.h_mm / 2 - sec.cover_mm - wl - lay.dia_mm / 2, yl = sec.b_mm / 2 - sec.cover_mm - wl - lay.dia_mm / 2;
          if (zl <= 0 || yl < 0 || (yl === 0 && (lay.n > 1 || lab === "side"))) throw new ModelError(`${where}: ${lab} bars lie outside the section; check cover and bar diameters`);
        }
      }
      sec.area_m2 = sec.b_mm * sec.h_mm * 1e-6;
      sec.Iy_m4 = sec.b_mm * sec.h_mm ** 3 / 12 * 1e-12;
      sec.Iz_m4 = sec.h_mm * sec.b_mm ** 3 / 12 * 1e-12;
      const A = Math.max(sec.b_mm, sec.h_mm), B = Math.min(sec.b_mm, sec.h_mm);
      sec.J_m4 = A * B ** 3 * (1 / 3 - 0.21 * (B / A) * (1 - B ** 4 / (12 * A ** 4))) * 1e-12;
      sec.bars = sectionBars(sec);
      sections[name] = sec;
    }
    const nodes = {};
    req(data, "nodes", "model").forEach((nd, k) => {
      const where = `nodes[${k}]`, id = String(req(nd, "id", where));
      if (id in nodes) throw new ModelError(`${where}: duplicate node id '${id}'`);
      nodes[id] = { id, x: num(req(nd, "x", where), where + ".x"), y: num(req(nd, "y", where), where + ".y"), z: num(nd.z ?? 0, where + ".z") };
    });
    if (Object.keys(nodes).length < 2) throw new ModelError("model needs at least two nodes");
    const members = {};
    req(data, "members", "model").forEach((md, k) => {
      const where = `members[${k}]`, id = String(req(md, "id", where));
      if (id in members) throw new ModelError(`${where}: duplicate member id '${id}'`);
      const i = String(req(md, "i", where)), j = String(req(md, "j", where));
      for (const [end, nid] of [["i", i], ["j", j]]) if (!(nid in nodes)) throw new ModelError(`${where}.${end}: unknown node '${nid}'`);
      const a = nodes[i], b = nodes[j], dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      if (Math.sqrt(dx * dx + dy * dy + dz * dz) < 1e-6) throw new ModelError(`${where}: member has zero length`);
      const sec = String(req(md, "section", where));
      if (!(sec in sections)) throw new ModelError(`${where}.section: unknown section '${sec}'`);
      const rel = md.releases || {}, dd = md.design || {};
      let type = dd.type;
      if (type == null) type = Math.abs(dz) > Math.hypot(dx, dy) * Math.tan(60 * Math.PI / 180) ? "column" : "beam";
      if (!I.DESIGN_TYPES.includes(type)) throw new ModelError(`${where}.design.type: expected one of ${I.DESIGN_TYPES.join(", ")}`);
      const system = dd.deflection_system || "simply_supported";
      if (!I.DEFLECTION_SYSTEMS.includes(system)) throw new ModelError(`${where}.design.deflection_system: expected one of ${I.DEFLECTION_SYSTEMS.join(", ")}`);
      const relp = (v, w) => {
        if (v == null || v === false) return { t: false, my: false, mz: false };
        if (v === true) return { t: false, my: true, mz: true };
        if (typeof v !== "object") throw new ModelError(`${w}: expected true/false or an object with 't', 'my', 'mz'`);
        return { t: !!v.t, my: !!v.my, mz: !!v.mz };
      };
      const l0 = dd.l0_factor ?? 1.0;
      members[id] = { id, i, j, section: sec, roll_deg: num(md.roll_deg ?? 0, where + ".roll_deg"),
        rel_i: relp(rel.i, where + ".releases.i"), rel_j: relp(rel.j, where + ".releases.j"),
        design: { type, deflection_system: system,
          l0_factor_y: num(dd.l0_factor_y ?? l0, where + ".design.l0_factor_y", { positive: true }),
          l0_factor_z: num(dd.l0_factor_z ?? l0, where + ".design.l0_factor_z", { positive: true }),
          creep_coefficient: num(dd.creep_coefficient ?? 2.0, where + ".design.creep_coefficient", { nonneg: true }) } };
    });
    if (!Object.keys(members).length) throw new ModelError("model needs at least one member");
    const supports = {};
    (data.supports || []).forEach((sd, k) => {
      const where = `supports[${k}]`, nid = String(req(sd, "node", where));
      if (!(nid in nodes)) throw new ModelError(`${where}.node: unknown node '${nid}'`);
      if (nid in supports) throw new ModelError(`${where}: node '${nid}' already has a support`);
      supports[nid] = { node: nid, fix: ["ux", "uy", "uz", "rx", "ry", "rz"].map((key) => !!sd[key]),
        springs: ["kx", "ky", "kz", "krx", "kry", "krz"].map((key) => num(sd[key] ?? 0, `${where}.${key}`, { nonneg: true })) };
    });
    if (!Object.keys(supports).length) throw new ModelError("model has no supports");
    const mlen = (mid) => { const m = members[mid], a = nodes[m.i], b = nodes[m.j]; return Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z); };
    const load_cases = {};
    req(data, "load_cases", "model").forEach((cd, k) => {
      const where = `load_cases[${k}]`, id = String(req(cd, "id", where));
      if (id in load_cases) throw new ModelError(`${where}: duplicate load case id '${id}'`);
      const type = req(cd, "type", where);
      if (type !== "permanent" && type !== "variable") throw new ModelError(`${where}.type: expected 'permanent' or 'variable'`);
      const cat = cd.category ?? null;
      if (type === "variable" && !I.VARIABLE_CATEGORIES.includes(cat)) throw new ModelError(`${where}.category: variable actions need one of ${I.VARIABLE_CATEGORIES.join(", ")}`);
      const nodal = [], dist = [], point = [];
      (cd.loads || []).forEach((ld, q) => {
        const lw = `${where}.loads[${q}]`, lt = req(ld, "type", lw);
        if (lt === "nodal") {
          const nid = String(req(ld, "node", lw));
          if (!(nid in nodes)) throw new ModelError(`${lw}.node: unknown node '${nid}'`);
          nodal.push({ node: nid, f: ["fx", "fy", "fz", "mx", "my", "mz"].map((key) => num(ld[key] ?? 0, `${lw}.${key}`)) });
          return;
        }
        const mid = String(req(ld, "member", lw));
        if (!(mid in members)) throw new ModelError(`${lw}.member: unknown member '${mid}'`);
        const L = mlen(mid), direction = ld.direction || "global_z";
        if (!DIRS.includes(direction)) throw new ModelError(`${lw}.direction: expected one of ${DIRS.join(", ")}`);
        if (lt === "udl") {
          let w1, w2;
          if ("w" in ld) w1 = w2 = num(ld.w, lw + ".w"); else { w1 = num(req(ld, "w1", lw), lw + ".w1"); w2 = num(req(ld, "w2", lw), lw + ".w2"); }
          const a = num(ld.a ?? 0, lw + ".a", { nonneg: true }), b = num(ld.b ?? L, lw + ".b", { nonneg: true });
          if (!(0 <= a && a < b && b <= L + 1e-9)) throw new ModelError(`${lw}: need 0 <= a < b <= member length (${L.toFixed(3)} m)`);
          dist.push({ member: mid, w1, w2, a, b: Math.min(b, L), direction });
        } else if (lt === "point") {
          const a = num(req(ld, "a", lw), lw + ".a", { nonneg: true });
          if (a > L + 1e-9) throw new ModelError(`${lw}.a: position ${a} m is beyond the member length (${L.toFixed(3)} m)`);
          point.push({ member: mid, p: num(req(ld, "p", lw), lw + ".p"), a: Math.min(a, L), direction });
        } else throw new ModelError(`${lw}.type: expected 'nodal', 'udl' or 'point', got ${JSON.stringify(lt)}`);
      });
      load_cases[id] = { id, type, name: String(cd.name ?? id), category: cat, group: cd.group ?? null, self_weight: !!cd.self_weight, nodal, distributed: dist, point };
    });
    if (!Object.keys(load_cases).length) throw new ModelError("model has no load cases");
    if (!Object.values(load_cases).some((lc) => lc.type === "permanent")) throw new ModelError("model needs at least one permanent load case (self weight counts)");
    const sd = data.design || {};
    const rule = sd.combination_rule || "6.10";
    if (rule !== "6.10" && rule !== "6.10ab") throw new ModelError("design.combination_rule: expected '6.10' or '6.10ab'");
    const code = sd.design_code || "EC2";
    if (code !== "EC2" && code !== "SIA262") throw new ModelError("design.design_code: expected 'EC2' or 'SIA262'");
    const settings = { design_code: code, combination_rule: rule, params: Object.assign({}, sd.params || {}),
      stations_per_member: Math.trunc(num(sd.stations_per_member ?? 20, "design.stations_per_member", { positive: true })) };
    const model = { is3d: true, name: String(data.name ?? "Untitled model"), nodes, members, sections, supports, load_cases, settings, raw: data };
    model.axes = (mid) => {
      const m = members[mid], a = nodes[m.i], b = nodes[m.j];
      const d = [b.x - a.x, b.y - a.y, b.z - a.z], L = norm(d);
      return [L, localAxes(scale(d, 1 / L), m.roll_deg)];
    };
    return model;
  }

  // ---------------------------------------------------------------- solver
  const NU = 0.2;
  const modulus = (sec) => 22000 * Math.pow((sec.concrete.fck + 8) / 10, 0.3) * 1000;
  const V_IDX = [1, 5, 7, 11], W_IDX = [2, 4, 8, 10], W_SIGN = [1, -1, 1, -1];
  function planeK(EI, L) {
    const b = EI / L ** 3;
    return [[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]].map((r) => r.map((v) => v * b));
  }
  function beamK3d(EA, GJ, EIy, EIz, L) {
    const k = Array.from({ length: 12 }, () => zeros(12));
    const a = EA / L, t = GJ / L;
    k[0][0] = a; k[0][6] = -a; k[6][0] = -a; k[6][6] = a;
    k[3][3] = t; k[3][9] = -t; k[9][3] = -t; k[9][9] = t;
    const kv = planeK(EIz, L), kw = planeK(EIy, L);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      k[V_IDX[r]][V_IDX[c]] = kv[r][c];
      k[W_IDX[r]][W_IDX[c]] = W_SIGN[r] * kw[r][c] * W_SIGN[c];
    }
    return k;
  }
  function loc(direction, w, lam) {
    if (direction.startsWith("local_")) { const v = [0, 0, 0]; v["xyz".indexOf(direction.slice(-1))] = w; return v; }
    const g = [0, 0, 0]; g["xyz".indexOf(direction.slice(-1))] = w;
    return lam.map((row) => row[0] * g[0] + row[1] * g[1] + row[2] * g[2]);
  }
  function eqLoads3d(L, loads) {
    const p = zeros(12);
    for (const [a, b, q1, q2] of loads.dist) {
      for (let g = 0; g < 3; g++) {
        const s = a + (b - a) * GP[g], w = (b - a) * GW[g], t = (s - a) / (b - a);
        const q = [0, 1, 2].map((k) => q1[k] + (q2[k] - q1[k]) * t);
        const N = shape(s / L, L);
        p[0] += N[0] * w * q[0]; p[6] += N[3] * w * q[0];
        const Nv = [N[1], N[2], N[4], N[5]];
        for (let r = 0; r < 4; r++) { p[V_IDX[r]] += Nv[r] * w * q[1]; p[W_IDX[r]] += W_SIGN[r] * Nv[r] * w * q[2]; }
      }
    }
    for (const [a, P] of loads.point) {
      const N = shape(a / L, L);
      p[0] += N[0] * P[0]; p[6] += N[3] * P[0];
      const Nv = [N[1], N[2], N[4], N[5]];
      for (let r = 0; r < 4; r++) { p[V_IDX[r]] += Nv[r] * P[1]; p[W_IDX[r]] += W_SIGN[r] * Nv[r] * P[2]; }
    }
    return p;
  }
  function cum3d(x, loads, side) {
    const S = [0, 0, 0];
    let My = 0, Mz = 0;
    for (const [a, b, q1, q2] of loads.dist) {
      if (x <= a) continue;
      const t = Math.min(x, b);
      const F = (s, al, be) => al * x * s + (be * x - al) * s * s / 2 - be * s ** 3 / 3;
      for (let k = 0; k < 3; k++) {
        const be = (q2[k] - q1[k]) / (b - a), al = q1[k] - be * a;
        S[k] += al * (t - a) + be * (t * t - a * a) / 2;
        if (k === 1) Mz += F(t, al, be) - F(a, al, be);
        if (k === 2) My += F(t, al, be) - F(a, al, be);
      }
    }
    for (const [a, P] of loads.point) {
      if (a < x || (side > 0 && a <= x)) { S[0] += P[0]; S[1] += P[1]; S[2] += P[2]; Mz += (x - a) * P[1]; My += (x - a) * P[2]; }
    }
    return [S, Mz, My];
  }
  function condense(k, cidx, n) {
    if (!cidx.length) return k.map((r) => r.slice());
    const aidx = [...Array(n).keys()].filter((i) => !cidx.includes(i));
    const kcc = cidx.map((r) => cidx.map((c) => k[r][c]));
    const kk = Array.from({ length: n }, () => zeros(n));
    for (const c of aidx) {
      const y = solveSmall(kcc, cidx.map((q) => k[q][c]));
      for (const r of aidx) kk[r][c] = k[r][c] - cidx.reduce((s, q, m) => s + k[r][q] * y[m], 0);
    }
    return kk;
  }
  function condenseLoad(k, cidx, p, n) {
    if (!cidx.length) return p.slice();
    const aidx = [...Array(n).keys()].filter((i) => !cidx.includes(i));
    const y = solveSmall(cidx.map((r) => cidx.map((c) => k[r][c])), cidx.map((c) => p[c]));
    const out = zeros(n);
    for (const r of aidx) out[r] = p[r] - cidx.reduce((s, c, m) => s + k[r][c] * y[m], 0);
    return out;
  }

  class FrameAnalysis3D {
    constructor(model) {
      this.model = model;
      this.nodeIds = Object.keys(model.nodes);
      this.nidx = Object.fromEntries(this.nodeIds.map((n, k) => [n, k]));
      const n = this.nodeIds.length;
      this.ndof = 6 * n;
      // sparse-ish global stiffness: map of rows
      const K = Array.from({ length: this.ndof }, () => new Map());
      const add = (r, c, v) => K[r].set(c, (K[r].get(c) || 0) + v);
      const connected = new Array(n).fill(false);
      this.members = {};
      for (const [mid, m] of Object.entries(model.members)) {
        const sec = model.sections[m.section];
        const [L, lam] = model.axes(mid);
        const E = modulus(sec) * sec.stiffness_factor, G = E / (2 * (1 + NU));
        const cidx = [];
        for (const [base, rel] of [[0, m.rel_i], [6, m.rel_j]]) { if (rel.t) cidx.push(base + 3); if (rel.my) cidx.push(base + 4); if (rel.mz) cidx.push(base + 5); }
        const kf = beamK3d(E * sec.area_m2, G * sec.J_m4 * sec.torsion_factor, E * sec.Iy_m4, E * sec.Iz_m4, L);
        const kl = condense(kf, cidx, 12);
        const md = { id: mid, ni: this.nidx[m.i], nj: this.nidx[m.j], L, lam, EA: E * sec.area_m2, GJ: G * sec.J_m4 * sec.torsion_factor,
          EIy: E * sec.Iy_m4, EIz: E * sec.Iz_m4, cidx, k_full: kf, k_local: kl };
        // global k = T^T k T with T = blockdiag(lam x4)
        const dofs = this.dofs(md);
        const kg = this.toGlobal(lam, kl);
        for (let r = 0; r < 12; r++) for (let c = 0; c < 12; c++) if (kg[r][c] !== 0) add(dofs[r], dofs[c], kg[r][c]);
        connected[md.ni] = connected[md.nj] = true;
        this.members[mid] = md;
      }
      connected.forEach((ok, k) => { if (!ok) throw new StructureUnstable(`node '${this.nodeIds[k]}' is not connected to any member`); });
      const fixed = new Array(this.ndof).fill(false);
      this.springs = {};
      for (const [nid, sp] of Object.entries(model.supports)) {
        const b = 6 * this.nidx[nid];
        for (let off = 0; off < 6; off++) {
          if (sp.fix[off]) fixed[b + off] = true;
          else if (sp.springs[off] > 0) { add(b + off, b + off, sp.springs[off]); this.springs[b + off] = sp.springs[off]; }
        }
      }
      let sc = 1;
      for (let d = 0; d < this.ndof; d++) sc = Math.max(sc, Math.abs(K[d].get(d) || 0));
      this.autoFixed = [];
      for (let k = 0; k < n; k++) for (const off of [3, 4, 5]) {
        const d = 6 * k + off;
        if (!fixed[d] && Math.abs(K[d].get(d) || 0) < 1e-12 * sc) { fixed[d] = true; this.autoFixed.push(d); }
      }
      this.K = K; this.fixed = fixed;
      this.free = []; for (let d = 0; d < this.ndof; d++) if (!fixed[d]) this.free.push(d);
      this.factorise();
      this.stations = this.buildStations();
    }
    dofs(md) { const out = []; for (let k = 0; k < 6; k++) out.push(6 * md.ni + k); for (let k = 0; k < 6; k++) out.push(6 * md.nj + k); return out; }
    toGlobal(lam, k) {
      const T = (r, c) => (Math.floor(r / 3) === Math.floor(c / 3) ? lam[r % 3][c % 3] : 0);
      const kT = Array.from({ length: 12 }, () => zeros(12)), out = Array.from({ length: 12 }, () => zeros(12));
      for (let r = 0; r < 12; r++) for (let c = 0; c < 12; c++) {
        let s = 0; const b = Math.floor(c / 3) * 3;
        for (let q = b; q < b + 3; q++) s += k[r][q] * T(q, c);
        kT[r][c] = s;
      }
      for (let r = 0; r < 12; r++) for (let c = 0; c < 12; c++) {
        let s = 0; const b = Math.floor(r / 3) * 3;
        for (let q = b; q < b + 3; q++) s += T(q, r) * kT[q][c];
        out[r][c] = s;
      }
      return out;
    }
    factorise() {
      // skyline (variable band) Cholesky on the free DOFs, rows stored from their first non-zero column
      const f = this.free, m = f.length, pos = new Int32Array(this.ndof).fill(-1);
      f.forEach((d, k) => { pos[d] = k; });
      const first = new Int32Array(m);
      for (let i = 0; i < m; i++) {
        let fc = i;
        for (const c of this.K[f[i]].keys()) { const j = pos[c]; if (j >= 0 && j < fc) fc = j; }
        first[i] = fc;
      }
      const rows = [], diag = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        const fi = first[i], row = new Float64Array(i - fi + 1);
        for (const [c, v] of this.K[f[i]]) { const j = pos[c]; if (j >= fi && j <= i) row[j - fi] = v; }
        for (let j = fi; j < i; j++) {
          const rj = rows[j], fj = first[j];
          const lo = Math.max(fi, fj);
          let s = row[j - fi];
          for (let k = lo; k < j; k++) s -= row[k - fi] * rj[k - fj];
          row[j - fi] = s / rj[j - fj];
        }
        let d = row[i - fi];
        for (let k = fi; k < i; k++) d -= row[k - fi] * row[k - fi];
        if (!(d > 0)) throw new StructureUnstable(this.mechanismMessage(f[i]));
        row[i - fi] = Math.sqrt(d);
        diag[i] = d;
        rows.push(row);
      }
      if (m) {
        let jmin = 0, dmax = 0;
        for (let j = 0; j < m; j++) { if (diag[j] < diag[jmin]) jmin = j; dmax = Math.max(dmax, diag[j]); }
        if (diag[jmin] < 1e-13 * dmax) throw new StructureUnstable(this.mechanismMessage(f[jmin]));
      }
      this.L = rows; this.first = first;
    }
    mechanismMessage(dof) {
      const names = ["move in X", "move in Y", "move in Z", "rotate about X", "rotate about Y", "rotate about Z"];
      return `structure is unstable (mechanism): node '${this.nodeIds[Math.floor(dof / 6)]}' is free to ${names[dof % 6]}. Check supports and member releases.`;
    }
    solve(rhs) {
      const L = this.L, fs = this.first, m = rhs.length, y = new Float64Array(m), x = new Float64Array(m);
      for (let i = 0; i < m; i++) { let s = rhs[i]; const fi = fs[i], r = L[i]; for (let k = fi; k < i; k++) s -= r[k - fi] * y[k]; y[i] = s / r[i - fi]; }
      for (let i = m - 1; i >= 0; i--) x[i] = y[i];
      for (let i = m - 1; i >= 0; i--) {
        const fi = fs[i], r = L[i];
        x[i] = x[i] / r[i - fi];
        for (let k = fi; k < i; k++) x[k] -= r[k - fi] * x[i];
      }
      return Array.from(x);
    }
    localLoads(lc) {
      const out = {};
      for (const [mid, md] of Object.entries(this.members)) {
        out[mid] = { dist: [], point: [] };
        if (lc.self_weight) {
          const sec = this.model.sections[this.model.members[mid].section];
          const q = loc("global_z", -sec.concrete.density * sec.area_m2, md.lam);
          out[mid].dist.push([0, md.L, q, q.slice()]);
        }
      }
      for (const ld of lc.distributed) { const md = this.members[ld.member]; out[ld.member].dist.push([ld.a, ld.b, loc(ld.direction, ld.w1, md.lam), loc(ld.direction, ld.w2, md.lam)]); }
      for (const ld of lc.point) { const md = this.members[ld.member]; out[ld.member].point.push([ld.a, loc(ld.direction, ld.p, md.lam)]); }
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
        for (const b of Array.from(new Set(breaks[mid])).sort((p, q) => p - q)) if (b > 1e-9 && b < md.L - 1e-9 && !inner.some((c) => Math.abs(b - c) < 1e-9)) inner.push(b);
        const grid = [];
        for (let i = 0; i <= nseg; i++) { const x = i === nseg ? md.L : (md.L / nseg) * i; if (!inner.some((b) => Math.abs(x - b) < 1e-9)) grid.push(x); }
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
      for (const ld of lc.nodal) { const b = 6 * this.nidx[ld.node]; for (let k = 0; k < 6; k++) P[b + k] += ld.f[k]; }
      for (const d of this.autoFixed) if (Math.abs(P[d]) > 1e-12) throw new StructureUnstable(`a moment is applied at node '${this.nodeIds[Math.floor(d / 6)]}' about a released axis`);
      const loads = this.localLoads(lc), peq = {};
      for (const [mid, md] of Object.entries(this.members)) {
        const p = eqLoads3d(md.L, loads[mid]);
        peq[mid] = p;
        const pc = condenseLoad(md.k_full, md.cidx, p, 12), dofs = this.dofs(md);
        for (let b = 0; b < 4; b++) for (let r = 0; r < 3; r++) {
          let s = 0; for (let q = 0; q < 3; q++) s += md.lam[q][r] * pc[3 * b + q];
          P[dofs[3 * b + r]] += s;
        }
      }
      const U = zeros(this.ndof);
      if (this.free.length) { const uf = this.solve(this.free.map((d) => P[d])); this.free.forEach((d, k) => { U[d] = uf[k]; }); }
      const KU = (d) => { let s = 0; for (const [c, v] of this.K[d]) s += v * U[c]; return s; };
      const reactions = {}, springs = {};
      for (const nid of Object.keys(this.model.supports)) {
        const b = 6 * this.nidx[nid];
        reactions[nid] = [0, 1, 2, 3, 4, 5].map((k) => (this.fixed[b + k] ? KU(b + k) - P[b + k] : 0));
        const sf = [0, 1, 2, 3, 4, 5].map((k) => -(this.springs[b + k] || 0) * U[b + k]);
        if (sf.some((v) => v !== 0)) springs[nid] = sf;
      }
      const members = {};
      for (const [mid, md] of Object.entries(this.members)) members[mid] = this.memberResult(md, U, loads[mid], peq[mid]);
      const [eq, applied] = this.equilibrium(lc, loads, reactions, springs);
      const disp = []; for (let k = 0; k < n; k++) disp.push(U.slice(6 * k, 6 * k + 6));
      return { caseId: lc.id, displacements: disp, reactions, springs, members, equilibrium: eq, applied };
    }
    memberResult(md, U, loads, peq) {
      const dofs = this.dofs(md), ug = dofs.map((d) => U[d]), lam = md.lam;
      const ul = zeros(12);
      for (let b = 0; b < 4; b++) for (let r = 0; r < 3; r++) ul[3 * b + r] = lam[r][0] * ug[3 * b] + lam[r][1] * ug[3 * b + 1] + lam[r][2] * ug[3 * b + 2];
      const k = md.k_full;
      if (md.cidx.length) {
        const aidx = [...Array(12).keys()].filter((i) => !md.cidx.includes(i));
        const uc = solveSmall(md.cidx.map((r) => md.cidx.map((c) => k[r][c])), md.cidx.map((r) => peq[r] - aidx.reduce((s, c) => s + k[r][c] * ul[c], 0)));
        md.cidx.forEach((c, q) => { ul[c] = uc[q]; });
      }
      const f = k.map((row, r) => row.reduce((s, v, c) => s + v * ul[c], 0) - peq[r]);
      const { x: xs, side } = this.stations[md.id];
      const N = [], Vy = [], Vz = [], My = [], Mz = [], Tq = [];
      xs.forEach((x, q) => {
        const [S, Iz, Iy] = cum3d(x, loads, side[q]);
        N.push(-f[0] - S[0]); Vy.push(f[1] + S[1]); Vz.push(f[2] + S[2]);
        Mz.push(-f[5] + x * f[1] + Iz); My.push(f[4] + x * f[2] + Iy); Tq.push(-f[3]);
      });
      const ux = []; xs.forEach((x) => { if (!ux.length || x !== ux[ux.length - 1]) ux.push(x); });
      const nU = ux.length, F1v = zeros(nU), Gv = zeros(nU), F1w = zeros(nU), Gw = zeros(nU), E = zeros(nU);
      for (let q = 1; q < nU; q++) {
        const x0 = ux[q - 1], x1 = ux[q];
        let a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0;
        for (let g = 0; g < 3; g++) {
          const s = x0 + (x1 - x0) * GP[g], w = (x1 - x0) * GW[g];
          const [S, Iz, Iy] = cum3d(s, loads, 1);
          const kv = (-f[5] + s * f[1] + Iz) / md.EIz, kw = (f[4] + s * f[2] + Iy) / md.EIy, ea = (-f[0] - S[0]) / md.EA;
          a1 += w * kv; a2 += w * s * kv; a3 += w * kw; a4 += w * s * kw; a5 += w * ea;
        }
        F1v[q] = F1v[q - 1] + a1; Gv[q] = Gv[q - 1] + a2; F1w[q] = F1w[q - 1] + a3; Gw[q] = Gw[q - 1] + a4; E[q] = E[q - 1] + a5;
      }
      const L = md.L, last = nU - 1;
      const thv = (ul[7] - ul[1] - (L * F1v[last] - Gv[last])) / L, thw = (ul[8] - ul[2] - (L * F1w[last] - Gw[last])) / L;
      const pos = new Map(ux.map((x, q) => [x, q]));
      const v = xs.map((x) => { const q = pos.get(x); return ul[1] + thv * x + x * F1v[q] - Gv[q]; });
      const w = xs.map((x) => { const q = pos.get(x); return ul[2] + thw * x + x * F1w[q] - Gw[q]; });
      const u = xs.map((x) => ul[0] + E[pos.get(x)]);
      return { x: xs, side, N, Vy, Vz, T: Tq, My, Mz, u, v, w,
        v_rel: xs.map((x, q) => v[q] - (ul[1] + (ul[7] - ul[1]) * x / L)), w_rel: xs.map((x, q) => w[q] - (ul[2] + (ul[8] - ul[2]) * x / L)), end_forces: f };
    }
    equilibrium(lc, loads, reactions, springs) {
      const F = [0, 0, 0], M = [0, 0, 0];
      const addF = (r, f, m = [0, 0, 0], s = 1) => { const c = cross(r, f); for (let k = 0; k < 3; k++) { F[k] += s * f[k]; M[k] += s * (c[k] + m[k]); } };
      for (const ld of lc.nodal) { const nd = this.model.nodes[ld.node]; addF([nd.x, nd.y, nd.z], ld.f.slice(0, 3), ld.f.slice(3)); }
      for (const [mid, md] of Object.entries(this.members)) {
        const m = this.model.members[mid], ni = this.model.nodes[m.i], ri = [ni.x, ni.y, ni.z], lam = md.lam;
        const toG = (q) => [0, 1, 2].map((c) => lam[0][c] * q[0] + lam[1][c] * q[1] + lam[2][c] * q[2]);
        for (const [a, b, q1, q2] of loads[mid].dist) {
          for (let g = 0; g < 3; g++) {
            const wg = (b - a) * GW[g], s = a + (b - a) * GP[g], t = (s - a) / (b - a);
            const gq = toG([0, 1, 2].map((k) => q1[k] + (q2[k] - q1[k]) * t));
            addF([0, 1, 2].map((k) => ri[k] + lam[0][k] * s), gq, [0, 0, 0], wg);
          }
        }
        for (const [a, P] of loads[mid].point) addF([0, 1, 2].map((k) => ri[k] + lam[0][k] * a), toG(P));
      }
      const applied = { Fx: F[0], Fy: F[1], Fz: F[2], Mx: M[0], My: M[1], Mz: M[2] };
      for (const [nid, r] of Object.entries(reactions).concat(Object.entries(springs))) {
        const nd = this.model.nodes[nid];
        addF([nd.x, nd.y, nd.z], r.slice(0, 3), r.slice(3));
      }
      return [{ Fx: F[0], Fy: F[1], Fz: F[2], Mx: M[0], My: M[1], Mz: M[2] }, applied];
    }
  }

  // ---------------------------------------------------------------- EC2 checks in 3D
  function biaxialExponent(r) {
    if (r <= 0.1) return 1.0;
    if (r <= 0.7) return 1.0 + (r - 0.1) / 0.6 * 0.5;
    if (r <= 1.0) return 1.5 + (r - 0.7) / 0.3 * 0.5;
    return 2.0;
  }
  function sectionModels(sec, p) {
    const vy = sectionView(sec, "y"), vz = sectionView(sec, "z"), s2 = sectionAs2d(sec);
    return { sec, sec2d: s2, view_y: vy, view_z: vz, rc_y: new I.RCSection(vy, p), rc_z: new I.RCSection(vz, p), rc_2d: new I.RCSection(s2, p) };
  }
  const map2 = (A, fn) => A.map((row, i) => row.map((v, j) => fn(v, i, j)));

  function checkBiaxial(member, S, f, p) {
    const sec = S.sec, isColumn = member.design.type === "column";
    const c = S.rc_y.c, st = S.rc_y.st, AsTot = S.rc_y.As.reduce((s, a) => s + a, 0);
    const NRd = (sec.b_mm * sec.h_mm * c.fcd + AsTot * st.fyd) * 1e-3;
    const e0y = Math.max(sec.h_mm / p.e0_h_divisor, p.e0_min_mm), e0z = Math.max(sec.b_mm / p.e0_h_divisor, p.e0_min_mm);
    const etaOf = (Nc, My, Mz, a) => {
      const uy = S.rc_y.utilisation(Nc, My), uz = S.rc_z.utilisation(Nc, Mz);
      return [!isFinite(uy) || !isFinite(uz) ? Infinity : (uy ** a + uz ** a) ** (1 / a), uy, uz];
    };
    let best = -Infinity, ci = 0, si = 0, bMy = 0, bMz = 0;
    f.N.forEach((row, i) => row.forEach((Nv, j) => {
      const Nc = -Nv, a = biaxialExponent(Math.max(Nc, 0) / NRd), My = f.My[i][j], Mz = f.Mz[i][j];
      let e, myd = My, mzd = Mz;
      if (isColumn) {
        const cands = [];
        for (const [e0, which] of [[e0y, "y"], [e0z, "z"]]) {
          const Mmin = Nc > 0 ? Nc * e0 * 1e-3 : 0, base = which === "y" ? My : Mz, small = Math.abs(base) < Mmin;
          for (const sg of [1, -1]) {
            const mod = small ? sg * Mmin : base;
            const [cy, cz] = which === "y" ? [mod, Mz] : [My, mod];
            cands.push([etaOf(Nc, cy, cz, a)[0], cy, cz]);
          }
        }
        const emax = Math.max(...cands.map((cd) => cd[0]));
        const tol = 1e-9 * (isFinite(emax) ? Math.max(Math.abs(emax), 1) : 1);
        const pick = cands.find((cd) => cd[0] >= emax - tol) || cands[0];
        [e, myd, mzd] = pick;
      } else e = etaOf(Nc, My, Mz, a)[0];
      if (e > best) { best = e; ci = i; si = j; bMy = myd; bMz = mzd; }
    }));
    const NEd = -f.N[ci][si], MyEd = bMy, MzEd = bMz, av = biaxialExponent(Math.max(NEd, 0) / NRd);
    const steps = [
      step("N_{Ed}", "axial force (compression +)", fmt(NEd), NEd, "kN"),
      step("M_{Edy}", "moment in the x-z plane (compression on +z face +)" + (isColumn ? ", at least N_{Ed}·e_{0}" : ""), fmt(MyEd), MyEd, "kNm"),
      step("M_{Edz}", "moment in the x-y plane (compression on +y face +)" + (isColumn ? ", at least N_{Ed}·e_{0}" : ""), fmt(MzEd), MzEd, "kNm"),
    ];
    const notes = [];
    const Nmin = S.rc_y.N_min, Nmax = S.rc_y.N_max;
    let util, capacity = null;
    if (NEd > Nmax || NEd < Nmin) {
      const cap = NEd > Nmax ? Nmax : Nmin;
      util = cap ? NEd / cap : Infinity;
      steps.push(step("N_{Rd}", "resistance to pure axial force", "", cap, "kN", "6.1"));
      notes.push("The axial force alone exceeds the section resistance.");
      capacity = cap;
    } else {
      const MRdy = S.rc_y.solveState(NEd, MyEd < -1e-9).M, MRdz = S.rc_z.solveState(NEd, MzEd < -1e-9).M;
      const uyv = Math.abs(MyEd) > 1e-12 ? Math.abs(MyEd / MRdy) : 0, uzv = Math.abs(MzEd) > 1e-12 ? Math.abs(MzEd / MRdz) : 0;
      util = (uyv ** av + uzv ** av) ** (1 / av);
      steps.push(
        step("M_{Rdy}", "uniaxial resistance at N_{Ed}, x-z plane (strain compatibility, Fig. 6.1)", "", MRdy, "kNm", "6.1"),
        step("M_{Rdz}", "uniaxial resistance at N_{Ed}, x-y plane", "", MRdz, "kNm", "6.1"),
        step("N_{Rd}", "A_{c}·f_{cd} + A_{s}·f_{yd}", `${fmt(sec.b_mm * sec.h_mm)}·${fmt(c.fcd)} + ${fmt(AsTot)}·${fmt(st.fyd)}`, NRd, "kN", "5.8.9(4)"),
        step("a", "1.0 / 1.5 / 2.0 at N_{Ed}/N_{Rd} = 0.1 / 0.7 / 1.0, interpolated", `N_{Ed}/N_{Rd} = ${fmt(Math.max(NEd, 0) / NRd)}`, av, "", "5.8.9(4)"),
        step("η", "[(M_{Edy}/M_{Rdy})^{a} + (M_{Edz}/M_{Rdz})^{a}]^{1/a}", `[${fmt(uyv)}^{${av.toFixed(2)}} + ${fmt(uzv)}^{${av.toFixed(2)}}]^{1/${av.toFixed(2)}}`, util, "", "Eq. 5.39"));
    }
    notes.push("Eq. 5.39 is applied to every member; with bending about one axis only it reduces to the uniaxial check of 6.1.");
    return check({ id: "ec2.biaxial", title: "Bending with axial force, both axes", clause: "EN 1992-1-1 6.1, 5.8.9", member: member.id,
      status: isFinite(util) ? statusU(util) : FAIL, utilisation: util, demand: null, capacity, x: f.x[si], combination: f.uls_ids[ci], combination_label: f.uls_labels[ci], steps, notes });
  }

  function checkTorsion(member, S, f, p) {
    const sec = S.sec, c = S.rc_y.c, st = S.rc_y.st, b = sec.b_mm, h = sec.h_mm, A = b * h, u = 2 * (b + h);
    const phil = Math.max(0, ...[sec.top, sec.bottom, sec.side].filter((l) => l && l.n > 0).map((l) => l.dia_mm));
    const edge = sec.cover_mm + (sec.links ? sec.links.dia_mm : 0) + phil / 2;
    const tef = Math.max(A / u, 2 * edge), Ak = (b - tef) * (h - tef), uk = 2 * ((b - tef) + (h - tef));
    const fctd = p.alpha_ct * 0.7 * c.fctm / p.gamma_c, TRdc = 2 * Ak * tef * fctd * 1e-6;
    let Tmax = 0; for (const r of f.T) for (const v of r) Tmax = Math.max(Tmax, Math.abs(v));
    if (Tmax < 0.1 * TRdc) return null;
    let best = -Infinity, ci = 0, si = 0, G = null;
    f.T.forEach((row, i) => row.forEach((Tv, j) => {
      const rv = I.shearResistance(S.rc_y, S.view_y, p, -f.N[i][j], f.My[i][j]);
      const cot = rv.cot !== undefined ? rv.cot : 1.0, sc = cot / (1 + cot * cot);
      const TRdmax = 2 * c.nu * 1.0 * c.fcd * Ak * tef * sc * 1e-6;
      const VRdmax = rv.VRdmax !== undefined ? rv.VRdmax : 0.5 * S.rc_y.b * 0.9 * rv.d * c.nu * c.fcd * 1e-3;
      const T = Math.abs(Tv), V = Math.abs(f.Vz[i][j]);
      const e631 = T / TRdc + V / rv.VRdc, e629 = T / TRdmax + V / VRdmax;
      let TRds = 0, elinks;
      if (sec.links) {
        const A1 = Math.PI * sec.links.dia_mm ** 2 / 4, fywd = (sec.links.fyk || sec.rebar.fyk) / p.gamma_s;
        TRds = 2 * Ak * A1 / sec.links.spacing_mm * fywd * cot * 1e-6;
        elinks = e631 > 1 ? T / TRds + V / rv.VRds : 0;
      } else elinks = e631 > 1 ? Infinity : 0;
      const e = Math.max(e629, elinks), key = isFinite(e) ? e : 1e9;
      if (key > best) { best = key; ci = i; si = j; G = { T, V, cot, TRdmax, VRdmax, e631, e629, TRds, elinks, e }; }
    }));
    const g = G;
    const steps = [
      step("T_{Ed}", "design torsional moment", fmt(g.T), g.T, "kNm"),
      step("V_{Ed}", "concurrent shear force in the x-z plane", fmt(g.V), g.V, "kN"),
      step("t_{ef}", "A/u, not less than twice the edge-to-bar-centre distance", `max(${fmt(A)}/${fmt(u)}, 2·${fmt(edge)})`, tef, "mm", "6.3.2(1)"),
      step("A_{k}", "(b − t_{ef})·(h − t_{ef})", `(${fmt(b)} − ${fmt(tef)})·(${fmt(h)} − ${fmt(tef)})`, Ak, "mm²", "6.3.2(1)"),
      step("T_{Rd,c}", "2·A_{k}·t_{ef}·f_{ctd} (cracking torque)", `2·${fmt(Ak)}·${fmt(tef)}·${fmt(fctd)}`, TRdc, "kNm", "6.3.2(5)"),
      step("check", "T_{Ed}/T_{Rd,c} + V_{Ed}/V_{Rd,c} ≤ 1: only minimum reinforcement needed", "", g.e631, "", "Eq. 6.31"),
      step("T_{Rd,max}", "2·ν·α_{cw}·f_{cd}·A_{k}·t_{ef}·sin θ·cos θ", `cot θ = ${fmt(g.cot)}`, g.TRdmax, "kNm", "Eq. 6.30"),
      step("η_{struts}", "T_{Ed}/T_{Rd,max} + V_{Ed}/V_{Rd,max} ≤ 1", "", g.e629, "", "Eq. 6.29"),
    ];
    const notes = ["Shown because T_{Ed} ≥ 0.1·T_{Rd,c}. Where torsion arises from compatibility only, 6.3.1(2) allows it to be neglected at ULS."];
    if (g.e631 > 1) {
      steps.push(step("T_{Rd,s}", "2·A_{k}·(A_{sw,1}/s)·f_{ywd}·cot θ (outer leg)", "", g.TRds, "kNm", "Eq. 6.28"));
      steps.push(step("η_{links}", "T_{Ed}/T_{Rd,s} + V_{Ed}/V_{Rd,s} ≤ 1 (links shared by shear and torsion)", "", g.elinks, "", "6.3.2(2)"));
      steps.push(step("ΣA_{sl}", "T_{Ed}·u_{k}·cot θ/(2·A_{k}·f_{yd}), additional longitudinal steel", "", g.T * 1e6 * uk * g.cot / (2 * Ak * st.fyd), "mm²", "Eq. 6.28"));
      notes.push("Distribute the additional longitudinal torsion steel around the perimeter; it is reported, not checked against the bars.");
    }
    return check({ id: "ec2.torsion", title: "Torsion with shear", clause: "EN 1992-1-1 6.3", member: member.id, status: isFinite(g.e) ? statusU(g.e) : FAIL,
      utilisation: g.e, demand: g.T, capacity: g.TRdmax, unit: "kNm", x: f.x[si], combination: f.uls_ids[ci], combination_label: f.uls_labels[ci], steps, notes });
  }

  function slenderness(member, S, L, NEd, axis) {
    const c = S.rc_y.c, st = S.rc_y.st, Ac = S.sec.b_mm * S.sec.h_mm;
    const depth = axis === "y" ? S.sec.h_mm : S.sec.b_mm, factor = axis === "y" ? member.design.l0_factor_y : member.design.l0_factor_z;
    const l0 = factor * L * 1000, i = depth / Math.sqrt(12), lam = l0 / i;
    const n = Math.max(NEd, 1e-9) * 1e3 / (Ac * c.fcd);
    const omega = S.rc_y.As.reduce((s, a) => s + a, 0) * st.fyd / (Ac * c.fcd), B = Math.sqrt(1 + 2 * omega);
    return { l0, i, lam, lam_lim: 20 * 0.7 * B * 0.7 / Math.sqrt(n), n, omega, B };
  }
  function checkSecondOrder(member, S, f, p) {
    const sec = S.sec;
    let NEdMax = -Infinity; for (const r of f.N) for (const v of r) NEdMax = Math.max(NEdMax, -v);
    if (NEdMax <= 0) return null;
    if (["y", "z"].every((ax) => { const s = slenderness(member, S, f.L, NEdMax, ax); return s.lam <= s.lam_lim; })) return null;
    const c = S.rc_y.c, st = S.rc_y.st, Ac = sec.b_mm * sec.h_mm, bars = sec.bars, AsTot = bars.reduce((s, b) => s + b[0], 0);
    const NRd = (Ac * c.fcd + AsTot * st.fyd) * 1e-3, phi = member.design.creep_coefficient;
    let bestRow = null;
    for (let k = 0; k < f.N.length; k++) {
      let jmax = 0; f.N[k].forEach((v, j) => { if (-v > -f.N[k][jmax]) jmax = j; });
      const NEd = -f.N[k][jmax];
      if (NEd <= 0) continue;
      const per = {};
      for (const ax of ["y", "z"]) {
        const sl = slenderness(member, S, f.L, NEd, ax);
        const M = (ax === "y" ? f.My : f.Mz)[k];
        const Ma = M[0], Mb = M[M.length - 1];
        const [M02, M01] = Math.abs(Ma) >= Math.abs(Mb) ? [Ma, Mb] : [Mb, Ma];
        let M0e = 0.6 * M02 + 0.4 * M01;
        if (Math.abs(M0e) < 0.4 * Math.abs(M02)) M0e = (M02 < 0 ? -1 : 1) * 0.4 * Math.abs(M02);
        const sgn = M0e >= 0 ? 1 : -1, ei = sl.l0 / 400;
        let M2 = 0, detail = {};
        if (sl.lam > sl.lam_lim) {
          const coord = bars.map((b) => (ax === "y" ? b[2] : b[1]));
          const is_ = AsTot ? Math.sqrt(bars.reduce((s, b, q) => s + b[0] * coord[q] * coord[q], 0) / AsTot) : 0;
          const depth = ax === "y" ? sec.h_mm : sec.b_mm, d = depth / 2 + is_, nu_ = 1 + sl.omega;
          const Kr = Math.min(Math.max((nu_ - sl.n) / (nu_ - 0.4), 0), 1);
          const qpM = ax === "y" ? f.qp_My : (f.qp_Mz || f.qp_My);
          let M0qp = 0; for (const r of qpM) for (const v of r) M0qp = Math.max(M0qp, Math.abs(v));
          const M0Ed = Math.max(Math.abs(M02), 1e-9), phiEf = phi * Math.min(1, M0qp / M0Ed);
          const beta = 0.35 + c.fck / 200 - sl.lam / 150, Kphi = Math.max(1, 1 + beta * phiEf);
          const invR = Kr * Kphi * st.eps_yd / (0.45 * d), e2 = invR * sl.l0 ** 2 / 10;
          M2 = NEd * e2 * 1e-3;
          detail = { i_s: is_, d, Kr, phi_ef: phiEf, beta, Kphi, inv_r: invR, e2 };
        }
        const e0 = Math.max((ax === "y" ? sec.h_mm : sec.b_mm) / p.e0_h_divisor, p.e0_min_mm);
        per[ax] = Object.assign({ lam: sl.lam, lam_lim: sl.lam_lim, l0: sl.l0, M01, M02, M0e, ei, M2, sgn, e0, n: sl.n, omega: sl.omega, B: sl.B }, detail);
      }
      const a = biaxialExponent(Math.max(NEd, 0) / NRd);
      let best = null;
      for (const imp of ["y", "z"]) {
        const Md = {};
        for (const ax of ["y", "z"]) {
          const q = per[ax], add = ax === imp ? NEd * q.ei * 1e-3 : 0;
          let MEd = Math.max(Math.abs(q.M02), Math.abs(q.M0e) + add + q.M2, Math.abs(q.M01) + 0.5 * q.M2);
          if (ax === imp) MEd = Math.max(MEd, NEd * q.e0 * 1e-3);
          Md[ax] = q.sgn * MEd;
        }
        const uy = S.rc_y.utilisation(NEd, Md.y), uz = S.rc_z.utilisation(NEd, Md.z);
        const e = !isFinite(uy) || !isFinite(uz) ? Infinity : (uy ** a + uz ** a) ** (1 / a);
        if (best === null || e > best.e + 1e-9 * Math.max(Math.abs(best.e), 1)) best = { e, imp, Md, uy, uz };
      }
      if (bestRow === null || best.e > bestRow.eta) bestRow = { eta: best.e, k, NEd, a, per, best };
    }
    if (!bestRow) return null;
    const { eta, k, NEd, a, per, best } = bestRow;
    const steps = [step("N_{Ed}", "maximum compression in the governing combination", fmt(NEd), NEd, "kN")];
    for (const ax of ["y", "z"]) {
      const q = per[ax], plane = ax === "y" ? "x-z" : "x-y";
      steps.push(step(`λ_{${ax}}`, `slenderness, ${plane} plane; λ_{lim} = 20·A·B·C/√n with A = C = 0.7, B = √(1 + 2ω) = ${fmt(q.B)}`, `λ_{lim} = ${fmt(q.lam_lim)}`, q.lam, "", "5.8.3.1"));
      if (q.M2 > 0) steps.push(
        step(`K_{r,${ax}}`, "(n_{u} − n)/(n_{u} − n_{bal}) ≤ 1, n_{u} = 1 + ω, n_{bal} = 0.4", `n = ${fmt(q.n)}, ω = ${fmt(q.omega)}`, q.Kr, "", "Eq. 5.36"),
        step(`K_{φ,${ax}}`, "1 + β·φ_{ef} ≥ 1, β = 0.35 + f_{ck}/200 − λ/150", `β = ${fmt(q.beta)}, φ_{ef} = ${fmt(q.phi_ef)}`, q.Kphi, "", "Eq. 5.37"),
        step(`1/r_{${ax}}`, "K_{r}·K_{φ}·ε_{yd}/(0.45·d), d = h/2 + i_{s}", `d = ${fmt(q.d)} mm`, q.inv_r * 1e3, "1/m", "Eq. 5.34, 5.35"),
        step(`e_{2,${ax}}`, "(1/r)·l_{0}²/c, c = 10", `l_{0} = ${fmt(q.l0)} mm`, q.e2, "mm", "Eq. 5.33"),
        step(`M_{2,${ax}}`, "N_{Ed}·e_{2}", "", q.M2, "kNm", "5.8.8.2(3)"));
      steps.push(step(`M_{0e,${ax}}`, "0.6·M_{02} + 0.4·M_{01} ≥ 0.4·M_{02}", `M_{02} = ${fmt(q.M02)}, M_{01} = ${fmt(q.M01)}`, q.M0e, "kNm", "Eq. 5.32"));
      steps.push(step(`M_{Ed,${ax}}`, "max(M_{02}, M_{0e} + N·e_{i} + M_{2}, M_{01} + M_{2}/2), e_{i} = l_{0}/400" + (ax === best.imp ? " (imperfection about this axis)" : ""), `e_{i} = ${fmt(q.ei)} mm`, best.Md[ax], "kNm", "5.8.8.2, 5.2(7)"));
    }
    steps.push(step("η_{y}, η_{z}", "M_{Ed}/M_{Rd} about each axis at N_{Ed}", `${fmt(best.uy)}, ${fmt(best.uz)}`, Math.max(best.uy, best.uz), "", "6.1"),
      step("η", "[(M_{Edy}/M_{Rdy})^{a} + (M_{Edz}/M_{Rdz})^{a}]^{1/a}", `a = ${a.toFixed(2)}`, eta, "", "Eq. 5.39"));
    return check({ id: "ec2.second_order", title: "Slender column, second-order effects", clause: "EN 1992-1-1 5.8.8, 5.8.9", member: member.id,
      status: isFinite(eta) ? statusU(eta) : FAIL, utilisation: eta, combination: f.uls_ids[k], combination_label: f.uls_labels[k], steps,
      notes: ["Nominal curvature method for an isolated member; l_{0} from the member's effective length factors. For an unbraced (sway) frame, set the factors accordingly.",
        "φ_{ef} = φ(∞,t_{0})·M_{0Eqp}/M_{0Ed} with M_{0Eqp} the largest quasi-permanent moment."] });
  }
  function checkSlenderness3d(member, S, f) {
    let NEd = -Infinity; for (const r of f.N) for (const v of r) NEd = Math.max(NEd, -v);
    return ["y", "z"].map((axis) => {
      const s = slenderness(member, S, f.L, NEd, axis), depth = axis === "y" ? S.sec.h_mm : S.sec.b_mm, ok = s.lam <= s.lam_lim;
      return check({ id: `ec2.slenderness_${axis}`, title: `Slenderness, x-${axis === "y" ? "z" : "y"} plane`, clause: "EN 1992-1-1 5.8.3.1", member: member.id,
        status: ok ? PASS : INFO, utilisation: null, demand: s.lam, capacity: s.lam_lim,
        steps: [step("l_{0}", `effective length for buckling in the x-${axis === "y" ? "z" : "y"} plane`, "", s.l0, "mm", "5.8.3.2"),
          step("i", `${axis === "y" ? "h" : "b"}/√12`, `${fmt(depth)}/√12`, s.i, "mm"), step("λ", "l_{0}/i", `${fmt(s.l0)}/${fmt(s.i)}`, s.lam, "", "5.8.3.2(1)"),
          step("n", "N_{Ed}/(A_{c}·f_{cd})", "", s.n, "", "5.8.3.1(1)"), step("B", "√(1 + 2ω), ω = A_{s}·f_{yd}/(A_{c}·f_{cd})", `ω = ${fmt(s.omega)}`, s.B, "", "5.8.3.1(1)"),
          step("λ_{lim}", "20·A·B·C/√n with A = 0.7, C = 0.7", `20·0.7·${fmt(s.B)}·0.7/√${fmt(s.n)}`, s.lam_lim, "", "Eq. 5.13N")],
        notes: [ok ? "λ ≤ λ_lim: second-order effects may be ignored." : "λ > λ_lim: second-order effects are included in the check 'Slender column, second-order effects' (5.8.8)."] });
    });
  }
  function plane(f, axis) {
    const [M, V] = axis === "y" ? [f.My, f.Vz] : [f.Mz, f.Vy];
    return { x: f.x, L: f.L, uls_N: f.N, uls_V: V, uls_M: M, uls_ids: f.uls_ids, uls_labels: f.uls_labels, qp_M: f.qp_My, qp_ids: f.qp_ids, qp_labels: f.qp_labels, tip_at_i: f.tip_at_i };
  }
  function checkMember3d(member, S, f, p) {
    if (member.design.type === "none") return [];
    const out = [checkBiaxial(member, S, f, p)];
    const sz = I.checkShear(member, S.rc_y, S.view_y, plane(f, "y"), p); sz.id = "ec2.shear_z"; sz.title = "Shear, x-z plane (V_{z})";
    const sy = I.checkShear(member, S.rc_z, S.view_z, plane(f, "z"), p); sy.id = "ec2.shear_y"; sy.title = "Shear, x-y plane (V_{y})";
    out.push(sz, sy);
    const tors = checkTorsion(member, S, f, p);
    if (tors) out.push(tors);
    const m2d = { id: member.id, design: { type: member.design.type, deflection_system: member.design.deflection_system, creep_coefficient: member.design.creep_coefficient, l0_factor: member.design.l0_factor_y } };
    out.push(...I.checkDetailing(m2d, S.rc_2d, S.sec2d, plane(f, "y"), p));
    if (member.design.type === "column") {
      const so = checkSecondOrder(member, S, f, p);
      if (so) out.push(so);
      out.push(...checkSlenderness3d(member, S, f));
    } else {
      const ld = I.checkSpanDepth(m2d, S.rc_2d, S.sec2d, plane(f, "y"), p), defl = I.checkDeflection(m2d, S.rc_2d, S.sec2d, plane(f, "y"), p);
      if (ld.status === FAIL && defl.status === PASS) { ld.status = INFO; ld.notes.push("Limit exceeded, so the calculated deflection (7.4.3) governs; it passes."); }
      out.push(ld, defl);
    }
    return out;
  }

  // ---------------------------------------------------------------- project
  const QS = ["N", "Vy", "Vz", "T", "My", "Mz"];
  function runProject3d(data, opts = {}) {
    const now = () => (typeof performance !== "undefined" ? performance : Date).now();
    const t0 = now();
    const model = parseModel3d(data);
    const fa = new FrameAnalysis3D(model);
    const cases = {};
    for (const [cid, lc] of Object.entries(model.load_cases)) cases[cid] = fa.solveCase(lc);
    const t1 = now();
    const cs = I.generateCombinations(model);
    const nUls = cs.combos.filter((c) => c.kind === "ULS").length;
    if (opts.maxCombinations && nUls > opts.maxCombinations)
      throw new ModelError(`${nUls} ULS combinations is more than this browser session handles (${opts.maxCombinations}); reduce the number of variable load cases, or run the model with the Python package`);
    const params = I.makeParams(model.settings.params);
    const warnings = [];
    const axn = ["X", "Y", "Z"];
    for (const d of fa.autoFixed) warnings.push(`Node '${fa.nodeIds[Math.floor(d / 6)]}': rotation about ${axn[d % 6 - 3]} has no stiffness (released on every member) and was restrained automatically.`);
    for (const [cid, cr] of Object.entries(cases)) {
      const scale = Math.max(1, ...Object.values(cr.applied).map(Math.abs));
      if (Math.max(...Object.values(cr.equilibrium).map(Math.abs)) > 1e-6 * scale) warnings.push(`Load case ${cid}: equilibrium residual larger than expected.`);
    }
    const uls = cs.combos.filter((c) => c.kind === "ULS"), qp = cs.combos.filter((c) => c.kind === "SLS_qp");
    const F = uls.map((c) => c.factors), Fq = qp.map((c) => c.factors);
    const envelopes = {}, forces = {};
    for (const mid of Object.keys(model.members)) {
      const comb = {};
      for (const q of QS) comb[q] = I.combine(F, cs.caseIds.map((c) => cases[c].members[mid][q]));
      forces[mid] = comb;
      const env = { x: fa.stations[mid].x, max: {}, min: {}, max_combo: {}, min_combo: {} };
      for (const q of QS) {
        env.max[q] = []; env.min[q] = []; env.max_combo[q] = []; env.min_combo[q] = [];
        comb[q][0].forEach((_, j) => {
          let mx = -Infinity, mn = Infinity, ix = 0, im = 0;
          comb[q].forEach((row, i) => { if (row[j] > mx) { mx = row[j]; ix = i; } if (row[j] < mn) { mn = row[j]; im = i; } });
          env.max[q].push(mx); env.min[q].push(mn); env.max_combo[q].push(uls[ix].id); env.min_combo[q].push(uls[im].id);
        });
      }
      envelopes[mid] = env;
    }
    const reactions = {};
    for (const nid of Object.keys(model.supports)) {
      const R = cs.caseIds.map((c) => cases[c].reactions[nid]);
      const RU = F.map((fr) => [0, 1, 2, 3, 4, 5].map((q) => fr.reduce((s, fv, c) => s + fv * R[c][q], 0)));
      const mx = [0, 1, 2, 3, 4, 5].map((q) => Math.max(...RU.map((r) => r[q]))), mn = [0, 1, 2, 3, 4, 5].map((q) => Math.min(...RU.map((r) => r[q])));
      const ixOf = (q, v) => uls[RU.findIndex((r) => r[q] === v)].id;
      reactions[nid] = { per_case: Object.fromEntries(cs.caseIds.map((c, k) => [c, R[k]])), uls_max: mx, uls_min: mn,
        uls_max_combo: mx.map((v, q) => ixOf(q, v)), uls_min_combo: mn.map((v, q) => ixOf(q, v)) };
    }
    const cache = {}, checks = {};
    if (!opts.skipChecks) {
      for (const [mid, m] of Object.entries(model.members)) {
        if (!cache[m.section]) {
          const sec = model.sections[m.section];
          const key = opts.sectionCache ? JSON.stringify([sec.b_mm, sec.h_mm, sec.cover_mm, sec.top, sec.bottom, sec.side, sec.links, sec.links_legs_y,
            sec.concrete, sec.rebar, sec.stiffness_factor, params]) : null;
          if (key && opts.sectionCache.has(key)) cache[m.section] = opts.sectionCache.get(key);
          else {
            cache[m.section] = sectionModels(sec, params);
            if (key) { if (opts.sectionCache.size > 64) opts.sectionCache.clear(); opts.sectionCache.set(key, cache[m.section]); }
          }
        }
        let tipAtI = false;
        if (m.design.deflection_system === "cantilever") {
          const isFree = (nid) => !(nid in model.supports) && !Object.values(model.members).some((o) => o.id !== mid && (o.i === nid || o.j === nid));
          tipAtI = isFree(m.i) && !isFree(m.j);
        }
        const fr = forces[mid];
        const f = { x: fa.stations[mid].x, L: fa.members[mid].L, N: fr.N, Vy: fr.Vy, Vz: fr.Vz, T: fr.T, My: fr.My, Mz: fr.Mz,
          uls_ids: uls.map((c) => c.id), uls_labels: uls.map((c) => c.label),
          qp_My: I.combine(Fq, cs.caseIds.map((c) => cases[c].members[mid].My)), qp_Mz: I.combine(Fq, cs.caseIds.map((c) => cases[c].members[mid].Mz)),
          qp_ids: qp.map((c) => c.id), qp_labels: qp.map((c) => c.label), tip_at_i: tipAtI };
        checks[mid] = model.settings.design_code === "EC2" ? checkMember3d(m, cache[m.section], f, params) : I.checkMemberSIA(m);
      }
    }
    const all = Object.values(checks).flat();
    let gov = null, bu = -1;
    for (const c of all) {
      const u = c.utilisation != null ? c.utilisation : (c.status === FAIL ? Infinity : -1);
      if ((c.status === PASS || c.status === FAIL) && u > bu) { gov = c; bu = u; }
    }
    const sts = new Set(all.map((c) => c.status));
    const status = sts.has(FAIL) ? FAIL : (sts.has(NOT_COVERED) || sts.has("warn") ? NOT_COVERED : PASS);
    const t2 = now();
    return { is3d: true, version: T.VERSION, model, analysis: fa, cases, combos: cs, params, envelopes, reactions, checks, warnings,
      summary: { status, max_utilisation: gov && gov.utilisation != null && isFinite(gov.utilisation) ? gov.utilisation : null,
        governing: gov ? { member: gov.member, check: gov.title, clause: gov.clause } : null, runtime_ms: t2 - t0, analysis_ms: t1 - t0 } };
  }

  // ---------------------------------------------------------------- building generator
  const CONCRETE_NAMES = { 20: "C20/25", 25: "C25/30", 30: "C30/37", 35: "C35/45", 40: "C40/50", 45: "C45/55", 50: "C50/60", 55: "C55/67", 60: "C60/75", 70: "C70/85", 80: "C80/95", 90: "C90/105" };
  const sec0 = (b, h, top, bottom, side, links, cover) => {
    const s = { type: "rc_rect", b_mm: b, h_mm: h, concrete: "concrete", rebar: "B500B", cover_mm: cover, top: { n: top[0], dia_mm: top[1] }, bottom: { n: bottom[0], dia_mm: bottom[1] },
      links: { dia_mm: links[0], legs: links[1], spacing_mm: links[2] } };
    if (side) s.side = { n: side[0], dia_mm: side[1] };
    return s;
  };
  const BUILDING_DEFAULTS = () => ({
    name: "Office building, 3 × 2 bays, 4 storeys", spans_x: [6.0, 6.0, 6.0], spans_y: [5.5, 5.5], storeys: [3.6, 3.2, 3.2, 3.2], base: "fixed", fck: 30.0,
    column: sec0(400, 400, [3, 20], [3, 20], [1, 20], [8, 2, 240], 35), beam_x: sec0(300, 550, [4, 20], [3, 20], null, [8, 2, 200], 30),
    beam_y: sec0(300, 500, [4, 20], [3, 16], null, [8, 2, 200], 30), slab_mm: 220.0, finishes: 1.5, imposed: 3.0, imposed_category: "B", snow: 1.0,
    facade: 6.0, wind: 0.9, distribution: "two_way", pattern_by_storey: false, column_l0_factor: 1.0, combination_rule: "6.10" });
  const r6 = (v) => Math.round(v * 1e6) / 1e6;
  function gridNames(n) { const L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; return [...Array(n).keys()].map((k) => (k < 26 ? L[k] : L[Math.floor(k / 26) - 1] + L[k % 26])); }
  function tributary(L, S, q, mode) {
    if (mode === "none" || q === 0) return [];
    if (mode === "full") return [[0, L, q * S / 2, q * S / 2]];
    const c = Math.min(L, S) / 2, p = q * c, segs = [[0, c, 0, p]];
    if (L - 2 * c > 1e-9) segs.push([c, L - c, p, p]);
    segs.push([L - c, L, p, 0]);
    return segs;
  }
  function generateBuilding(specIn) {
    const spec = Object.assign(BUILDING_DEFAULTS(), JSON.parse(JSON.stringify(specIn || {})));
    const gx = gridNames(spec.spans_x.length + 1), gy = [...Array(spec.spans_y.length + 1).keys()].map((k) => String(k + 1));
    const xs = [0], ys = [0], zs = [0];
    for (const s of spec.spans_x) xs.push(r6(xs[xs.length - 1] + Number(s)));
    for (const s of spec.spans_y) ys.push(r6(ys[ys.length - 1] + Number(s)));
    for (const h of spec.storeys) zs.push(r6(zs[zs.length - 1] + Number(h)));
    const nlev = zs.length - 1, nid = (i, j, k) => `${gx[i]}${gy[j]}-L${k}`;
    const nodes = [];
    zs.forEach((z, k) => ys.forEach((y, j) => xs.forEach((x, i) => nodes.push({ id: nid(i, j, k), x, y, z }))));
    const fixRot = spec.base === "fixed", supports = [];
    ys.forEach((_, j) => xs.forEach((__, i) => supports.push({ node: nid(i, j, 0), ux: true, uy: true, uz: true, rx: fixRot, ry: fixRot, rz: true })));
    const system = (idx, n) => (n === 1 ? "simply_supported" : (idx === 0 || idx === n - 1) ? "end_span" : "interior_span");
    const members = [];
    for (let k = 1; k <= nlev; k++) {
      ys.forEach((_, j) => xs.forEach((__, i) => members.push({ id: `C-${gx[i]}${gy[j]}-L${k}`, i: nid(i, j, k - 1), j: nid(i, j, k), section: "Column", design: { type: "column", l0_factor: spec.column_l0_factor } })));
      for (let j = 0; j < ys.length; j++) for (let i = 0; i < xs.length - 1; i++) members.push({ id: `BX-${gx[i]}${gy[j]}-L${k}`, i: nid(i, j, k), j: nid(i + 1, j, k), section: "Beam X", design: { type: "beam", deflection_system: system(i, xs.length - 1) } });
      for (let i = 0; i < xs.length; i++) for (let j = 0; j < ys.length - 1; j++) members.push({ id: `BY-${gx[i]}${gy[j]}-L${k}`, i: nid(i, j, k), j: nid(i, j + 1, k), section: "Beam Y", design: { type: "beam", deflection_system: system(j, ys.length - 1) } });
    }
    const areaLoads = (q, levels) => {
      const loads = [];
      if (q === 0) return loads;
      const mx = { two_way: "two_way", one_way_x: "none", one_way_y: "full" }[spec.distribution], my = { two_way: "two_way", one_way_x: "full", one_way_y: "none" }[spec.distribution];
      for (const k of levels) for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
        const Lx = xs[i + 1] - xs[i], Ly = ys[j + 1] - ys[j];
        for (const jj of [j, j + 1]) for (const [a, b, w1, w2] of tributary(Lx, Ly, q, mx)) loads.push({ type: "udl", member: `BX-${gx[i]}${gy[jj]}-L${k}`, w1: -w1, w2: -w2, a: r6(a), b: r6(b), direction: "global_z" });
        for (const ii of [i, i + 1]) for (const [a, b, w1, w2] of tributary(Ly, Lx, q, my)) loads.push({ type: "udl", member: `BY-${gx[ii]}${gy[j]}-L${k}`, w1: -w1, w2: -w2, a: r6(a), b: r6(b), direction: "global_z" });
      }
      return loads;
    };
    const floors = [...Array(nlev).keys()].map((k) => k + 1), occupied = nlev > 1 ? floors.slice(0, -1) : floors, roof = [nlev];
    const gSlab = areaLoads(25 * spec.slab_mm / 1000, floors), gFin = areaLoads(spec.finishes, floors), facade = [];
    if (spec.facade) for (const k of (nlev > 1 ? occupied : floors)) {
      for (const j of [0, ys.length - 1]) for (let i = 0; i < xs.length - 1; i++) facade.push({ type: "udl", member: `BX-${gx[i]}${gy[j]}-L${k}`, w: -spec.facade, direction: "global_z" });
      for (const i of [0, xs.length - 1]) for (let j = 0; j < ys.length - 1; j++) facade.push({ type: "udl", member: `BY-${gx[i]}${gy[j]}-L${k}`, w: -spec.facade, direction: "global_z" });
    }
    const cases = [{ id: "G1", name: "Self weight of frame and slabs", type: "permanent", self_weight: true, loads: gSlab },
      { id: "G2", name: "Finishes and facade", type: "permanent", loads: gFin.concat(facade) }];
    if (spec.imposed) {
      if (spec.pattern_by_storey) for (const k of occupied) cases.push({ id: `Q-L${k}`, name: `Imposed load, level ${k}`, type: "variable", category: spec.imposed_category, loads: areaLoads(spec.imposed, [k]) });
      else cases.push({ id: "Q", name: "Imposed load, all floors", type: "variable", category: spec.imposed_category, loads: areaLoads(spec.imposed, occupied) });
    }
    if (spec.snow && nlev > 1) cases.push({ id: "S", name: "Snow on the roof", type: "variable", category: "snow", loads: areaLoads(spec.snow, roof) });
    if (spec.wind) {
      const htrib = (k) => (zs[k] - zs[k - 1]) / 2 + (k < nlev ? zs[k + 1] - zs[k] : 0) / 2;
      for (const [cid, axis, sgn, label] of [["W+X", 0, 1, "Wind in +X"], ["W-X", 0, -1, "Wind in -X"], ["W+Y", 1, 1, "Wind in +Y"], ["W-Y", 1, -1, "Wind in -Y"]]) {
        const loads = [];
        for (const k of floors) {
          if (axis === 0) {
            const i = sgn > 0 ? 0 : xs.length - 1;
            ys.forEach((_, j) => { const bt = (j > 0 ? ys[j] - ys[j - 1] : 0) / 2 + (j < ys.length - 1 ? ys[j + 1] - ys[j] : 0) / 2; loads.push({ type: "nodal", node: nid(i, j, k), fx: r6(sgn * spec.wind * htrib(k) * bt) }); });
          } else {
            const j = sgn > 0 ? 0 : ys.length - 1;
            xs.forEach((_, i) => { const bt = (i > 0 ? xs[i] - xs[i - 1] : 0) / 2 + (i < xs.length - 1 ? xs[i + 1] - xs[i] : 0) / 2; loads.push({ type: "nodal", node: nid(i, j, k), fy: r6(sgn * spec.wind * htrib(k) * bt) }); });
          }
        }
        cases.push({ id: cid, name: label, type: "variable", category: "wind", group: "wind", loads });
      }
    }
    const cname = CONCRETE_NAMES[Math.trunc(spec.fck)] || `C${spec.fck}`;
    const sections = { Column: JSON.parse(JSON.stringify(spec.column)), "Beam X": JSON.parse(JSON.stringify(spec.beam_x)), "Beam Y": JSON.parse(JSON.stringify(spec.beam_y)) };
    for (const s of Object.values(sections)) s.concrete = cname;
    const slabs = [];
    for (const k of floors) for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) slabs.push({ level: k, x0: xs[i], x1: xs[i + 1], y0: ys[j], y1: ys[j + 1], z: zs[k], t_mm: spec.slab_mm });
    return { schema: SCHEMA3D, name: spec.name, design: { design_code: "EC2", combination_rule: spec.combination_rule },
      materials: { [cname]: { type: "concrete", fck: spec.fck }, B500B: { type: "rebar", fyk: 500 } }, sections, nodes, supports, members, load_cases: cases,
      meta: { generator: "tragwerk.building", spec, grid: { x: Object.fromEntries(gx.map((g, k) => [g, xs[k]])), y: Object.fromEntries(gy.map((g, k) => [g, ys[k]])), levels: zs }, slabs } };
  }

  Object.assign(T, { SCHEMA3D, parseModel3d, runProject3d, FrameAnalysis3D, generateBuilding, BUILDING_DEFAULTS, localAxes, sectionBars, biaxialExponent });
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
