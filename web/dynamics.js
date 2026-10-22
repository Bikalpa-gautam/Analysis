/*
 * Tragwerk dynamics for the browser: modal analysis (consistent or lumped mass,
 * subspace iteration) and the EN 1998-1 response spectrum method. A port of
 * tragwerk.modal and tragwerk.seismic; tests/test_js_parity_dyn.py compares the two.
 * Load after engine.js and engine3d.js.
 */
(function (root) {
  "use strict";
  const T = typeof module !== "undefined" && module.exports ? require("./engine3d.js") : root.Tragwerk;
  const I = T._internal;
  const { ModelError, shape, solveSmall, zeros, fmt, step, check, statusU, PASS, FAIL, V_IDX, W_IDX, W_SIGN } = I;
  const GRAV = 9.81;
  // 5-point Gauss-Legendre on [0, 1]
  const GX = [-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664];
  const GWT = [0.23692688505618908, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.23692688505618908];
  const GP5 = GX.map((x) => (x + 1) / 2), GW5 = GWT.map((w) => w / 2);
  const AXES = ["X", "Y", "Z"];

  // ================================================================ refined model
  function refine(model, segments) {
    const raw = model.raw;
    const nodes = Object.values(model.nodes).map((n) => ({ id: n.id, x: n.x, y: n.y, z: n.z }));
    const taken = new Set([...Object.keys(model.nodes), ...Object.keys(model.members)]);
    const fresh = (name) => { while (taken.has(name)) name += "'"; taken.add(name); return name; };
    const members = [], submap = {};
    for (const [mid, m] of Object.entries(model.members)) {
      const [L] = model.axes(mid), a = model.nodes[m.i], b = model.nodes[m.j], s = segments;
      const ids = [m.i];
      for (let k = 1; k < s; k++) ids.push(fresh(`${mid}@${k}`));
      ids.push(m.j);
      for (let k = 1; k < s; k++) { const t = k / s; nodes.push({ id: ids[k], x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t }); }
      const subs = [];
      for (let k = 0; k < s; k++) {
        const eid = s === 1 ? mid : fresh(`${mid}#${k + 1}`);
        const releases = {};
        if (k === 0) releases.i = Object.assign({}, m.rel_i);
        if (k === s - 1) releases.j = Object.assign({}, m.rel_j);
        members.push({ id: eid, i: ids[k], j: ids[k + 1], section: m.section, roll_deg: m.roll_deg, releases, design: { type: "none" } });
        subs.push([eid, L * k / s, L * (k + 1) / s]);
      }
      submap[mid] = subs;
    }
    const lmap = (lc) => {
      const out = [];
      for (const ld of lc.nodal) out.push({ type: "nodal", node: ld.node, fx: ld.f[0], fy: ld.f[1], fz: ld.f[2], mx: ld.f[3], my: ld.f[4], mz: ld.f[5] });
      for (const ld of lc.distributed) {
        for (const [eid, x0, x1] of submap[ld.member]) {
          const lo = Math.max(ld.a, x0), hi = Math.min(ld.b, x1);
          if (hi - lo > 1e-9) {
            const w = (x) => ld.w1 + (ld.w2 - ld.w1) * (x - ld.a) / (ld.b - ld.a);
            out.push({ type: "udl", member: eid, w1: w(lo), w2: w(hi), a: lo - x0, b: Math.min(hi - x0, x1 - x0), direction: ld.direction });
          }
        }
      }
      for (const ld of lc.point) {
        for (const [eid, x0, x1] of submap[ld.member]) {
          if (x0 - 1e-12 <= ld.a && ld.a <= x1 + 1e-12) { out.push({ type: "point", member: eid, p: ld.p, a: Math.min(Math.max(ld.a - x0, 0), x1 - x0), direction: ld.direction }); break; }
        }
      }
      return out;
    };
    const cases = Object.values(model.load_cases).map((lc) => ({ id: lc.id, type: lc.type, name: lc.name, category: lc.category, group: lc.group, self_weight: lc.self_weight, loads: lmap(lc) }));
    const keys = ["ux", "uy", "uz", "rx", "ry", "rz"], skeys = ["kx", "ky", "kz", "krx", "kry", "krz"];
    const supports = Object.values(model.supports).map((sp) => {
      const d = { node: sp.node };
      keys.forEach((k, q) => { d[k] = sp.fix[q]; });
      skeys.forEach((k, q) => { if (sp.springs[q]) d[k] = sp.springs[q]; });
      return d;
    });
    const data = { schema: raw.schema, name: model.name, materials: raw.materials, sections: raw.sections,
      design: { design_code: model.settings.design_code, combination_rule: model.settings.combination_rule, params: model.settings.params, stations_per_member: 1 },
      nodes, supports, members, load_cases: cases };
    return { ref: T.parseModel3d(data), submap };
  }

  // ================================================================ mass
  function massFactors(model, dyn) {
    if (dyn.mass_cases) return Object.fromEntries(Object.keys(model.load_cases).map((c) => [c, Number(dyn.mass_cases[c] ?? 0)]));
    const psi = Object.assign({}, I.PSI);
    for (const [cat, vals] of Object.entries(model.settings.params.psi || {})) psi[cat] = vals.map(Number);
    return Object.fromEntries(Object.entries(model.load_cases).map(([c, lc]) => [c, lc.type === "permanent" ? 1.0 : psi[lc.category][2]]));
  }
  const vertical = (direction, lam) => (direction.startsWith("global_") ? (direction === "global_z" ? 1 : 0) : lam["xyz".indexOf(direction.slice(-1))][2]);

  function elementMasses(ref, fa, factors) {
    const ems = {};
    for (const mid of Object.keys(fa.members)) {
      const sec = ref.sections[ref.members[mid].section];
      ems[mid] = { msw: 0, r2: (sec.Iy_m4 + sec.Iz_m4) / sec.area_m2, dist: [], point: [] };
    }
    const nodal = zeros(fa.nodeIds.length);
    for (const [cid, f] of Object.entries(factors)) {
      if (!(f > 0)) continue;
      const lc = ref.load_cases[cid];
      if (lc.self_weight) for (const mid of Object.keys(fa.members)) { const sec = ref.sections[ref.members[mid].section]; ems[mid].msw += f * sec.concrete.density * sec.area_m2 / GRAV; }
      for (const ld of lc.distributed) { const vz = vertical(ld.direction, fa.members[ld.member].lam); if (vz) ems[ld.member].dist.push([ld.a, ld.b, -f * ld.w1 * vz / GRAV, -f * ld.w2 * vz / GRAV]); }
      for (const ld of lc.point) { const vz = vertical(ld.direction, fa.members[ld.member].lam); if (vz) ems[ld.member].point.push([ld.a, -f * ld.p * vz / GRAV]); }
      for (const ld of lc.nodal) nodal[fa.nidx[ld.node]] += Math.max(-f * ld.f[2] / GRAV, 0);
    }
    return { ems, nodal };
  }

  function elementMassLocal(L, em, kind) {
    const M = Array.from({ length: 12 }, () => zeros(12));
    const addPoint = (x, m, mr = 0) => {
      const xi = x / L, N = shape(xi, L);
      if (kind === "consistent") {
        const Nu = [N[0], N[3]], H = [N[1], N[2], N[4], N[5]], ax = [0, 6], tq = [3, 9];
        for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) { M[ax[r]][ax[c]] += m * Nu[r] * Nu[c]; if (mr) M[tq[r]][tq[c]] += mr * Nu[r] * Nu[c]; }
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
          M[V_IDX[r]][V_IDX[c]] += m * H[r] * H[c];
          M[W_IDX[r]][W_IDX[c]] += m * (W_SIGN[r] * H[r]) * (W_SIGN[c] * H[c]);
        }
      } else {
        for (const [base, f] of [[0, 1 - xi], [6, xi]]) {
          for (let q = 0; q < 3; q++) M[base + q][base + q] += m * f;
          if (mr) M[base + 3][base + 3] += mr * f;
        }
      }
    };
    if (em.msw > 0) for (let g = 0; g < 5; g++) addPoint(GP5[g] * L, em.msw * GW5[g] * L, em.msw * em.r2 * GW5[g] * L);
    for (const [a, b, m1, m2] of em.dist) for (let g = 0; g < 5; g++) { const m = m1 + (m2 - m1) * GP5[g]; if (m > 0) addPoint(a + (b - a) * GP5[g], m * GW5[g] * (b - a)); }
    for (const [a, m] of em.point) if (m > 0) addPoint(a, m);
    return M;
  }

  function releaseTransform(md) {
    const Tr = Array.from({ length: 12 }, (_, r) => zeros(12).map((_, c) => (r === c ? 1 : 0)));
    if (md.cidx.length) {
      const aidx = [...Array(12).keys()].filter((i) => !md.cidx.includes(i)), k = md.k_full;
      const kcc = md.cidx.map((r) => md.cidx.map((c) => k[r][c]));
      for (const c of md.cidx) Tr[c][c] = 0;
      for (const a of aidx) {
        const y = solveSmall(kcc, md.cidx.map((r) => k[r][a]));
        md.cidx.forEach((c, q) => { Tr[c][a] = -y[q]; });
      }
    }
    return Tr;
  }
  const matMul = (A, B) => A.map((row) => B[0].map((_, c) => row.reduce((s, v, k) => s + v * B[k][c], 0)));
  const transpose = (A) => A[0].map((_, c) => A.map((row) => row[c]));
  function blockT(lam) {
    const Tm = Array.from({ length: 12 }, () => zeros(12));
    for (let b = 0; b < 4; b++) for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) Tm[3 * b + r][3 * b + c] = lam[r][c];
    return Tm;
  }
  const dofsOf = (md) => { const o = []; for (let k = 0; k < 6; k++) o.push(6 * md.ni + k); for (let k = 0; k < 6; k++) o.push(6 * md.nj + k); return o; };

  function assembleMass(ref, fa, factors, dyn) {
    const { ems, nodal } = elementMasses(ref, fa, factors);
    for (const [nid, m] of dyn.masses) nodal[fa.nidx[nid]] += m;
    const M = Array.from({ length: fa.ndof }, () => new Map());
    const add = (r, c, v) => { if (v !== 0) M[r].set(c, (M[r].get(c) || 0) + v); };
    for (const [mid, md] of Object.entries(fa.members)) {
      const Me = elementMassLocal(md.L, ems[mid], dyn.mass);
      const Tr = releaseTransform(md), Tm = blockT(md.lam);
      const TT = matMul(Tr, Tm);                       // u_local_full = Tr · T · u_global
      const Mg = matMul(transpose(TT), matMul(Me, TT));
      const d = dofsOf(md);
      for (let r = 0; r < 12; r++) for (let c = 0; c < 12; c++) add(d[r], d[c], Mg[r][c]);
    }
    nodal.forEach((m, k) => { for (let q = 0; q < 3; q++) add(6 * k + q, 6 * k + q, m); });
    return M;
  }

  // ================================================================ small dense linear algebra
  function jacobiEig(A) {
    const n = A.length, a = A.map((r) => r.slice()), V = Array.from({ length: n }, (_, i) => zeros(n).map((_, j) => (i === j ? 1 : 0)));
    let norm = 0; for (const r of a) for (const v of r) norm += v * v;
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
      if (off <= 1e-32 * norm) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    return { vals: a.map((r, i) => r[i]), V };
  }
  function cholesky(A) {
    const n = A.length, L = Array.from({ length: n }, () => zeros(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) { if (!(s > 0)) throw new Error("subspace lost positive definiteness"); L[i][i] = Math.sqrt(s); } else L[i][j] = s / L[j][j];
    }
    return L;
  }
  function genEig(Kp, Mp) {   // Kp Q = Mp Q Λ, Qᵀ Mp Q = I, ascending
    const n = Kp.length, L = cholesky(Mp);
    const fwd = (b) => { const y = zeros(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; } return y; };
    const bwd = (b) => { const x = zeros(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; };
    const W = transpose(Kp).map(fwd);                  // columns of L⁻¹ Kp, stored as rows
    const C = transpose(W).map(fwd);                   // L⁻¹ (L⁻¹ Kp)ᵀ = L⁻¹ Kp L⁻ᵀ
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = 0.5 * (C[i][j] + C[j][i]); C[i][j] = C[j][i] = v; }
    const { vals, V } = jacobiEig(C);
    const order = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
    const Q = order.map((i) => bwd(V.map((r) => r[i])));   // Q[k] = k-th eigenvector (column)
    return { vals: order.map((i) => vals[i]), Q };
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  // ================================================================ subspace iteration
  function subspace(nf, solveK, mulM, mulK, diagM, diagK, p, nmass) {
    const q = Math.min(nf, nmass, Math.max(2 * p, p + 8));
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 * 2 - 1; };
    let X = [Float64Array.from(diagM)];
    const cand = [...Array(nf).keys()].filter((i) => diagM[i] > 0).sort((a, b) => diagM[b] / diagK[b] - diagM[a] / diagK[a] || a - b);
    for (let k = 0; X.length < q - 1 && k < cand.length; k++) { const v = new Float64Array(nf); v[cand[k]] = 1; X.push(v); }
    while (X.length < q) { const v = new Float64Array(nf); for (let i = 0; i < nf; i++) v[i] = diagM[i] > 0 ? rnd() : 0; X.push(v); }
    let old = null, vals = null, it = 0, stall = 0, lastRes = null;
    for (it = 0; it < 500; it++) {
      const Y = X.map(mulM), Xb = Y.map(solveK), MXb = Xb.map(mulM);
      const Kp = Xb.map((xi) => Y.map((yj) => dot(xi, yj)));
      const Mp = Xb.map((xi) => MXb.map((mj) => dot(xi, mj)));
      for (let i = 0; i < q; i++) for (let j = i + 1; j < q; j++) { const a = 0.5 * (Kp[i][j] + Kp[j][i]), b = 0.5 * (Mp[i][j] + Mp[j][i]); Kp[i][j] = Kp[j][i] = a; Mp[i][j] = Mp[j][i] = b; }
      const ge = genEig(Kp, Mp);
      vals = ge.vals;
      X = ge.Q.map((Qc) => { const v = new Float64Array(nf); for (let r = 0; r < q; r++) { const c = Qc[r]; if (c) { const xr = Xb[r]; for (let i = 0; i < nf; i++) v[i] += c * xr[i]; } } return v; });
      if (old) {
        let dl = 0; for (let i = 0; i < p; i++) dl = Math.max(dl, Math.abs(vals[i] - old[i]) / vals[i]);
        if (dl < 1e-13) {
          let res = 0;
          for (let i = 0; i < p; i++) {
            const kx = mulK(X[i]), mx = mulM(X[i]);
            let num2 = 0, den2 = 0; for (let j = 0; j < nf; j++) { const d = kx[j] - vals[i] * mx[j]; num2 += d * d; den2 += kx[j] * kx[j]; }
            res = Math.max(res, Math.sqrt(num2 / den2));
          }
          if (res < 1e-11) break;
          // converged to round-off: the residual of an ill-conditioned model stalls above 1e-11
          stall = res < 1e-8 && lastRes !== null && res > 0.8 * lastRes ? stall + 1 : 0;
          if (stall >= 4) break;
          lastRes = res;
        }
      }
      old = vals;
    }
    return { lam: vals.slice(0, p), X: X.slice(0, p), iterations: it + 1 };
  }

  // ================================================================ modal analysis
  function normaliseSign(v) {
    let mx = 0; for (const x of v) mx = Math.max(mx, Math.abs(x));
    for (let i = 0; i < v.length; i++) if (Math.abs(v[i]) >= mx * (1 - 1e-6)) { if (v[i] < 0) for (let j = 0; j < v.length; j++) v[j] = -v[j]; break; }
  }
  function modal(model, dynIn) {
    const dyn = dynIn || model.dynamics || { modes: 12, mass: "consistent", segments: 2, mass_cases: null, masses: [] };
    const { ref, submap } = refine(model, dyn.segments);
    const fa = new T.FrameAnalysis3D(ref);
    const factors = massFactors(model, dyn);
    const M = assembleMass(ref, fa, factors, dyn);
    const free = fa.free, nf = free.length, pos = new Int32Array(fa.ndof).fill(-1);
    free.forEach((d, k) => { pos[d] = k; });
    const Mff = free.map((d) => { const row = []; for (const [c, v] of M[d]) if (pos[c] >= 0) row.push([pos[c], v]); return row; });
    const Kff = free.map((d) => { const row = []; for (const [c, v] of fa.K[d]) if (pos[c] >= 0) row.push([pos[c], v]); return row; });
    const mul = (rows) => (x) => { const y = new Float64Array(nf); for (let i = 0; i < nf; i++) { let s = 0; for (const [c, v] of rows[i]) s += v * x[c]; y[i] = s; } return y; };
    const diag = (rows) => rows.map((row, i) => { let d = 0; for (const [c, v] of row) if (c === i) d += v; return d; });
    const diagM = diag(Mff), diagK = diag(Kff);
    const mmax = Math.max(...diagM, 1e-300);
    const nmass = diagM.filter((v) => v > 1e-12 * mmax).length;
    const warnings = [];
    const k = Math.min(dyn.modes, nmass);
    if (k === 0) throw new ModelError("the model has no mass: give load cases a mass factor or switch on self weight");
    if (k < dyn.modes) warnings.push(`Only ${k} modes exist with mass; ${dyn.modes} were requested.`);
    const sol = subspace(nf, (b) => Float64Array.from(fa.solve(Array.from(b))), mul(Mff), mul(Kff), diagM, diagK, k, nmass);
    const omega = [], phi = [];
    sol.lam.forEach((w2, i) => {
      if (!(w2 > 0)) return;
      const v = new Float64Array(fa.ndof);
      free.forEach((d, j) => { v[d] = sol.X[i][j]; });
      normaliseSign(v);
      omega.push(Math.sqrt(w2)); phi.push(v);
    });
    const gamma = {}, meff = {}, mtot = {};
    AXES.forEach((d, q) => {
      const Mi = new Float64Array(fa.ndof);
      for (let r = 0; r < fa.ndof; r++) { let s = 0; for (const [c, v] of M[r]) if (c % 6 === q) s += v; Mi[r] = s; }
      gamma[d] = phi.map((v) => dot(v, Mi));
      meff[d] = gamma[d].map((g) => g * g);
      let t = 0; for (let r = q; r < fa.ndof; r += 6) t += Mi[r]; mtot[d] = t;
    });
    const res = { settings: dyn, model: ref, analysis: fa, submap, M, omega, phi, gamma, meff, mtot, factors, warnings, iterations: sol.iterations };
    res.period = omega.map((w) => 2 * Math.PI / w);
    res.freq = omega.map((w) => w / (2 * Math.PI));
    res.ratio = (d) => meff[d].map((m) => (mtot[d] > 0 ? m / mtot[d] : 0));
    return res;
  }

  // shape of mode vector `vec` along an original member: points and global displacements
  function memberShape(mr, mid, vec, ppe = 6) {
    const fa = mr.analysis, P = [], D = [];
    for (const [eid] of mr.submap[mid]) {
      const md = fa.members[eid], d = dofsOf(md), lam = md.lam;
      let ul = zeros(12);
      for (let b = 0; b < 4; b++) for (let r = 0; r < 3; r++) ul[3 * b + r] = lam[r][0] * vec[d[3 * b]] + lam[r][1] * vec[d[3 * b + 1]] + lam[r][2] * vec[d[3 * b + 2]];
      if (md.cidx.length) { const Tr = releaseTransform(md); ul = Tr.map((row) => row.reduce((s, v, c) => s + v * ul[c], 0)); }
      const ni = mr.model.nodes[mr.model.members[eid].i];
      for (let k = P.length ? 1 : 0; k <= ppe; k++) {
        const t = k / ppe, N = shape(t, md.L);
        const u = N[0] * ul[0] + N[3] * ul[6];
        const v = N[1] * ul[1] + N[2] * ul[5] + N[4] * ul[7] + N[5] * ul[11];
        const w = N[1] * ul[2] - N[2] * ul[4] + N[4] * ul[8] - N[5] * ul[10];
        P.push([ni.x + lam[0][0] * md.L * t, ni.y + lam[0][1] * md.L * t, ni.z + lam[0][2] * md.L * t]);
        D.push([0, 1, 2].map((c) => lam[0][c] * u + lam[1][c] * v + lam[2][c] * w));
      }
    }
    return { P, D };
  }

  // ================================================================ EN 1998-1 response spectrum
  const SPECTRUM = {
    1: { A: [1.0, 0.15, 0.4, 2.0], B: [1.2, 0.15, 0.5, 2.0], C: [1.15, 0.20, 0.6, 2.0], D: [1.35, 0.20, 0.8, 2.0], E: [1.4, 0.15, 0.5, 2.0] },
    2: { A: [1.0, 0.05, 0.25, 1.2], B: [1.35, 0.05, 0.25, 1.2], C: [1.5, 0.10, 0.25, 1.2], D: [1.8, 0.10, 0.30, 1.2], E: [1.6, 0.05, 0.25, 1.2] },
  };
  const QUANT = ["N", "Vy", "Vz", "T", "My", "Mz"];
  const CASE_E = "AEd", CASE_E2 = "AEd'";
  function designSpectrum(Tp, s) {
    const [S, TB, TC, TD] = SPECTRUM[s.spectrum_type][s.ground], ag = s.agR * s.importance, q = s.q;
    if (Tp <= TB) return ag * S * (2 / 3 + Tp / TB * (2.5 / q - 2 / 3));
    if (Tp <= TC) return ag * S * 2.5 / q;
    if (Tp <= TD) return Math.max(ag * S * 2.5 / q * TC / Math.max(Tp, 1e-12), s.beta * ag);
    return Math.max(ag * S * 2.5 / q * TC * TD / Math.max(Tp, 1e-12) ** 2, s.beta * ag);
  }
  function cqcRho(omega, zeta) {
    return omega.map((wi) => omega.map((wj) => { const r = wj / wi; return 8 * zeta * zeta * (1 + r) * r ** 1.5 / ((1 - r * r) ** 2 + 4 * zeta * zeta * r * (1 + r) ** 2); }));
  }
  function cqc(vals, rho) {   // vals: modal values (array over modes)
    let s = 0; const k = vals.length;
    for (let i = 0; i < k; i++) { const vi = vals[i]; if (!vi) continue; const ri = rho[i]; for (let j = 0; j < k; j++) s += vi * ri[j] * vals[j]; }
    return Math.sqrt(Math.max(s, 0));
  }
  const combineDirs = (perDir) => {
    const ks = Object.keys(perDir);
    if (ks.length === 1) return perDir[ks[0]];
    const a = perDir.X, b = perDir.Y;
    return Array.isArray(a) || a instanceof Float64Array ? Array.from(a, (v, i) => Math.max(v + 0.3 * b[i], 0.3 * v + b[i])) : Math.max(a + 0.3 * b, 0.3 * a + b);
  };
  function modalEndForces(mr, U) {   // U: array over modes of Float64Array(ndof) -> {eid: [mode][12]}
    const fa = mr.analysis, out = {};
    for (const [eid, md] of Object.entries(fa.members)) {
      const d = dofsOf(md), lam = md.lam, Tr = md.cidx.length ? releaseTransform(md) : null;
      out[eid] = U.map((u) => {
        let ul = zeros(12);
        for (let b = 0; b < 4; b++) for (let r = 0; r < 3; r++) ul[3 * b + r] = lam[r][0] * u[d[3 * b]] + lam[r][1] * u[d[3 * b + 1]] + lam[r][2] * u[d[3 * b + 2]];
        if (Tr) ul = Tr.map((row) => row.reduce((s, v, c) => s + v * ul[c], 0));
        return md.k_full.map((row) => row.reduce((s, v, c) => s + v * ul[c], 0));
      });
    }
    return out;
  }
  function stationsFromElements(mr, mid, xs, F) {
    const subs = mr.submap[mid], k = F[subs[0][0]].length;
    const res = Object.fromEntries(QUANT.map((q) => [q, Array.from({ length: k }, () => zeros(xs.length))]));
    xs.forEach((x, s) => {
      const e = subs.find((sb) => x <= sb[2] + 1e-9) || subs[subs.length - 1], xl = x - e[1], f = F[e[0]];
      for (let i = 0; i < k; i++) {
        const fi = f[i];
        res.N[i][s] = -fi[0]; res.Vy[i][s] = fi[1]; res.Vz[i][s] = fi[2]; res.T[i][s] = -fi[3];
        res.Mz[i][s] = -fi[5] + xl * fi[1]; res.My[i][s] = fi[4] + xl * fi[2];
      }
    });
    return res;
  }
  function gravityFactors(model) {
    const psi = Object.assign({}, I.PSI);
    for (const [cat, vals] of Object.entries(model.settings.params.psi || {})) psi[cat] = vals.map(Number);
    return Object.fromEntries(Object.entries(model.load_cases).map(([c, lc]) => [c, lc.type === "permanent" ? 1.0 : psi[lc.category][2]]));
  }
  function rsa(model, faS, cases, mr) {
    const s = model.seismic, Tm = mr.period, Sd = Tm.map((t) => designSpectrum(t, s)), rho = cqcRho(mr.omega, s.damping), k = Tm.length;
    const warnings = [];
    for (const d of s.directions) {
      const cum = mr.ratio(d).reduce((a, b) => a + b, 0);
      if (cum < 0.9) warnings.push(`The ${k} modes capture ${Math.round(100 * cum)} % of the mass in ${d}; EN 1998-1 4.3.3.3.1(3) asks for 90 %. Increase the number of modes.`);
    }
    warnings.push("Accidental torsional effects (EN 1998-1 4.3.3.3.3) are not included.");
    const fa = mr.analysis, sup = Object.keys(model.supports);
    const perM = {}, perR = {}, perD = {}, modalU = {}, modalF = {}, baseShear = {}, modalVb = {};
    const col = (A, s) => A.map((row) => row[s]);
    for (const d of s.directions) {
      const U = mr.phi.map((v, i) => { const f = mr.gamma[d][i] * Sd[i] / (mr.omega[i] ** 2); return v.map((x) => x * f); });
      modalU[d] = U;
      const F = modalEndForces(mr, U);
      modalF[d] = F;
      perM[d] = {};
      for (const mid of Object.keys(model.members)) {
        const xs = faS.stations[mid].x, mv = stationsFromElements(mr, mid, xs, F);
        perM[d][mid] = Object.fromEntries(QUANT.map((q) => [q, xs.map((_, st) => cqc(col(mv[q], st), rho))]));
      }
      perR[d] = {};
      const qd = "XY".indexOf(d), vb = zeros(k);
      for (const nid of sup) {
        const b = 6 * fa.nidx[nid];
        const Rm = [0, 1, 2, 3, 4, 5].map((c) => U.map((u) => { if (!fa.fixed[b + c]) return 0; let t = 0; for (const [cc, v] of fa.K[b + c]) t += v * u[cc]; return t; }));
        perR[d][nid] = Rm.map((row) => cqc(row, rho));
        Rm[qd].forEach((v, i) => { vb[i] += v; });
      }
      modalVb[d] = vb;
      baseShear[d] = cqc(vb, rho);
      perD[d] = {};
      for (const nid of Object.keys(model.nodes)) { const b = 6 * fa.nidx[nid]; perD[d][nid] = [0, 1, 2, 3, 4, 5].map((c) => cqc(U.map((u) => u[b + c]), rho)); }
    }
    const members = {}, reactions = {}, displacements = {};
    for (const mid of Object.keys(model.members)) members[mid] = Object.fromEntries(QUANT.map((q) => [q, combineDirs(Object.fromEntries(s.directions.map((d) => [d, perM[d][mid][q]])))]));
    for (const nid of sup) reactions[nid] = combineDirs(Object.fromEntries(s.directions.map((d) => [d, perR[d][nid]])));
    for (const nid of Object.keys(model.nodes)) displacements[nid] = combineDirs(Object.fromEntries(s.directions.map((d) => [d, perD[d][nid]])));
    // drift per column and storey sensitivity
    const gf = gravityFactors(model), drift = {}, storeys = new Map();
    const r3 = (v) => Math.round(v * 1000) / 1000;
    for (const [mid, m] of Object.entries(model.members)) {
      if (m.design.type !== "column") continue;
      const a = model.nodes[m.i], b = model.nodes[m.j], h = Math.abs(b.z - a.z);
      if (h < 1e-6) continue;
      const rel = {};
      for (const [comp, q] of [["x", 0], ["y", 1]]) {
        const vals = {};
        for (const d of s.directions) vals[d] = cqc(modalU[d].map((u) => u[6 * fa.nidx[m.j] + q] - u[6 * fa.nidx[m.i] + q]), rho);
        rel[comp] = combineDirs(vals);
      }
      const up = b.z >= a.z;
      let Ng = 0; for (const [cid, f] of Object.entries(gf)) { const N = cases[cid].members[mid].N; Ng += f * N[up ? N.length - 1 : 0]; }
      const subs = mr.submap[mid], eid = up ? subs[0][0] : subs[subs.length - 1][0], lam = fa.members[eid].lam, shear = {};
      for (const d of s.directions) {
        shear[d] = [0, 1].map((c) => modalF[d][eid].map((f) => { const fb = up ? [f[0], f[1], f[2]] : [f[6], f[7], f[8]]; return lam[0][c] * fb[0] + lam[1][c] * fb[1] + lam[2][c] * fb[2]; }));
      }
      drift[mid] = { h, dre_x: rel.x, dre_y: rel.y, N_grav: Ng, shear, top_z: Math.max(a.z, b.z), bot_z: Math.min(a.z, b.z) };
      const key = `${r3(Math.min(a.z, b.z))}|${r3(Math.max(a.z, b.z))}`;
      if (!storeys.has(key)) storeys.set(key, []);
      storeys.get(key).push(mid);
    }
    const storeyList = [];
    const keys = [...storeys.keys()].sort((p, q) => { const [a0, a1] = p.split("|").map(Number), [b0, b1] = q.split("|").map(Number); return a0 - b0 || a1 - b1; });
    for (const key of keys) {
      const mids = storeys.get(key), [z0, z1] = key.split("|").map(Number), h = z1 - z0;
      const P = mids.reduce((t, m) => t + Math.max(-drift[m].N_grav, 0), 0);
      const V = {}, dr = {};
      for (const [comp, q] of [["x", 0], ["y", 1]]) {
        const vals = {};
        for (const d of s.directions) vals[d] = cqc(Array.from({ length: k }, (_, i) => mids.reduce((t, m) => t + drift[m].shear[d][q][i], 0)), rho);
        V[comp] = combineDirs(vals);
        dr[comp] = s.q * mids.reduce((t, m) => t + drift[m][`dre_${comp}`], 0) / mids.length;
      }
      const theta = Object.fromEntries(["x", "y"].map((c) => [c, V[c] > 1e-9 ? P * dr[c] / (V[c] * h) : 0]));
      storeyList.push({ z0, z1, h, columns: mids.length, P, V, dr, theta });
      const th = Math.max(theta.x, theta.y);
      for (const m of mids) drift[m].theta = th;
      if (th > 0.1) warnings.push(`Storey +${z0.toFixed(2)} to +${z1.toFixed(2)} m: θ = ${th.toFixed(3)} > 0.10, second-order effects are not included (EN 1998-1 4.4.2.2(3) asks for amplification by 1/(1 - θ)).`);
    }
    return { settings: s, modal: mr, Sd, base_shear: baseShear, modal_base_shear: modalVb, members, reactions, displacements, drift, storeys: storeyList, gravity_factors: gf, warnings,
      spectrumCurve: (n = 121) => Array.from({ length: n }, (_, i) => { const t = 4 * i / (n - 1); return [t, designSpectrum(t, s)]; }) };
  }

  function driftCheck(mid, sr) {
    const s = sr.settings, d = sr.drift[mid];
    const dre = Math.max(d.dre_x, d.dre_y), comp = d.dre_x >= d.dre_y ? "x" : "y", dr = s.q * dre, lim = s.drift_limit * d.h, util = s.nu * dr / lim, th = d.theta || 0;
    const steps = [
      step("d_{r,e}", `elastic inter-storey drift, global ${comp} (CQC, 30 % rule)`, "", dre * 1000, "mm", "4.3.3.3.2"),
      step("d_{r}", "q·d_{r,e}", `${fmt(s.q)}·${fmt(dre * 1000)}`, dr * 1000, "mm", "4.3.4(1)"),
      step("h", "storey height", "", d.h * 1000, "mm"),
      step("ν", "reduction factor for the damage limitation seismic action", "", s.nu, "", "4.4.3.2(2)"),
      step("α·h", "drift limit", `${fmt(s.drift_limit)}·${fmt(d.h * 1000)}`, lim * 1000, "mm", "4.4.3.2(1)"),
      step("η", "ν·d_{r}/(α·h)", `${fmt(s.nu)}·${fmt(dr * 1000)}/${fmt(lim * 1000)}`, util, "", "Eq. 4.31-4.33"),
      step("θ", "P_{tot}·d_{r}/(V_{tot}·h) of the storey", "", th, "", "4.4.2.2(2)"),
    ];
    const notes = ["Limit α = 0.005 for brittle non-structural elements, 0.0075 for ductile ones, 0.010 where they do not interfere."];
    if (th > 0.1) notes.push("θ > 0.10: second-order effects should be included by 1/(1 - θ); this check does not apply it.");
    let st = statusU(util);
    if (th > 0.3) { st = FAIL; notes.push("θ > 0.30 is not permitted (4.4.2.2(4))."); }
    return check({ id: "ec8.drift", title: "Damage limitation, inter-storey drift", clause: "EN 1998-1 4.4.3.2", member: mid, status: st, utilisation: util,
      demand: dr * 1000, capacity: lim * 1000 / s.nu, unit: "mm", combination: CASE_E, combination_label: "design seismic action (A_Ed)", steps, notes });
  }

  function pseudoCases(faS, sr, model) {
    const out = {};
    const disp = faS.nodeIds.map((nid) => Array.from(sr.displacements[nid]));
    for (const [cid, sg] of [[CASE_E, 1], [CASE_E2, -1]]) {
      const mem = {};
      for (const mid of Object.keys(model.members)) {
        const { x, side } = faS.stations[mid], e = sr.members[mid], z = zeros(x.length);
        mem[mid] = { x, side, N: e.N.map((v) => sg * v), Vy: e.Vy, Vz: e.Vz, T: e.T, My: e.My, Mz: e.Mz, u: z, v: z, w: z, v_rel: z, w_rel: z, end_forces: zeros(12) };
      }
      const zero = { Fx: 0, Fy: 0, Fz: 0, Mx: 0, My: 0, Mz: 0 };
      out[cid] = { caseId: cid, displacements: disp.map((r) => r.slice()), reactions: Object.fromEntries(Object.entries(sr.reactions).map(([n, v]) => [n, Array.from(v)])),
        springs: {}, members: mem, equilibrium: Object.assign({}, zero), applied: Object.assign({}, zero), pseudo: true };
    }
    return out;
  }

  function seismicCombinations(model, cs, gf) {
    const ids = cs.caseIds.concat([CASE_E, CASE_E2]);
    const base = cs.combos.map((c) => Object.assign({}, c, { factors: c.factors.concat([0, 0]) }));
    const groups = new Map(), free = [];
    for (const [cid, lc] of Object.entries(model.load_cases)) {
      if (lc.type === "variable" && gf[cid] > 0) { if (lc.group) { if (!groups.has(lc.group)) groups.set(lc.group, []); groups.get(lc.group).push(cid); } else free.push(cid); }
    }
    let picks = [[]];
    for (const members of groups.values()) { const next = []; for (const pre of picks) for (const c of members) next.push(pre.concat([c])); picks = next; }
    const out = [], seen = new Set();
    let n = 0;
    for (const pick of picks) {
      const vec = zeros(ids.length);
      for (const [cid, lc] of Object.entries(model.load_cases)) if (lc.type === "permanent") vec[ids.indexOf(cid)] = 1.0;
      for (const cid of free.concat(pick)) vec[ids.indexOf(cid)] = gf[cid];
      for (const [e, sg] of [[CASE_E, 1], [CASE_E, -1], [CASE_E2, 1], [CASE_E2, -1]]) {
        const v = vec.slice();
        v[ids.indexOf(e)] = sg;
        const key = v.map((x) => Math.round(x * 1e9) / 1e9).join(",");
        if (seen.has(key)) continue;
        seen.add(key); n += 1;
        const parts = []; v.slice(0, -2).forEach((f, i) => { if (Math.abs(f) > 1e-12) parts.push(`${f.toFixed(2)}·${ids[i]}`); });
        out.push({ id: `SEI${String(n).padStart(3, "0")}`, kind: "ULS", factors: v, leading: e, label: parts.join(" + ") + (sg > 0 ? " + " : " − ") + e });
      }
    }
    return { caseIds: ids, combos: base.concat(out) };
  }

  T.runDynamics = function (model, fa, cases, cs) {
    const mr = modal(model, model.dynamics);
    const warnings = mr.warnings.slice();
    let sr = null;
    if (model.seismic) {
      sr = rsa(model, fa, cases, mr);
      warnings.push(...sr.warnings);
      Object.assign(cases, pseudoCases(fa, sr, model));
      cs = seismicCombinations(model, cs, sr.gravity_factors);
    }
    return { modal: mr, seismic: sr, combos: cs, warnings };
  };
  Object.assign(T, { modal, refine, memberShape, designSpectrum, cqcRho, driftCheck, SPECTRUM, CASE_E, CASE_E2, elementMassLocal, releaseTransform });
  T.driftCheck = driftCheck;
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
