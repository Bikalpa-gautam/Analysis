/* Tragwerk FE lab: plane stress / plane strain continuum elements in the browser.
 *
 * A port of tragwerk/fe2d.py (same meshes, node numbering, quadrature and stress
 * recovery; the test suite compares both to round-off). Notation of the ETH
 * Method of Finite Elements I course: x = Σ N_i x_i, J = ∂x/∂ξ, B = ∂N/∂x,
 * k_e = Σ_g w_g Bᵀ C B t det J. Elements T3, T6, Q4, Q8, Q8R. Units: m, kN, MPa.
 */
(function (root) {
  "use strict";
  const T = root.Tragwerk || (root.Tragwerk = {});
  const SCHEMA_FE = "tragwerk-fe/0.1";
  const TYPES = ["T3", "T6", "Q4", "Q8", "Q8R"];
  const NN = { T3: 3, T6: 6, Q4: 4, Q8: 8, Q8R: 8 };
  const EDGES = { T3: [[0, 1], [1, 2], [2, 0]], T6: [[0, 3, 1], [1, 4, 2], [2, 5, 0]], Q4: [[0, 1], [1, 2], [2, 3], [3, 0]], Q8: [[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]] };
  EDGES.Q8R = EDGES.Q8;
  const QUADRATIC = ["T6", "Q8", "Q8R"];
  const Q4N = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  const Q8N = [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]];
  const G2 = 1 / Math.sqrt(3), G3 = Math.sqrt(0.6);
  const MAX_NODES = 40000;
  class FEError extends Error {}

  // ------------------------------------------------------------ shape functions and quadrature
  function shape(et, xi, eta) {
    if (et === "T3") return { N: [1 - xi - eta, xi, eta], dN: [[-1, 1, 0], [-1, 0, 1]] };
    if (et === "T6") {
      const L1 = 1 - xi - eta, L2 = xi, L3 = eta, dL = [[-1, 1, 0], [-1, 0, 1]];
      const N = [L1 * (2 * L1 - 1), L2 * (2 * L2 - 1), L3 * (2 * L3 - 1), 4 * L1 * L2, 4 * L2 * L3, 4 * L3 * L1];
      const dN = dL.map(([d1, d2, d3]) => [(4 * L1 - 1) * d1, (4 * L2 - 1) * d2, (4 * L3 - 1) * d3, 4 * (d1 * L2 + L1 * d2), 4 * (d2 * L3 + L2 * d3), 4 * (d3 * L1 + L3 * d1)]);
      return { N, dN };
    }
    if (et === "Q4") {
      return { N: Q4N.map(([a, b]) => 0.25 * (1 + xi * a) * (1 + eta * b)), dN: [Q4N.map(([a, b]) => 0.25 * a * (1 + eta * b)), Q4N.map(([a, b]) => 0.25 * b * (1 + xi * a))] };
    }
    if (et === "Q8" || et === "Q8R") {
      const N = new Array(8), d0 = new Array(8), d1 = new Array(8);
      Q8N.forEach(([a, b], i) => {
        if (i < 4) {
          N[i] = 0.25 * (1 + xi * a) * (1 + eta * b) * (xi * a + eta * b - 1);
          d0[i] = 0.25 * a * (1 + eta * b) * (2 * xi * a + eta * b);
          d1[i] = 0.25 * b * (1 + xi * a) * (xi * a + 2 * eta * b);
        } else if (a === 0) {
          N[i] = 0.5 * (1 - xi * xi) * (1 + eta * b); d0[i] = -xi * (1 + eta * b); d1[i] = 0.5 * b * (1 - xi * xi);
        } else {
          N[i] = 0.5 * (1 + xi * a) * (1 - eta * eta); d0[i] = 0.5 * a * (1 - eta * eta); d1[i] = -eta * (1 + xi * a);
        }
      });
      return { N, dN: [d0, d1] };
    }
    throw new FEError(`unknown element type ${et}`);
  }
  function gauss(et, purpose = "stiffness") {
    if (et === "Q4" || et === "Q8" || et === "Q8R") {
      const two = (et === "Q4" || et === "Q8R") && purpose === "stiffness";
      const p = two ? [-G2, G2] : [-G3, 0, G3], w = two ? [1, 1] : [5 / 9, 8 / 9, 5 / 9];
      const out = [];
      for (let j = 0; j < p.length; j++) for (let i = 0; i < p.length; i++) out.push([p[i], p[j], w[i] * w[j]]);
      return out;
    }
    if (purpose === "mass") {
      const a = 0.445948490915965, b = 0.091576213509771, wa = 0.223381589678011 / 2, wb = 0.109951743655322 / 2;
      return [[a, a, wa], [1 - 2 * a, a, wa], [a, 1 - 2 * a, wa], [b, b, wb], [1 - 2 * b, b, wb], [b, 1 - 2 * b, wb]];
    }
    if (et === "T3") return [[1 / 3, 1 / 3, 0.5]];
    return [[1 / 6, 1 / 6, 1 / 6], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6]];
  }
  function constitutive(E, nu, analysis) {
    if (analysis === "plane_stress") { const c = E / (1 - nu * nu); return [[c, c * nu, 0], [c * nu, c, 0], [0, 0, c * (1 - nu) / 2]]; }
    if (analysis === "plane_strain") { const c = E / ((1 + nu) * (1 - 2 * nu)); return [[c * (1 - nu), c * nu, 0], [c * nu, c * (1 - nu), 0], [0, 0, c * (1 - 2 * nu) / 2]]; }
    throw new FEError(`analysis must be plane_stress or plane_strain, not ${analysis}`);
  }
  function gaussPoint(et, xy, xi, eta, w = 0) {
    const { N, dN } = shape(et, xi, eta), n = N.length;
    const J = [[0, 0], [0, 0]];
    for (let i = 0; i < n; i++) { J[0][0] += dN[0][i] * xy[i][0]; J[0][1] += dN[0][i] * xy[i][1]; J[1][0] += dN[1][i] * xy[i][0]; J[1][1] += dN[1][i] * xy[i][1]; }
    const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    if (!(det > 0)) throw new FEError(`element is inverted or badly distorted: det J = ${det.toPrecision(3)} at (ξ, η) = (${xi.toFixed(3)}, ${eta.toFixed(3)})`);
    const Ji = [[J[1][1] / det, -J[0][1] / det], [-J[1][0] / det, J[0][0] / det]];
    const dx = [new Array(n), new Array(n)];
    for (let i = 0; i < n; i++) { dx[0][i] = Ji[0][0] * dN[0][i] + Ji[0][1] * dN[1][i]; dx[1][i] = Ji[1][0] * dN[0][i] + Ji[1][1] * dN[1][i]; }
    const B = [new Array(2 * n).fill(0), new Array(2 * n).fill(0), new Array(2 * n).fill(0)];
    let x = 0, y = 0;
    for (let i = 0; i < n; i++) { B[0][2 * i] = dx[0][i]; B[1][2 * i + 1] = dx[1][i]; B[2][2 * i] = dx[1][i]; B[2][2 * i + 1] = dx[0][i]; x += N[i] * xy[i][0]; y += N[i] * xy[i][1]; }
    return { xi, eta, w, N, dN, J, detJ: det, dNdx: dx, B, x, y };
  }
  function elementStiffness(et, xy, C, t) {
    const n2 = 2 * NN[et], k = Array.from({ length: n2 }, () => new Float64Array(n2)), gps = [];
    for (const [xi, eta, w] of gauss(et)) {
      const g = gaussPoint(et, xy, xi, eta, w), B = g.B, f = w * t * g.detJ;
      const CB = [0, 1, 2].map((r) => { const row = new Float64Array(n2); for (let c = 0; c < n2; c++) row[c] = C[r][0] * B[0][c] + C[r][1] * B[1][c] + C[r][2] * B[2][c]; return row; });
      for (let a = 0; a < n2; a++) {
        const b0 = B[0][a], b1 = B[1][a], b2 = B[2][a];
        if (!b0 && !b1 && !b2) continue;
        const ka = k[a];
        for (let c = 0; c < n2; c++) ka[c] += f * (b0 * CB[0][c] + b1 * CB[1][c] + b2 * CB[2][c]);
      }
      gps.push(g);
    }
    return { k, gps };
  }
  function elementMass(et, xy, rho, t) {
    const n = NN[et], m = Array.from({ length: 2 * n }, () => new Float64Array(2 * n));
    for (const [xi, eta, w] of gauss(et, "mass")) {
      const g = gaussPoint(et, xy, xi, eta, w), f = w * rho * t * g.detJ;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = f * g.N[i] * g.N[j]; m[2 * i][2 * j] += v; m[2 * i + 1][2 * j + 1] += v; }
    }
    return m;
  }
  function bodyForce(et, xy, b, t) {
    const n = NN[et], f = new Float64Array(2 * n);
    for (const [xi, eta, w] of gauss(et, "mass")) {
      const g = gaussPoint(et, xy, xi, eta, w), s = w * t * g.detJ;
      for (let i = 0; i < n; i++) { f[2 * i] += s * g.N[i] * b[0]; f[2 * i + 1] += s * g.N[i] * b[1]; }
    }
    return f;
  }
  function extrapolation(et) {
    if (et === "T3") return { pts: [[1 / 3, 1 / 3]], E: [[1], [1], [1]] };
    if (et === "T6") return { pts: [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], E: [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 2 : 0) - 1 / 3)) };
    const r3 = Math.sqrt(3);
    return { pts: [[-G2, -G2], [G2, -G2], [G2, G2], [-G2, G2]], E: Q4N.map(([a, b]) => shape("Q4", a * r3, b * r3).N) };
  }
  const MIDS = { T6: [[3, 0, 1], [4, 1, 2], [5, 2, 0]], Q8: [[4, 0, 1], [5, 1, 2], [6, 2, 3], [7, 3, 0]], T3: [], Q4: [] };
  MIDS.Q8R = MIDS.Q8;

  // ------------------------------------------------------------ meshes
  function lines(breaks, h) {
    const out = [breaks[0]];
    for (let k = 0; k + 1 < breaks.length; k++) {
      const a = breaks[k], b = breaks[k + 1], n = Math.max(1, Math.floor((b - a) / h + 0.5));
      for (let q = 1; q <= n; q++) out.push(a + (b - a) * q / n);
    }
    return out;
  }
  function assembleGrid(point, keep, nx, ny, et) {
    const quad = QUADRATIC.includes(et), conn = [];
    const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
    for (const [i, j] of keep) {
      const a = [2 * i, 2 * j], b = [2 * i + 2, 2 * j], c = [2 * i + 2, 2 * j + 2], d = [2 * i, 2 * j + 2];
      if (et === "Q4") conn.push([a, b, c, d]);
      else if (et === "Q8" || et === "Q8R") conn.push([a, b, c, d, mid(a, b), mid(b, c), mid(c, d), mid(d, a)]);
      else {
        const tris = (i + j) % 2 === 0 ? [[a, b, c], [a, c, d]] : [[a, b, d], [b, c, d]];
        for (const [p, q, r] of tris) conn.push(quad ? [p, q, r, mid(p, q), mid(q, r), mid(r, p)] : [p, q, r]);
      }
    }
    const key = (k) => `${k[0]},${k[1]}`, uniq = new Map();
    for (const e of conn) for (const k of e) uniq.set(key(k), k);
    const used = [...uniq.values()].sort(nx >= ny ? (p, q) => p[0] - q[0] || p[1] - q[1] : (p, q) => p[1] - q[1] || p[0] - q[0]);
    const index = new Map(used.map((k, n) => [key(k), n]));
    return { nodes: used.map((k) => point(k[0], k[1])), elements: conn.map((e) => e.map((k) => index.get(key(k)))) };
  }
  function meshRectangle(Lx, Ly, h, et, openings) {
    const bx = new Set([0, Lx]), by = new Set([0, Ly]), cl = (v, hi) => Math.min(Math.max(v, 0), hi);
    for (const o of openings) { bx.add(cl(o.x0, Lx)); bx.add(cl(o.x1, Lx)); by.add(cl(o.y0, Ly)); by.add(cl(o.y1, Ly)); }
    const xs = lines([...bx].sort((a, b) => a - b), h), ys = lines([...by].sort((a, b) => a - b), h);
    const nx = xs.length - 1, ny = ys.length - 1, keep = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const cx = (xs[i] + xs[i + 1]) / 2, cy = (ys[j] + ys[j + 1]) / 2;
      if (!openings.some((o) => o.x0 < cx && cx < o.x1 && o.y0 < cy && cy < o.y1)) keep.push([i, j]);
    }
    if (!keep.length) throw new FEError("the openings remove the whole wall");
    const X = Array.from({ length: 2 * nx + 1 }, (_, k) => (k % 2 === 0 ? xs[k / 2] : (xs[(k - 1) / 2] + xs[(k + 1) / 2]) / 2));
    const Y = Array.from({ length: 2 * ny + 1 }, (_, k) => (k % 2 === 0 ? ys[k / 2] : (ys[(k - 1) / 2] + ys[(k + 1) / 2]) / 2));
    return assembleGrid((p, q) => [X[p], Y[q]], keep, nx, ny, et);
  }
  function meshPlateHole(W, a, nTheta, nR, et, grading = 1.6) {
    if (!(a > 0 && a < W)) throw new FEError("the hole radius must be smaller than the half width");
    const nt = 2 * nTheta, rs = (j) => (j / nR) ** grading;
    const point = (p, q) => {
      const th = (Math.PI / 2) * p / (2 * nt);
      const s = q % 2 === 0 ? rs(q / 2) : 0.5 * (rs((q - 1) / 2) + rs((q + 1) / 2));
      const hx = p < 2 * nt ? a * Math.cos(th) : 0, hy = p < 2 * nt ? a * Math.sin(th) : a;
      let ox, oy;
      if (th <= Math.PI / 4 + 1e-12) { ox = W; oy = W * Math.tan(th); } else { ox = W / Math.tan(th); oy = W; }
      if (p === 2 * nt) ox = 0;
      return [(1 - s) * hx + s * ox, (1 - s) * hy + s * oy];
    };
    const keep = [];
    for (let i = 0; i < nt; i++) for (let j = 0; j < nR; j++) keep.push([j, i]);
    return assembleGrid((p, q) => point(q, p), keep, nR, nt, et);
  }
  function meshRing(a, b, nTheta, nR, et, grading = 1.0) {
    if (!(a > 0 && a < b)) throw new FEError("the inner radius must be smaller than the outer radius");
    const nt = nTheta, rs = (j) => (j / nR) ** grading;
    const point = (p, q) => {
      const th = (Math.PI / 2) * p / (2 * nt);
      const s = q % 2 === 0 ? rs(q / 2) : 0.5 * (rs((q - 1) / 2) + rs((q + 1) / 2)), r = a + (b - a) * s;
      if (p === 0) return [r, 0];
      if (p === 2 * nt) return [0, r];
      return [r * Math.cos(th), r * Math.sin(th)];
    };
    const keep = [];
    for (let i = 0; i < nt; i++) for (let j = 0; j < nR; j++) keep.push([j, i]);
    return assembleGrid((p, q) => point(q, p), keep, nR, nt, et);
  }
  function edgeSets(nodes, elements, et, geometry) {
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const [x, y] of nodes) { lo = [Math.min(lo[0], x), Math.min(lo[1], y)]; hi = [Math.max(hi[0], x), Math.max(hi[1], y)]; }
    const tol = 1e-9 * Math.max(hi[0] - lo[0], hi[1] - lo[1], 1);
    const on = { left: (p) => Math.abs(p[0] - lo[0]) < tol, right: (p) => Math.abs(p[0] - hi[0]) < tol, bottom: (p) => Math.abs(p[1] - lo[1]) < tol, top: (p) => Math.abs(p[1] - hi[1]) < tol };
    if (geometry.type === "plate_hole") { const a = geometry.a; on.hole = (p) => Math.abs(Math.hypot(p[0], p[1]) - a) < 1e-7 * a; }
    if (geometry.type === "ring") { const ra = geometry.a, rb = geometry.b; on.inner = (p) => Math.abs(Math.hypot(p[0], p[1]) - ra) < 1e-7 * rb; on.outer = (p) => Math.abs(Math.hypot(p[0], p[1]) - rb) < 1e-7 * rb; }
    const out = Object.fromEntries(Object.keys(on).map((k) => [k, []]));
    for (const e of elements) for (const ed of EDGES[et]) {
      const ids = ed.map((k) => e[k]);
      for (const [name, f] of Object.entries(on)) if (ids.every((i) => f(nodes[i]))) out[name].push(ids);
    }
    return out;
  }
  function inRange(nodes, edges, name, spec, what) {
    if (!("from" in spec) && !("to" in spec)) return edges;
    if (!["left", "right", "bottom", "top"].includes(name)) throw new FEError(`${what}: from/to are available on the straight edges left, right, bottom and top`);
    const c = name === "bottom" || name === "top" ? 0 : 1;
    const lo = num(spec.from ?? -1e300, `${what}.from`), hi = num(spec.to ?? 1e300, `${what}.to`);
    let mn = Infinity, mx = -Infinity;
    for (const p of nodes) { mn = Math.min(mn, p[c]); mx = Math.max(mx, p[c]); }
    const tol = 1e-9 * Math.max(mx - mn, 1);
    return edges.filter((ed) => ed.every((i) => lo - tol <= nodes[i][c] && nodes[i][c] <= hi + tol));
  }
  const SUPPORT_KEYS = ["edge", "at", "ux", "uy", "from", "to", "label"];
  const LOAD_KEYS = ["edge", "at", "qx", "qy", "p", "fx", "fy", "case", "profile", "P", "dir", "from", "to", "label"];
  function known(spec, keys, what) {
    const bad = Object.keys(spec).filter((k) => !keys.includes(k));
    if (bad.length) throw new FEError(`${what}: unknown key '${bad[0]}' (allowed: ${keys.join(", ")})`);
  }
  function nearest(nodes, at) {
    let best = 0, bd = Infinity;
    nodes.forEach(([x, y], i) => { const d = Math.hypot(x - at[0], y - at[1]); if (d < bd) { bd = d; best = i; } });
    return best;
  }
  const GL5 = [[-0.906179845938664, 0.236926885056189], [-0.538469310105683, 0.478628670499366], [0, 0.568888888888889], [0.538469310105683, 0.478628670499366], [0.906179845938664, 0.236926885056189]];
  function edgeLoad(nodes, edge, qfun, p = 0) {
    const pts = edge.map((i) => nodes[i]), out = edge.map(() => [0, 0]);
    if (edge.length === 2) {
      for (const [s, w] of [[-G3, 5 / 9], [0, 8 / 9], [G3, 5 / 9]]) {
        const N = [(1 - s) / 2, (1 + s) / 2], dx = 0.5 * (pts[1][0] - pts[0][0]), dy = 0.5 * (pts[1][1] - pts[0][1]), ds = Math.hypot(dx, dy);
        const x = N[0] * pts[0][0] + N[1] * pts[1][0], y = N[0] * pts[0][1] + N[1] * pts[1][1];
        let [qx, qy] = qfun(x, y);
        qx = qx - p * dy / ds; qy = qy + p * dx / ds;
        N.forEach((Nk, k) => { out[k][0] += w * Nk * qx * ds; out[k][1] += w * Nk * qy * ds; });
      }
    } else {
      for (const [s, w] of GL5) {
        const N = [s * (s - 1) / 2, 1 - s * s, s * (s + 1) / 2], dN = [s - 0.5, -2 * s, s + 0.5];
        const dx = dN.reduce((t, d, k) => t + d * pts[k][0], 0), dy = dN.reduce((t, d, k) => t + d * pts[k][1], 0), ds = Math.hypot(dx, dy);
        const x = N.reduce((t, v, k) => t + v * pts[k][0], 0), y = N.reduce((t, v, k) => t + v * pts[k][1], 0);
        let [qx, qy] = qfun(x, y);
        qx = qx - p * dy / ds; qy = qy + p * dx / ds;
        N.forEach((Nk, k) => { out[k][0] += w * Nk * qx * ds; out[k][1] += w * Nk * qy * ds; });
      }
    }
    return edge.map((i, k) => [i, out[k][0], out[k][1]]);
  }
  function num(v, what, lo, hi) {
    const x = Number(v);
    if (v === null || v === undefined || v === "" || !isFinite(x) || (lo !== undefined && x < lo) || (hi !== undefined && x > hi)) throw new FEError(`${what} = ${v} is out of range`);
    return x;
  }

  // ------------------------------------------------------------ material, reinforcement, design
  function concreteProps(fck) {
    const fcm = fck + 8, fctm = fck <= 50 ? 0.30 * fck ** (2 / 3) : 2.12 * Math.log(1 + fcm / 10);
    return { fcm, fctm, fctk005: 0.7 * fctm, Ecm: 22000 * (fcm / 10) ** 0.3 };
  }
  function parseDesign(d) {
    const member = d.member ?? "wall";
    if (!["wall", "deep_beam", "none"].includes(member)) throw new FEError("design.member: wall, deep_beam or none");
    return { gamma_G: num(d.gamma_G ?? 1.35, "design.gamma_G", 0), gamma_Q: num(d.gamma_Q ?? 1.5, "design.gamma_Q", 0), psi2: num(d.psi2 ?? 0.3, "design.psi2", 0, 1),
      gamma_c: num(d.gamma_c ?? 1.5, "design.gamma_c", 1), gamma_s: num(d.gamma_s ?? 1.15, "design.gamma_s", 1), alpha_cc: num(d.alpha_cc ?? 1.0, "design.alpha_cc", 0.5, 1), member,
      w_max_mm: num(d.w_max_mm ?? 0.3, "design.w_max_mm", 0.01) };
  }
  function parseMaterial(mat, analysis, design, reinf) {
    const model = mat.model ?? "elastic";
    if (!["elastic", "j2", "rc"].includes(model)) throw new FEError("material.model must be elastic, j2 or rc");
    const out = { model };
    if (model === "j2") { out.fy = num(mat.fy_MPa ?? 355, "material.fy_MPa", 1e-3) * 1000; out.H = num(mat.H_MPa ?? 0, "material.H_MPa", 0) * 1000; }
    if (model === "rc") {
      if (analysis !== "plane_stress") throw new FEError("the reinforced concrete model is a plane stress (membrane) model");
      const fck = num(mat.fck_MPa ?? 30, "material.fck_MPa", 12, 90), cp = concreteProps(fck), values = mat.values ?? "design";
      if (!["design", "mean"].includes(values)) throw new FEError("material.values must be design or mean");
      const fyk = reinf.fyk;
      let fc, fct, fy;
      if (values === "design") { fc = design.alpha_cc * fck / design.gamma_c; fct = cp.fctk005 / design.gamma_c; fy = fyk / design.gamma_s; }
      else { fc = cp.fcm; fct = cp.fctm; fy = 1.1 * fyk; }
      const soft = mat.softening ?? "mcft";
      if (!["mcft", "ec2"].includes(soft)) throw new FEError("material.softening must be mcft or ec2");
      const tension = mat.tension ?? "stiffening";
      if (!["stiffening", "none"].includes(tension)) throw new FEError("material.tension must be stiffening or none");
      const Es = reinf.Es, eud = 0.9 * reinf.euk;
      Object.assign(out, { fck, values, fc: fc * 1000, fct: fct * 1000, eps_c0: num(mat.eps_c0 ?? 0.002, "material.eps_c0", 1e-4, 0.01), softening: soft, tension, nu_ec2: 0.6 * (1 - fck / 250),
        Gf: num(mat.Gf_Nmm ?? 0.073 * cp.fcm ** 0.18, "material.Gf_Nmm", 1e-3), fy: fy * 1000, Es: Es * 1000, Esh: (reinf.k - 1) * fy * 1000 / Math.max(eud - fy / Es, 1e-6), eud });
    }
    return out;
  }
  function parseReinforcement(r) {
    const out = { cover_mm: num(r.cover_mm ?? 30, "reinforcement.cover_mm", 0), fyk: num(r.fyk_MPa ?? 500, "reinforcement.fyk_MPa", 100, 1000), Es: num(r.Es_MPa ?? 200000, "reinforcement.Es_MPa", 1e3),
      k: num(r.k ?? 1.08, "reinforcement.k", 1, 1.5), euk: num(r.euk ?? 0.05, "reinforcement.euk", 0.01, 0.2), layers: [] };
    (r.layers || []).forEach((L, k) => {
      const w = `reinforcement.layers[${k}]`, d = L.dir;
      if (d !== "x" && d !== "y") throw new FEError(`${w}.dir must be x or y`);
      const dia = num(L.dia_mm, `${w}.dia_mm`, 4, 50), area = Math.PI * dia * dia / 4, region = L.region || L.band;
      let reg = null;
      if (region) {
        reg = Object.fromEntries(["x0", "y0", "x1", "y1"].map((q) => [q, num(region[q] ?? (q === "x0" || q === "y0" ? -1e9 : 1e9), `${w}.region.${q}`)]));
        if (reg.x1 <= reg.x0 || reg.y1 <= reg.y0) throw new FEError(`${w}: region needs x1 > x0 and y1 > y0`);
      }
      let As;
      if ("n" in L) {
        if (!reg) throw new FEError(`${w}: a tie of n bars needs its band (region)`);
        const width = d === "x" ? reg.y1 - reg.y0 : reg.x1 - reg.x0;
        As = num(L.n, `${w}.n`, 1, 200) * area / width;
      } else As = num(L.faces ?? 2, `${w}.faces`, 1, 2) * area * 1000 / num(L.spacing_mm, `${w}.spacing_mm`, 25, 1000);
      out.layers.push({ dir: d, dia, As, region: reg, label: L.label || "" });
    });
    return out;
  }
  function elementCentroids(m) { return m.elements.map((e) => { let x = 0, y = 0; for (const i of e) { x += m.nodes[i][0]; y += m.nodes[i][1]; } return [x / e.length, y / e.length]; }); }
  function elementReinforcement(m) {
    const c = elementCentroids(m), ne = m.elements.length, Asx = new Float64Array(ne), Asy = new Float64Array(ne), dx = new Float64Array(ne), dy = new Float64Array(ne);
    for (const L of m.reinforcement.layers) for (let q = 0; q < ne; q++) {
      const r = L.region;
      if (r && !(c[q][0] >= r.x0 && c[q][0] <= r.x1 && c[q][1] >= r.y0 && c[q][1] <= r.y1)) continue;
      if (L.dir === "x") { Asx[q] += L.As; dx[q] = Math.max(dx[q], L.dia); } else { Asy[q] += L.As; dy[q] = Math.max(dy[q], L.dia); }
    }
    return { Asx, Asy, diax: dx, diay: dy };
  }

  // ------------------------------------------------------------ model
  function parseFE(data) {
    if (data.schema !== SCHEMA_FE) throw new FEError(`expected schema '${SCHEMA_FE}'`);
    const analysis = data.analysis || "plane_stress", et = data.element || "Q8";
    if (!TYPES.includes(et)) throw new FEError(`element must be one of ${TYPES.join(", ")}`);
    const mat = data.material || {};
    const E = num(mat.E_MPa ?? 30000, "material.E_MPa", 1e-6) * 1000, nu = num(mat.nu ?? 0.2, "material.nu", 0, 0.499);
    const t = num(mat.thickness_m ?? 0.25, "material.thickness_m", 1e-6), gamma = num(mat.density_kNm3 ?? 25, "material.density_kNm3", 0);
    const C = constitutive(E, nu, analysis);
    const design = parseDesign(data.design || {}), reinforcement = parseReinforcement(data.reinforcement || {});
    design.fck = num(mat.fck_MPa ?? 30, "material.fck_MPa", 12, 90);
    const material = parseMaterial(mat, analysis, design, reinforcement);
    const g = data.geometry || {}, kind = g.type || "rectangle";
    let mesh;
    if (kind === "rectangle") {
      const Lx = num(g.Lx, "geometry.Lx", 1e-6), Ly = num(g.Ly, "geometry.Ly", 1e-6), h = num(g.h ?? Math.min(Lx, Ly) / 8, "geometry.h", 1e-6);
      if ((Lx / h) * (Ly / h) > MAX_NODES / (QUADRATIC.includes(et) ? 4 : 1)) throw new FEError("mesh too fine: increase the element size h");
      const openings = (g.openings || []).map((o, k) => {
        const q = Object.fromEntries(["x0", "y0", "x1", "y1"].map((key) => [key, num(o[key], `geometry.openings[${k}].${key}`)]));
        if (q.x1 <= q.x0 || q.y1 <= q.y0) throw new FEError(`opening ${k + 1}: x1 > x0 and y1 > y0 required`);
        return q;
      });
      mesh = meshRectangle(Lx, Ly, h, et, openings);
    } else if (kind === "plate_hole") {
      const W = num(g.W, "geometry.W", 1e-6), a = num(g.a, "geometry.a", 1e-6);
      const nt = Math.trunc(num(g.n_theta ?? 8, "geometry.n_theta", 1, 200)), nr = Math.trunc(num(g.n_r ?? 8, "geometry.n_r", 1, 200));
      mesh = meshPlateHole(W, a, nt, nr, et, num(g.grading ?? 1.6, "geometry.grading", 1, 4));
    } else if (kind === "ring") {
      const a = num(g.a, "geometry.a", 1e-6), b = num(g.b, "geometry.b", 1e-6);
      const nt = Math.trunc(num(g.n_theta ?? 8, "geometry.n_theta", 1, 200)), nr = Math.trunc(num(g.n_r ?? 6, "geometry.n_r", 1, 200));
      mesh = meshRing(a, b, nt, nr, et, num(g.grading ?? 1.0, "geometry.grading", 1, 4));
    } else if (kind === "mesh") {
      const nodes = (g.nodes || []).map((p) => [num(p[0], "node x"), num(p[1], "node y")]), elements = g.elements || [];
      if (!elements.length || elements.some((e) => e.length !== NN[et] || e.some((i) => !(i >= 0 && i < nodes.length) || i !== Math.trunc(i)))) throw new FEError(`geometry.elements must list ${NN[et]} valid node indices per ${et} element`);
      mesh = { nodes, elements: elements.map((e) => e.slice()) };
    } else throw new FEError(`geometry.type must be rectangle, plate_hole, ring or mesh, not ${kind}`);
    const { nodes, elements } = mesh;
    if (nodes.length > MAX_NODES) throw new FEError(`${nodes.length} nodes: at most ${MAX_NODES}`);
    const edges = edgeSets(nodes, elements, et, kind === "mesh" ? {} : g);
    const n = nodes.length, fixed = new Map();
    (data.supports || []).forEach((s, k) => {
      known(s, SUPPORT_KEYS, `supports[${k}]`);
      let ids;
      if ("edge" in s) {
        if (!(s.edge in edges)) throw new FEError(`supports[${k}]: unknown edge '${s.edge}' (one of ${Object.keys(edges).join(", ")})`);
        ids = [...new Set(inRange(nodes, edges[s.edge], s.edge, s, `supports[${k}]`).flat())].sort((p, q) => p - q);
        if (!ids.length) throw new FEError(`supports[${k}]: no element edges on '${s.edge}' in the given range`);
      } else if ("at" in s) ids = [nearest(nodes, s.at)];
      else throw new FEError(`supports[${k}] needs an edge or a point 'at'`);
      for (const i of ids) ["ux", "uy"].forEach((key, c) => {
        const v = s[key];
        if (v === true) fixed.set(2 * i + c, 0);
        else if (v !== false && v !== undefined && v !== null) fixed.set(2 * i + c, num(v, `supports[${k}].${key}`));
      });
    });
    if (!fixed.size) throw new FEError("the model has no supports");
    const f = new Float64Array(2 * n), fq = new Float64Array(2 * n);
    (data.loads || []).forEach((ld, k) => {
      known(ld, LOAD_KEYS, `loads[${k}]`);
      const lcase = ld.case ?? "G";
      if (lcase !== "G" && lcase !== "Q") throw new FEError(`loads[${k}].case must be G (permanent) or Q (variable)`);
      const add = (i, fx, fy) => { f[2 * i] += fx; f[2 * i + 1] += fy; if (lcase === "Q") { fq[2 * i] += fx; fq[2 * i + 1] += fy; } };
      if ("edge" in ld) {
        if (!edges[ld.edge] || !edges[ld.edge].length) throw new FEError(`loads[${k}]: unknown or empty edge '${ld.edge}'`);
        const eds = inRange(nodes, edges[ld.edge], ld.edge, ld, `loads[${k}]`);
        if (!eds.length) throw new FEError(`loads[${k}]: no element edges on '${ld.edge}' in the given range`);
        let qfun, pr = 0;
        if (ld.profile === "parabolic") {
          const P = num(ld.P ?? 0, `loads[${k}].P`), c = (ld.dir || "y") === "y" ? 1 : 0;
          const ids = [...new Set(eds.flat())], xs = ids.map((i) => nodes[i][0]), ys = ids.map((i) => nodes[i][1]);
          const sAx = Math.max(...xs) - Math.min(...xs) < Math.max(...ys) - Math.min(...ys) ? 1 : 0;
          const vals = sAx ? ys : xs, lo = Math.min(...vals), hi = Math.max(...vals), D = hi - lo;
          qfun = (x, y) => { const s = sAx ? y : x, q = 6 * P * (s - lo) * (hi - s) / D ** 3; return c === 0 ? [q, 0] : [0, q]; };
        } else {
          const qx = num(ld.qx ?? 0, `loads[${k}].qx`), qy = num(ld.qy ?? 0, `loads[${k}].qy`);
          pr = num(ld.p ?? 0, `loads[${k}].p`);
          qfun = () => [qx, qy];
        }
        for (const ed of eds) for (const [i, fx, fy] of edgeLoad(nodes, ed, qfun, pr)) add(i, fx, fy);
      } else if ("at" in ld) {
        add(nearest(nodes, ld.at), num(ld.fx ?? 0, `loads[${k}].fx`), num(ld.fy ?? 0, `loads[${k}].fy`));
      } else throw new FEError(`loads[${k}] needs an edge or a point 'at'`);
    });
    return { name: String(data.name || ""), analysis, etype: et, E, nu, t, gamma, C, nodes, elements, fixed, forces: f, forcesQ: fq, selfWeight: !!data.self_weight,
      geometry: Object.assign({}, g), edges, material, reinforcement, design, raw: data };
  }
  function combinationFactors(m, combo) {
    if (combo == null || combo === "char") return [1, 1];
    if (combo === "uls") return [m.design.gamma_G, m.design.gamma_Q];
    if (combo === "qp") return [1, m.design.psi2];
    throw new FEError("combination must be one of char, uls, qp");
  }
  function selfWeightVector(m) {
    const f = new Float64Array(2 * m.nodes.length);
    if (m.selfWeight && m.gamma > 0) m.elements.forEach((e) => {
      const fb = bodyForce(m.etype, e.map((i) => m.nodes[i]), [0, -m.gamma], m.t);
      e.forEach((i, k) => { f[2 * i] += fb[2 * k]; f[2 * i + 1] += fb[2 * k + 1]; });
    });
    return f;
  }
  function loadVector(m, combo) {
    const [aG, aQ] = combinationFactors(m, combo), sw = selfWeightVector(m), out = new Float64Array(sw.length);
    for (let d = 0; d < out.length; d++) out[d] = aG * (m.forces[d] - m.forcesQ[d] + sw[d]) + aQ * m.forcesQ[d];
    return out;
  }

  // ------------------------------------------------------------ profile (skyline) LDLᵀ
  const dofsOf = (e) => { const d = new Int32Array(2 * e.length); e.forEach((i, k) => { d[2 * k] = 2 * i; d[2 * k + 1] = 2 * i + 1; }); return d; };
  function makeProfile(m, fixed) {
    const n2 = 2 * m.nodes.length, eq = new Int32Array(n2).fill(-1);
    let nf = 0;
    for (let d = 0; d < n2; d++) if (!fixed.has(d)) eq[d] = nf++;
    const first = new Int32Array(nf);
    for (let q = 0; q < nf; q++) first[q] = q;
    for (const e of m.elements) {
      const d = dofsOf(e);
      let mn = Infinity;
      for (const x of d) if (eq[x] >= 0) mn = Math.min(mn, eq[x]);
      for (const x of d) if (eq[x] >= 0 && mn < first[eq[x]]) first[eq[x]] = mn;
    }
    const ptr = new Float64Array(nf + 1);                   // row q stores columns first[q] .. q
    for (let q = 0; q < nf; q++) ptr[q + 1] = ptr[q] + (q - first[q] + 1);
    return { n2, nf, eq, first, ptr, A: new Float64Array(ptr[nf]), diag0: new Float64Array(nf), neg: 0 };
  }
  function profileAdd(P, d, k) {
    const { eq, ptr, first, A } = P;
    for (let a = 0; a < d.length; a++) {
      const ra = eq[d[a]];
      if (ra < 0) continue;
      const base = ptr[ra] - first[ra], row = k[a];
      for (let b = 0; b < d.length; b++) { const cb = eq[d[b]]; if (cb >= 0 && cb <= ra) A[base + cb] += row[b]; }
    }
  }
  function profileFactor(P, requirePositive = true) {
    // LDLᵀ by rows: g_ij = a_ij − Σ_k L_jk g_ik, L_ij = g_ij / D_j, D_i = a_ii − Σ_j L_ij g_ij
    const { nf, first, ptr, A, diag0 } = P;
    let neg = 0;
    for (let i = 0; i < nf; i++) {
      const fi = first[i], pi = ptr[i] - fi;
      diag0[i] = A[pi + i];
      for (let j = fi; j < i; j++) {
        const fj = first[j], pj = ptr[j] - fj, k0 = Math.max(fi, fj);
        let s = A[pi + j];
        for (let k = k0; k < j; k++) s -= A[pj + k] * A[pi + k];
        A[pi + j] = s;
      }
      let d = A[pi + i];
      for (let j = fi; j < i; j++) { const g = A[pi + j], L = g / A[ptr[j] - first[j] + j]; A[pi + j] = L; d -= L * g; }
      if (!(Math.abs(d) > 1e-11 * Math.abs(diag0[i])) || (requirePositive && d <= 0))
        throw new FEError("the structure is a mechanism: add supports so that it cannot move as a rigid body");
      if (d < 0) neg++;
      A[pi + i] = d;
    }
    P.neg = neg;
    return neg;
  }
  function profileSolve(P, b) {
    const { nf, first, ptr, A } = P, x = Float64Array.from(b);
    for (let i = 0; i < nf; i++) { const fi = first[i], pi = ptr[i] - fi; let s = x[i]; for (let k = fi; k < i; k++) s -= A[pi + k] * x[k]; x[i] = s; }
    for (let i = 0; i < nf; i++) x[i] /= A[ptr[i] - first[i] + i];
    for (let i = nf - 1; i >= 0; i--) { const fi = first[i], pi = ptr[i] - fi, v = x[i]; for (let k = fi; k < i; k++) x[k] -= A[pi + k] * v; }
    return x;
  }

  function solveSystem(m, prescribed, combo) {
    const et = m.etype, C = m.C, fixed = new Map(m.fixed);
    if (prescribed) for (const [d, v] of prescribed) fixed.set(d, v);
    const P = makeProfile(m, fixed), { n2, nf, eq } = P;
    const u = new Float64Array(n2);
    for (const [d, v] of fixed) u[d] = v;
    const fApplied = loadVector(m, combo), rhs = new Float64Array(nf);
    for (let d = 0; d < n2; d++) if (eq[d] >= 0) rhs[eq[d]] = fApplied[d];
    const elemK = [];
    for (const e of m.elements) {
      const xy = e.map((i) => m.nodes[i]), { k } = elementStiffness(et, xy, C, m.t), d = dofsOf(e);
      elemK.push(k);
      profileAdd(P, d, k);
      for (let a = 0; a < d.length; a++) { const ra = eq[d[a]]; if (ra < 0) continue; for (let b = 0; b < d.length; b++) if (eq[d[b]] < 0 && u[d[b]]) rhs[ra] -= k[a][b] * u[d[b]]; }
    }
    profileFactor(P, true);
    const x = profileSolve(P, rhs);
    for (let d = 0; d < n2; d++) if (eq[d] >= 0) u[d] = x[eq[d]];
    const Ku = new Float64Array(n2);
    m.elements.forEach((e, q) => {
      const d = dofsOf(e), k = elemK[q];
      for (let a = 0; a < d.length; a++) { let s = 0; for (let b = 0; b < d.length; b++) s += k[a][b] * u[d[b]]; Ku[d[a]] += s; }
    });
    const R = new Float64Array(n2);
    let energy = 0;
    for (let d = 0; d < n2; d++) { if (eq[d] < 0) R[d] = Ku[d] - fApplied[d]; energy += 0.5 * u[d] * Ku[d]; }
    return { u, R, f: fApplied, energy, nf };
  }
  function recover(m, u) {
    const et = m.etype, C = m.C, { pts, E } = extrapolation(et), mids = MIDS[et];
    const nn = NN[et], ne = m.elements.length, elemNodal = new Array(ne), gp = new Array(ne);
    m.elements.forEach((e, q) => {
      const xy = e.map((i) => m.nodes[i]), ue = new Float64Array(2 * nn);
      e.forEach((i, k) => { ue[2 * k] = u[2 * i]; ue[2 * k + 1] = u[2 * i + 1]; });
      const sig = (g) => { const eps = [0, 1, 2].map((r) => g.B[r].reduce((s, b, c) => s + b * ue[c], 0)); return [0, 1, 2].map((r) => C[r][0] * eps[0] + C[r][1] * eps[1] + C[r][2] * eps[2]); };
      const sp = pts.map(([xi, eta]) => sig(gaussPoint(et, xy, xi, eta)));
      const corners = E.map((row) => [0, 1, 2].map((c) => row.reduce((s, w, j) => s + w * sp[j][c], 0)));
      const vals = new Array(nn);
      corners.forEach((v, i) => { vals[i] = v.map((x) => x / 1000); });
      for (const [mi, a, b] of mids) vals[mi] = [0, 1, 2].map((c) => 0.5 * (corners[a][c] + corners[b][c]) / 1000);
      elemNodal[q] = vals;
      gp[q] = gauss(et).map(([xi, eta, w]) => { const g = gaussPoint(et, xy, xi, eta, w), s = sig(g); return [g.x, g.y, s[0] / 1000, s[1] / 1000, s[2] / 1000]; });
    });
    const nodal = m.nodes.map(() => [0, 0, 0]), cnt = new Float64Array(m.nodes.length);
    m.elements.forEach((e, q) => e.forEach((i, k) => { for (let c = 0; c < 3; c++) nodal[i][c] += elemNodal[q][k][c]; cnt[i] += 1; }));
    nodal.forEach((v, i) => { const c = Math.max(cnt[i], 1); v[0] /= c; v[1] /= c; v[2] /= c; });
    return { elemNodal, nodal, gp };
  }
  function fieldOf(m, s, name) {
    const [sx, sy, txy] = s;
    if (name === "sxx") return sx;
    if (name === "syy") return sy;
    if (name === "txy") return txy;
    const c = (sx + sy) / 2, r = Math.sqrt(((sx - sy) / 2) ** 2 + txy * txy);
    if (name === "s1") return c + r;
    if (name === "s2") return c - r;
    if (name === "mises") { const sz = m.analysis === "plane_strain" ? m.nu * (sx + sy) : 0; return Math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * txy * txy); }
    throw new FEError(`unknown field ${name}`);
  }
  function solveFE(m, prescribed, combo) {
    const t0 = (typeof performance !== "undefined" ? performance : Date).now();
    const sol = solveSystem(m, prescribed, combo), t1 = (typeof performance !== "undefined" ? performance : Date).now();
    const rec = recover(m, sol.u);
    const n = m.nodes.length, U = Array.from({ length: n }, (_, i) => [sol.u[2 * i], sol.u[2 * i + 1]]), R = Array.from({ length: n }, (_, i) => [sol.R[2 * i], sol.R[2 * i + 1]]);
    const res = { model: m, u: U, reactions: R, f: sol.f, energy: sol.energy, nodal: rec.nodal, elemNodal: rec.elemNodal, gp: rec.gp, dofs: 2 * n, free: sol.nf,
      solve_ms: t1 - t0, total_ms: (typeof performance !== "undefined" ? performance : Date).now() - t0 };
    res.field = (name, averaged = true) => {
      if (name === "umag") return U.map(([a, b]) => Math.hypot(a, b));
      if (name === "ux") return U.map((v) => v[0]);
      if (name === "uy") return U.map((v) => v[1]);
      return averaged ? rec.nodal.map((s) => fieldOf(m, s, name)) : rec.elemNodal.map((vals) => vals.map((s) => fieldOf(m, s, name)));
    };
    return res;
  }
  function elementDetails(m, eid, u) {
    const et = m.etype, e = m.elements[eid], xy = e.map((i) => m.nodes[i]);
    const { k, gps } = elementStiffness(et, xy, m.C, m.t);
    const out = { type: et, nodes: e.slice(), xy, C: m.C, t: m.t, k: k.map((r) => Array.from(r)), m: elementMass(et, xy, m.gamma / 9.81, m.t).map((r) => Array.from(r)),
      gauss: gps.map((g) => ({ xi: g.xi, eta: g.eta, w: g.w, x: g.x, y: g.y, N: g.N, dN_dxi: g.dN, J: g.J, detJ: g.detJ, B: g.B })) };
    if (u) {
      const ue = []; e.forEach((i) => { ue.push(u[i][0], u[i][1]); });
      out.u_e = ue;
      out.f_e = out.k.map((row) => row.reduce((s, v, c) => s + v * ue[c], 0));
      out.gauss.forEach((g) => {
        const eps = [0, 1, 2].map((r) => g.B[r].reduce((s, b, c) => s + b * ue[c], 0));
        g.strain = eps; g.stress_MPa = [0, 1, 2].map((r) => (m.C[r][0] * eps[0] + m.C[r][1] * eps[1] + m.C[r][2] * eps[2]) / 1000);
      });
    }
    return out;
  }
  // symmetric eigenvalues (cyclic Jacobi) for the element stiffness rank check
  function symEig(Ain) {
    const n = Ain.length, A = Ain.map((r) => Array.from(r));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-30 * n * n) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      }
    }
    return A.map((r, i) => r[i]).sort((a, b) => a - b);
  }
  // closed-form references
  function timoshenkoCantilever(x, y, P, L, D, E, nu, t = 1) {
    const I = t * D ** 3 / 12;
    return [-P * y / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * (y * y - D * D / 4)),
      P / (6 * E * I) * (3 * nu * y * y * (L - x) + (4 + 5 * nu) * D * D * x / 4 + (3 * L - x) * x * x),
      -P * (L - x) * y / I, 0, P / (2 * I) * (D * D / 4 - y * y)];
  }
  const heywoodKt = (r) => (2 + (1 - r) ** 3) / (1 - r);

  T.FE = { SCHEMA_FE, TYPES, NN, EDGES, MIDS, QUADRATIC, FEError, shape, gauss, constitutive, gaussPoint, elementStiffness, elementMass, bodyForce, extrapolation,
    meshRectangle, meshPlateHole, meshRing, parseFE, solveFE, recover, fieldOf, elementDetails, symEig, timoshenkoCantilever, heywoodKt,
    concreteProps, elementCentroids, elementReinforcement, combinationFactors, selfWeightVector, loadVector, dofsOf, makeProfile, profileAdd, profileFactor, profileSolve, num,
    run: (data) => solveFE(parseFE(data)) };
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
