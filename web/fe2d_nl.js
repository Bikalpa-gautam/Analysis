// Nonlinear plane finite elements in the browser: a port of tragwerk.fe2d_nl (Python), checked against it in the
// test suite. Geometric nonlinearity (total Lagrangian, St. Venant–Kirchhoff), J2 plasticity with radial return,
// a reinforced concrete membrane model (rotating smeared cracks, MCFT), and load, displacement and arc-length control.
// Units kN, m; stresses reported in MPa. Needs fe2d.js.
(function (root) {
  "use strict";
  const T = root.Tragwerk || (root.Tragwerk = {});
  const F = T.FE;
  const SQ23 = Math.sqrt(2 / 3);
  const now = () => (typeof performance !== "undefined" ? performance : Date).now();
  const Q4C = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

  // ------------------------------------------------------------ precomputed reference data of all Gauss points
  function nlExtrapolation(et) {
    const rule = F.gauss(et), r3 = Math.sqrt(3);
    let E;
    if (et === "T3") E = [[1], [1], [1]];
    else if (et === "T6") E = [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 2 : 0) - 1 / 3));
    else if (et === "Q4" || et === "Q8R") {
      const Nc = Q4C.map(([a, b]) => F.shape("Q4", a * r3, b * r3).N);
      const order = rule.map(([xi, eta]) => Q4C.findIndex(([a, b]) => a === Math.sign(xi) && b === Math.sign(eta)));
      E = Nc.map((row) => order.map((k) => row[k]));
    } else {
      E = Q4C.map(() => new Array(9).fill(0));
      const G3 = Math.sqrt(0.6);
      Q4C.forEach(([a, b], i) => { const N = F.shape("Q4", a / G3, b / G3).N; [0, 2, 8, 6].forEach((g, k) => { E[i][g] = N[k]; }); });
    }
    return { E, mids: F.MIDS[et] || [] };
  }

  function makeKernel(m) {
    const et = m.etype, rule = F.gauss(et), ne = m.elements.length, nn = F.NN[et], ng = rule.length;
    const dN = new Float64Array(ne * ng * 2 * nn), wdet = new Float64Array(ne * ng), xy = new Float64Array(ne * ng * 2), area = new Float64Array(ne);
    m.elements.forEach((e, q) => {
      const X = e.map((i) => m.nodes[i]);
      rule.forEach(([xi, eta, w], g) => {
        const gp = F.gaussPoint(et, X, xi, eta, w), o = ((q * ng + g) * 2) * nn;
        for (let a = 0; a < nn; a++) { dN[o + a] = gp.dNdx[0][a]; dN[o + nn + a] = gp.dNdx[1][a]; }
        wdet[q * ng + g] = w * gp.detJ * m.t;
        xy[2 * (q * ng + g)] = gp.x; xy[2 * (q * ng + g) + 1] = gp.y;
        area[q] += w * gp.detJ;
      });
    });
    const ex = nlExtrapolation(et);
    return { et, ne, nn, ng, N: ne * ng, dN, wdet, xy, area, dofs: m.elements.map(F.dofsOf), n2: 2 * m.nodes.length, extrap: ex.E, mids: ex.mids };
  }

  // ------------------------------------------------------------ J2 plasticity (radial return, consistent tangent)
  function j2Return(e, ep, alpha, E, nu, sy, H) {
    const G = E / (2 * (1 + nu)), K = E / (3 * (1 - 2 * nu));
    const ee = [e[0] - ep[0], e[1] - ep[1], e[2] - ep[2], e[3] - ep[3]], tr = ee[0] + ee[1] + ee[2];
    const s = [2 * G * ee[0] - 2 * G * tr / 3, 2 * G * ee[1] - 2 * G * tr / 3, 2 * G * ee[2] - 2 * G * tr / 3, 2 * G * ee[3]];
    const ns = Math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2 * s[3] * s[3]);
    const f = ns - SQ23 * (sy + H * alpha), plastic = f > 1e-12 * sy;
    const dg = plastic ? f / (2 * G + 2 * H / 3) : 0, safe = ns > 0 ? ns : 1, n = s.map((v) => v / safe);
    const sig = s.map((v, i) => v - 2 * G * dg * n[i]);
    sig[0] += K * tr; sig[1] += K * tr; sig[2] += K * tr;
    const theta = 1 - 2 * G * dg / safe, thetab = plastic ? 1 / (1 + H / (3 * G)) - (1 - theta) : 0;
    const mv = [1, 1, 1, 0], Ie = [1, 1, 1, 0.5], D = new Float64Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
      D[4 * i + j] = K * mv[i] * mv[j] + 2 * G * theta * ((i === j ? Ie[i] : 0) - mv[i] * mv[j] / 3) - 2 * G * thetab * n[i] * n[j];
    return { sig, D, ep: ep.map((v, i) => v + dg * n[i]), alpha: alpha + SQ23 * dg, dg };
  }
  // one Gauss point; eps engineering [εx, εy, γ]; returns σ (3), D (3×3 flat), new state values
  function j2Point(eps, st, i, mat, E, nu, analysis) {
    const e4 = [eps[0], eps[1], 0, eps[2] / 2], ep = [st.epsp[4 * i], st.epsp[4 * i + 1], st.epsp[4 * i + 2], st.epsp[4 * i + 3]];
    let r, ezz = 0;
    if (analysis === "plane_strain") r = j2Return(e4, ep, st.alpha[i], E, nu, mat.fy, mat.H);
    else {
      ezz = st.ezz[i];
      for (let it = 0; it < 40; it++) {
        e4[2] = ezz;
        r = j2Return(e4, ep, st.alpha[i], E, nu, mat.fy, mat.H);
        if (Math.abs(r.sig[2]) <= 1e-10 * mat.fy) break;
        ezz -= r.sig[2] / r.D[10];
      }
    }
    const a = [0, 1, 3], D = new Float64Array(9), Dr = r.D;
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {
      D[3 * p + q] = Dr[4 * a[p] + a[q]];
      if (analysis !== "plane_strain") D[3 * p + q] -= Dr[4 * a[p] + 2] * Dr[8 + a[q]] / Dr[10];
    }
    return { sig: [r.sig[0], r.sig[1], r.sig[3]], D, ep: r.ep, alpha: r.alpha, ezz, yielded: r.dg > 0, szz: r.sig[2] };
  }

  // ------------------------------------------------------------ reinforced concrete membrane (rotating smeared cracks)
  // concrete in a principal direction: [σ, dσ/dε] (tension positive)
  function concreteEnv(e, et, mat, Ec, plain, hcb) {
    const fc = mat.fc, fct = mat.fct, e0 = mat.eps_c0, ecr = fct / Ec;
    if (e >= 0) {
      if (e <= ecr) return [Ec * e, Ec];
      let eu = Math.max(2 * mat.Gf / (fct * hcb), 2 * ecr);
      if (mat.tension === "none") eu = 2 * ecr;
      if (plain || mat.tension === "none") return [Math.max(fct * (eu - e) / (eu - ecr), 0), e < eu ? -fct / (eu - ecr) : 0];
      if (e < 2 * ecr) { const c2 = fct / (1 + Math.sqrt(1000 * ecr)); return [fct + (c2 - fct) * (e - ecr) / ecr, (c2 - fct) / ecr]; }
      const sq = Math.sqrt(500 * e);
      return [fct / (1 + sq), -fct * 250 / (sq * (1 + sq) ** 2)];
    }
    const trans = Math.max(et, 0);
    const beta = mat.softening === "mcft" ? Math.min(1 / (0.8 + 170 * trans), 1) : (trans > ecr ? mat.nu_ec2 : 1);
    const eta = -e / e0;
    return eta <= 2 ? [-beta * fc * (2 * eta - eta * eta), beta * fc * (2 - 2 * eta) / e0] : [0, 0];
  }
  function concrete(e, et, mat, Ec, plain, hcb, emax, emin, smax) {
    const ecr = mat.fct / Ec;
    if (e >= 0 && e < emax && emax > ecr) { const se = smax / emax; return [se * e, se]; }
    if (e < 0 && e > emin && emin < 0) { const se = concreteEnv(emin, et, mat, Ec, plain, hcb)[0] / emin; return [se * e, se]; }
    return concreteEnv(e, et, mat, Ec, plain, hcb);
  }
  function steel(e, ep, q, mat) {
    const Es = mat.Es, fy = mat.fy, Esh = mat.Esh, Hk = Esh < Es ? Es * Esh / (Es - Esh) : 1e30;
    const str = Es * (e - ep), xi = str - q, f = Math.abs(xi) - fy, pl = f > 1e-12 * fy;
    const dg = pl ? f / (Es + Hk) : 0, n = Math.sign(xi);
    return [str - Es * dg * n, pl ? Es * Hk / (Es + Hk) : Es, ep + dg * n, q + Hk * dg * n];
  }
  function rcPoint(eps, st, i, mat, Ec, rhox, rhoy, hcb, plain, tangent = true) {
    const ex = eps[0], ey = eps[1], g = eps[2];
    const cen = 0.5 * (ex + ey), rad = Math.sqrt((0.5 * (ex - ey)) ** 2 + (0.5 * g) ** 2), e1 = cen + rad, e2 = cen - rad;
    const th = 0.5 * Math.atan2(g, ex - ey), c = Math.cos(th), s = Math.sin(th);
    const e1max = st.e1max[i], e2min = st.e2min[i], s1max = st.s1max[i];
    let [s1, d1] = concrete(e1, e2, mat, Ec, plain, hcb, e1max, e2min, s1max);
    const [s2, d2] = concrete(e2, e1, mat, Ec, plain, hcb, e1max, e2min, s1max);
    const [ssx, Dsx, spx, qx] = steel(ex, st.spx[i], st.qx[i], mat), [ssy, Dsy, spy, qy] = steel(ey, st.spy[i], st.qy[i], mat);
    const ecr = mat.fct / Ec;
    const cap = rhox * (mat.fy - ssx) * c * c + rhoy * (mat.fy - ssy) * s * s;
    if (!plain && e1 > ecr && s1 > Math.max(cap, 0)) { s1 = Math.max(cap, 0); d1 = 0; }
    const sig = [s1 * c * c + s2 * s * s + rhox * ssx, s1 * s * s + s2 * c * c + rhoy * ssy, (s1 - s2) * s * c];
    let E1, E2;
    if (tangent) { E1 = Math.abs(d1) < 1e-4 * Ec ? 1e-4 * Ec : d1; E2 = Math.abs(d2) < 1e-4 * Ec ? 1e-4 * Ec : d2; }
    else {
      const E0c = 2 * mat.fc / mat.eps_c0;
      E1 = Math.abs(e1) > 1e-14 ? s1 / e1 : (e1 >= 0 ? Ec : E0c); E2 = Math.abs(e2) > 1e-14 ? s2 / e2 : (e2 >= 0 ? Ec : E0c);
      E1 = Math.max(E1, 1e-4 * Ec); E2 = Math.max(E2, 1e-4 * Ec);
    }
    const dd = e1 - e2;
    const Gs = Math.max(dd > 1e-12 ? (s1 - s2) / (2 * dd) : 0.25 * (E1 + E2), 1e-3 * Ec);
    const Tm = [c * c, s * s, c * s, s * s, c * c, -c * s, -2 * c * s, 2 * c * s, c * c - s * s], Dp = [E1, E2, Gs], D = new Float64Array(9);
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) { let v = 0; for (let k = 0; k < 3; k++) v += Tm[3 * k + p] * Dp[k] * Tm[3 * k + q]; D[3 * p + q] = v; }
    D[0] += rhox * Math.max(Dsx, 1e-3 * mat.Es);
    D[4] += rhoy * Math.max(Dsy, 1e-3 * mat.Es);
    const nE1 = Math.max(e1max, e1), nE2 = Math.min(e2min, e2);
    return { sig, D, e1max: nE1, e2min: nE2, s1max: e1 >= e1max ? Math.max(s1, 0) : s1max, spx, spy, qx, qy,
      ex: { e1, e2, theta: th, sc1: s1, sc2: s2, ssx, ssy, cracked: nE1 > ecr, open: e1 > ecr, yield_x: Math.abs(spx) > 0 && rhox > 0, yield_y: Math.abs(spy) > 0 && rhoy > 0,
        crushed: nE2 < -mat.eps_c0, rupture: Math.max(rhox > 0 ? Math.abs(ex) : 0, rhoy > 0 ? Math.abs(ey) : 0) > mat.eud } };
  }
  const RC_FLOAT = ["e1", "e2", "theta", "sc1", "sc2", "ssx", "ssy"], RC_FLAG = ["cracked", "open", "yield_x", "yield_y", "crushed", "rupture"];

  // ------------------------------------------------------------ problem
  function makeProblem(m, o) {
    const k = makeKernel(m), kind = (m.material && m.material.model) || "elastic", N = k.N;
    if (o.geometric && kind !== "elastic") throw new F.FEError("geometric nonlinearity is available with the elastic (St. Venant–Kirchhoff) material");
    const re = F.elementReinforcement(m), rho = new Float64Array(2 * N), plain = new Uint8Array(N), hcb = new Float64Array(N);
    for (let q = 0; q < k.ne; q++) for (let g = 0; g < k.ng; g++) {
      const i = q * k.ng + g;
      rho[2 * i] = re.Asx[q] * 1e-6 / m.t; rho[2 * i + 1] = re.Asy[q] * 1e-6 / m.t;
      plain[i] = re.Asx[q] + re.Asy[q] <= 0 ? 1 : 0; hcb[i] = Math.sqrt(k.area[q]);
    }
    const state0 = kind === "rc"
      ? { e1max: new Float64Array(N), e2min: new Float64Array(N), s1max: new Float64Array(N), spx: new Float64Array(N), spy: new Float64Array(N), qx: new Float64Array(N), qy: new Float64Array(N) }
      : { epsp: new Float64Array(4 * N), alpha: new Float64Array(N), ezz: new Float64Array(N) };
    return { m, o, k, kind, C: m.C, rho, plain, hcb, state0, re, secant: false };
  }
  function cloneState(st) { const o = {}; for (const key in st) o[key] = new Float64Array(st[key]); return o; }

  // internal forces, iteration matrix (assembled into the profile Pm, free dofs), K_fc·ū, trial state, Gauss-point output
  function respond(P, u, state, Pm, ubar) {
    const { m, k, kind, C } = P, { ne, nn, ng, dN, wdet } = k, nd = 2 * nn, geo = P.o.geometric;
    const fi = new Float64Array(k.n2), sig = new Float64Array(3 * k.N);
    const wantK = !!Pm;
    let kfu = null;
    if (wantK) { Pm.A.fill(0); kfu = new Float64Array(Pm.nf); }
    const ns = kind === "elastic" ? state : cloneState(state);
    const extra = {};
    if (kind === "rc") { for (const key of RC_FLOAT) extra[key] = new Float64Array(k.N); for (const key of RC_FLAG) extra[key] = new Uint8Array(k.N); }
    if (kind === "j2") { extra.yield = new Uint8Array(k.N); extra.szz = new Float64Array(k.N); }
    if (geo) extra.detF = new Float64Array(k.N);
    const ue = new Float64Array(nd), fe = new Float64Array(nd), ke = new Float64Array(nd * nd), Bt = new Float64Array(3 * nd), DB = new Float64Array(3 * nd);
    const mat = m.material, eq = wantK ? Pm.eq : null;
    for (let q = 0; q < ne; q++) {
      const d = k.dofs[q];
      for (let a = 0; a < nd; a++) ue[a] = u[d[a]];
      fe.fill(0);
      if (wantK) ke.fill(0);
      for (let g = 0; g < ng; g++) {
        const i = q * ng + g, o = 2 * i * nn, w = wdet[i];
        let s0, s1, s2, D = null;
        if (geo) {
          let h00 = 0, h01 = 0, h10 = 0, h11 = 0;
          for (let a = 0; a < nn; a++) { const nx = dN[o + a], ny = dN[o + nn + a]; h00 += ue[2 * a] * nx; h01 += ue[2 * a] * ny; h10 += ue[2 * a + 1] * nx; h11 += ue[2 * a + 1] * ny; }
          const F11 = 1 + h00, F12 = h01, F21 = h10, F22 = 1 + h11;
          const E11 = 0.5 * (F11 * F11 + F21 * F21 - 1), E22 = 0.5 * (F12 * F12 + F22 * F22 - 1), E12 = 0.5 * (F11 * F12 + F21 * F22);
          const ev = [E11, E22, 2 * E12];
          const S = [0, 1, 2].map((r) => C[r][0] * ev[0] + C[r][1] * ev[1] + C[r][2] * ev[2]);
          for (let a = 0; a < nn; a++) {
            const nx = dN[o + a], ny = dN[o + nn + a];
            Bt[2 * a] = F11 * nx; Bt[2 * a + 1] = F21 * nx;
            Bt[nd + 2 * a] = F12 * ny; Bt[nd + 2 * a + 1] = F22 * ny;
            Bt[2 * nd + 2 * a] = F11 * ny + F12 * nx; Bt[2 * nd + 2 * a + 1] = F21 * ny + F22 * nx;
          }
          for (let a = 0; a < nd; a++) fe[a] += (Bt[a] * S[0] + Bt[nd + a] * S[1] + Bt[2 * nd + a] * S[2]) * w;
          if (wantK) {
            for (let r = 0; r < 3; r++) for (let a = 0; a < nd; a++) DB[r * nd + a] = C[r][0] * Bt[a] + C[r][1] * Bt[nd + a] + C[r][2] * Bt[2 * nd + a];
            for (let a = 0; a < nd; a++) { const b0 = Bt[a], b1 = Bt[nd + a], b2 = Bt[2 * nd + a]; for (let b = 0; b < nd; b++) ke[a * nd + b] += (b0 * DB[b] + b1 * DB[nd + b] + b2 * DB[2 * nd + b]) * w; }
            for (let a = 0; a < nn; a++) {
              const nxa = dN[o + a], nya = dN[o + nn + a];
              for (let b = 0; b < nn; b++) {
                const nxb = dN[o + b], nyb = dN[o + nn + b];
                const gab = (nxa * (S[0] * nxb + S[2] * nyb) + nya * (S[2] * nxb + S[1] * nyb)) * w;
                ke[(2 * a) * nd + 2 * b] += gab; ke[(2 * a + 1) * nd + 2 * b + 1] += gab;
              }
            }
          }
          const detF = F11 * F22 - F12 * F21;
          // Cauchy stress σ = F S Fᵀ / det F
          const Sm00 = S[0], Sm11 = S[1], Sm01 = S[2];
          const a00 = F11 * Sm00 + F12 * Sm01, a01 = F11 * Sm01 + F12 * Sm11, a10 = F21 * Sm00 + F22 * Sm01, a11 = F21 * Sm01 + F22 * Sm11;
          s0 = (a00 * F11 + a01 * F12) / detF; s1 = (a10 * F21 + a11 * F22) / detF; s2 = (a00 * F21 + a01 * F22) / detF;
          extra.detF[i] = detF;
          sig[3 * i] = s0 / 1000; sig[3 * i + 1] = s1 / 1000; sig[3 * i + 2] = s2 / 1000;
          continue;
        }
        let e0 = 0, e1 = 0, e2 = 0;
        for (let a = 0; a < nn; a++) { const nx = dN[o + a], ny = dN[o + nn + a]; e0 += nx * ue[2 * a]; e1 += ny * ue[2 * a + 1]; e2 += ny * ue[2 * a] + nx * ue[2 * a + 1]; }
        const eps = [e0, e1, e2];
        if (kind === "elastic") {
          s0 = C[0][0] * e0 + C[0][1] * e1 + C[0][2] * e2; s1 = C[1][0] * e0 + C[1][1] * e1 + C[1][2] * e2; s2 = C[2][0] * e0 + C[2][1] * e1 + C[2][2] * e2;
        } else if (kind === "j2") {
          const r = j2Point(eps, state, i, mat, m.E, m.nu, m.analysis);
          [s0, s1, s2] = r.sig; D = r.D;
          for (let c = 0; c < 4; c++) ns.epsp[4 * i + c] = r.ep[c];
          ns.alpha[i] = r.alpha; ns.ezz[i] = r.ezz; extra.yield[i] = r.yielded ? 1 : 0; extra.szz[i] = r.szz;
        } else {
          const r = rcPoint(eps, state, i, mat, m.E, P.rho[2 * i], P.rho[2 * i + 1], P.hcb[i], !!P.plain[i], !P.secant);
          [s0, s1, s2] = r.sig; D = r.D;
          ns.e1max[i] = r.e1max; ns.e2min[i] = r.e2min; ns.s1max[i] = r.s1max; ns.spx[i] = r.spx; ns.spy[i] = r.spy; ns.qx[i] = r.qx; ns.qy[i] = r.qy;
          for (const key of RC_FLOAT) extra[key][i] = r.ex[key];
          for (const key of RC_FLAG) extra[key][i] = r.ex[key] ? 1 : 0;
        }
        sig[3 * i] = s0 / 1000; sig[3 * i + 1] = s1 / 1000; sig[3 * i + 2] = s2 / 1000;
        for (let a = 0; a < nn; a++) { const nx = dN[o + a], ny = dN[o + nn + a]; fe[2 * a] += (nx * s0 + ny * s2) * w; fe[2 * a + 1] += (ny * s1 + nx * s2) * w; }
        if (wantK) {
          const Dm = D || [C[0][0], C[0][1], C[0][2], C[1][0], C[1][1], C[1][2], C[2][0], C[2][1], C[2][2]];
          for (let a = 0; a < nn; a++) {
            const nx = dN[o + a], ny = dN[o + nn + a];
            Bt[2 * a] = nx; Bt[2 * a + 1] = 0; Bt[nd + 2 * a] = 0; Bt[nd + 2 * a + 1] = ny; Bt[2 * nd + 2 * a] = ny; Bt[2 * nd + 2 * a + 1] = nx;
          }
          for (let r = 0; r < 3; r++) for (let a = 0; a < nd; a++) DB[r * nd + a] = Dm[3 * r] * Bt[a] + Dm[3 * r + 1] * Bt[nd + a] + Dm[3 * r + 2] * Bt[2 * nd + a];
          for (let a = 0; a < nd; a++) { const b0 = Bt[a], b1 = Bt[nd + a], b2 = Bt[2 * nd + a]; if (!b0 && !b1 && !b2) continue; for (let b = 0; b < nd; b++) ke[a * nd + b] += (b0 * DB[b] + b1 * DB[nd + b] + b2 * DB[2 * nd + b]) * w; }
        }
      }
      for (let a = 0; a < nd; a++) fi[d[a]] += fe[a];
      if (wantK) {
        const { ptr, first, A } = Pm;
        for (let a = 0; a < nd; a++) {
          const ra = eq[d[a]];
          if (ra < 0) continue;
          const base = ptr[ra] - first[ra];
          for (let b = 0; b < nd; b++) {
            const cb = eq[d[b]];
            if (cb >= 0) { if (cb <= ra) A[base + cb] += ke[a * nd + b]; }
            else if (ubar && ubar[d[b]]) kfu[ra] += ke[a * nd + b] * ubar[d[b]];
          }
        }
      }
    }
    return { fi, kfu, state: ns, sig, extra };
  }

  // ------------------------------------------------------------ small dense solve and Anderson mixing
  function smallSolve(Ain, bin) {
    const n = bin.length, A = Ain.map((r) => r.slice()), b = bin.slice();
    for (let k = 0; k < n; k++) {
      let p = k;
      for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
      if (p !== k) { [A[k], A[p]] = [A[p], A[k]]; [b[k], b[p]] = [b[p], b[k]]; }
      if (Math.abs(A[k][k]) < 1e-300) return new Array(n).fill(0);
      for (let i = k + 1; i < n; i++) { const f = A[i][k] / A[k][k]; for (let j = k; j < n; j++) A[i][j] -= f * A[k][j]; b[i] -= f * b[k]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j]; x[i] = s / A[i][i]; }
    return x;
  }
  function anderson(depth = 5) {
    let G = [], Fh = [];
    return {
      reset() { G = []; Fh = []; },
      step(x, f) {
        const g = new Float64Array(x.length);
        for (let i = 0; i < x.length; i++) g[i] = x[i] + f[i];
        G.push(g); Fh.push(Float64Array.from(f));
        if (Fh.length > depth + 1) { G.shift(); Fh.shift(); }
        if (Fh.length < 2) return g;
        const k = Fh.length - 1, n = f.length, dF = [], dG = [];
        for (let j = 0; j < k; j++) {
          const a = new Float64Array(n), b = new Float64Array(n);
          for (let i = 0; i < n; i++) { a[i] = Fh[j + 1][i] - Fh[j][i]; b[i] = G[j + 1][i] - G[j][i]; }
          dF.push(a); dG.push(b);
        }
        const M = dF.map((a) => dF.map((b) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }));
        let tr = 0;
        for (let j = 0; j < k; j++) tr += M[j][j];
        const reg = 1e-10 * Math.max(tr / k, 1e-300);
        for (let j = 0; j < k; j++) M[j][j] += reg;
        const rhs = dF.map((a) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * f[i]; return s; });
        const gam = smallSolve(M, rhs), out = Float64Array.from(g);
        for (let j = 0; j < k; j++) for (let i = 0; i < n; i++) out[i] -= dG[j][i] * gam[j];
        for (let i = 0; i < n; i++) if (!isFinite(out[i])) { G = []; Fh = []; return g; }
        return out;
      },
    };
  }

  // ------------------------------------------------------------ solver
  const DEFAULTS = { geometric: false, combination: "uls", control: "load", iteration: "newton", steps: 10, lam_max: 1.0, to_failure: false, tol: null,
    max_iter: 40, max_steps: 400, max_bisections: 8, monitor: null, accelerate: true, keep_states: true, load_pattern: "all", line_search: true, accept_tol: null };
  const norm = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); };

  // λ multiplies the loads of the combination (load_pattern "all"), or – "variable", a pushover – the permanent loads
  // are applied first and λ then multiplies only the variable ones
  function runNonlinear(m, opts = {}, onStep = null, stage = null) {
    const t0 = now(), o = Object.assign({}, DEFAULTS, opts);
    const P = makeProblem(m, o), k = P.k, n2 = k.n2;
    let fExt, fConst, uInit = new Float64Array(n2), stInit = P.state0, uScale = 1;
    if (stage) [fConst, fExt, uInit, stInit, uScale] = stage;
    else if (o.load_pattern === "variable") {
      const [, aQ] = F.combinationFactors(m, o.combination), fAll = F.loadVector(m, o.combination);
      const fVar = Float64Array.from(m.forcesQ, (v) => aQ * v), fPerm = fAll.map((v, i) => v - fVar[i]);
      if (!fVar.some((v) => v !== 0)) throw new F.FEError("a pushover (permanent loads first, then λ × variable loads) needs variable loads (case Q)");
      const g = runNonlinear(m, Object.assign({}, o, { load_pattern: "all", control: "load", to_failure: false, lam_max: 1, steps: 4, max_steps: 200 }), null,
        [new Float64Array(n2), fPerm, new Float64Array(n2), P.state0, 0]);
      if (g.status !== "complete") throw new F.FEError(`the permanent loads could not be applied: ${g.message || g.status}`);
      return runNonlinear(m, o, onStep, [fPerm, fVar, g.steps[g.steps.length - 1].u, g.state, 1]);
    } else { fExt = F.loadVector(m, o.combination); fConst = new Float64Array(n2); }
    const fixed = [...m.fixed.keys()].sort((a, b) => a - b), ubar = new Float64Array(n2);
    for (const d of fixed) ubar[d] = m.fixed.get(d) * uScale;
    const hasUbar = fixed.some((d) => ubar[d] !== 0);
    const Pm = F.makeProfile(m, new Map(m.fixed)), { nf, eq } = Pm;
    const free = new Int32Array(nf);
    for (let d = 0; d < n2; d++) if (eq[d] >= 0) free[eq[d]] = d;
    const ff = new Float64Array(nf), fcf = new Float64Array(nf);
    for (let j = 0; j < nf; j++) { ff[j] = fExt[free[j]]; fcf[j] = fConst[free[j]]; }
    const res = { model: m, options: o, steps: [], status: "", message: "", lam_peak: 0, monitor_dof: 0, kernel: k, rho: P.rho, elapsed_ms: 0 };
    const factor = () => { try { F.profileFactor(Pm, false); return true; } catch (e) { return false; } };
    const solveFree = (rhs) => F.profileSolve(Pm, rhs);
    const rhsLoad = (kfu) => { const b = new Float64Array(nf); for (let j = 0; j < nf; j++) b[j] = ff[j] - (kfu ? kfu[j] : 0); return b; };
    // initial stiffness and monitor
    let u = Float64Array.from(uInit), state = stInit;
    const r0 = respond(P, u, state, Pm, hasUbar ? ubar : null);
    if (!factor()) throw new F.FEError("the structure is a mechanism: add supports so that it cannot move as a rigid body");
    const du = solveFree(rhsLoad(r0.kfu));
    let mon;
    if (o.monitor == null) {
      let best = 0, bi = 0;
      for (let j = 0; j < nf; j++) best = Math.max(best, Math.abs(du[j]));
      for (let j = 0; j < nf; j++) if (Math.abs(du[j]) >= (1 - 1e-6) * best) { bi = j; break; }
      mon = nf ? free[bi] : 0;
    }
    else mon = 2 * o.monitor[0] + o.monitor[1];
    res.monitor_dof = mon;
    const monF = eq[mon];
    if (o.control === "displacement" && monF < 0) throw new F.FEError("displacement control needs a free monitor degree of freedom");
    let lam = 0;
    const lamTarget = o.to_failure ? 1e9 : o.lam_max;
    const record = (uu, lamv, it, resid, st, converged = true) => {
      const r = respond(P, uu, st, null, null);
      let Rx = 0, Ry = 0;
      for (const d of fixed) { if (d % 2 === 0) Rx += r.fi[d]; else Ry += r.fi[d]; }
      const ex = {};
      if (P.kind === "j2") { ex.alpha = Float64Array.from(st.alpha); ex.yield = r.extra.yield; }
      else if (P.kind === "rc") Object.assign(ex, r.extra);
      const step = { lam: lamv, u: Float64Array.from(uu), iterations: it, residuals: resid.slice(), monitor: uu[mon], reaction: [Rx, Ry], gp_stress: r.sig, extra: ex,
        matrix: P.secant ? "secant" : "tangent", converged };
      res.steps.push(step);
      if (onStep) onStep(step, res);
      return step;
    };
    record(u, 0, 0, [], state);
    const dlam0 = o.lam_max / Math.max(o.steps, 1);
    let dlam = dlam0, dl = null, dmon = null, duPrev = null, bis = 0, nsteps = 0, status = "complete";
    const baseSecant = o.iteration === "secant";
    P.secant = baseSecant;
    const targetIt = P.kind === "rc" ? (baseSecant ? 12 : 6) : 5, tol = o.tol != null ? o.tol : (P.kind === "rc" ? 1e-3 : 1e-6), accel = P.kind === "rc" && o.accelerate, aa = anderson();
    // a failed Newton increment of a reinforced concrete model is tried once more with secant matrices
    const retrySecant = () => { if (P.kind === "rc" && !P.secant) { P.secant = true; return true; } return false; };
    // reinforced concrete: energy line search on Newton steps, and increments whose residual stays below accept_tol
    // (default 1 %, as in the Dutch NLFEA guidelines RTD 1016) are accepted and flagged when the iterations stall
    const lineSearch = P.kind === "rc" && o.line_search !== false;
    const acceptTol = o.accept_tol != null ? o.accept_tol : (P.kind === "rc" ? 1e-2 : null);
    const dotF = (a, b) => { let v = 0; for (let j = 0; j < nf; j++) v += a[j] * b[j]; return v; };
    let unconv = 0;
    let low = 0;
    const adapt = (it) => (it < targetIt ? Math.min(Math.sqrt(targetIt / Math.max(it, 1)), 1.5) : (it <= 2 * targetIt ? 1.0 : 0.7));
    let jump = false;
    // a converged increment that loses more than half the peak load at once is not a path that can be followed (a snap
    // to a far branch, typical after strain localisation): reject it and try a smaller increment
    const dropped = (lamOld, lamNew) => {
      if (!(o.to_failure && res.steps.length)) return false;
      const peak = res.steps.reduce((v, s) => Math.max(v, s.lam), -Infinity);
      jump = peak > 0 && lamOld - lamNew > 0.5 * peak;
      return jump;
    };
    const finished = (lamNow) => {
      const last = res.steps[res.steps.length - 1];
      if (P.kind === "rc" && last.extra.rupture && last.extra.rupture.some((v) => v)) return "rupture";
      const peak = res.steps.reduce((v, s) => Math.max(v, s.lam), -Infinity);
      const prev = res.steps.length > 1 ? res.steps[res.steps.length - 2].lam : 0;
      const rising = lamNow > prev + 0.01 * Math.abs(peak);   // a recovering load restarts the count
      low = peak > 0 && lamNow < 0.5 * peak && !rising ? low + 1 : 0;
      if (low >= 10 || (peak > 0 && lamNow <= 0)) return "post-peak";
      return "";
    };
    const residual = (fi, lt) => {
      const r = new Float64Array(nf);
      for (let j = 0; j < nf; j++) r[j] = fcf[j] + lt * ff[j] - fi[free[j]];
      let fx = 0;
      for (const d of fixed) fx += fi[d] * fi[d];
      let lf = 0;
      for (let j = 0; j < nf; j++) lf += (fcf[j] + lt * ff[j]) ** 2;
      return { r, rn: norm(r), ref: Math.max(Math.sqrt(lf), Math.sqrt(fx), 1e-9) };
    };
    const setFixed = (uu, lt) => { for (const d of fixed) uu[d] = lt * ubar[d]; };
    while (nsteps < o.max_steps) {
      if (lam >= lamTarget - 1e-12) break;
      nsteps++;
      jump = false;
      if (o.control === "load") {
        const lt = Math.min(lam + dlam, lamTarget), ut = Float64Array.from(u);
        setFixed(ut, lt);
        const resid = [];
        let conv = false, it = 0, st = null, haveK = false, best = null;
        aa.reset();
        for (it = 1; it <= o.max_iter; it++) {
          const need = o.iteration !== "modified" || !haveK;
          const rr = respond(P, ut, state, need ? Pm : null, null);
          st = rr.state;
          const { r, rn, ref } = residual(rr.fi, lt);
          resid.push(rn / ref);
          if (acceptTol != null && it > 1 && (!best || rn / ref < best.r)) best = { r: rn / ref, ut: Float64Array.from(ut), st, it };
          if ((rn <= tol * ref && it > 1) || rn <= 1e-14 * Math.max(ref, 1)) { conv = true; break; }
          if (need) { if (!factor()) break; haveK = true; }
          const dx = solveFree(r);
          if (!dx.every(isFinite)) break;
          if (accel && P.secant) {
            if (resid.length > 2 && resid[resid.length - 1] > 10 * Math.min(...resid)) aa.reset();
            const xf = new Float64Array(nf);
            for (let j = 0; j < nf; j++) xf[j] = ut[free[j]];
            const xn = aa.step(xf, dx);
            for (let j = 0; j < nf; j++) ut[free[j]] = xn[j];
          } else if (lineSearch && !P.secant) {
            const g0 = dotF(dx, r), ut1 = Float64Array.from(ut);
            for (let j = 0; j < nf; j++) ut1[free[j]] += dx[j];
            const g1 = dotF(dx, residual(respond(P, ut1, state, null, null).fi, lt).r);
            const eta = g0 > 0 && g1 < -0.5 * g0 ? Math.max(0.1, g0 / (g0 - g1)) : 1;
            for (let j = 0; j < nf; j++) ut[free[j]] += eta * dx[j];
          } else for (let j = 0; j < nf; j++) ut[free[j]] += dx[j];
        }
        let accepted = false;
        if (!conv && best && best.r <= acceptTol && P.secant) { conv = true; accepted = true; ut.set(best.ut); st = best.st; it = best.it; unconv++; }
        if (conv) {
          u = ut; state = st; lam = lt;
          const step = record(u, lam, it, resid, state, !accepted);
          P.secant = baseSecant;
          bis = 0;
          if (it <= targetIt - 1) dlam = Math.min(dlam * 1.5, 4 * dlam0);
          if (P.kind === "rc" && step.extra.rupture.some((v) => v)) { status = "rupture"; break; }
          continue;
        }
        if (retrySecant()) { nsteps--; continue; }
        P.secant = baseSecant;
        dlam /= 2; bis++; nsteps--;
        if (bis > o.max_bisections) { status = lam > 0 ? "limit" : "failed"; break; }
        continue;
      }
      if (o.control === "displacement") {
        const r0s = respond(P, u, state, Pm, hasUbar ? ubar : null);
        if (!factor()) { status = "failed"; break; }
        const dxf = solveFree(rhsLoad(r0s.kfu));
        if (!(Math.abs(dxf[monF]) >= 1e-300)) { status = "failed"; break; }
        if (dmon === null) dmon = dlam0 * du[monF];
        let dL = dmon / dxf[monF];
        const ut = Float64Array.from(u);
        for (let j = 0; j < nf; j++) ut[free[j]] += dL * dxf[j];
        setFixed(ut, lam + dL);
        const resid = [];
        let conv = false, it = 0, st = null, best = null;
        aa.reset();
        for (it = 1; it <= o.max_iter; it++) {
          const rr = respond(P, ut, state, Pm, hasUbar ? ubar : null);
          st = rr.state;
          const { r, rn, ref } = residual(rr.fi, lam + dL);
          resid.push(rn / ref);
          if (acceptTol != null && it > 1 && (!best || rn / ref < best.r)) best = { r: rn / ref, ut: Float64Array.from(ut), st, it, dL };
          if (rn <= tol * ref && it > 1) { conv = true; break; }
          if (!factor()) break;
          const dr = solveFree(r), df = solveFree(rhsLoad(rr.kfu));
          if (!(Math.abs(df[monF]) >= 1e-300)) break;
          const dli = -dr[monF] / df[monF];
          if (accel && P.secant) {
            if (resid.length > 2 && resid[resid.length - 1] > 10 * Math.min(...resid)) aa.reset();
            const xf = new Float64Array(nf + 1), fv = new Float64Array(nf + 1);
            for (let j = 0; j < nf; j++) { xf[j] = ut[free[j]]; fv[j] = dr[j] + dli * df[j]; }
            xf[nf] = dL; fv[nf] = dli;
            const xn = aa.step(xf, fv);
            for (let j = 0; j < nf; j++) ut[free[j]] = xn[j];
            dL = xn[nf];
          } else { for (let j = 0; j < nf; j++) ut[free[j]] += dr[j] + dli * df[j]; dL += dli; }
          setFixed(ut, lam + dL);
          if (!ut.every(isFinite)) break;
        }
        let acc2 = false;
        if (!conv && best && best.r <= acceptTol && P.secant) { conv = true; acc2 = true; ut.set(best.ut); st = best.st; it = best.it; dL = best.dL; unconv++; }
        if (conv && dropped(lam, lam + dL)) conv = false;
        if (conv) {
          u = ut; state = st; lam = lam + dL;
          record(u, lam, it, resid, state, !acc2);
          P.secant = baseSecant;
          bis = 0;
          dmon *= adapt(it);
          if (!o.to_failure && lam >= o.lam_max - 1e-12) break;
          const stop = finished(lam);
          if (stop) { status = stop; break; }
          continue;
        }
        if (!jump && retrySecant()) { nsteps--; continue; }
        P.secant = baseSecant;
        dmon /= 2; bis++; nsteps--;
        if (bis > (jump ? 3 : o.max_bisections)) { status = jump ? "drop" : lam > 0 ? "limit" : "failed"; break; }
        continue;
      }
      // ---- arc-length (Crisfield, cylindrical)
      const ra = respond(P, u, state, Pm, hasUbar ? ubar : null);
      if (!factor()) { status = "failed"; break; }
      const dxf = solveFree(rhsLoad(ra.kfu));
      if (dl === null) dl = Math.abs(dlam0) * norm(dxf);
      let sgn = 1;
      if (duPrev) { let dp = 0; for (let j = 0; j < nf; j++) dp += duPrev[j] * dxf[j]; sgn = dp >= 0 ? 1 : -1; }
      let dL = sgn * dl / Math.max(norm(dxf), 1e-300);
      let Du = new Float64Array(nf);
      for (let j = 0; j < nf; j++) Du[j] = dL * dxf[j];
      const ut = Float64Array.from(u);
      for (let j = 0; j < nf; j++) ut[free[j]] += Du[j];
      setFixed(ut, lam + dL);
      const resid = [];
      let conv = false, it = 0, st = null, best = null;
      for (it = 1; it <= o.max_iter; it++) {
        const rr = respond(P, ut, state, Pm, hasUbar ? ubar : null);
        st = rr.state;
        const { r, rn, ref } = residual(rr.fi, lam + dL);
        resid.push(rn / ref);
        if (acceptTol != null && it > 1 && (!best || rn / ref < best.r)) best = { r: rn / ref, ut: Float64Array.from(ut), st, it, dL, Du: Float64Array.from(Du) };
        if (rn <= tol * ref && it > 1) { conv = true; break; }
        if (!factor()) break;
        const dr = solveFree(r), df = solveFree(rhsLoad(rr.kfu));
        let a = 0, b = 0, c = -dl * dl;
        for (let j = 0; j < nf; j++) { const v = Du[j] + dr[j]; a += df[j] * df[j]; b += 2 * df[j] * v; c += v * v; }
        if (a <= 0) break;
        const disc = b * b - 4 * a * c;
        const roots = disc >= 0 ? [(-b + Math.sqrt(disc)) / (2 * a), (-b - Math.sqrt(disc)) / (2 * a)] : [-b / (2 * a)];
        let root = roots[0], rootv = -Infinity;
        for (const x of roots) { let v = 0; for (let j = 0; j < nf; j++) v += (Du[j] + dr[j] + x * df[j]) * Du[j]; if (v > rootv) { rootv = v; root = x; } }
        for (let j = 0; j < nf; j++) Du[j] = Du[j] + dr[j] + root * df[j];
        dL += root;
        for (let j = 0; j < nf; j++) ut[free[j]] = u[free[j]] + Du[j];
        setFixed(ut, lam + dL);
      }
      let acc3 = false;
      if (!conv && best && best.r <= acceptTol && P.secant) { conv = true; acc3 = true; ut.set(best.ut); st = best.st; it = best.it; dL = best.dL; Du = best.Du; unconv++; }
      if (conv && dropped(lam, lam + dL)) conv = false;
      if (conv) {
        u = ut; state = st; lam = lam + dL; duPrev = Du;
        record(u, lam, it, resid, state, !acc3);
        P.secant = baseSecant;
        bis = 0;
        dl *= adapt(it);
        if (!o.to_failure && lam >= o.lam_max - 1e-12) break;
        const stop = finished(lam);
        if (stop) { status = stop; break; }
        continue;
      }
      if (!jump && retrySecant()) { nsteps--; continue; }
      P.secant = baseSecant;
      dl /= 2; bis++; nsteps--;
      if (bis > (jump ? 3 : o.max_bisections)) { status = jump ? "drop" : lam > 0 ? "limit" : "failed"; break; }
    }
    if (status === "complete" && nsteps >= o.max_steps && (o.to_failure || lam < o.lam_max - 1e-12)) status = "max-steps";
    res.status = status;
    res.state = state;
    res.unconverged = unconv;
    res.lam_peak = res.steps.reduce((v, s) => Math.max(v, s.lam), -Infinity);
    const fmt = (x) => Number(x.toPrecision(4)).toString();
    if (status === "limit") res.message = o.control === "load"
      ? `No equilibrium beyond λ = ${fmt(lam)}: a limit point (load-carrying capacity) was reached. Displacement control or the arc-length method follow the path beyond it.`
      : `The analysis stopped at λ = ${fmt(lam)}: no converged increment, even after ${o.max_bisections} step reductions.`;
    else if (status === "drop") res.message = `Beyond the peak λ_u = ${fmt(res.lam_peak)} equilibrium was found only after losing more than half of it in one increment, also with increments down to 1/8: a sudden, brittle loss of capacity. λ_u is the capacity.`;
    else if (status === "rupture") res.message = `Reinforcement ruptured (steel strain above ε_ud) at λ = ${fmt(lam)}.`;
    else if (status === "post-peak") res.message = `The load fell below half the peak (λ_peak = ${fmt(res.lam_peak)}): the structure has failed.`;
    else if (status === "failed") res.message = "The first increment did not converge; reduce the step size or check the supports.";
    res.elapsed_ms = now() - t0;
    return res;
  }

  // ------------------------------------------------------------ post-processing
  function nodalStress(res, step) {
    const m = res.model, k = res.kernel, E = k.extrap, nc = E.length, nn = k.nn, ng = k.ng;
    const out = m.nodes.map(() => [0, 0, 0]), cnt = new Float64Array(m.nodes.length);
    m.elements.forEach((e, q) => {
      const vals = new Array(nn);
      for (let c = 0; c < nc; c++) { const v = [0, 0, 0]; for (let g = 0; g < ng; g++) for (let s = 0; s < 3; s++) v[s] += E[c][g] * step.gp_stress[3 * (q * ng + g) + s]; vals[c] = v; }
      for (const [mi, a, b] of k.mids) vals[mi] = [0, 1, 2].map((s) => 0.5 * (vals[a][s] + vals[b][s]));
      e.forEach((i, j) => { for (let s = 0; s < 3; s++) out[i][s] += vals[j][s]; cnt[i] += 1; });
    });
    out.forEach((v, i) => { const c = Math.max(cnt[i], 1); v[0] /= c; v[1] /= c; v[2] /= c; });
    return out;
  }
  // per-element nodal values (not averaged) of any Gauss-point quantity: get(i) → number, i = Gauss point index
  function elementNodal(res, get) {
    const m = res.model, k = res.kernel, E = k.extrap, nc = E.length, nn = k.nn, ng = k.ng;
    return m.elements.map((e, q) => {
      const vals = new Array(nn);
      for (let c = 0; c < nc; c++) { let v = 0; for (let g = 0; g < ng; g++) v += E[c][g] * get(q * ng + g); vals[c] = v; }
      for (const [mi, a, b] of k.mids) vals[mi] = 0.5 * (vals[a] + vals[b]);
      return vals;
    });
  }
  function averageNodal(m, elemVals) {
    const out = new Float64Array(m.nodes.length), cnt = new Float64Array(m.nodes.length);
    m.elements.forEach((e, q) => e.forEach((i, j) => { out[i] += elemVals[q][j]; cnt[i] += 1; }));
    for (let i = 0; i < out.length; i++) out[i] /= Math.max(cnt[i], 1);
    return Array.from(out);
  }
  // crack width (mm) at every Gauss point: w = s_r,max,θ (ε1 − σc1/Ec) (EN 1992-1-1 7.3.4, 7.15)
  function crackWidths(res, step) {
    const m = res.model, k = res.kernel, out = new Float64Array(k.N);
    if (!m.material || m.material.model !== "rc" || !step.extra.e1) return out;
    const re = F.elementReinforcement(m), tmm = m.t * 1000, c = m.reinforcement.cover_mm;
    const sr = (As, dia) => {
      if (!(As > 0)) return 1.3 * tmm;
      const hef = Math.min(2.5 * (c + dia / 2), tmm / 2), rho = (As / 2) / (hef * 1000);
      return 3.4 * c + 0.425 * 0.8 * 1.0 * dia / rho;
    };
    for (let q = 0; q < k.ne; q++) {
      const sx = sr(re.Asx[q], re.diax[q]), sy = sr(re.Asy[q], re.diay[q]);
      for (let g = 0; g < k.ng; g++) {
        const i = q * k.ng + g, th = step.extra.theta[i];
        const srt = 1 / (Math.abs(Math.cos(th)) / sx + Math.abs(Math.sin(th)) / sy);
        out[i] = step.extra.cracked[i] ? srt * Math.max(step.extra.e1[i] - step.extra.sc1[i] / m.E, 0) : 0;
      }
    }
    return out;
  }
  function summary(res) {
    const last = res.steps[res.steps.length - 1], m = res.model;
    const out = { status: res.status, message: res.message, steps: res.steps.length - 1, lam_final: last.lam, lam_peak: res.lam_peak, monitor_dof: res.monitor_dof,
      curve: res.steps.map((s) => [s.lam, s.monitor]), iterations: res.steps.slice(1).map((s) => s.iterations), elapsed_ms: res.elapsed_ms, unconverged: res.unconverged || 0 };
    const cnt = (a) => { let n = 0; for (let i = 0; i < a.length; i++) if (a[i]) n++; return n; };
    const amax = (a) => { let v = 0; for (let i = 0; i < a.length; i++) if (a[i] > v) v = a[i]; return v; };
    if (m.material && m.material.model === "rc") {
      const ex = last.extra, yl = ex.yield_x.map((v, i) => (v || ex.yield_y[i] ? 1 : 0));
      out.rc = { cracked_points: cnt(ex.cracked), yielded_points: cnt(yl), crushed_points: cnt(ex.crushed), rupture: ex.rupture.some((v) => v),
        max_crack_width_mm: amax(crackWidths(res, last)) };
    }
    if (m.material && m.material.model === "j2") out.j2 = { max_plastic_strain: amax(last.extra.alpha), yielded_points: cnt(last.extra.alpha.map((v) => (v > 0 ? 1 : 0))) };
    return out;
  }

  T.FENL = { DEFAULTS, runNonlinear, nodalStress, crackWidths, summary, j2Return, rcPoint, concreteEnv, steel, makeKernel, nlExtrapolation, smallSolve, anderson, RC_FLOAT, RC_FLAG,
    makeProblem, respond, elementNodal, averageNodal };
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
