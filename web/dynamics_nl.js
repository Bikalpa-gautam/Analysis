// Time-history analysis in the browser: a port of tragwerk.dynamics_nl (Python), checked against it in the test suite.
// Spectrum-compatible synthetic accelerograms (EN 1998-1 3.2.3.1.2), exact linear SDOF response (Nigam–Jennings),
// the Bouc–Wen oscillator, and implicit Newmark integration of the nonlinear plane FE model. Needs fe2d.js, fe2d_nl.js.
(function (root) {
  "use strict";
  const T = root.Tragwerk || (root.Tragwerk = {});
  const F = T.FE, NL = T.FENL;
  const G_ACC = 9.81;
  const SPEC = {
    1: { A: [1.0, 0.15, 0.4, 2.0], B: [1.2, 0.15, 0.5, 2.0], C: [1.15, 0.20, 0.6, 2.0], D: [1.35, 0.20, 0.8, 2.0], E: [1.4, 0.15, 0.5, 2.0] },
    2: { A: [1.0, 0.05, 0.25, 1.2], B: [1.35, 0.05, 0.25, 1.2], C: [1.5, 0.10, 0.25, 1.2], D: [1.8, 0.10, 0.30, 1.2], E: [1.6, 0.05, 0.25, 1.2] },
  };
  const now = () => (typeof performance !== "undefined" ? performance : Date).now();

  function elasticSpectrum(Tv, ag, soil = "B", stype = 1, xi = 0.05) {
    if (!SPEC[stype] || !SPEC[stype][soil]) throw new F.FEError("spectrum: type 1 or 2 and ground type A–E");
    const [S, TB, TC, TD] = SPEC[stype][soil], eta = Math.max(Math.sqrt(10 / (5 + 100 * xi)), 0.55);
    const one = (t) => (t <= TB ? ag * S * (1 + t / TB * (eta * 2.5 - 1)) : t <= TC ? ag * S * eta * 2.5 : t <= TD ? ag * S * eta * 2.5 * TC / Math.max(t, 1e-12)
      : ag * S * eta * 2.5 * TC * TD / Math.max(t, 1e-12) ** 2);
    return Array.isArray(Tv) || ArrayBuffer.isView(Tv) ? Array.from(Tv, one) : one(Tv);
  }
  function LCG(seed) {
    let x = seed >>> 0;
    return { next() { x = (Math.imul(1664525, x) + 1013904223) >>> 0; return x / 4294967296; } };
  }

  // ------------------------------------------------------------ exact linear SDOF (Nigam–Jennings, Chopra 5.2)
  function njCoefficients(Tp, xi, dt) {
    const w = 2 * Math.PI / Tp, wd = w * Math.sqrt(1 - xi * xi), e = Math.exp(-xi * w * dt), s = Math.sin(wd * dt), c = Math.cos(wd * dt);
    const r = xi / Math.sqrt(1 - xi * xi), k = w * w, sq = Math.sqrt(1 - xi * xi);
    return [e * (r * s + c), e * s / wd,
      (2 * xi / (w * dt) + e * (((1 - 2 * xi * xi) / (wd * dt) - r) * s - (1 + 2 * xi / (w * dt)) * c)) / k,
      (1 - 2 * xi / (w * dt) + e * ((2 * xi * xi - 1) / (wd * dt) * s + 2 * xi / (w * dt) * c)) / k,
      -e * w / sq * s, e * (c - r * s),
      (-1 / dt + e * ((w / sq + xi / (dt * sq)) * s + c / dt)) / k,
      (1 - e * (r * s + c)) / (k * dt)];
  }
  function responseSpectrum(a, dt, periods, xi = 0.05) {
    const nT = periods.length, co = Array.from(periods, (Tp) => njCoefficients(Tp, xi, dt));
    const Sd = new Float64Array(nT), Sv = new Float64Array(nT), Sa = new Float64Array(nT);
    for (let j = 0; j < nT; j++) {
      const [A, B, C, D, A1, B1, C1, D1] = co[j];
      let u = 0, v = 0, um = 0, vm = 0;
      for (let i = 0; i + 1 < a.length; i++) {
        const p0 = -a[i], p1 = -a[i + 1], un = A * u + B * v + C * p0 + D * p1, vn = A1 * u + B1 * v + C1 * p0 + D1 * p1;
        u = un; v = vn;
        if (Math.abs(u) > um) um = Math.abs(u);
        if (Math.abs(v) > vm) vm = Math.abs(v);
      }
      const w = 2 * Math.PI / periods[j];
      Sd[j] = um; Sv[j] = vm; Sa[j] = w * w * um;
    }
    return { T: Array.from(periods), Sd, Sv, Sa };
  }
  function sdofHistory(a, dt, Tp, xi = 0.05) {
    const [A, B, C, D, A1, B1, C1, D1] = njCoefficients(Tp, xi, dt), n = a.length, u = new Float64Array(n), v = new Float64Array(n);
    for (let i = 0; i + 1 < n; i++) { u[i + 1] = A * u[i] + B * v[i] - C * a[i] - D * a[i + 1]; v[i + 1] = A1 * u[i] + B1 * v[i] - C1 * a[i] - D1 * a[i + 1]; }
    return { u, v };
  }

  // ------------------------------------------------------------ synthetic accelerograms
  function baseline(a, dt) {
    const n = a.length, Tn = (n - 1) * dt;
    let v = 0, d = 0;
    for (let i = 0; i + 1 < n; i++) { d += v * dt + (2 * a[i] + a[i + 1]) * dt * dt / 6; v += 0.5 * (a[i] + a[i + 1]) * dt; }
    // [[T, T²/2], [T²/2, T³/6]] [c0, c1] = [v, d]
    const m00 = Tn, m01 = Tn * Tn / 2, m11 = Tn ** 3 / 6, det = m00 * m11 - m01 * m01;
    const c0 = (v * m11 - m01 * d) / det, c1 = (m00 * d - m01 * v) / det;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = a[i] - c0 - c1 * (i * dt);
    return out;
  }
  function geomspace(a, b, n) { const out = new Float64Array(n), r = Math.log(b / a) / (n - 1); for (let i = 0; i < n; i++) out[i] = a * Math.exp(r * i); out[n - 1] = b; return out; }
  function syntheticAccelerogram(agG = 0.25, soil = "B", stype = 1, duration = 20, dt = 0.01, seed = 1, rise = 2, strong = 10, iterations = 15, xi = 0.05) {
    if (!(agG > 0 && agG <= 2) || !(duration >= 2 && duration <= 120) || !(dt >= 0.002 && dt <= 0.05)) throw new F.FEError("record: 0 < a_g ≤ 2 g, 2 s ≤ duration ≤ 120 s, 0.002 s ≤ Δt ≤ 0.05 s");
    if (rise + strong >= duration) throw new F.FEError("record: rise + strong phase must be shorter than the duration");
    const ag = agG * G_ACC, n = Math.round(duration / dt) + 1, t2 = rise + strong, decay = -Math.log(0.05) / (duration - rise - strong);
    const env = new Float64Array(n);
    for (let i = 0; i < n; i++) { const t = i * dt; env[i] = t < rise ? (t / rise) ** 2 : t <= t2 ? 1 : Math.exp(-decay * (t - t2)); }
    const fmax = Math.min(25, 0.4 / dt), fl = [0.1];
    while (fl[fl.length - 1] < fmax) fl.push(fl[fl.length - 1] + Math.max(0.02 * fl[fl.length - 1], 0.5 / duration));
    fl.pop();
    const nf = fl.length, w = fl.map((f) => 2 * Math.PI * f), dw = w.map((_, k) => (k === 0 ? w[1] - w[0] : k === nf - 1 ? w[nf - 1] - w[nf - 2] : 0.5 * (w[k + 1] - w[k - 1])));
    const rng = LCG(seed), phi = fl.map(() => 2 * Math.PI * rng.next()), Tk = fl.map((f) => 1 / f);
    const target = elasticSpectrum(Tk, ag, soil, stype, xi);
    let amp = target.map((sa, k) => Math.sqrt(2 * (xi * sa * sa / (Math.PI * w[k] * 2.5 * 2.5)) * dw[k]));
    // sin table per component (n × nf would be large; recompute per iteration in place)
    const synth = () => {
      const a = new Float64Array(n);
      for (let k = 0; k < nf; k++) { const A = amp[k], wk = w[k], pk = phi[k]; for (let i = 0; i < n; i++) a[i] += A * Math.sin(wk * (i * dt) + pk); }
      for (let i = 0; i < n; i++) a[i] *= env[i];
      return baseline(a, dt);
    };
    for (let it = 0; it < iterations; it++) {
      const sp = responseSpectrum(synth(), dt, Tk, xi).Sa;
      amp = amp.map((A, k) => A * Math.min(Math.max(target[k] / Math.max(sp[k], 1e-12), 0.5), 2.0));
    }
    let a = synth(), pga = 0;
    for (let i = 0; i < n; i++) pga = Math.max(pga, Math.abs(a[i]));
    const sc = Math.max(1, ag * SPEC[stype][soil][0] / pga);
    for (let i = 0; i < n; i++) a[i] *= sc;
    const mT = geomspace(Math.max(0.05, 1.5 / fmax), 4.0, 60);
    return { t: Float64Array.from({ length: n }, (_, i) => i * dt), a, dt, name: `synthetic, a_g = ${agG} g, ground ${soil}, type ${stype}, seed ${seed}`,
      target: { ag_g: agG, soil, type: stype, xi, duration, seed }, match: { T: Array.from(mT), target: elasticSpectrum(mT, ag, soil, stype, xi), achieved: Array.from(responseSpectrum(a, dt, mT, xi).Sa) },
      get pga() { let p = 0; for (const v of this.a) p = Math.max(p, Math.abs(v)); return p; } };
  }
  function recordFromValues(valuesG, dt, name = "record") {
    const a = Float64Array.from(valuesG, (v) => Number(v) * G_ACC);
    if (a.length < 3 || !a.every(isFinite)) throw new F.FEError("record: at least three finite acceleration values (in g) are needed");
    if (!(dt >= 0.0005 && dt <= 0.1)) throw new F.FEError("record: 0.0005 s ≤ Δt ≤ 0.1 s");
    return { t: Float64Array.from({ length: a.length }, (_, i) => i * dt), a, dt, name, target: null, match: null,
      get pga() { let p = 0; for (const v of this.a) p = Math.max(p, Math.abs(v)); return p; } };
  }
  function parseRecord(spec) {
    const kind = spec.type || "synthetic";
    if (kind === "synthetic") return syntheticAccelerogram(Number(spec.ag_g ?? 0.25), String(spec.soil ?? "B"), Number(spec.spectrum_type ?? 1), Number(spec.duration_s ?? 20),
      Number(spec.dt_s ?? 0.01), Number(spec.seed ?? 1), Number(spec.rise_s ?? 2), Number(spec.strong_s ?? 10));
    if (kind === "values") return recordFromValues(spec.values_g || [], Number(spec.dt_s ?? 0.01), String(spec.name || "record"));
    throw new F.FEError("record.type must be synthetic or values");
  }

  // ------------------------------------------------------------ Bouc–Wen oscillator
  const BW_DEFAULTS = { m: 100, k: 4e4, uy: 0.01, alpha: 0.05, n: 2, beta: 0.5, gamma: 0.5, A: 1, xi: 0.05 };
  const bwDz = (z, du, p) => (p.A - Math.abs(z) ** p.n * (p.beta * Math.sign(du * z) + p.gamma)) / p.uy;
  function bwAdvance(z0, du, p, sub = null) {
    const m = sub || Math.max(1, Math.min(200, Math.ceil(8 * Math.abs(du) / p.uy))), h = du / m;
    let z = z0;
    for (let i = 0; i < m; i++) {
      const k1 = bwDz(z, du, p), k2 = bwDz(z + 0.5 * h * k1, du, p), k3 = bwDz(z + 0.5 * h * k2, du, p), k4 = bwDz(z + h * k3, du, p);
      z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    return [z, bwDz(z, du !== 0 ? du : 1, p)];
  }
  function boucWenSdof(pin, rec, newmark = [0.5, 0.25], tol = 1e-10, maxIter = 30) {
    const p = Object.assign({}, BW_DEFAULTS, pin), [gam, bet] = newmark, dt = rec.dt, ag = rec.a, n = ag.length, c = 2 * p.xi * Math.sqrt(p.k * p.m);
    const u = new Float64Array(n), v = new Float64Array(n), acc = new Float64Array(n), z = new Float64Array(n), fs = new Float64Array(n);
    const Ein = new Float64Array(n), Ed = new Float64Array(n), Es = new Float64Array(n);
    acc[0] = (-p.m * ag[0] - c * v[0]) / p.m;
    let itTot = 0;
    for (let i = 0; i + 1 < n; i++) {
      const fext = -p.m * ag[i + 1];
      let du = 0, u1 = 0, v1 = 0, a1 = 0, zi = z[i], f1 = 0;
      for (let it = 0; it < maxIter; it++) {
        u1 = u[i] + du;
        v1 = gam / (bet * dt) * du + (1 - gam / bet) * v[i] + dt * (1 - gam / (2 * bet)) * acc[i];
        a1 = du / (bet * dt * dt) - v[i] / (bet * dt) - (1 / (2 * bet) - 1) * acc[i];
        const [zz, dzdu] = bwAdvance(z[i], du, p);
        zi = zz;
        f1 = p.alpha * p.k * u1 + (1 - p.alpha) * p.k * p.uy * zi;
        const r = fext - p.m * a1 - c * v1 - f1, kt = p.alpha * p.k + (1 - p.alpha) * p.k * p.uy * dzdu;
        const keff = p.m / (bet * dt * dt) + c * gam / (bet * dt) + kt;
        if (Math.abs(r) <= tol * Math.max(Math.abs(fext), p.k * p.uy * 1e-3) && it > 0) break;
        du += r / keff;
        itTot++;
      }
      u[i + 1] = u1; v[i + 1] = v1; acc[i + 1] = a1; z[i + 1] = zi; fs[i + 1] = f1;
      const dU = u[i + 1] - u[i];
      Ein[i + 1] = Ein[i] - p.m * 0.5 * (ag[i] + ag[i + 1]) * dU;
      Ed[i + 1] = Ed[i] + c * 0.5 * (v[i] + v[i + 1]) * dU;
      Es[i + 1] = Es[i] + 0.5 * (fs[i] + fs[i + 1]) * dU;
    }
    const Ek = v.map((x) => 0.5 * p.m * x * x), Estr = u.map((x, i) => 0.5 * p.alpha * p.k * x * x + 0.5 * (1 - p.alpha) * p.k * p.uy * p.uy * z[i] * z[i]);
    let umax = 0, fmax = 0, emax = 0, err = 0;
    for (let i = 0; i < n; i++) { umax = Math.max(umax, Math.abs(u[i])); fmax = Math.max(fmax, Math.abs(fs[i])); emax = Math.max(emax, Math.abs(Ein[i])); }
    for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(Ein[i] - Ek[i] - Ed[i] - Es[i]));
    return { t: rec.t, u, v, a_rel: acc, a_abs: acc.map((x, i) => x + ag[i]), z, fs, E_in: Ein, E_kin: Ek, E_damp: Ed, E_s: Es, E_hyst: Es.map((x, i) => x - Estr[i]), E_strain: Estr,
      u_max: umax, ductility: umax / p.uy, fs_max: fmax, iterations: itTot, energy_error: err / Math.max(emax, 1e-30), period: 2 * Math.PI * Math.sqrt(p.m / p.k) };
  }

  // ------------------------------------------------------------ profile helpers for the FE dynamics
  function profileMatvec(P, A, x) {
    const { nf, first, ptr } = P, y = new Float64Array(nf);
    for (let i = 0; i < nf; i++) {
      const fi = first[i], base = ptr[i] - fi;
      let s = A[base + i] * x[i];
      for (let j = fi; j < i; j++) { const a = A[base + j]; s += a * x[j]; y[j] += a * x[i]; }
      y[i] += s;
    }
    return y;
  }
  // lowest eigenpairs of K φ = λ M φ by subspace iteration with Rayleigh–Ritz (Bathe)
  function lowestModes(Pm, KA, MA, nmodes) {
    const nf = Pm.nf, p = Math.min(nf, Math.max(2 * nmodes, nmodes + 6));
    const KF = Object.assign({}, Pm, { A: Float64Array.from(KA) });
    F.profileFactor(KF, true);
    let X = [];
    const md = new Float64Array(nf);
    for (let i = 0; i < nf; i++) md[i] = MA[Pm.ptr[i] - Pm.first[i] + i];
    for (let j = 0; j < p; j++) { const x = new Float64Array(nf); for (let i = 0; i < nf; i++) x[i] = j === 0 ? md[i] : (((i * 7919 + j * 104729) % 1009) / 1009 - 0.5) * (md[i] || 1); X.push(x); }
    let lam = new Array(p).fill(0), vecs = X;
    for (let it = 0; it < 200; it++) {
      const Y = X.map((x) => F.profileSolve(KF, profileMatvec(Pm, MA, x)));
      const KY = Y.map((y) => profileMatvec(Pm, KA, y)), MY = Y.map((y) => profileMatvec(Pm, MA, y));
      const Kr = Y.map((a) => KY.map((b) => a.reduce((s, v, i) => s + v * b[i], 0))), Mr = Y.map((a) => MY.map((b) => a.reduce((s, v, i) => s + v * b[i], 0)));
      const { values, vectors } = genEig(Kr, Mr);
      const nx = vectors.map((q) => { const x = new Float64Array(nf); q.forEach((c, k) => { const y = Y[k]; for (let i = 0; i < nf; i++) x[i] += c * y[i]; }); return x; });
      const conv = values.slice(0, nmodes).every((v, k) => Math.abs(v - lam[k]) <= 1e-13 * Math.abs(v));
      lam = values; vecs = nx; X = nx;
      if (conv && it > 2) break;
    }
    return { values: lam.slice(0, nmodes), vectors: vecs.slice(0, nmodes) };
  }
  // small generalized symmetric eigenproblem A q = λ B q (B SPD): Cholesky reduction + Jacobi
  function genEig(A, B) {
    const n = A.length, L = B.map((r) => r.map(() => 0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = B[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
    }
    const Li = L.map((r) => r.map(() => 0));
    for (let i = 0; i < n; i++) { Li[i][i] = 1 / L[i][i]; for (let j = 0; j < i; j++) { let s = 0; for (let k = j; k < i; k++) s -= L[i][k] * Li[k][j]; Li[i][j] = s / L[i][i]; } }
    const C = Li.map((ri) => Li.map((rj) => { let s = 0; for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) s += ri[a] * A[a][b] * rj[b]; return s; }));
    const V = C.map((r, i) => r.map((_, j) => (i === j ? 1 : 0))), M = C.map((r) => r.slice());
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += M[p][q] * M[p][q];
      if (off < 1e-32) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(M[p][q]) < 1e-300) continue;
        const th = (M[q][q] - M[p][p]) / (2 * M[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const a = M[k][p], b = M[k][q]; M[k][p] = c * a - s * b; M[k][q] = s * a + c * b; }
        for (let k = 0; k < n; k++) { const a = M[p][k], b = M[q][k]; M[p][k] = c * a - s * b; M[q][k] = s * a + c * b; }
        for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
      }
    }
    const order = M.map((r, i) => i).sort((a, b) => M[a][a] - M[b][b]);
    return { values: order.map((i) => M[i][i]), vectors: order.map((i) => { const y = V.map((r) => r[i]); return Li.map((_, a) => { let s = 0; for (let b = 0; b < n; b++) s += Li[b][a] * y[b]; return s; }); }) };
  }

  // ------------------------------------------------------------ nonlinear FE dynamics
  const DYN_DEFAULTS = { combination: "qp", mass_from_loads: true, xi: 0.05, newmark: [0.5, 0.25], tol: null, max_iter: 30, direction: 0, snapshots: 120, monitor: null };
  function assembleMass(m, o, Pm) {
    const MA = new Float64Array(Pm.A.length), rho = m.selfWeight ? m.gamma / G_ACC : 0, n2 = 2 * m.nodes.length, full = new Float64Array(n2);   // full: M·r per dof
    let any = false;
    if (rho > 0) {
      m.elements.forEach((e) => {
        const Me = F.elementMass(m.etype, e.map((i) => m.nodes[i]), rho, m.t), d = F.dofsOf(e);
        F.profileAdd(Pm.eq ? { eq: Pm.eq, ptr: Pm.ptr, first: Pm.first, A: MA } : null, d, Me);
        for (let a = 0; a < d.length; a++) for (let b = 0; b < d.length; b++) if (d[a] % 2 === d[b] % 2) full[d[a]] += Me[a][b];
        any = true;
      });
    }
    if (o.mass_from_loads) {
      const [aG, aQ] = F.combinationFactors(m, o.combination);
      for (let i = 0; i < m.nodes.length; i++) {
        const fy = aG * (m.forces[2 * i + 1] - m.forcesQ[2 * i + 1]) + aQ * m.forcesQ[2 * i + 1], ml = Math.max(-fy, 0) / G_ACC;
        if (!(ml > 0)) continue;
        any = true;
        for (const d of [2 * i, 2 * i + 1]) { full[d] += ml; const q = Pm.eq[d]; if (q >= 0) MA[Pm.ptr[q] - Pm.first[q] + q] += ml; }
      }
    }
    if (!any) throw new F.FEError("the model has no mass: give the material a density (self weight) or gravity loads");
    return { MA, Mr_full: full };
  }

  function feDynamic(m, rec, opts = {}, onProgress = null) {
    const t0 = now(), o = Object.assign({}, DYN_DEFAULTS, opts), [gam, bet] = o.newmark, dt = rec.dt;
    const stat = NL.runNonlinear(m, { combination: o.combination, steps: 4, lam_max: 1.0 });
    if (stat.status !== "complete") throw new F.FEError(`the gravity loads of the seismic combination could not be applied: ${stat.message}`);
    const P = NL.makeProblem(m, Object.assign({}, NL.DEFAULTS)), k = P.k, n2 = k.n2;
    const fGrav = F.loadVector(m, o.combination);
    const fixed = [...m.fixed.keys()].sort((a, b) => a - b);
    const Pm = F.makeProfile(m, new Map(m.fixed)), { nf, eq } = Pm, free = new Int32Array(nf);
    for (let d = 0; d < n2; d++) if (eq[d] >= 0) free[eq[d]] = d;
    const { MA, Mr_full } = assembleMass(m, o, Pm);
    NL.respond(P, new Float64Array(n2), P.state0, Pm, null);
    const K0A = Float64Array.from(Pm.A);
    const nm = Math.min(3, nf - 1), modes = lowestModes(Pm, K0A, MA, nm), w = modes.values.map((x) => Math.sqrt(Math.max(x, 1e-30)));
    const w1 = w[0], w2 = w.length > 1 ? w[1] : 3 * w[0], a0 = o.xi * 2 * w1 * w2 / (w1 + w2), a1 = o.xi * 2 / (w1 + w2);
    let mon;
    if (o.monitor == null) {
      const phi = new Float64Array(n2);
      for (let j = 0; j < nf; j++) phi[free[j]] = modes.vectors[0][j];
      let best = 0;
      for (let i = 0; i < m.nodes.length; i++) best = Math.max(best, Math.abs(phi[2 * i + o.direction]));
      let bi = 0;
      for (let i = 0; i < m.nodes.length; i++) if (Math.abs(phi[2 * i + o.direction]) >= (1 - 1e-6) * best) { bi = i; break; }
      mon = 2 * bi + o.direction;
    } else mon = 2 * o.monitor[0] + o.monitor[1];
    const mf = eq[mon];
    const rFree = new Float64Array(nf);
    for (let j = 0; j < nf; j++) rFree[j] = free[j] % 2 === o.direction ? 1 : 0;
    // M·r on the free dofs (r = 1 on the dofs of the ground motion direction, also at the supports)
    const rAll = new Float64Array(n2);
    for (let d = o.direction; d < n2; d += 2) rAll[d] = 1;
    const Mr = (() => {
      // free rows of M·r: M_ff r_f + M_fc r_c; assemble from the element matrices to include the coupling to the supports
      const out = new Float64Array(nf), rho = m.selfWeight ? m.gamma / G_ACC : 0;
      if (rho > 0) m.elements.forEach((e) => {
        const Me = F.elementMass(m.etype, e.map((i) => m.nodes[i]), rho, m.t), d = F.dofsOf(e);
        for (let a = 0; a < d.length; a++) { const q = eq[d[a]]; if (q < 0) continue; for (let b = 0; b < d.length; b++) out[q] += Me[a][b] * rAll[d[b]]; }
      });
      if (o.mass_from_loads) {
        const [aG, aQ] = F.combinationFactors(m, o.combination);
        for (let i = 0; i < m.nodes.length; i++) {
          const fy = aG * (m.forces[2 * i + 1] - m.forcesQ[2 * i + 1]) + aQ * m.forcesQ[2 * i + 1], ml = Math.max(-fy, 0) / G_ACC;
          const q = eq[2 * i + o.direction];
          if (ml > 0 && q >= 0) out[q] += ml;
        }
      }
      return out;
    })();
    const tol = o.tol != null ? o.tol : (P.kind === "rc" ? 1e-4 : 1e-8);
    const CA = new Float64Array(MA.length);
    for (let i = 0; i < CA.length; i++) CA[i] = a0 * MA[i] + a1 * K0A[i];
    const nT0 = rec.a.length;
    let u = Float64Array.from(stat.steps[stat.steps.length - 1].u);
    const uBase = Float64Array.from(u), uGrav = Float64Array.from(u);
    let state = stat.state;
    let uf = new Float64Array(nf), vf = new Float64Array(nf), af = new Float64Array(nf);
    for (let j = 0; j < nf; j++) { uf[j] = u[free[j]]; af[j] = -rFree[j] * rec.a[0]; }
    const r0 = NL.respond(P, u, state, null, null);
    const fxFixed = fixed.filter((d) => d % 2 === o.direction);
    const tMon = new Float64Array(nT0), vMon = new Float64Array(nT0), aMon = new Float64Array(nT0), Vb = new Float64Array(nT0);
    aMon[0] = af[mf] + rec.a[0];
    Vb[0] = fxFixed.reduce((s, d) => s + r0.fi[d], 0);
    const Ein = new Float64Array(nT0), Ed = new Float64Array(nT0), Es = new Float64Array(nT0), Ek = new Float64Array(nT0);
    let massTotal = 0;
    for (let d = 0; d < n2; d++) massTotal += Mr_full[d] * rAll[d];
    const res = { t: rec.t, u_mon: tMon, v_mon: vMon, a_mon_abs: aMon, base_shear: Vb, periods: w.map((x) => 2 * Math.PI / x), rayleigh: [a0, a1], monitor_dof: mon,
      frames: [], energy: {}, status: "complete", message: "", iterations: [], substeps: [], mass_total: massTotal, model: m, kernel: k };
    const every = Math.max(1, Math.floor((nT0 - 1) / Math.max(o.snapshots, 1)));
    const st0 = stat.steps[stat.steps.length - 1];
    res.frames.push({ i: 0, u: Float64Array.from(u), gp_stress: st0.gp_stress, extra: st0.extra });
    let fintPrev = new Float64Array(nf);
    for (let j = 0; j < nf; j++) fintPrev[j] = r0.fi[free[j]];
    const c1 = 1 / (bet * dt * dt), c2 = gam / (bet * dt);
    const linear = P.kind === "elastic" && !P.o.geometric;
    const fgF = new Float64Array(nf);
    for (let j = 0; j < nf; j++) fgF[j] = fGrav[free[j]];
    const dot = (x, y) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; };
    const nrm = (x) => Math.sqrt(dot(x, x));
    let KL = null;
    if (linear) {
      KL = Object.assign({}, Pm, { A: new Float64Array(K0A.length) });
      for (let i = 0; i < K0A.length; i++) KL.A[i] = K0A[i] + c1 * MA[i] + c2 * CA[i];
      F.profileFactor(KL, true);
    }
    const KE = Object.assign({}, Pm, { A: new Float64Array(K0A.length) });
    const toFull = (x) => { const ut = Float64Array.from(uBase); for (let j = 0; j < nf; j++) ut[free[j]] = x[j]; return ut; };
    function nlStep(uf0, vf0, af0, st, ag1, h) {
      const c1h = 1 / (bet * h * h), c2h = gam / (bet * h), fext = new Float64Array(nf), mra = new Float64Array(nf);
      for (let j = 0; j < nf; j++) { mra[j] = Mr[j] * ag1; fext[j] = fgF[j] - mra[j]; }
      const ref = Math.max(nrm(fext), nrm(mra), 1e-9);
      const pv = new Float64Array(nf), pa = new Float64Array(nf);
      for (let j = 0; j < nf; j++) { pv[j] = (1 - gam / bet) * vf0[j] + h * (1 - gam / (2 * bet)) * af0[j]; pa[j] = -vf0[j] / (bet * h) - (1 / (2 * bet) - 1) * af0[j]; }
      for (const secant of (P.kind === "rc" ? [false, true] : [false])) {
        P.secant = secant;
        let du = new Float64Array(nf), ok = true;
        for (let it = 1; it <= o.max_iter; it++) {
          const u1 = new Float64Array(nf), v1 = new Float64Array(nf), a1v = new Float64Array(nf);
          for (let j = 0; j < nf; j++) { u1[j] = uf0[j] + du[j]; v1[j] = c2h * du[j] + pv[j]; a1v[j] = c1h * du[j] + pa[j]; }
          const ut = toFull(u1), rr = NL.respond(P, ut, st, Pm, null);
          const Ma = profileMatvec(Pm, MA, a1v), Cv = profileMatvec(Pm, CA, v1), R = new Float64Array(nf);
          for (let j = 0; j < nf; j++) R[j] = fext[j] - Ma[j] - Cv[j] - rr.fi[free[j]];
          if (nrm(R) <= tol * ref && it > 1) { P.secant = false; return { u1, v1, a1v, ut, st: rr.state, fi: rr.fi, sig: rr.sig, ex: rr.extra, it }; }
          for (let i = 0; i < KE.A.length; i++) KE.A[i] = Pm.A[i] + c1h * MA[i] + c2h * CA[i];
          try { F.profileFactor(KE, false); } catch (e) { ok = false; break; }
          const dx = F.profileSolve(KE, R);
          for (let j = 0; j < nf; j++) du[j] += dx[j];
          if (!du.every(isFinite)) { ok = false; break; }
        }
        if (!ok) continue;
      }
      P.secant = false;
      return null;
    }
    function advance(uf0, vf0, af0, st, ag0, ag1, h, depth) {
      const r1 = nlStep(uf0, vf0, af0, st, ag1, h);
      if (r1) return [[r1, ag0, ag1]];
      if (depth >= 3) return null;
      const agm = 0.5 * (ag0 + ag1), first = advance(uf0, vf0, af0, st, ag0, agm, h / 2, depth + 1);
      if (!first) return null;
      const q = first[first.length - 1][0], second = advance(q.u1, q.v1, q.a1v, q.st, agm, ag1, h / 2, depth + 1);
      return second ? first.concat(second) : null;
    }
    let nT = nT0, lastSig = st0.gp_stress, lastEx = st0.extra, lastFi = r0.fi;
    for (let i = 0; i + 1 < nT0; i++) {
      if (linear) {
        const pv = new Float64Array(nf), pa = new Float64Array(nf);
        for (let j = 0; j < nf; j++) { pv[j] = (1 - gam / bet) * vf[j] + dt * (1 - gam / (2 * bet)) * af[j]; pa[j] = -vf[j] / (bet * dt) - (1 / (2 * bet) - 1) * af[j]; }
        const Ma = profileMatvec(Pm, MA, pa), Cv = profileMatvec(Pm, CA, pv), Ku = profileMatvec(Pm, K0A, uf), R = new Float64Array(nf);
        for (let j = 0; j < nf; j++) R[j] = fgF[j] - Mr[j] * rec.a[i + 1] - Ma[j] - Cv[j] - Ku[j];
        const du = F.profileSolve(KL, R), u1 = new Float64Array(nf), v1 = new Float64Array(nf), a1v = new Float64Array(nf);
        for (let j = 0; j < nf; j++) { u1[j] = uf[j] + du[j]; v1[j] = c2 * du[j] + pv[j]; a1v[j] = c1 * du[j] + pa[j]; }
        const ut = toFull(u1), fiF = profileMatvec(Pm, K0A, u1);
        // full internal forces K0·u (also at the supports, for the base shear) from the element matrices
        const fi = NL.respond(P, ut, P.state0, null, null).fi;
        const Cbar = profileMatvec(Pm, CA, v1.map((x, j) => 0.5 * (vf[j] + x)));
        Ein[i + 1] = Ein[i]; Ed[i + 1] = Ed[i]; Es[i + 1] = Es[i];
        for (let j = 0; j < nf; j++) {
          const dU = du[j];
          Ein[i + 1] += 0.5 * (-Mr[j] * rec.a[i] - Mr[j] * rec.a[i + 1]) * dU;
          Ed[i + 1] += Cbar[j] * dU;
          Es[i + 1] += (0.5 * (fintPrev[j] + fiF[j]) - fgF[j]) * dU;
        }
        fintPrev = fiF;
        uf = u1; vf = v1; af = a1v; u = ut;
        res.iterations.push(1); res.substeps.push(1);
        Ek[i + 1] = 0.5 * dot(vf, profileMatvec(Pm, MA, vf));
        tMon[i + 1] = u[mon] - uGrav[mon]; vMon[i + 1] = vf[mf]; aMon[i + 1] = af[mf] + rec.a[i + 1];
        Vb[i + 1] = fxFixed.reduce((s, d) => s + fi[d], 0);
        if ((i + 1) % every === 0 || i + 1 === nT0 - 1) res.frames.push({ i: i + 1, u: Float64Array.from(u), gp_stress: null, extra: {} });
      } else {
        const subs = advance(uf, vf, af, state, rec.a[i], rec.a[i + 1], dt, 0);
        if (!subs) {
          res.status = "failed";
          res.message = `no convergence at t = ${rec.t[i + 1].toFixed(3)} s, even with the time step divided by 8` + failureHint(P, m, u, state);
          nT = i + 1;
          break;
        }
        Ein[i + 1] = Ein[i]; Ed[i + 1] = Ed[i]; Es[i + 1] = Es[i];
        let its = 0;
        for (const [q, ag0, ag1] of subs) {
          const vbar = q.v1.map((x, j) => 0.5 * (vf[j] + x)), Cbar = profileMatvec(Pm, CA, vbar);
          for (let j = 0; j < nf; j++) {
            const dU = q.u1[j] - uf[j], fij = q.fi[free[j]];
            Ein[i + 1] += 0.5 * (-Mr[j] * ag0 - Mr[j] * ag1) * dU;
            Ed[i + 1] += Cbar[j] * dU;
            Es[i + 1] += (0.5 * (fintPrev[j] + fij) - fgF[j]) * dU;
          }
          for (let j = 0; j < nf; j++) fintPrev[j] = q.fi[free[j]];
          uf = q.u1; vf = q.v1; af = q.a1v; u = q.ut; state = q.st; lastSig = q.sig; lastEx = q.ex; lastFi = q.fi;
          its += q.it;
        }
        res.iterations.push(its); res.substeps.push(subs.length);
        Ek[i + 1] = 0.5 * dot(vf, profileMatvec(Pm, MA, vf));
        tMon[i + 1] = u[mon] - uGrav[mon]; vMon[i + 1] = vf[mf]; aMon[i + 1] = af[mf] + rec.a[i + 1];
        Vb[i + 1] = fxFixed.reduce((s, d) => s + lastFi[d], 0);
        if ((i + 1) % every === 0 || i + 1 === nT0 - 1) res.frames.push({ i: i + 1, u: Float64Array.from(u), gp_stress: lastSig, extra: lastEx });
      }
      if (onProgress && (i % 20 === 0)) onProgress(i + 1, nT0, res);
    }
    const cut = (x) => x.slice(0, nT);
    res.t = cut(rec.t); res.u_mon = cut(tMon); res.v_mon = cut(vMon); res.a_mon_abs = cut(aMon); res.base_shear = cut(Vb);
    res.energy = { E_in: cut(Ein), E_damp: cut(Ed), E_internal: cut(Es), E_kin: cut(Ek) };
    let err = 0, emax = 0;
    for (let i = 0; i < nT; i++) { err = Math.max(err, Math.abs(Ein[i] - Ed[i] - Es[i] - Ek[i])); emax = Math.max(emax, Math.abs(Ein[i])); }
    res.energy.error = err / Math.max(emax, 1e-30);
    res.elapsed_ms = now() - t0;
    return res;
  }
  function failureHint(P, m, u, state) {
    if (P.kind !== "rc") return "";
    const r = NL.respond(P, u, state, null, null), e2 = r.extra.e2, xy = P.k.xy, out = [];
    let k = 0;
    for (let i = 1; i < e2.length; i++) if (e2[i] < e2[k]) k = i;
    if (e2[k] < -0.8 * m.material.eps_c0) out.push(`concrete near its compressive strength at (${xy[2 * k].toFixed(2)}, ${xy[2 * k + 1].toFixed(2)}) m (ε₂ = ${(e2[k] * 1000).toFixed(2)} ‰): crushing`);
    if (r.extra.rupture.some((v) => v)) out.push("reinforcement strain above ε_ud");
    return out.length ? ": " + out.join("; ") : "";
  }

  T.DYN = { G_ACC, SPEC, elasticSpectrum, LCG, njCoefficients, responseSpectrum, sdofHistory, baseline, syntheticAccelerogram, recordFromValues, parseRecord,
    BW_DEFAULTS, bwAdvance, boucWenSdof, DYN_DEFAULTS, feDynamic, lowestModes, profileMatvec, genEig };
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
