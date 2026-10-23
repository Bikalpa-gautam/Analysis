// Reinforcement design of plane concrete regions from FE stresses (EN 1992-1-1 Annex F, 9.6, 9.7, 8.4): a port of
// tragwerk.fe2d_design (Python), checked against it in the test suite. Needs fe2d.js.
(function (root) {
  "use strict";
  const T = root.Tragwerk || (root.Tragwerk = {});
  const F = T.FE;
  const BAR_DIAS = [8, 10, 12, 14, 16, 20, 25], SPACINGS = [300, 250, 200, 150, 125, 100];

  // (f'_tdx, f'_tdy, σ_cd, case) in MPa from stresses in MPa with tension positive
  function annexF(sx, sy, txy) {
    let cx = -sx, cy = -sy;
    const t = Math.abs(txy), swap = cx < cy;
    if (swap) [cx, cy] = [cy, cx];
    let ftx, fty, scd, kase;
    if (cx > 0 && cy > 0 && cx * cy > t * t) { ftx = fty = 0; scd = 0.5 * (cx + cy) + Math.sqrt(0.25 * (cx - cy) ** 2 + t * t); kase = "biaxial compression (F.1(3))"; }
    else if (cx <= t) { ftx = t - cx; fty = t - cy; scd = 2 * t; kase = "F.2–F.4"; }
    else { ftx = 0; fty = t * t / cx - cy; scd = cx * (1 + (t / cx) ** 2); kase = "F.5–F.7"; }
    if (swap) [ftx, fty] = [fty, ftx];
    return [ftx, fty, scd, kase];
  }
  function elementAverageStress(m, u) {
    const et = m.etype, C = m.C, ne = m.elements.length, S = m.elements.map(() => [0, 0, 0]), A = new Float64Array(ne);
    m.elements.forEach((e, q) => {
      const xy = e.map((i) => m.nodes[i]), ue = [];
      e.forEach((i) => { ue.push(u[2 * i], u[2 * i + 1]); });
      for (const [xi, eta, w] of F.gauss(et)) {
        const g = F.gaussPoint(et, xy, xi, eta, w), a = w * g.detJ;
        const eps = [0, 1, 2].map((r) => g.B[r].reduce((s, b, c) => s + b * ue[c], 0));
        for (let r = 0; r < 3; r++) S[q][r] += a * (C[r][0] * eps[0] + C[r][1] * eps[1] + C[r][2] * eps[2]) / 1000;
        A[q] += a;
      }
      for (let r = 0; r < 3; r++) S[q][r] /= A[q];
    });
    return { S, A };
  }
  function patchAverage(cent, S, A, r) {
    if (r <= 0) return S.map((v) => v.slice());
    return cent.map((cq, q) => {
      let w = 0;
      const v = [0, 0, 0];
      cent.forEach((c, j) => {
        if (j !== q && (c[0] - cq[0]) ** 2 + (c[1] - cq[1]) ** 2 > r * r) return;
        w += A[j];
        for (let k = 0; k < 3; k++) v[k] += S[j][k] * A[j];
      });
      return v.map((x) => x / w);
    });
  }
  // [As_min,x, As_min,y] in mm²/m (both faces) and notes
  function minimumReinforcement(member, t) {
    const Ac = t * 1e6;
    if (member === "wall") {
      const v = 0.002 * Ac, h = Math.max(0.25 * v, 0.001 * Ac);
      return [h, v, [`vertical A_s,vmin = 0.002·A_c = ${v.toFixed(0)} mm²/m (9.6.2(1))`, `horizontal A_s,hmin = max(25 % of vertical, 0.001·A_c) = ${h.toFixed(0)} mm²/m (9.6.3(1))`]];
    }
    if (member === "deep_beam") {
      const face = Math.max(0.001 * Ac, 150);
      return [2 * face, 2 * face, [`A_s,dbmin = max(0.1 % of A_c, 150 mm²/m) in each face and direction = ${face.toFixed(0)} mm²/m per face (9.7(1))`]];
    }
    return [0, 0, ["no minimum reinforcement (member type none)"]];
  }
  const barArea = (d, s) => 2 * Math.PI * d * d / 4 * 1000 / s;
  function barOptions() {
    const o = [];
    for (const d of BAR_DIAS) for (const s of SPACINGS) o.push([barArea(d, s), d, s]);
    return o.sort((a, b) => (Math.round(a[0] * 1e6) - Math.round(b[0] * 1e6)) || (b[2] - a[2]));
  }
  function pickBars(As) { if (As <= 0) return null; for (const o of barOptions()) if (o[0] >= As - 1e-9) return o; return null; }

  function designReinforcement(m, rAvg = null) {
    const d = m.design, fck = d.fck ?? 30, fyd = m.reinforcement.fyk / d.gamma_s, fcd = d.alpha_cc * fck / d.gamma_c, nu = 0.6 * (1 - fck / 250);
    const res = F.solveFE(m, null, "uls"), u = new Float64Array(2 * m.nodes.length);
    res.u.forEach((v, i) => { u[2 * i] = v[0]; u[2 * i + 1] = v[1]; });
    const { S: S0, A } = elementAverageStress(m, u), cent = F.elementCentroids(m), r = rAvg == null ? m.t : Number(rAvg);
    const S = patchAverage(cent, S0, A, r), ne = m.elements.length;
    const ftd = [], scd = new Float64Array(ne), cases = [];
    S.forEach((s, q) => { const [fx, fy, sc, k] = annexF(s[0], s[1], s[2]); ftd.push([fx, fy]); scd[q] = sc; cases.push(k); });
    const AsF = ftd.map(([a, b]) => [a * m.t * 1e6 / fyd, b * m.t * 1e6 / fyd]);
    const [minx, miny, notes] = minimumReinforcement(d.member, m.t);
    const minxArr = AsF.map((a) => (d.member === "wall" ? Math.max(0.25 * Math.max(a[1], miny), 0.001 * m.t * 1e6) : minx));
    const AsReq = AsF.map((a, q) => [Math.max(a[0], minxArr[q]), Math.max(a[1], miny)]);
    const re = F.elementReinforcement(m), prov = AsReq.map((_, q) => [re.Asx[q], re.Asy[q]]);
    const util = AsReq.map((a, q) => a.map((v, j) => (prov[q][j] > 0 ? v / prov[q][j] : (v > 0 ? Infinity : 0))));
    const utilC = Array.from(scd, (v, q) => (cases[q].startsWith("biaxial") ? v / fcd : v / (nu * fcd)));
    return { r_avg: r, fyd, fcd, nu, stress: S, stress_element: S0, ftd, scd, cases, As_annex_f: AsF, As_min: [minx, miny], As_min_x: minxArr, As_req: AsReq,
      As_max: 0.04 * m.t * 1e6, As_prov: prov, util_steel: util, util_concrete: utilC, notes, result: res };
  }
  function barsFor(As, dia) { for (const sp of SPACINGS) if (barArea(dia, sp) >= As - 1e-9) return sp; return SPACINGS[SPACINGS.length - 1]; }
  // l_bd = (φ/4)(f_yd/f_bd), f_bd = 2.25 · 0.7 · f_ctd (poor bond, 8.4.2), at least 10φ and 100 mm; in m
  function anchorageLength(dia, m) {
    const d = m.design, fctd = F.concreteProps(d.fck ?? 30).fctk005 / d.gamma_c, fbd = 2.25 * 0.7 * 1.0 * fctd, fyd = m.reinforcement.fyk / d.gamma_s;
    return Math.max(dia / 4 * fyd / fbd, 10 * dia, 100) / 1000;
  }
  const r6 = (v) => Math.round(v * 1e6) / 1e6;
  function proposeLayers(m, des) {
    const layers = [], warns = [], nc = m.etype[0] === "T" ? 3 : 4;
    const ext = m.elements.map((e) => { const p = e.slice(0, nc).map((i) => m.nodes[i]); return [Math.min(...p.map((v) => v[0])), Math.min(...p.map((v) => v[1])), Math.max(...p.map((v) => v[0])), Math.max(...p.map((v) => v[1]))]; });
    ["x", "y"].forEach((dname, j) => {
      const req = des.As_req.map((a) => a[j]);
      const baseNeed = j === 0 ? Math.max(...des.As_min_x) : des.As_min[1];
      const base = baseNeed > 0 ? pickBars(baseNeed) : null, baseArea = base ? base[0] : 0;
      if (base) layers.push({ dir: dname, dia_mm: base[1], spacing_mm: base[2], faces: 2, label: `base mesh, ${dname} bars` });
      const need = req.map((v) => v - baseArea), top = Math.max(...need);
      if (top <= 1e-6) return;
      let dia = BAR_DIAS.find((dd) => barArea(dd, SPACINGS[SPACINGS.length - 1]) >= top) ?? BAR_DIAS[BAR_DIAS.length - 1];
      if (barArea(dia, SPACINGS[SPACINGS.length - 1]) < top)
        warns.push(`${dname}: ${(top + baseArea).toFixed(0)} mm²/m exceeds Ø${dia} @ ${SPACINGS[SPACINGS.length - 1]} mm in both faces plus the base mesh; use a thicker wall or concentrated ties`);
      const across = j === 0 ? [1, 3] : [0, 2], along = j === 0 ? [0, 2] : [1, 3];
      const rows = new Map();
      ext.forEach((x, q) => { const key = `${r6(x[across[0]])}|${r6(x[across[1]])}`; if (!rows.has(key)) rows.set(key, [r6(x[across[0]]), r6(x[across[1]]), []]); rows.get(key)[2].push(q); });
      const keys = [...rows.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const strips = [];
      for (const [a0, a1, qsAll] of keys) {
        const qs = qsAll.filter((q) => need[q] > 1e-6);
        if (!qs.length) continue;
        strips.push([a0, a1, Math.min(...qs.map((q) => ext[q][along[0]])), Math.max(...qs.map((q) => ext[q][along[1]])), barsFor(Math.max(...qs.map((q) => need[q])), dia)]);
      }
      const merged = [];
      for (const st of strips) {
        const mm = merged[merged.length - 1];
        if (mm && mm[4] === st[4] && Math.abs(mm[1] - st[0]) < 1e-6) { mm[1] = st[1]; mm[2] = Math.min(mm[2], st[2]); mm[3] = Math.max(mm[3], st[3]); }
        else merged.push(st.slice());
      }
      const lbd = anchorageLength(dia, m), coord = m.nodes.map((p) => p[along[0]]), lo0 = Math.min(...coord), hi0 = Math.max(...coord);
      let short = 0;
      for (const [a0, a1, lo, hi, sp] of merged) {
        const lo2 = Math.max(lo - lbd, lo0), hi2 = Math.min(hi + lbd, hi0);
        if (lo - lbd < lo0 - 1e-9 || hi + lbd > hi0 + 1e-9) short++;
        const reg = j === 0 ? { x0: lo2, x1: hi2, y0: a0, y1: a1 } : { x0: a0, x1: a1, y0: lo2, y1: hi2 };
        layers.push({ dir: dname, dia_mm: dia, spacing_mm: sp, faces: 2, label: `additional ${dname} bars`, region: { x0: r6(reg.x0), x1: r6(reg.x1), y0: r6(reg.y0), y1: r6(reg.y1) } });
      }
      if (merged.length) warns.push(`additional ${dname} bars Ø${dia}: extended by l_bd = ${(lbd * 1000).toFixed(0)} mm beyond their zones (EN 1992-1-1 8.4.3, poor bond, σ_sd = f_yd)`
        + (short ? `; ${short} zone(s) reach the edge with less than l_bd: anchor them there with U-bars, loops or plates (8.4, 9.7(2))` : ""));
    });
    return { layers, warns };
  }

  T.FEDesign = { BAR_DIAS, SPACINGS, annexF, elementAverageStress, patchAverage, minimumReinforcement, barOptions, pickBars, designReinforcement, proposeLayers, anchorageLength };
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof window !== "undefined" ? window : globalThis);
