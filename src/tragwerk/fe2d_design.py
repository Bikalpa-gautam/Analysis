"""Reinforcement design of plane concrete regions (walls, deep beams) from FE stresses.

EN 1992-1-1 Annex F, "Tension reinforcement expressions for in-plane stress
conditions": for design membrane stresses σ_Edx, σ_Edy, τ_Edxy (compression
positive, σ_Edx ≥ σ_Edy) the tensile strengths the reinforcement must provide are

    σ_Edx ≤ |τ_Edxy|:  f'_tdx = |τ| − σ_Edx,  f'_tdy = |τ| − σ_Edy,  σ_cd = 2|τ|          (F.2 – F.4)
    σ_Edx > |τ_Edxy|:  f'_tdx = 0,  f'_tdy = τ²/σ_Edx − σ_Edy,  σ_cd = σ_Edx (1 + (τ/σ_Edx)²)  (F.5 – F.7)

with no reinforcement needed where both stresses are compressive and
σ_Edx·σ_Edy > τ². The reinforcement is A_s = f'_td · t / f_yd per unit length,
and the concrete stress must satisfy σ_cd ≤ ν f_cd, ν = 0.6 (1 − f_ck/250).

Stresses are the linear-elastic ULS stresses averaged over a design patch of
radius r_avg around each element (default: the wall thickness), so that the
requirement does not depend on the mesh near singular points. Minimum
reinforcement follows 9.6 (walls) or 9.7 (deep beams).
"""
from __future__ import annotations

import math

import numpy as np

from .fe2d import FEError, FEModel, element_reinforcement, gauss, gauss_point, solve_fe

BAR_DIAS = (8, 10, 12, 14, 16, 20, 25)
SPACINGS = (300, 250, 200, 150, 125, 100)


def annex_f(sx: float, sy: float, txy: float) -> tuple[float, float, float, str]:
    """(f'_tdx, f'_tdy, σ_cd, case) in MPa from stresses in MPa with tension positive."""
    cx, cy, t = -sx, -sy, abs(txy)
    swap = cx < cy
    if swap:
        cx, cy = cy, cx
    if cx > 0 and cy > 0 and cx * cy > t * t:
        c = 0.5 * (cx + cy) + math.sqrt(0.25 * (cx - cy) ** 2 + t * t)
        ftx = fty = 0.0
        scd, case = c, "biaxial compression (F.1(3))"
    elif cx <= t:
        ftx, fty, scd, case = t - cx, t - cy, 2 * t, "F.2–F.4"
    else:
        ftx, fty, scd, case = 0.0, t * t / cx - cy, cx * (1 + (t / cx) ** 2), "F.5–F.7"
    if swap:
        ftx, fty = fty, ftx
    return ftx, fty, scd, case


def element_average_stress(m: FEModel, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted mean stress (MPa) and area of every element."""
    et, C = m.etype, m.C
    ne, nn = m.elements.shape
    S, A = np.zeros((ne, 3)), np.zeros(ne)
    uf = np.asarray(u).reshape(-1)
    for q, e in enumerate(m.elements):
        xy = m.nodes[e]
        dofs = np.empty(2 * nn, dtype=int)
        dofs[0::2], dofs[1::2] = 2 * e, 2 * e + 1
        ue = uf[dofs]
        for xi, eta, w in gauss(et):
            g = gauss_point(et, xy, xi, eta, w)
            a = w * g.detJ
            S[q] += a * (C @ (g.B @ ue)) / 1000.0
            A[q] += a
        S[q] /= A[q]
    return S, A


def patch_average(cent: np.ndarray, S: np.ndarray, A: np.ndarray, r: float) -> np.ndarray:
    if r <= 0:
        return S.copy()
    out = np.zeros_like(S)
    for q in range(len(cent)):
        d2 = (cent[:, 0] - cent[q, 0]) ** 2 + (cent[:, 1] - cent[q, 1]) ** 2
        sel = d2 <= r * r
        sel[q] = True
        w = A[sel]
        out[q] = (S[sel] * w[:, None]).sum(axis=0) / w.sum()
    return out


def minimum_reinforcement(member: str, t: float, As_req_y: np.ndarray | None = None) -> tuple[float, float, list[str]]:
    """(As_min,x, As_min,y) in mm²/m, both faces together, and explanatory notes."""
    Ac = t * 1e6                                   # mm² per m of wall
    if member == "wall":
        v = 0.002 * Ac
        hmin = max(0.25 * v, 0.001 * Ac)
        return hmin, v, [f"vertical A_s,vmin = 0.002·A_c = {v:.0f} mm²/m (9.6.2(1))",
                         f"horizontal A_s,hmin = max(25 % of vertical, 0.001·A_c) = {hmin:.0f} mm²/m (9.6.3(1))"]
    if member == "deep_beam":
        face = max(0.001 * Ac, 150.0)
        return 2 * face, 2 * face, [f"A_s,dbmin = max(0.1 % of A_c, 150 mm²/m) in each face and direction = {face:.0f} mm²/m per face (9.7(1))"]
    return 0.0, 0.0, ["no minimum reinforcement (member type none)"]


def bar_options() -> list[tuple[float, int, int]]:
    """(area mm²/m both faces, Ø, spacing) sorted by area, then wider spacing first."""
    opts = [(2 * math.pi * d * d / 4 * 1000.0 / s, d, s) for d in BAR_DIAS for s in SPACINGS]
    return sorted(opts, key=lambda o: (round(o[0], 6), -o[2]))


def pick_bars(As: float) -> tuple[float, int, int] | None:
    if As <= 0:
        return None
    for o in bar_options():
        if o[0] >= As - 1e-9:
            return o
    return None


def design_reinforcement(m: FEModel, r_avg: float | None = None) -> dict:
    """Design of the reinforcement for the ULS combination of the model; returns per-element arrays."""
    d = m.design
    fck = d.get("fck", 30.0)
    fyd = m.reinforcement["fyk"] / d["gamma_s"]
    fcd = d["alpha_cc"] * fck / d["gamma_c"]
    nu = 0.6 * (1 - fck / 250.0)
    res = solve_fe(m, combo="uls")
    S0, A = element_average_stress(m, res.u)
    cent = np.array([m.nodes[e].mean(axis=0) for e in m.elements])
    r = m.t if r_avg is None else float(r_avg)
    S = patch_average(cent, S0, A, r)
    ne = len(m.elements)
    ftd = np.zeros((ne, 2))
    scd = np.zeros(ne)
    cases = []
    for q in range(ne):
        fx, fy, sc, case = annex_f(*S[q])
        ftd[q] = (fx, fy)
        scd[q] = sc
        cases.append(case)
    As_F = ftd * m.t * 1e6 / fyd                   # mm²/m, both faces
    minx, miny, notes = minimum_reinforcement(d["member"], m.t)
    if d["member"] == "wall":                      # horizontal ≥ 25 % of the vertical reinforcement
        minx_arr = np.maximum(0.25 * np.maximum(As_F[:, 1], miny), 0.001 * m.t * 1e6)
    else:
        minx_arr = np.full(ne, minx)
    As_req = np.column_stack([np.maximum(As_F[:, 0], minx_arr), np.maximum(As_F[:, 1], miny)])
    As_max = 0.04 * m.t * 1e6
    Asx, Asy, _, _ = element_reinforcement(m)
    prov = np.column_stack([Asx, Asy])
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(prov > 0, As_req / prov, np.where(As_req > 0, np.inf, 0.0))
    util_c = scd / (nu * fcd)
    biax = np.array([c.startswith("biaxial") for c in cases])
    util_c[biax] = scd[biax] / fcd                 # both compressive: f_cd (F.1(3))
    return {"r_avg": r, "fyd": fyd, "fcd": fcd, "nu": nu, "stress": S, "stress_element": S0, "ftd": ftd, "scd": scd, "cases": cases,
            "As_annex_f": As_F, "As_min": (minx, miny), "As_min_x": minx_arr, "As_req": As_req, "As_max": As_max, "As_prov": prov,
            "util_steel": util, "util_concrete": util_c, "notes": notes, "result": res}


def _bars_for(As: float, dia: int) -> int:
    """Widest spacing from the table giving at least As (mm²/m, both faces) with bars of diameter dia."""
    for sp in SPACINGS:
        if 2 * math.pi * dia * dia / 4 * 1000.0 / sp >= As - 1e-9:
            return sp
    return SPACINGS[-1]


def propose_layers(m: FEModel, des: dict) -> tuple[list[dict], list[str]]:
    """A practical bar layout: a base mesh for the minimum reinforcement in each direction, and additional
    bars of one diameter per direction in strips along the bar direction (rows for x bars, columns for y bars)
    where the design asks for more. Strips of neighbouring rows with the same spacing are merged."""
    layers, warns = [], []
    corner = m.elements[:, :3] if m.etype[0] == "T" else m.elements[:, :4]
    ext = np.array([[m.nodes[c][:, 0].min(), m.nodes[c][:, 1].min(), m.nodes[c][:, 0].max(), m.nodes[c][:, 1].max()] for c in corner])
    for j, dname in enumerate(("x", "y")):
        req = des["As_req"][:, j]
        base_need = float(des["As_min_x"].max()) if j == 0 else float(des["As_min"][1])
        base = pick_bars(base_need) if base_need > 0 else None
        base_area = base[0] if base else 0.0
        if base:
            layers.append({"dir": dname, "dia_mm": base[1], "spacing_mm": base[2], "faces": 2, "label": f"base mesh, {dname} bars"})
        need = req - base_area
        if need.max() <= 1e-6:
            continue
        top = float(need.max())
        dia = next((d for d in BAR_DIAS if 2 * math.pi * d * d / 4 * 1000.0 / SPACINGS[-1] >= top), BAR_DIAS[-1])
        if 2 * math.pi * dia * dia / 4 * 1000.0 / SPACINGS[-1] < top:
            warns.append(f"{dname}: {top + base_area:.0f} mm²/m exceeds Ø{dia} @ {SPACINGS[-1]} mm in both faces plus the base mesh; "
                         "use a thicker wall or concentrated ties")
        across = (1, 3) if j == 0 else (0, 2)            # rows for x bars, columns for y bars
        along = (0, 2) if j == 0 else (1, 3)
        rows: dict[tuple[float, float], list[int]] = {}
        for q in range(len(m.elements)):
            rows.setdefault((round(ext[q, across[0]], 6), round(ext[q, across[1]], 6)), []).append(q)
        strips = []
        for key in sorted(rows):
            qs = [q for q in rows[key] if need[q] > 1e-6]
            if not qs:
                continue
            lo, hi = min(ext[q, along[0]] for q in qs), max(ext[q, along[1]] for q in qs)
            strips.append([key[0], key[1], lo, hi, _bars_for(max(need[q] for q in qs), dia)])
        merged = []
        for st in strips:
            if merged and merged[-1][4] == st[4] and abs(merged[-1][1] - st[0]) < 1e-6:
                mm = merged[-1]
                mm[1], mm[2], mm[3] = st[1], min(mm[2], st[2]), max(mm[3], st[3])
            else:
                merged.append(list(st))
        lbd = anchorage_length(dia, m)
        edge_lo, edge_hi = float(m.nodes[:, along[0]].min()), float(m.nodes[:, along[0]].max())
        short = 0
        for a0, a1, lo, hi, sp in merged:
            # the bars run on by the anchorage length beyond the zone where they are needed (8.4.3), up to the edge
            lo2, hi2 = max(lo - lbd, edge_lo), min(hi + lbd, edge_hi)
            if lo - lbd < edge_lo - 1e-9 or hi + lbd > edge_hi + 1e-9:
                short += 1
            reg = {"x0": lo2, "x1": hi2, "y0": a0, "y1": a1} if j == 0 else {"x0": a0, "x1": a1, "y0": lo2, "y1": hi2}
            layers.append({"dir": dname, "dia_mm": dia, "spacing_mm": sp, "faces": 2, "label": f"additional {dname} bars",
                           "region": {k: round(float(v), 6) for k, v in reg.items()}})
        if merged:
            warns.append(f"additional {dname} bars Ø{dia}: extended by l_bd = {lbd * 1000:.0f} mm beyond their zones (EN 1992-1-1 8.4.3, poor bond, σ_sd = f_yd)"
                         + (f"; {short} zone(s) reach the edge with less than l_bd: anchor them there with U-bars, loops or plates (8.4, 9.7(2))" if short else ""))
    return layers, warns


def anchorage_length(dia_mm: float, m: FEModel) -> float:
    """Design anchorage length l_bd = (φ/4)(σ_sd/f_bd) in m (EN 1992-1-1 8.4.3) with σ_sd = f_yd and the bond strength
    f_bd = 2.25 η₁ η₂ f_ctd (8.4.2), η₁ = 0.7 (poor bond conditions, on the safe side), η₂ = 1 (φ ≤ 32 mm)."""
    from .fe2d import concrete_props
    d = m.design
    fctd = concrete_props(d.get("fck", 30.0))["fctk005"] / d["gamma_c"]
    fbd = 2.25 * 0.7 * 1.0 * fctd
    fyd = m.reinforcement["fyk"] / d["gamma_s"]
    lb = dia_mm / 4 * fyd / fbd
    return max(lb, 10 * dia_mm, 100.0) / 1000.0


def to_dict_design(m: FEModel, des: dict) -> dict:
    return {"r_avg_m": des["r_avg"], "fyd_MPa": des["fyd"], "fcd_MPa": des["fcd"], "nu": des["nu"],
            "As_req_mm2_per_m": des["As_req"].tolist(), "As_prov_mm2_per_m": des["As_prov"].tolist(),
            "util_steel": [[float(v) if math.isfinite(v) else None for v in row] for row in des["util_steel"]],
            "util_concrete": des["util_concrete"].tolist(), "notes": des["notes"],
            "summary": {"max_As_req_x": float(des["As_req"][:, 0].max()), "max_As_req_y": float(des["As_req"][:, 1].max()),
                        "max_util_concrete": float(des["util_concrete"].max()),
                        "max_util_steel": (lambda v: float(v) if math.isfinite(v) else None)(float(np.max(des["util_steel"])))}}


__all__ = ["annex_f", "design_reinforcement", "propose_layers", "anchorage_length", "pick_bars", "bar_options", "minimum_reinforcement", "to_dict_design", "FEError"]
