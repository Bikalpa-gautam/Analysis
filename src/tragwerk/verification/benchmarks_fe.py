"""Benchmarks for the plane continuum elements (tragwerk.fe2d)."""
from __future__ import annotations

import math

import numpy as np

from ..fe2d import (SCHEMA_FE, constitutive, element_stiffness, kirsch, parse_fe, run_fe, solve_fe,
                    timoshenko_cantilever)
from .benchmarks import Comparison, benchmark

_PATCH = np.array([[0, 0], [0.24, 0], [0.24, 0.12], [0, 0.12], [0.04, 0.02], [0.18, 0.03], [0.16, 0.08], [0.08, 0.08]])
_PATCH_Q4 = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]


def _patch_mesh(et):
    base = _PATCH_Q4 if et[0] == "Q" else [[e[0], e[1], e[2]] for e in _PATCH_Q4] + [[e[0], e[2], e[3]] for e in _PATCH_Q4]
    if et in ("Q4", "T3"):
        return _PATCH.tolist(), base
    nodes, mids, els = [list(p) for p in _PATCH], {}, []
    for e in base:
        extra = []
        for a, b in zip(e, e[1:] + e[:1]):
            key = tuple(sorted((a, b)))
            if key not in mids:
                mids[key] = len(nodes)
                nodes.append(list((_PATCH[a] + _PATCH[b]) / 2))
            extra.append(mids[key])
        els.append(list(e) + extra)
    return nodes, els


def _patch(et):
    nodes, els = _patch_mesh(et)
    m = parse_fe({"schema": SCHEMA_FE, "element": et, "material": {"E_MPa": 1000, "nu": 0.25, "thickness_m": 0.001},
                  "geometry": {"type": "mesh", "nodes": nodes, "elements": els}, "supports": [{"at": [0, 0], "ux": True}]})
    lin = lambda x, y: (1e-3 * (1 + 2 * x + 3 * y), 1e-3 * (-1 + 4 * x - 2 * y))   # noqa: E731
    lo, hi = m.nodes.min(0), m.nodes.max(0)
    pres = {}
    for i, (x, y) in enumerate(m.nodes):
        if min(abs(x - lo[0]), abs(x - hi[0]), abs(y - lo[1]), abs(y - hi[1])) < 1e-12:
            pres[2 * i], pres[2 * i + 1] = lin(x, y)
    m.fixed = {}
    r = solve_fe(m, pres)
    du = max(abs(r.u[i, 0] - lin(*m.nodes[i])[0]) + abs(r.u[i, 1] - lin(*m.nodes[i])[1]) for i in range(len(m.nodes)))
    sig = m.C @ np.array([2e-3, -2e-3, 7e-3]) / 1000
    ds = max(abs(g[2 + c] - sig[c]) for gl in r.gp for g in gl for c in range(3))
    return du, ds, sig


@benchmark("F01", "Continuum FE", "Patch test: T3, T6, Q4 and Q8 on a distorted patch reproduce a linear displacement field",
           "MacNeal & Harder (1985) patch; Irons patch test; MFE I (Chatzi), convergence requirements")
def _f01():
    out = []
    for et in ("T3", "T6", "Q4", "Q8"):
        du, ds, sig = _patch(et)
        out.append(Comparison(f"{et}: interior displacement error", du, 0.0, "m", abs_tol=1e-15))
        out.append(Comparison(f"{et}: stress error at every Gauss point", ds, 0.0, "MPa", abs_tol=1e-12 * abs(sig).max()))
    return out


@benchmark("F02", "Continuum FE", "Element stiffness: CST in closed form, rigid-body modes of Q4 and Q8",
           "k = t A Bᵀ C B with b_i = y_j − y_k, c_i = x_k − x_j (Zienkiewicz & Taylor); 2n − 3 non-zero eigenvalues")
def _f02():
    X = np.array([[0.0, 0.0], [3.0, 0.5], [1.0, 2.0]])
    C = constitutive(30e6, 0.25, "plane_strain")
    A = 0.5 * ((X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1]) - (X[2, 0] - X[0, 0]) * (X[1, 1] - X[0, 1]))
    B = np.zeros((3, 6))
    for i in range(3):
        j, k_ = (i + 1) % 3, (i + 2) % 3
        b, c = X[j, 1] - X[k_, 1], X[k_, 0] - X[j, 0]
        B[:, 2 * i:2 * i + 2] = np.array([[b, 0], [0, c], [c, b]]) / (2 * A)
    k, _ = element_stiffness("T3", X, C, 0.2)
    ref = 0.2 * A * B.T @ C @ B
    out = [Comparison("T3: largest deviation from t·A·BᵀCB", float(np.max(np.abs(k - ref))), 0.0, "kN/m", abs_tol=1e-6)]
    for et, X in (("Q4", np.array([[0, 0], [2, 0.2], [2.2, 1.4], [-0.1, 1.2]])),
                  ("Q8", np.array([[0, 0], [2, 0.2], [2.2, 1.4], [-0.1, 1.2], [1, 0.1], [2.1, 0.8], [1.05, 1.3], [-0.05, 0.6]]))):
        k, _ = element_stiffness(et, X, constitutive(30e6, 0.2, "plane_stress"), 0.3)
        lam = np.linalg.eigvalsh(k)
        out.append(Comparison(f"{et}: number of zero eigenvalues (rigid-body modes)", float(np.sum(np.abs(lam) < 1e-9 * lam.max())), 3.0, ""))
    return out


def _cantilever(et, nx, P=100.0, L=8.0, D=1.0, E=3e7, nu=0.25, t=0.3):
    d = {"schema": SCHEMA_FE, "element": et, "material": {"E_MPa": E / 1000, "nu": nu, "thickness_m": t},
         "geometry": {"type": "rectangle", "Lx": L, "Ly": D, "h": L / nx}, "supports": [{"edge": "left", "ux": True, "uy": True}],
         "loads": [{"edge": "right", "profile": "parabolic", "P": P, "dir": "y"}]}
    m = parse_fe(d)
    m.nodes[:, 1] -= D / 2
    pres = {}
    for i, (x, y) in enumerate(m.nodes):
        if abs(x) < 1e-12:
            ux, uy, *_ = timoshenko_cantilever(x, y, P, L, D, E, nu, t)
            pres[2 * i], pres[2 * i + 1] = ux, uy
    m.fixed = {}
    r = solve_fe(m, pres)
    k = int(np.argmin(np.hypot(m.nodes[:, 0] - L, m.nodes[:, 1])))
    return r, float(r.u[k, 1]), timoshenko_cantilever(L, 0.0, P, L, D, E, nu, t)[1]


@benchmark("F03", "Continuum FE", "Cantilever under end shear: Q8 and T6 against the exact plane stress solution",
           "Timoshenko & Goodier, Theory of Elasticity, §21; exact displacements imposed on the fixed end (Augarde & Deeks 2008)")
def _f03():
    out = []
    for et in ("Q8", "T6"):
        r, tip, ex = _cantilever(et, 32)
        out.append(Comparison(f"{et}, 32 × 4 cells: tip deflection", tip * 1000, ex * 1000, "mm", rel_tol=2e-5))
        g = min((g for gl in r.gp for g in gl), key=lambda g: math.hypot(g[0] - 2.0, g[1] - 0.2))
        x, y, sxx, _, txy = g
        ref = timoshenko_cantilever(x, y, 100.0, 8.0, 1.0, 3e7, 0.25, 0.3)
        out.append(Comparison(f"{et}: σ_xx at the Gauss point nearest (L/4, D/5)", sxx, ref[2] / 1000, "MPa", rel_tol=1e-3))
        m = r.model
        k = int(np.argmin(np.hypot(m.nodes[:, 0] - 2.0, m.nodes[:, 1])))
        out.append(Comparison(f"{et}: τ_xy,max at mid-depth, nodal average (4 elements through the depth: the parabola "
                              "is approximated piecewise linearly)", float(r.nodal[k, 2]), 1.5 * 100.0 / (0.3 * 1.0) / 1000, "MPa", rel_tol=0.05))
    return out


@benchmark("F04", "Continuum FE", "Convergence under mesh refinement: Q4 and T3 at rate h², Q8 and T6 faster",
           "A priori error estimates, |e| ∝ h^(p+1) for displacements (Strang & Fix); MFE I")
def _f04():
    out = []
    for et, lo, hi in (("Q4", 1.7, 2.3), ("T3", 1.4, 2.3), ("Q8", 3.0, 5.0), ("T6", 3.0, 5.0)):
        errs = []
        for nx in (16, 32):
            _, tip, ex = _cantilever(et, nx)
            errs.append(abs(tip / ex - 1))
        rate = math.log2(errs[0] / errs[1])
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        out.append(Comparison(f"{et}: observed rate of the tip deflection error (h = L/16 → L/32)", rate, mid, "", rel_tol=half / mid))
    return out


@benchmark("F05", "Continuum FE", "Plate with a circular hole under uniaxial tension (Kirsch)",
           "Kirsch (1898): σ_θθ = 3σ₀ at θ = 90°, −σ₀ at θ = 0; plate 80a × 80a, quarter model, Q8 isoparametric mesh")
def _f05():
    a, W, s0 = 1.0, 40.0, 1.0
    r = run_fe({"schema": SCHEMA_FE, "element": "Q8", "material": {"E_MPa": 200000, "nu": 0.3, "thickness_m": 1.0},
                "geometry": {"type": "plate_hole", "W": W, "a": a, "n_theta": 24, "n_r": 24, "grading": 2.0},
                "supports": [{"edge": "left", "ux": True}, {"edge": "bottom", "uy": True}], "loads": [{"edge": "right", "qx": s0 * 1000}]})
    m = r.model
    at = lambda x, y: int(np.argmin(np.hypot(m.nodes[:, 0] - x, m.nodes[:, 1] - y)))   # noqa: E731
    k1, k2 = at(0, a), at(a, 0)
    k3 = at(0, 2 * a)
    _, stt, _ = kirsch(math.hypot(*m.nodes[k3]), math.atan2(m.nodes[k3][1], m.nodes[k3][0]), a, s0)
    return [Comparison("σ_xx at (0, a), stress concentration", float(r.nodal[k1, 0]), 3 * s0, "MPa", rel_tol=0.01),
            Comparison("σ_yy at (a, 0)", float(r.nodal[k2, 1]), -s0, "MPa", rel_tol=0.01),
            Comparison("σ_xx at (0, 2a)", float(r.nodal[k3, 0]), stt, "MPa", rel_tol=0.01),
            Comparison("Σ reactions in x + applied load", float(r.reactions[:, 0].sum() + r.f[0::2].sum()), 0.0, "kN", abs_tol=1e-8)]


@benchmark("F06", "Continuum FE", "Global equilibrium of a wall with a door opening under self weight and edge loads",
           "Σ R + Σ F = 0 for every element type; self weight γ t A")
def _f06():
    out = []
    for et in ("T3", "T6", "Q4", "Q8"):
        r = run_fe({"schema": SCHEMA_FE, "element": et, "material": {"E_MPa": 30000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25},
                    "geometry": {"type": "rectangle", "Lx": 6, "Ly": 3, "h": 0.25, "openings": [{"x0": 2, "y0": 0, "x1": 3, "y1": 2.2}]},
                    "supports": [{"edge": "bottom", "ux": True, "uy": True}], "loads": [{"edge": "top", "qx": 20, "qy": -80}], "self_weight": True})
        out.append(Comparison(f"{et}: Σ applied vertical load", float(r.f[1::2].sum()), -80 * 6 - 25 * 0.25 * (18 - 2.2), "kN", rel_tol=1e-12))
        out.append(Comparison(f"{et}: Σ R_y + Σ F_y", float(r.reactions[:, 1].sum() + r.f[1::2].sum()), 0.0, "kN", abs_tol=1e-8))
    return out
