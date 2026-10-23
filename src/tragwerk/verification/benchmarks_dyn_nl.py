"""Benchmarks for time-history analysis (tragwerk.dynamics_nl): ground motions, response spectra, the Bouc–Wen
oscillator and Newmark integration of FE models. References are closed-form solutions, SciPy's exact discretisation of
linear systems (scipy.signal.lsim, independent of the code under test) and modal superposition."""
from __future__ import annotations

import math

import numpy as np

from ..dynamics_nl import (G_ACC, LCG, BWParams, DynOptions, Record, assemble_mass, bouc_wen_sdof, bw_advance, elastic_spectrum, fe_dynamic,
                           response_spectrum, sdof_history, synthetic_accelerogram)
from ..fe2d import SCHEMA_FE, parse_fe
from ..fe2d_nl import NLOptions, Problem
from .benchmarks import Comparison, benchmark

CAT = "Time-history analysis"


@benchmark("N01", CAT, "Linear SDOF response (Nigam–Jennings recurrence) against the exact solution",
           "scipy.signal.lsim (matrix exponential, linear interpolation of the input); resonant steady state u = (A/ω²)/(2ξ) "
           "(Chopra, Dynamics of Structures, 3.2)")
def _n01():
    from scipy import signal
    out = []
    rec = synthetic_accelerogram(0.3, "C", 1, 12.0, 0.01, seed=5, rise=1.5, strong=6.0)
    for T in (0.1, 0.5, 2.0):
        w, xi = 2 * math.pi / T, 0.05
        sysm = signal.lti([[0, 1], [-w * w, -2 * xi * w]], [[0], [-1]], [[1, 0]], [[0]])
        _, y, _ = signal.lsim(sysm, rec.a, rec.t, interp=True)
        out.append(Comparison(f"S_d(T = {T:g} s), 5 %", float(response_spectrum(rec.a, rec.dt, [T])["Sd"][0]), float(np.abs(y).max()), "m", rel_tol=1e-9))
    # resonance: a_g = A sin(ωt), steady-state amplitude (A/ω²)/(2ξ)
    T, xi, A, dt = 0.5, 0.05, 1.0, 0.0005
    t = np.arange(0, 40.0 + dt / 2, dt)
    u, _ = sdof_history(A * np.sin(2 * math.pi / T * t), dt, T, xi)
    last = u[t > 40.0 - T]
    out.append(Comparison("resonant steady-state amplitude", float(np.abs(last).max()), A / (2 * math.pi / T) ** 2 / (2 * xi), "m", rel_tol=2e-4))
    return out


@benchmark("N02", CAT, "Newmark average acceleration: linear SDOF and Bouc–Wen hysteresis",
           "Peak response of a linear SDOF (α = 1) against the exact recurrence; Bouc–Wen loading branch with A = 1, β + γ = 1: "
           "z = 1 − e^(−u/u_y) for n = 1, z = tanh(u/u_y) for n = 2 (Wen 1976); discrete energy balance of the average acceleration method")
def _n02():
    out = []
    rec = synthetic_accelerogram(0.3, "C", 1, 12.0, 0.002, seed=5, rise=1.5, strong=6.0)
    p = BWParams(m=100.0, k=100.0 * (2 * math.pi / 0.5) ** 2, uy=1.0, alpha=1.0, xi=0.05)
    r = bouc_wen_sdof(p, rec)
    out.append(Comparison("linear SDOF T = 0.5 s: peak displacement, Newmark vs exact", r["u_max"],
                          float(response_spectrum(rec.a, rec.dt, [0.5])["Sd"][0]), "m", rel_tol=2e-3))
    for n, ref in ((1, 1 - math.exp(-2.0)), (2, math.tanh(2.0))):
        z, _ = bw_advance(0.0, 0.02, BWParams(n=n, uy=0.01))
        out.append(Comparison(f"Bouc–Wen n = {n}: z at u = 2u_y on the loading branch", z, ref, "", rel_tol=1e-5))
    rb = bouc_wen_sdof(BWParams(m=100.0, k=4e4, uy=0.01, alpha=0.05, n=2), synthetic_accelerogram(0.35, "C", 1, 12.0, 0.01, seed=5, rise=1.5, strong=6.0))
    out.append(Comparison("energy balance E_in − E_kin − E_damp − ∫f_s du, relative", rb["energy_error"], 0.0, "", abs_tol=1e-9))
    out.append(Comparison(f"hysteretic energy {rb['E_hyst'][-1]:.2f} kNm at ductility μ = {rb['ductility']:.2f}, checked > 0",
                          1.0 if rb["E_hyst"][-1] > 0 and rb["ductility"] > 1 else 0.0, 1.0, "", abs_tol=0.0))
    return out


@benchmark("N03", CAT, "Spectrum-compatible synthetic accelerograms (EN 1998-1 3.2.3.1.2)",
           "Three records (seeds 1–3): mean 5 % spectrum ≥ 0.9 S_e between 0.06 s and 4 s, every peak ground acceleration ≥ a_g S; "
           "S_e from EN 1998-1 Eqs. (3.2)–(3.5); random phases from the LCG x ← (1664525 x + 1013904223) mod 2³²")
def _n03():
    ag, soil = 0.25 * G_ACC, "B"
    T = np.geomspace(0.06, 4.0, 120)
    tg = elastic_spectrum(T, ag, soil, 1)
    recs = [synthetic_accelerogram(0.25, soil, 1, 20.0, 0.01, seed=s) for s in (1, 2, 3)]
    mean = np.mean([response_spectrum(r.a, r.dt, T)["Sa"] for r in recs], axis=0) / tg
    out = [Comparison(f"lowest ratio of the mean spectrum to S_e = {mean.min():.3f}, checked ≥ 0.9", min(mean.min(), 0.9), 0.9, "", abs_tol=0.0)]
    for k, r in enumerate(recs, 1):
        out.append(Comparison(f"record {k}: PGA = {r.pga / G_ACC:.3f} g, checked ≥ a_g S = 0.300 g", min(r.pga, 1.2 * ag), 1.2 * ag, "m/s²", rel_tol=1e-12))
    out.append(Comparison("S_e plateau a_g S η 2.5 (ground B, type 1)", float(elastic_spectrum([0.3], ag, soil, 1)[0]), ag * 1.2 * 2.5, "m/s²", rel_tol=1e-12))
    out.append(Comparison("LCG: first number for seed 1", LCG(1).next(), ((1664525 + 1013904223) % 2 ** 32) / 2 ** 32, "", rel_tol=1e-15))
    return out


def _wall(material=None, reinforcement=None, Lx=2.0, Ly=8.0, h=0.5):
    return {"schema": SCHEMA_FE, "element": "Q8", "analysis": "plane_stress",
            "material": material or {"E_MPa": 30000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25},
            "reinforcement": reinforcement or {},
            "geometry": {"type": "rectangle", "Lx": Lx, "Ly": Ly, "h": h}, "self_weight": True,
            "supports": [{"edge": "bottom", "ux": True, "uy": True}],
            "loads": [{"edge": "top", "qy": -100}, {"edge": "top", "qy": -50, "case": "Q"}]}


@benchmark("N04", CAT, "Direct integration of an FE wall (Newmark) against modal superposition of the same model",
           "Modal superposition with all modes below 200 Hz, modal damping ξᵢ = a₀/(2ωᵢ) + a₁ωᵢ/2 of the Rayleigh matrix, "
           "each mode integrated exactly; mass = ρ t ∫NᵀN + (G + ψ₂Q)/g lumped")
def _n04():
    from scipy.linalg import eigh
    m = parse_fe(_wall())
    rec = synthetic_accelerogram(0.25, "B", 1, 8.0, 0.005, seed=2, rise=1.0, strong=4.0)
    r = fe_dynamic(m, rec, DynOptions())
    P = Problem(m, NLOptions())
    _, K0, _, _, _ = P.respond(np.zeros(P.k.n2), P.state0)
    M = assemble_mass(m, DynOptions())
    fixed = np.array(sorted(m.fixed))
    free = np.setdiff1d(np.arange(P.k.n2), fixed)
    Kd, Md = K0[free][:, free].toarray(), M[free][:, free].toarray()
    lam, Phi = eigh(Kd, Md)
    w = np.sqrt(lam)
    a0, a1 = r.rayleigh
    rv = (free % 2 == 0).astype(float)
    mf = int(np.searchsorted(free, r.monitor_dof))
    u = np.zeros(len(rec.a))
    for i in range(len(w)):
        if w[i] > 2 * math.pi * 200:
            break
        phi = Phi[:, i]
        q, _ = sdof_history(rec.a, rec.dt, 2 * math.pi / w[i], min(a0 / (2 * w[i]) + a1 * w[i] / 2, 0.99))
        u += (phi @ Md @ rv) / (phi @ Md @ phi) * phi[mf] * q
    mass = float(M.sum()) / 2
    return [Comparison("first period T₁", r.periods[0], float(2 * math.pi / w[0]), "s", rel_tol=1e-8),
            Comparison("peak top displacement, Newmark vs modal superposition", float(np.abs(r.u_mon).max()), float(np.abs(u).max()), "m", rel_tol=5e-3),
            Comparison("total mass = ρ t A + (G + ψ₂Q)/g", r.mass_total, 25 / G_ACC * 0.25 * 16 + (100 + 0.3 * 50) * 2 / G_ACC, "t", rel_tol=1e-9),
            Comparison("mass matrix: Σ M_ij over both directions / 2", mass, r.mass_total, "t", rel_tol=1e-9),
            Comparison("energy balance E_in − E_kin − E_damp − E_int, relative", r.energy["error"], 0.0, "", abs_tol=1e-8)]


@benchmark("N05", CAT, "Nonlinear FE time history (J2 steel wall): energy balance and plastic dissipation",
           "Average acceleration with Newton–Raphson: E_in = E_kin + E_damp + ∫f_int·du exactly when every step converges; "
           "the plastic part of ∫f_int·du is the dissipated energy")
def _n05():
    mat = {"E_MPa": 210000, "nu": 0.3, "thickness_m": 0.008, "density_kNm3": 78.5, "model": "j2", "fy_MPa": 235, "H_MPa": 1000}
    m = parse_fe(_wall(mat, Lx=2.0, Ly=6.0, h=1.0))
    rec = synthetic_accelerogram(0.6, "C", 1, 6.0, 0.01, seed=4, rise=1.0, strong=3.0)
    r = fe_dynamic(m, rec, DynOptions())
    alpha = np.asarray(r.frames[-1][3]["alpha"])
    return [Comparison("analysis completed", 1.0 if r.status == "complete" else 0.0, 1.0, "", abs_tol=0.0),
            Comparison("energy balance, relative", r.energy["error"], 0.0, "", abs_tol=1e-6),
            Comparison(f"equivalent plastic strain up to {alpha.max() * 1000:.2f} ‰ at {np.mean(alpha > 0) * 100:.0f} % of the points, checked > 0",
                       1.0 if alpha.max() > 0 else 0.0, 1.0, "", abs_tol=0.0)]
