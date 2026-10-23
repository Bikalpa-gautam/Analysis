"""Benchmarks for nonlinear plane FE (tragwerk.fe2d_nl) and the reinforcement design of FE regions (tragwerk.fe2d_design).

Every expected value comes from a closed-form solution, an independent numerical solution of the governing equation
(the elastica ODE, integrated here with SciPy) or a hand calculation following the code clause; none is produced by
the code under test.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..fe2d import SCHEMA_FE, concrete_props, parse_fe
from ..fe2d_design import anchorage_length, annex_f, design_reinforcement, minimum_reinforcement, propose_layers
from ..fe2d_nl import NLOptions, crack_widths, run_nonlinear
from .benchmarks import Comparison, benchmark

CAT = "Nonlinear FE and FE reinforcement design"


# ------------------------------------------------------------------ references
def elastica(k: float) -> tuple[float, float, float]:
    """Cantilever under a dead tip load with k = PL²/EI: (shortening u/L, deflection v/L, tip rotation) from
    EI θ'' = −P cos θ, θ(0) = 0, θ'(L) = 0, solved by shooting."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def shoot(k0):
        sol = solve_ivp(lambda s, y: [y[1], -k * math.cos(y[0]), math.cos(y[0]), math.sin(y[0])], (0, 1), [0, k0, 0, 0],
                        rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
    k0 = brentq(lambda k0: shoot(k0)[1], 0.0, k + 1e-9, xtol=1e-14)
    th, _, x, y = shoot(k0)
    return 1 - x, y, th


def _panel(sx_mm, sy_mm, tension="stiffening", k=1.0, t=0.2, loads=None, supports=None):
    q = 1000 * t                      # 1 MPa of shear on every edge (kN/m)
    return {"schema": SCHEMA_FE, "element": "Q4", "analysis": "plane_stress",
            "material": {"E_MPa": 33000, "nu": 0.0, "thickness_m": t, "density_kNm3": 0, "model": "rc", "fck_MPa": 30, "values": "mean",
                         "softening": "mcft", "tension": tension},
            "reinforcement": {"k": k, "layers": [L for L in ({"dir": "x", "dia_mm": 12, "spacing_mm": sx_mm, "faces": 2} if sx_mm else None,
                                                             {"dir": "y", "dia_mm": 12, "spacing_mm": sy_mm, "faces": 2} if sy_mm else None) if L]},
            "geometry": {"type": "rectangle", "Lx": 1, "Ly": 1, "h": 1},
            "supports": supports or [{"at": [0, 0], "ux": True, "uy": True}, {"at": [1, 0], "uy": True}],
            "loads": loads if loads is not None else [{"edge": "right", "qy": q}, {"edge": "left", "qy": -q}, {"edge": "top", "qx": q}, {"edge": "bottom", "qx": -q}]}


def _rho(s_mm, t=0.2, dia=12):
    return 2 * math.pi * dia * dia / 4 * 1000 / s_mm * 1e-6 / t


DEEP_BEAM = {"schema": SCHEMA_FE, "name": "Deep beam on short supports", "element": "Q8", "analysis": "plane_stress",
             "material": {"E_MPa": 33000, "nu": 0.2, "thickness_m": 0.3, "density_kNm3": 25, "model": "rc", "fck_MPa": 30, "values": "design"},
             "design": {"member": "deep_beam"}, "reinforcement": {"cover_mm": 35, "layers": []},
             "geometry": {"type": "rectangle", "Lx": 6.0, "Ly": 3.0, "h": 0.5},
             "supports": [{"edge": "bottom", "from": 0.0, "to": 0.5, "uy": True}, {"edge": "bottom", "from": 5.5, "to": 6.0, "uy": True},
                          {"at": [0, 0], "ux": True}],
             "loads": [{"edge": "top", "qy": -250}, {"edge": "top", "qy": -120, "case": "Q"}], "self_weight": True}


def designed_deep_beam() -> dict:
    """The deep beam with the reinforcement proposed by the Annex F design."""
    d = copy.deepcopy(DEEP_BEAM)
    m = parse_fe(d)
    layers, _ = propose_layers(m, design_reinforcement(m))
    d["reinforcement"]["layers"] = layers
    return d


# ------------------------------------------------------------------ geometric nonlinearity
@benchmark("F07", CAT, "Large deflection of a cantilever under a dead tip load (elastica), total Lagrangian Q8",
           "Elastica EI θ'' = −P cos θ solved by shooting; tabulated by Mattiasson (1981): PL²/EI = 1 → v/L = 0.30172, u/L = 0.05643; "
           "PL²/EI = 2 → v/L = 0.49346, u/L = 0.16064. MFE II (Chatzi), geometric nonlinearity")
def _f07():
    L, hb, t, E = 10.0, 0.25, 0.1, 10000.0
    EI = E * 1000 * t * hb ** 3 / 12
    out = []
    for k in (1.0, 2.0):
        d = {"schema": SCHEMA_FE, "element": "Q8", "analysis": "plane_stress", "material": {"E_MPa": E, "nu": 0.0, "thickness_m": t, "density_kNm3": 0},
             "geometry": {"type": "rectangle", "Lx": L, "Ly": hb, "h": 0.125}, "supports": [{"edge": "left", "ux": True, "uy": True}],
             "loads": [{"edge": "right", "qy": -k * EI / L ** 2 / hb}]}
        m = parse_fe(d)
        r = run_nonlinear(m, NLOptions(geometric=True, steps=8, combination="char"))
        tip = int(np.argmin(np.hypot(m.nodes[:, 0] - L, m.nodes[:, 1] - hb / 2)))
        ux, uy = r.steps[-1].u.reshape(-1, 2)[tip]
        uL, vL, _ = elastica(k)
        out.append(Comparison(f"PL²/EI = {k:g}: tip deflection v/L", -uy / L, vL, "", rel_tol=0.003))
        out.append(Comparison(f"PL²/EI = {k:g}: tip shortening u/L", -ux / L, uL, "", rel_tol=0.005))
        out.append(Comparison(f"PL²/EI = {k:g}: Σ R_y + P (equilibrium in the deformed state)", r.steps[-1].reaction[1] - k * EI / L ** 2, 0.0, "kN",
                              abs_tol=1e-6 * k * EI / L ** 2))
    return out


# ------------------------------------------------------------------ plasticity
@benchmark("F08", CAT, "J2 plasticity, uniaxial plane stress: stress–strain curve and quadratic convergence of the consistent tangent",
           "σ = E ε up to f_y, then σ = f_y + H ε_p with ε = σ/E + ε_p (linear isotropic hardening); Simo & Hughes (1998) radial return")
def _f08():
    E, sy, H = 210000.0, 355.0, 2000.0
    d = {"schema": SCHEMA_FE, "element": "Q4", "analysis": "plane_stress", "material": {"E_MPa": E, "nu": 0.3, "thickness_m": 0.01, "model": "j2", "fy_MPa": sy, "H_MPa": H},
         "geometry": {"type": "rectangle", "Lx": 1, "Ly": 1, "h": 1}, "supports": [{"edge": "left", "ux": True}, {"at": [0, 0], "uy": True}, {"edge": "right", "ux": 0.005}]}
    r = run_nonlinear(parse_fe(d), NLOptions(steps=20, combination="char"))
    err = 0.0
    for s in r.steps[1:]:
        eps = s.lam * 0.005
        ref = E * eps if eps * E <= sy else sy + H * (eps * E - sy) / (E + H)
        err = max(err, float(np.abs(s.gp_stress[0, :, 0] - ref).max()))
    last = r.steps[-1]
    out = [Comparison(f"largest |σ_xx − σ_ref| over all {len(r.steps) - 1} increments", err, 0.0, "MPa", abs_tol=1e-6 * sy),
           Comparison("σ_xx at ε = 0.005", float(last.gp_stress[0, :, 0].mean()), sy + H * (0.005 * E - sy) / (E + H), "MPa", rel_tol=1e-6),
           Comparison("σ_yy (free lateral contraction)", float(np.abs(last.gp_stress[0, :, 1]).max()), 0.0, "MPa", abs_tol=1e-6 * sy)]
    res = next(s.residuals for s in r.steps[1:] if len(s.residuals) >= 4)
    p = math.log(res[-1] / res[-2]) / math.log(res[-2] / res[-3])
    out.append(Comparison("observed order of convergence of Newton–Raphson (consistent tangent)", p, 2.0, "", rel_tol=0.25))
    return out


@benchmark("F09", CAT, "Thick cylinder under internal pressure: limit pressure of a perfectly plastic von Mises material (Hill)",
           "Hill (1950): p_L = (2/√3) σ_y ln(b/a), plane strain; Q8R (reduced integration avoids volumetric locking); "
           "load control and the arc-length method (Crisfield)")
def _f09():
    sy, a, b = 240.0, 1.0, 2.0
    pL = 2 / math.sqrt(3) * sy * math.log(b / a)
    d = {"schema": SCHEMA_FE, "element": "Q8R", "analysis": "plane_strain",
         "material": {"E_MPa": 210000, "nu": 0.3, "thickness_m": 1.0, "density_kNm3": 0, "model": "j2", "fy_MPa": sy, "H_MPa": 0.0},
         "geometry": {"type": "ring", "a": a, "b": b, "n_theta": 4, "n_r": 8},
         "supports": [{"edge": "left", "ux": True}, {"edge": "bottom", "uy": True}],
         "loads": [{"edge": "inner", "p": pL * 1000}]}
    m = parse_fe(d)
    out = []
    for ctl in ("load", "arc"):
        r = run_nonlinear(m, NLOptions(steps=10, combination="char", control=ctl, lam_max=1.2, to_failure=(ctl == "arc"), max_steps=60))
        out.append(Comparison(f"p_peak / p_L ({'load control' if ctl == 'load' else 'arc-length'})", r.lam_peak, 1.0, "", rel_tol=0.01))
    return out


# ------------------------------------------------------------------ reinforced concrete membrane
@benchmark("F10", CAT, "Reinforced concrete panel in pure shear: ultimate shear stress (plasticity theory and MCFT)",
           "Isotropic reinforcement: τ_u = ρ f_y; orthotropic reinforcement, no concrete tension: τ_u = √(ρ_x ρ_y) f_y at tan²θ = ρ_x/ρ_y "
           "(Nielsen; Vecchio & Collins 1986); perfectly plastic steel, mean strengths (f_y = 1.1 f_yk)")
def _f10():
    fy = 1.1 * 500
    out = []
    r = run_nonlinear(parse_fe(_panel(150, 150)), NLOptions(steps=10, combination="char", control="arc", to_failure=True, max_steps=60, tol=1e-6))
    out.append(Comparison("isotropic ρ = 0.75 %, tension stiffening (arc-length): τ_u", r.lam_peak, _rho(150) * fy, "MPa", rel_tol=1e-4))
    s = max(r.steps, key=lambda q: q.lam)
    out.append(Comparison("isotropic: crack angle θ", math.degrees(float(s.extra["theta"][0, 0])), 45.0, "°", abs_tol=0.05))
    r = run_nonlinear(parse_fe(_panel(100, 250, tension="none")), NLOptions(steps=10, combination="char", control="displacement", to_failure=True,
                                                                          max_steps=60, tol=1e-6))
    rx, ry = _rho(100), _rho(250)
    out.append(Comparison("orthotropic ρ_x = 1.13 %, ρ_y = 0.45 %, no tension (displacement control): τ_u", r.lam_peak, math.sqrt(rx * ry) * fy, "MPa",
                          rel_tol=1e-4))
    th = max(r.steps, key=lambda q: q.lam).extra["theta"][0, 0]
    out.append(Comparison("orthotropic: compression field angle at failure, tan²θ = ρ_x/ρ_y", math.degrees(float(th)),
                          math.degrees(math.atan(math.sqrt(rx / ry))), "°", abs_tol=0.5))
    return out


@benchmark("F11", CAT, "Reinforced concrete tie: tension stiffening, yield plateau and crack width",
           "σ = ρ E_s ε + f_ct/(1 + √(500 ε)) (Collins & Mitchell 1991) while the steel is elastic; σ = ρ f_y once it yields "
           "(MCFT crack check); w = s_r,max (ε₁ − σ_c1/E_c), s_r,max = 3.4c + 0.425 k₁ φ/ρ_p,eff (EN 1992-1-1 7.3.4)")
def _f11():
    t = 0.2

    def tie(ux):
        d = _panel(150, 0, t=t, loads=[], supports=[{"edge": "left", "ux": True}, {"at": [0, 0], "uy": True}, {"edge": "right", "ux": ux}])
        m = parse_fe(d)
        return m, run_nonlinear(m, NLOptions(steps=10, combination="char"))
    m, _ = tie(1e-4)
    mat = m.material
    rho = _rho(150, t)
    Ec, fct, fy, Es = m.E / 1000, mat["fct"] / 1000, mat["fy"] / 1000, mat["Es"] / 1000
    out = []
    e = 0.5 * fct / Ec
    _, r = tie(e)
    out.append(Comparison(f"uncracked, ε = {e:.2e}: σ = (E_c + ρ E_s) ε", float(r.steps[-1].gp_stress[0, :, 0].mean()), (Ec + rho * Es) * e, "MPa", rel_tol=1e-9))
    _, r = tie(0.001)
    s1 = r.steps[-1]
    out.append(Comparison("cracked, ε = 0.001, steel elastic: σ = ρ E_s ε + f_ct/(1 + √(500 ε))", float(s1.gp_stress[0, :, 0].mean()),
                          rho * Es * 0.001 + fct / (1 + math.sqrt(0.5)), "MPa", rel_tol=1e-6))
    c, dia = m.reinforcement["cover_mm"], 12.0
    hef = min(2.5 * (c + dia / 2), t * 1000 / 2)
    rho_eff = (rho * t * 1e6 / 2) / (hef * 1000)
    sr = 3.4 * c + 0.425 * 0.8 * dia / rho_eff
    out.append(Comparison("crack width at ε = 0.001: s_r,max (ε₁ − σ_c1/E_c)", float(crack_widths(r, s1)[0, 0]),
                          sr * (0.001 - fct / (1 + math.sqrt(0.5)) / Ec), "mm", rel_tol=1e-6))
    _, r = tie(0.01)
    out.append(Comparison("ε = 0.01, steel yielded: σ = ρ f_y (concrete tension released by the crack check)", float(r.steps[-1].gp_stress[0, :, 0].mean()),
                          rho * fy, "MPa", rel_tol=1e-6))
    return out


@benchmark("F12", CAT, "Deep beam designed with EN 1992-1-1 Annex F, verified by nonlinear analysis to failure (lower-bound theorem)",
           "Lower-bound theorem of plasticity: reinforcement designed for an equilibrium stress field with σ ≤ f_d carries at least the "
           "design load (λ_u ≥ 1 with design strengths); load control and arc-length agree on the peak")
def _f12():
    m = parse_fe(designed_deep_beam())
    des = design_reinforcement(m)
    us, uc = float(np.max(des["util_steel"])), float(des["util_concrete"].max())
    out = [Comparison(f"steel utilisation of the provided reinforcement {us:.3f}, checked ≤ 1", max(us, 1.0), 1.0, "", abs_tol=0.0),
           Comparison(f"concrete utilisation σ_cd/(ν f_cd) {uc:.3f}, checked ≤ 1", max(uc, 1.0), 1.0, "", abs_tol=0.0)]
    ra = run_nonlinear(m, NLOptions(combination="uls", steps=10, to_failure=True, control="arc", max_steps=80))
    out.append(Comparison(f"λ_u = {ra.lam_peak:.3f} with design strengths (arc-length), checked ≥ 1", min(ra.lam_peak, 1.0), 1.0, "", abs_tol=0.0))
    rl = run_nonlinear(m, NLOptions(combination="uls", steps=10, to_failure=True, control="load", max_steps=80))
    out.append(Comparison("λ_u by load control vs arc-length", rl.lam_peak, ra.lam_peak, "", rel_tol=0.03))
    return out


# ------------------------------------------------------------------ design of FE regions
@benchmark("F13", CAT, "In-plane reinforcement design: EN 1992-1-1 Annex F expressions, minimum reinforcement and anchorage length",
           "EN 1992-1-1 F.2–F.7, 9.6.2, 9.6.3, 9.7, 8.4.2–8.4.3; hand calculations")
def _f13():
    out = []
    # (i) σ_Edx = 2 ≤ |τ| = 3 (compression positive): f'_tdx = 1, f'_tdy = 4, σ_cd = 6
    fx, fy, sc, _ = annex_f(-2.0, 1.0, 3.0)
    out += [Comparison("F.2: f'_tdx = |τ| − σ_Edx", fx, 1.0, "MPa", rel_tol=1e-12), Comparison("F.3: f'_tdy = |τ| − σ_Edy", fy, 4.0, "MPa", rel_tol=1e-12),
            Comparison("F.4: σ_cd = 2|τ|", sc, 6.0, "MPa", rel_tol=1e-12)]
    # (ii) σ_Edx = 5 > |τ| = 2, σ_Edy = −1: f'_tdx = 0, f'_tdy = τ²/σ_Edx − σ_Edy = 1.8, σ_cd = σ_Edx (1 + (τ/σ_Edx)²) = 5.8
    fx, fy, sc, _ = annex_f(-5.0, 1.0, -2.0)
    out += [Comparison("F.5: f'_tdx = 0", fx, 0.0, "MPa", abs_tol=1e-12), Comparison("F.6: f'_tdy = τ²/σ_Edx − σ_Edy", fy, 1.8, "MPa", rel_tol=1e-12),
            Comparison("F.7: σ_cd = σ_Edx (1 + (τ/σ_Edx)²)", sc, 5.8, "MPa", rel_tol=1e-12)]
    # (iii) both compressive, σ_Edx σ_Edy = 12 > τ² = 4: no reinforcement
    fx, fy, sc, _ = annex_f(-4.0, -3.0, 2.0)
    out += [Comparison("biaxial compression: f'_tdx + f'_tdy", fx + fy, 0.0, "MPa", abs_tol=1e-12),
            Comparison("biaxial compression: σ_cd = principal compression", sc, 3.5 + math.sqrt(0.25 + 4.0), "MPa", rel_tol=1e-12)]
    # (iv) the more compressive direction is y: the expressions apply with x and y exchanged
    fx, fy, _, _ = annex_f(1.0, -2.0, 3.0)
    out += [Comparison("σ_Edy > σ_Edx: f'_tdx", fx, 4.0, "MPa", rel_tol=1e-12), Comparison("σ_Edy > σ_Edx: f'_tdy", fy, 1.0, "MPa", rel_tol=1e-12)]
    hx, v, _ = minimum_reinforcement("wall", 0.25)
    out += [Comparison("wall t = 250 mm: A_s,vmin = 0.002 A_c", v, 500.0, "mm²/m", rel_tol=1e-12),
            Comparison("wall: A_s,hmin = max(25 % A_s,v, 0.001 A_c)", hx, 250.0, "mm²/m", rel_tol=1e-12)]
    bx, by, _ = minimum_reinforcement("deep_beam", 0.3)
    out.append(Comparison("deep beam t = 300 mm: A_s,dbmin = 0.1 % A_c per face, both faces", bx + by, 2 * 600.0, "mm²/m", rel_tol=1e-12))
    m = parse_fe(DEEP_BEAM)
    fctd = 0.7 * 0.30 * 30 ** (2 / 3) / 1.5
    out.append(Comparison("l_bd of Ø12, C30/37, poor bond: (φ/4) f_yd / (2.25 · 0.7 · f_ctd)", anchorage_length(12, m) * 1000,
                          12 / 4 * (500 / 1.15) / (2.25 * 0.7 * fctd), "mm", rel_tol=1e-12))
    out.append(Comparison("f_ctm of C30/37 (Table 3.1)", concrete_props(30)["fctm"], 0.30 * 30 ** (2 / 3), "MPa", rel_tol=1e-12))
    return out
