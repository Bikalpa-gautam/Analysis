"""Benchmarks for modal analysis and the EN 1998-1 response spectrum method."""
from __future__ import annotations

import copy
import math

import numpy as np

from ..model3d import SeismicSettings, parse_model3d
from ..modal import ElementMass, element_mass_local, modal
from ..project3d import run3d
from ..seismic import cqc_rho, design_spectrum
from ..solver3d import analysis_modulus3d
from .benchmarks import Comparison, benchmark

MAT = {"C": {"type": "concrete", "fck": 30.0}, "R": {"type": "rebar", "fyk": 500}}
SQ = {"type": "rc_rect", "b_mm": 400, "h_mm": 400, "concrete": "C", "rebar": "R", "cover_mm": 35,
      "top": {"n": 3, "dia_mm": 20}, "bottom": {"n": 3, "dia_mm": 20}, "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200}}
FIX6 = {"ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True}
GRAV = 9.81


def _model(nodes, members, supports, loads=(), dynamics=None, seismic=None, sections=None, mass_cases=None):
    d = {"schema": "tragwerk3d/0.1", "materials": MAT, "sections": copy.deepcopy(sections or {"S": SQ}),
         "nodes": [{"id": k, "x": v[0], "y": v[1], "z": v[2]} for k, v in nodes.items()],
         "members": [dict({"id": k, "i": v[0], "j": v[1], "section": v[2] if len(v) > 2 else "S"}) for k, v in members.items()],
         "supports": supports, "load_cases": [{"id": "G", "type": "permanent", "self_weight": True, "loads": list(loads)}],
         "dynamics": dict({"modes": 6}, **(dynamics or {}))}
    if mass_cases is not None:
        d["dynamics"]["mass_cases"] = mass_cases
    if seismic:
        d["seismic"] = seismic
    return parse_model3d(d)


def _props():
    m = _model({"A": (0, 0, 0), "B": (1, 0, 0)}, {"M": ("A", "B")}, [{"node": "A", **FIX6}])
    s = m.sections["S"]
    E = analysis_modulus3d(s)
    return E, E * s.Iy_m4, E * s.area_m2, 25.0 * s.area_m2 / GRAV


@benchmark("D01", "Dynamics", "Consistent mass matrix of a uniform beam element",
           "Textbook consistent mass: m̄L/420·[156, 22L, 54, −13L; …; 4L²] for bending, m̄L/6·[2, 1; 1, 2] axial (e.g. Przemieniecki; MFE I)")
def _d01():
    L, m = 3.0, 2.0
    M = element_mass_local(L, ElementMass(msw=m), "consistent")
    ref = m * L / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                  [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
    v = M[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])]
    w = M[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])]
    sgn = np.array([1, -1, 1, -1])
    return [
        Comparison("bending v-θz block, largest deviation", float(np.max(np.abs(v - ref))), 0.0, "t", abs_tol=1e-12),
        Comparison("bending w-θy block (θy = −dw/dx), largest deviation", float(np.max(np.abs(w - sgn[:, None] * ref * sgn[None, :]))), 0.0, "t", abs_tol=1e-12),
        Comparison("axial M11", float(M[0, 0]), m * L / 3, "t"),
        Comparison("axial M12", float(M[0, 6]), m * L / 6, "t"),
        Comparison("total translational mass (sum of axial block)", float(M[0, 0] + 2 * M[0, 6] + M[6, 6]), m * L, "t"),
    ]


@benchmark("D02", "Dynamics", "Cantilever column: first two bending frequencies and effective mass",
           "Euler–Bernoulli: ω_n = (β_nL)²·√(EI/(m̄L⁴)), β₁L = 1.87510, β₂L = 4.69409; effective mass of mode 1 = 0.6131·m̄L")
def _d02():
    L = 4.0
    m = _model({"A": (0, 0, 0), "B": (0, 0, L)}, {"C": ("A", "B")}, [{"node": "A", **FIX6}], dynamics={"segments": 8})
    r = modal(m)
    E, EI, EA, mb = _props()
    w = lambda bl: bl ** 2 * math.sqrt(EI / (mb * L ** 4))  # noqa: E731
    return [
        Comparison("ω₁ (sway in X and Y, square section)", float(r.omega[0]), w(1.8751040687), "rad/s", rel_tol=1e-3),
        Comparison("ω₁, second direction", float(r.omega[1]), w(1.8751040687), "rad/s", rel_tol=1e-3),
        Comparison("ω₂", float(r.omega[2]), w(4.6940911330), "rad/s", rel_tol=2e-3),
        Comparison("effective mass mode 1 / m̄L (X or Y)", float(max(r.meff["X"][0], r.meff["Y"][0]) / (mb * L)), 0.6131, "", rel_tol=2e-3),
    ]


@benchmark("D03", "Dynamics", "Simply supported beam: vertical bending frequencies",
           "ω_n = (nπ)²·√(EI/(m̄L⁴)); torsion restrained at both ends; 8 elements")
def _d03():
    L = 8.0
    sup = [{"node": "A", "ux": True, "uy": True, "uz": True, "rx": True}, {"node": "B", "uy": True, "uz": True, "rx": True}]
    m = _model({"A": (0, 0, 0), "B": (L, 0, 0)}, {"M": ("A", "B")}, sup, dynamics={"segments": 8, "modes": 8})
    r = modal(m)
    E, EI, EA, mb = _props()
    zr = r.ratio("Z")
    # vertical modes: those with motion in Z (mode 1 carries most of the mass; mode 2 is antisymmetric, zero participation)
    vert = [k for k in range(len(r.omega)) if np.max(np.abs(r.node_displacements(k)[:, 2])) > 0.5 * np.max(np.abs(r.node_displacements(k)[:, :3]))]
    return [
        Comparison("ω₁ vertical", float(r.omega[vert[0]]), math.pi ** 2 * math.sqrt(EI / (mb * L ** 4)), "rad/s", rel_tol=1e-3),
        Comparison("ω₂ vertical", float(r.omega[vert[1]]), 4 * math.pi ** 2 * math.sqrt(EI / (mb * L ** 4)), "rad/s", rel_tol=3e-3),
        Comparison("effective mass of ω₁ in Z / m̄L", float(zr[vert[0]] * r.mtot["Z"] / (mb * L)), 8 / math.pi ** 2, "", rel_tol=2e-3),
    ]


@benchmark("D04", "Dynamics", "Fixed–free bar: first axial frequency",
           "ω₁ = π/(2L)·√(EA/m̄); vertical member, all lateral motion restrained; 8 consistent elements")
def _d04():
    L = 6.0
    sup = [{"node": "A", **FIX6}, {"node": "B", "ux": True, "uy": True, "rx": True, "ry": True, "rz": True}]
    m = _model({"A": (0, 0, 0), "B": (0, 0, L)}, {"C": ("A", "B")}, sup, dynamics={"segments": 8})
    r = modal(m)
    E, EI, EA, mb = _props()
    k = int(np.argmax(r.ratio("Z")))
    return [Comparison("ω₁ axial", float(r.omega[k]), math.pi / (2 * L) * math.sqrt(EA / mb), "rad/s", rel_tol=3e-3)]


@benchmark("D05", "Dynamics", "Two-storey shear frame with floor masses",
           "Rigid beams, columns fixed at both ends, floors restrained vertically (no rocking): k = 2·12EI/h³ per storey, m per floor; ω² = (k/m)·(3 ∓ √5)/2")
def _d05():
    h, Lb, mf = 3.0, 6.0, 40.0
    nodes = {"A0": (0, 0, 0), "B0": (Lb, 0, 0), "A1": (0, 0, h), "B1": (Lb, 0, h), "A2": (0, 0, 2 * h), "B2": (Lb, 0, 2 * h)}
    rigid = dict(SQ, b_mm=600, h_mm=900, stiffness_factor=1e4)
    members = {"CA1": ("A0", "A1"), "CB1": ("B0", "B1"), "CA2": ("A1", "A2"), "CB2": ("B1", "B2"), "B1": ("A1", "B1", "R"), "B2": ("A2", "B2", "R")}
    sup = [{"node": "A0", **FIX6}, {"node": "B0", **FIX6}] + [{"node": n, "uz": True} for n in ("A1", "B1", "A2", "B2")]
    m = _model(nodes, members, sup, dynamics={"segments": 1, "mass": "lumped", "masses": [{"node": n, "m": mf / 2} for n in ("A1", "B1", "A2", "B2")]},
               sections={"S": SQ, "R": rigid}, mass_cases={"G": 0.0})
    r = modal(m)
    E, EI, EA, mb = _props()
    k = 2 * 12 * EI / h ** 3
    xs = [i for i in range(len(r.omega)) if r.ratio("X")[i] > 1e-3]
    return [
        Comparison("ω₁ in X", float(r.omega[xs[0]]), math.sqrt(k / mf * (3 - math.sqrt(5)) / 2), "rad/s", rel_tol=2e-3),
        Comparison("ω₂ in X", float(r.omega[xs[1]]), math.sqrt(k / mf * (3 + math.sqrt(5)) / 2), "rad/s", rel_tol=2e-3),
        Comparison("mass ratio of mode 1 in X", float(r.ratio("X")[xs[0]]), (1 + (1 + math.sqrt(5)) / 2) ** 2 / (2 * (1 + ((1 + math.sqrt(5)) / 2) ** 2)), "", rel_tol=2e-3),
    ]


@benchmark("D06", "Seismic", "EN 1998-1 design spectrum, type 1, ground C",
           "Eqs. (3.13)-(3.16) by hand: a_g = 1.2 m/s² (γ_I = 1.2, a_gR = 1.0), S = 1.15, T_B = 0.2, T_C = 0.6, T_D = 2.0 s, q = 1.5, β = 0.2")
def _d06():
    s = SeismicSettings(agR=1.0, importance=1.2, ground="C", spectrum_type=1, q=1.5)
    ag, S = 1.2, 1.15
    vals = {0.0: ag * S * 2 / 3, 0.1: ag * S * (2 / 3 + 0.5 * (2.5 / 1.5 - 2 / 3)), 0.4: ag * S * 2.5 / 1.5,
            1.2: ag * S * 2.5 / 1.5 * 0.6 / 1.2, 3.0: max(ag * S * 2.5 / 1.5 * 0.6 * 2.0 / 9.0, 0.2 * ag), 4.0: 0.2 * ag}
    return [Comparison(f"S_d(T = {T} s)", float(design_spectrum(T, s)), v, "m/s²") for T, v in vals.items()]


@benchmark("D07", "Seismic", "CQC correlation coefficient",
           "Der Kiureghian (1981), equal damping ζ = 5 %: ρ = 8ζ²(1+r)r^{3/2}/((1−r²)² + 4ζ²r(1+r)²); by hand r = 0.9 gives 0.03245/0.06859 = 0.4730, r = 0.5 gives 0.010607/0.57375 = 0.01849, r = 1 gives 1")
def _d07():
    rho = cqc_rho(np.array([10.0, 9.0, 5.0]), 0.05)
    return [Comparison("ρ(r = 1)", float(rho[0, 0]), 1.0), Comparison("ρ(r = 0.9)", float(rho[0, 1]), 0.47300, rel_tol=1e-4),
            Comparison("ρ(r = 0.5)", float(rho[0, 2]), 0.010607 / 0.57375, rel_tol=1e-4), Comparison("symmetry", float(rho[1, 0] - rho[0, 1]), 0.0)]


@benchmark("D08", "Seismic", "Single-storey column with a top mass: base shear and drift",
           "SDOF by hand: k = 3EI/h³, T = 2π√(m/k), V = m·S_d(T), d_e = S_d/ω², d_r = q·d_e (EN 1998-1 4.3.4)")
def _d08():
    h, mt = 4.0, 30.0
    sei = {"agR": 2.0, "ground": "B", "q": 1.5, "directions": ["X"]}
    m = _model({"A": (0, 0, 0), "B": (0, 0, h)}, {"C": ("A", "B")}, [{"node": "A", **FIX6}],
               dynamics={"segments": 1, "mass": "lumped", "masses": [{"node": "B", "m": mt}], "modes": 2}, seismic=sei, mass_cases={"G": 0.0})
    res = run3d(m)
    E, EI, EA, mb = _props()
    k = 3 * EI / h ** 3
    w = math.sqrt(k / mt)
    T = 2 * math.pi / w
    Sd = float(design_spectrum(T, m.seismic))
    d = res.seismic.drift["C"]
    return [
        Comparison("period T", float(res.modal.period[0]), T, "s", rel_tol=1e-9),
        Comparison("base shear V_x", res.seismic.base_shear["X"], mt * Sd, "kN", rel_tol=1e-9),
        Comparison("base moment |M_y| at the foot", float(res.seismic.members["C"]["My"][0]), mt * Sd * h, "kNm", rel_tol=1e-9),
        Comparison("elastic drift d_r,e", d["dre_x"] * 1000, Sd / w ** 2 * 1000, "mm", rel_tol=1e-9),
    ]


# ===================================================================== linear buckling (tragwerk.buckling)
def _column(L, base, top, P=1000.0, b=400, h=400):
    d = {"schema": "tragwerk3d/0.1", "materials": MAT, "sections": {"S": dict(SQ, b_mm=b, h_mm=h)},
         "nodes": [{"id": "A", "x": 0, "y": 0, "z": 0}, {"id": "B", "x": 0, "y": 0, "z": L}],
         "members": [{"id": "C1", "i": "A", "j": "B", "section": "S"}],
         "supports": [dict({"node": "A"}, **base)] + ([dict({"node": "B"}, **top)] if top else []),
         "load_cases": [{"id": "G", "type": "permanent", "self_weight": False, "loads": [{"type": "nodal", "node": "B", "fz": -P}]}]}
    return parse_model3d(d)


@benchmark("D12", "Dynamics", "Linear buckling: Euler columns and a sway portal frame (elastic critical load factor α_cr)",
           "Euler: P_cr = π²EI/(βL)², β = 1 (pinned), 2 (cantilever), 0.5 (fixed–fixed), 0.699 (fixed–pinned, π² · 2.046); "
           "fixed-base portal with a rigid beam: P_cr = π²EI/h² per column; α_cr = P_cr/P_Ed with P_Ed = 1.35 G")
def _d12():
    from ..buckling import buckling
    out = []
    PIN = {"ux": True, "uy": True, "uz": True, "rz": True}
    FIX = {"ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True}
    L, P = 5.0, 1000.0
    for name, base, top, beta in (("pinned–pinned", PIN, {"ux": True, "uy": True}, 1.0), ("cantilever", FIX, None, 2.0),
                                  ("fixed–fixed", FIX, {"ux": True, "uy": True, "rx": True, "ry": True}, 0.5),
                                  ("fixed–pinned", FIX, {"ux": True, "uy": True}, 1 / math.sqrt(2.045752))):
        m = _column(L, base, top, P)
        sec = m.sections["S"]
        EI = analysis_modulus3d(sec) * sec.stiffness_factor * 0.4 ** 4 / 12
        br = buckling(m, modes=2, segments=8)
        out.append(Comparison(f"{name}: α_cr", br.alpha_cr, math.pi ** 2 * EI / (beta * L) ** 2 / (1.35 * P), "", rel_tol=2e-3))
    # portal frame in the x-z plane, columns 400×400 h = 4 m, beam 400×3000 (practically rigid), out-of-plane restrained
    h, span = 4.0, 6.0
    d = {"schema": "tragwerk3d/0.1", "materials": MAT, "sections": {"C": dict(SQ), "B": dict(SQ, b_mm=400, h_mm=3000)},
         "nodes": [{"id": "A", "x": 0, "y": 0, "z": 0}, {"id": "B", "x": 0, "y": 0, "z": h}, {"id": "C", "x": span, "y": 0, "z": h}, {"id": "D", "x": span, "y": 0, "z": 0}],
         "members": [{"id": "c1", "i": "A", "j": "B", "section": "C"}, {"id": "b", "i": "B", "j": "C", "section": "B"}, {"id": "c2", "i": "D", "j": "C", "section": "C"}],
         "supports": [dict({"node": "A"}, **FIX), dict({"node": "D"}, **FIX), {"node": "B", "uy": True}, {"node": "C", "uy": True}],
         "load_cases": [{"id": "G", "type": "permanent", "self_weight": False,
                         "loads": [{"type": "nodal", "node": "B", "fz": -P}, {"type": "nodal", "node": "C", "fz": -P}]}]}
    m = parse_model3d(d)
    sec = m.sections["C"]
    EI = analysis_modulus3d(sec) * sec.stiffness_factor * 0.4 ** 4 / 12
    br = buckling(m, modes=2, segments=8)
    out.append(Comparison("portal, rigid beam: sway α_cr", br.alpha_cr, math.pi ** 2 * EI / h ** 2 / (1.35 * P), "", rel_tol=1e-2))
    out.append(Comparison("amplification 1/(1 − 1/α_cr) reported for α_cr < 10", 1.0 if (br.alpha_cr >= 10 or any("amplified" in w for w in br.warnings)) else 0.0,
                          1.0, "", abs_tol=0.0))
    return out
