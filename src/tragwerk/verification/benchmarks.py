"""Verification benchmarks: closed-form solutions and hand calculations.

Every benchmark states its reference. Expected values are computed from the
published formula or written out as hand-calculated numbers; they never call
the code they check. The test suite runs all of them, and
``tragwerk verify`` publishes them as an HTML verification report.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..codes.ec2 import checks as ec2c
from ..codes.ec2.materials import concrete_props, steel_props
from ..codes.ec2.params import EC2Params
from ..codes.ec2.section import RCSection
from ..combinations import generate
from ..model import parse_model
from ..solver import FrameAnalysis, _cum_dist


@dataclass
class Comparison:
    quantity: str
    computed: float
    expected: float
    unit: str = ""
    rel_tol: float = 1e-9
    abs_tol: float = 1e-9

    @property
    def error(self) -> float:
        return abs(self.computed - self.expected)

    @property
    def rel_error(self) -> float:
        return self.error / abs(self.expected) if self.expected else self.error

    @property
    def passed(self) -> bool:
        return self.error <= max(self.rel_tol * abs(self.expected), self.abs_tol)


@dataclass
class Benchmark:
    id: str
    category: str
    title: str
    reference: str
    run: Callable[[], list[Comparison]] = field(repr=False)


BENCHMARKS: list[Benchmark] = []


def benchmark(id: str, category: str, title: str, reference: str):
    def deco(fn):
        BENCHMARKS.append(Benchmark(id, category, title, reference, fn))
        return fn
    return deco


# ------------------------------------------------------------------ helpers
FCK = 30.0
B, H = 300.0, 500.0
E = 22_000.0 * ((FCK + 8) / 10) ** 0.3 * 1000.0         # kN/m2, EN 1992-1-1 Table 3.1
EI = E * (B * H ** 3 / 12) * 1e-12                       # kNm2
EA = E * (B * H) * 1e-6                                  # kN

_SECTION = {"type": "rc_rect", "b_mm": B, "h_mm": H, "concrete": "C", "rebar": "R", "cover_mm": 30,
            "bottom": {"n": 4, "dia_mm": 20}, "top": {"n": 2, "dia_mm": 12},
            "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200}}


def _model(nodes, members, supports, loads, sections=None, extra_cases=None):
    secs = sections or {"S": _SECTION}
    ml = []
    for mid, spec in members.items():
        d = {"id": mid, "i": spec[0], "j": spec[1], "section": spec[2] if len(spec) > 2 else "S"}
        if len(spec) > 3:
            d.update(spec[3])
        ml.append(d)
    cases = [{"id": "G", "type": "permanent", "loads": loads}] + (extra_cases or [])
    return parse_model({
        "materials": {"C": {"type": "concrete", "fck": FCK}, "R": {"type": "rebar", "fyk": 500}},
        "sections": copy.deepcopy(secs),
        "nodes": [{"id": k, "x": v[0], "y": v[1]} for k, v in nodes.items()],
        "members": ml, "supports": supports, "load_cases": cases,
        "design": {"stations_per_member": 40},
    })


def _solve(model):
    fa = FrameAnalysis(model)
    return fa, fa.solve_case(model.load_cases["G"])


def _at(mr, x, q):
    """Value of quantity q at position x (exact station required)."""
    k = int(np.argmin(np.abs(mr.x - x)))
    assert abs(mr.x[k] - x) < 1e-9, f"no station at x = {x}"
    return float(getattr(mr, q)[k])


def _eq(r) -> list[Comparison]:
    return [Comparison(f"equilibrium residual {k}", float(v), 0.0, "kN" if k != "Mz_origin" else "kNm", abs_tol=1e-8)
            for k, v in r.equilibrium.items()]


PIN = {"ux": True, "uy": True}
ROLLER = {"uy": True}
FIX = {"ux": True, "uy": True, "rz": True}


# ================================================================ analysis
@benchmark("A01", "Linear analysis", "Simply supported beam, uniform load",
           "Classical beam theory: M = wL²/8, δ = 5wL⁴/(384EI), θ = wL³/(24EI)")
def _a01():
    L, w = 8.0, 25.0
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}], [{"type": "udl", "member": "M", "w": -w}])
    fa, r = _solve(m)
    mr = r.members["M"]
    return [
        Comparison("R_A", float(r.reactions["A"][1]), w * L / 2, "kN"),
        Comparison("M at midspan", _at(mr, L / 2, "M"), w * L ** 2 / 8, "kNm"),
        Comparison("V at A", _at(mr, 0.0, "V"), w * L / 2, "kN"),
        Comparison("deflection at midspan", _at(mr, L / 2, "v") * 1e3, -5 * w * L ** 4 / (384 * EI) * 1e3, "mm", rel_tol=1e-9),
        Comparison("rotation at A", float(r.displacements[0, 2]), -w * L ** 3 / (24 * EI), "rad"),
    ] + _eq(r)


@benchmark("A02", "Linear analysis", "Simply supported beam, eccentric point load",
           "Classical beam theory: M = Pab/L, δ under the load = Pa²b²/(3EIL)")
def _a02():
    L, P, a = 7.0, 80.0, 2.5
    b = L - a
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}], [{"type": "point", "member": "M", "p": -P, "a": a}])
    fa, r = _solve(m)
    mr = r.members["M"]
    k = int(np.flatnonzero((np.abs(mr.x - a) < 1e-9) & (mr.side < 0))[0])
    return [
        Comparison("R_A", float(r.reactions["A"][1]), P * b / L, "kN"),
        Comparison("M under the load", float(mr.M[k]), P * a * b / L, "kNm"),
        Comparison("V left of the load", float(mr.V[k]), P * b / L, "kN"),
        Comparison("V right of the load", float(mr.V[k + 1]), -P * a / L, "kN"),
        Comparison("deflection under the load", float(mr.v[k]) * 1e3, -P * a ** 2 * b ** 2 / (3 * EI * L) * 1e3, "mm"),
    ] + _eq(r)


@benchmark("A03", "Linear analysis", "Cantilever, tip load",
           "Classical beam theory: δ = PL³/(3EI), θ = PL²/(2EI), M = −PL")
def _a03():
    L, P = 3.0, 40.0
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")}, [{"node": "A", **FIX}],
               [{"type": "nodal", "node": "B", "fy": -P}])
    fa, r = _solve(m)
    mr = r.members["M"]
    return [
        Comparison("tip deflection", float(r.displacements[1, 1]) * 1e3, -P * L ** 3 / (3 * EI) * 1e3, "mm"),
        Comparison("tip rotation", float(r.displacements[1, 2]), -P * L ** 2 / (2 * EI), "rad"),
        Comparison("M at the root", _at(mr, 0.0, "M"), -P * L, "kNm"),
        Comparison("fixed-end moment reaction", float(r.reactions["A"][2]), P * L, "kNm"),
    ] + _eq(r)


@benchmark("A04", "Linear analysis", "Fixed-ended beam, uniform load",
           "Classical beam theory: M_end = −wL²/12, M_mid = wL²/24, δ = wL⁴/(384EI)")
def _a04():
    L, w = 6.0, 30.0
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")},
               [{"node": "A", **FIX}, {"node": "B", **FIX}], [{"type": "udl", "member": "M", "w": -w}])
    fa, r = _solve(m)
    mr = r.members["M"]
    return [
        Comparison("M at the ends", _at(mr, 0.0, "M"), -w * L ** 2 / 12, "kNm"),
        Comparison("M at midspan", _at(mr, L / 2, "M"), w * L ** 2 / 24, "kNm"),
        Comparison("deflection at midspan", _at(mr, L / 2, "v") * 1e3, -w * L ** 4 / (384 * EI) * 1e3, "mm"),
    ] + _eq(r)


@benchmark("A05", "Linear analysis", "Propped cantilever, uniform load",
           "Classical beam theory: R_prop = 3wL/8, M_fixed = −wL²/8, M_max = 9wL²/128 at 3L/8 from the prop")
def _a05():
    L, w = 8.0, 20.0
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")},
               [{"node": "A", **FIX}, {"node": "B", **ROLLER}], [{"type": "udl", "member": "M", "w": -w}])
    fa, r = _solve(m)
    mr = r.members["M"]
    return [
        Comparison("R at the prop", float(r.reactions["B"][1]), 3 * w * L / 8, "kN"),
        Comparison("M at the fixed end", _at(mr, 0.0, "M"), -w * L ** 2 / 8, "kNm"),
        Comparison("M at 3L/8 from the prop", _at(mr, 5 * L / 8, "M"), 9 * w * L ** 2 / 128, "kNm"),
    ] + _eq(r)


@benchmark("A06", "Linear analysis", "Two equal spans, uniform load",
           "Three-moment equation: M_B = −wL²/8, R_B = 10wL/8, R_A = 3wL/8")
def _a06():
    L, w = 5.0, 18.0
    m = _model({"A": (0, 0), "B": (L, 0), "C": (2 * L, 0)}, {"M1": ("A", "B"), "M2": ("B", "C")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}, {"node": "C", **ROLLER}],
               [{"type": "udl", "member": "M1", "w": -w}, {"type": "udl", "member": "M2", "w": -w}])
    fa, r = _solve(m)
    return [
        Comparison("M over the middle support", _at(r.members["M1"], L, "M"), -w * L ** 2 / 8, "kNm"),
        Comparison("R_B", float(r.reactions["B"][1]), 10 * w * L / 8, "kN"),
        Comparison("R_A", float(r.reactions["A"][1]), 3 * w * L / 8, "kN"),
    ] + _eq(r)


@benchmark("A07", "Linear analysis", "Simply supported beam, triangular load",
           "Statics and beam theory: R_A = w₀L/6, R_B = w₀L/3, M_max = w₀L²/(9√3) at L/√3, δ_mid = 5w₀L⁴/(768EI)")
def _a07():
    L, w0 = 6.0, 30.0
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}],
               [{"type": "udl", "member": "M", "w1": 0.0, "w2": -w0}])
    fa, r = _solve(m)
    mr = r.members["M"]
    # M at L/sqrt(3) from statics: R_A x - w0 x^3 / (6L)
    x = L / math.sqrt(3)
    Sx, Sy, My = _cum_dist(x, fa.local_loads(m.load_cases["G"])["M"], 1)
    M_x = -mr.end_forces[2] + x * mr.end_forces[1] + My
    return [
        Comparison("R_A", float(r.reactions["A"][1]), w0 * L / 6, "kN"),
        Comparison("R_B", float(r.reactions["B"][1]), w0 * L / 3, "kN"),
        Comparison("M_max at L/√3", M_x, w0 * L ** 2 / (9 * math.sqrt(3)), "kNm"),
        Comparison("deflection at midspan", _at(mr, L / 2, "v") * 1e3, -5 * w0 * L ** 4 / (768 * EI) * 1e3, "mm"),
    ] + _eq(r)


@benchmark("A08", "Linear analysis", "Simply supported beam, partial uniform load",
           "Statics: load c = b − a with resultant at (a + b)/2")
def _a08():
    L, w, a, b = 9.0, 12.0, 2.0, 5.0
    c = b - a
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}],
               [{"type": "udl", "member": "M", "w": -w, "a": a, "b": b}])
    fa, r = _solve(m)
    mr = r.members["M"]
    RA = w * c * (L - (a + b) / 2) / L
    x = 4.5  # inside the loaded length? no: 4.5 in [2, 5]
    Mx = RA * x - w * (x - a) ** 2 / 2
    return [
        Comparison("R_A", float(r.reactions["A"][1]), RA, "kN"),
        Comparison("M at x = 4.5 m", _at(mr, x, "M"), Mx, "kNm"),
        Comparison("M at x = 7.2 m", _at(mr, 7.2, "M"), RA * 7.2 - w * c * (7.2 - (a + b) / 2), "kNm"),
    ] + _eq(r)


@benchmark("A09", "Linear analysis", "Beam with an internal hinge (Gerber beam)",
           "Statics: the hinge carries V = wL₂/2 into the cantilever; M_A = −(wL₂/2)·L₁")
def _a09():
    L1, L2, w = 3.0, 5.0, 16.0
    m = _model({"A": (0, 0), "B": (L1, 0), "C": (L1 + L2, 0)},
               {"AB": ("A", "B"), "BC": ("B", "C", "S", {"releases": {"i": True}})},
               [{"node": "A", **FIX}, {"node": "C", **ROLLER}], [{"type": "udl", "member": "BC", "w": -w}])
    fa, r = _solve(m)
    return [
        Comparison("R_C", float(r.reactions["C"][1]), w * L2 / 2, "kN"),
        Comparison("M at A", _at(r.members["AB"], 0.0, "M"), -w * L2 / 2 * L1, "kNm"),
        Comparison("M at the hinge", _at(r.members["BC"], 0.0, "M"), 0.0, "kNm", abs_tol=1e-9),
        Comparison("M at midspan of BC", _at(r.members["BC"], L2 / 2, "M"), w * L2 ** 2 / 8, "kNm"),
    ] + _eq(r)


@benchmark("A10", "Linear analysis", "Pinned-base portal frame, horizontal load",
           "Antisymmetric load H/2 at each column head: H_A = H_D = H/2, M at the heads = Hh/2, V_A = −V_D = Hh/L")
def _a10():
    h, L, Hf = 4.0, 6.0, 50.0
    m = _model({"A": (0, 0), "B": (0, h), "C": (L, h), "D": (L, 0)},
               {"c1": ("A", "B"), "bm": ("B", "C"), "c2": ("D", "C")},
               [{"node": "A", **PIN}, {"node": "D", **PIN}],
               [{"type": "nodal", "node": "B", "fx": Hf / 2}, {"type": "nodal", "node": "C", "fx": Hf / 2}])
    fa, r = _solve(m)
    return [
        Comparison("H_A", float(r.reactions["A"][0]), -Hf / 2, "kN"),
        Comparison("V_A", float(r.reactions["A"][1]), -Hf * h / L, "kN"),
        Comparison("|M| at the head of column 1", abs(_at(r.members["c1"], h, "M")), Hf * h / 2, "kNm"),
    ] + _eq(r)


@benchmark("A11", "Linear analysis", "Column, axial load",
           "Hooke's law: Δ = PL/(EA)")
def _a11():
    L, P = 3.5, 900.0
    m = _model({"A": (0, 0), "B": (0, L)}, {"col": ("A", "B")}, [{"node": "A", **FIX}],
               [{"type": "nodal", "node": "B", "fy": -P}])
    fa, r = _solve(m)
    return [
        Comparison("shortening", float(r.displacements[1, 1]) * 1e3, -P * L / EA * 1e3, "mm"),
        Comparison("N in the column", _at(r.members["col"], L / 2, "N"), -P, "kN"),
    ] + _eq(r)


@benchmark("A12", "Linear analysis", "Cantilever with an elastic tip spring",
           "Springs in parallel: δ = H/(3EI/L³ + k)")
def _a12():
    L, Hf, k = 4.0, 30.0, 5000.0
    m = _model({"A": (0, 0), "B": (L, 0)}, {"M": ("A", "B")},
               [{"node": "A", **FIX}, {"node": "B", "ky": k}], [{"type": "nodal", "node": "B", "fy": -Hf}])
    fa, r = _solve(m)
    d = -Hf / (3 * EI / L ** 3 + k)
    return [
        Comparison("tip deflection", float(r.displacements[1, 1]) * 1e3, d * 1e3, "mm"),
        Comparison("spring force", float(r.spring_forces["B"][1]), -k * d, "kN"),
    ] + _eq(r)


@benchmark("A13", "Linear analysis", "Inclined simply supported beam, vertical load",
           "Statics: M_mid = w·L²·cos α / 8 for a load w per metre of member")
def _a13():
    Lh, Hv, w = 6.0, 3.0, 10.0
    L = math.hypot(Lh, Hv)
    m = _model({"A": (0, 0), "B": (Lh, Hv)}, {"M": ("A", "B")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}], [{"type": "udl", "member": "M", "w": -w}])
    fa, r = _solve(m)
    return [
        Comparison("M at midspan", _at(r.members["M"], L / 2, "M"), w * L ** 2 * (Lh / L) / 8, "kNm"),
        Comparison("vertical reaction at B", float(r.reactions["B"][1]), w * L / 2, "kN"),
    ] + _eq(r)


@benchmark("A14", "Linear analysis", "Fixed-base portal with a rigid beam, horizontal load",
           "Shear frame: column end moments = Hh/4 for a rigid beam. Beam stiffness ×10⁴ and column axial strain "
           "are not in the hand formula, so agreement within 0.5 % is expected")
def _a14():
    h, L, Hf = 4.0, 6.0, 40.0
    stiff = dict(_SECTION, stiffness_factor=1e4)
    m = _model({"A": (0, 0), "B": (0, h), "C": (L, h), "D": (L, 0)},
               {"c1": ("A", "B"), "bm": ("B", "C", "R"), "c2": ("D", "C")},
               [{"node": "A", **FIX}, {"node": "D", **FIX}],
               [{"type": "nodal", "node": "B", "fx": Hf / 2}, {"type": "nodal", "node": "C", "fx": Hf / 2}],
               sections={"S": _SECTION, "R": stiff})
    fa, r = _solve(m)
    return [
        Comparison("|M| at the column base", abs(_at(r.members["c1"], 0.0, "M")), Hf * h / 4, "kNm", rel_tol=5e-3),
        Comparison("|M| at the column head", abs(_at(r.members["c1"], h, "M")), Hf * h / 4, "kNm", rel_tol=5e-3),
    ] + _eq(r)


# ============================================================ combinations
@benchmark("C01", "Combinations", "EN 1990 Eq. 6.10 for G, office load and wind",
           "EN 1990 Table A1.1 and A1.2(B): γ_G = 1.35/1.00, γ_Q = 1.5, ψ₀ = 0.7 (cat. B), 0.6 (wind)")
def _c01():
    m = _model({"A": (0, 0), "B": (5, 0)}, {"M": ("A", "B")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}], [{"type": "udl", "member": "M", "w": -10}],
               extra_cases=[{"id": "Q", "type": "variable", "category": "B", "loads": []},
                            {"id": "W", "type": "variable", "category": "wind", "loads": []}])
    cs = generate(m)
    uls = {tuple(round(f, 6) for f in c.factors) for c in cs.combos if c.kind == "ULS"}
    expected = {
        (1.35, 0, 0), (1.0, 0, 0),
        (1.35, 1.5, 0), (1.0, 1.5, 0), (1.35, 0, 1.5), (1.0, 0, 1.5),
        (1.35, 1.5, 0.9), (1.0, 1.5, 0.9), (1.35, 1.05, 1.5), (1.0, 1.05, 1.5),
    }
    qp = {tuple(round(f, 6) for f in c.factors) for c in cs.combos if c.kind == "SLS_qp"}
    return [
        Comparison("number of ULS combinations", len(uls), len(expected)),
        Comparison("ULS set matches the hand list", float(uls == expected), 1.0),
        Comparison("quasi-permanent G + 0.3Q + 0·W present", float((1.0, 0.3, 0.0) in qp), 1.0),
    ]


# ============================================================== EC2
P = EC2Params()


def _sec(**kw):
    d = dict(_SECTION)
    d.update(kw)
    m = _model({"A": (0, 0), "B": (5, 0)}, {"M": ("A", "B")},
               [{"node": "A", **PIN}, {"node": "B", **ROLLER}], [], sections={"S": d})
    return m.sections["S"]


@benchmark("E01", "EC2 materials", "Concrete properties C30/37 and C60/75",
           "EN 1992-1-1 Table 3.1 (tabulated values are rounded)")
def _e01():
    s30 = _sec()
    c30 = concrete_props(s30.concrete, P)
    from ..model import Concrete
    c60 = concrete_props(Concrete("C60/75", 60.0), P)
    return [
        Comparison("f_ctm C30/37", c30.fctm, 2.9, "MPa", abs_tol=0.05),
        Comparison("E_cm C30/37", c30.Ecm / 1000, 33.0, "GPa", abs_tol=0.5),
        Comparison("f_ctm C60/75", c60.fctm, 4.4, "MPa", abs_tol=0.05),
        Comparison("E_cm C60/75", c60.Ecm / 1000, 39.0, "GPa", abs_tol=0.5),
        Comparison("ε_c2 C60/75", c60.eps_c2 * 1e3, 2.3, "‰", abs_tol=0.05),
        Comparison("ε_cu2 C60/75", c60.eps_cu2 * 1e3, 2.9, "‰", abs_tol=0.05),
        Comparison("n C60/75", c60.n, 1.6, "", abs_tol=0.05),
        Comparison("f_cd C30/37", c30.fcd, 20.0, "MPa"),
    ]


@benchmark("E02", "EC2 bending", "Singly reinforced rectangle, pure bending",
           "Closed-form parabola-rectangle block for f_ck ≤ 50: C = (17/21)·b·x·f_cd at (99/238)·x")
def _e02():
    sec = _sec(top=None, links=None)
    rc = RCSection(sec, P)
    As = 4 * math.pi * 20 ** 2 / 4
    d = H - 30 - 10
    fyd = 500 / 1.15
    x = As * fyd / (17 / 21 * B * 20.0)
    MRd = As * fyd * (d - 99 / 238 * x) * 1e-6
    st = rc.solve_state(0.0)
    return [
        Comparison("neutral axis x", st.x, x, "mm", rel_tol=1e-7),
        Comparison("M_Rd", st.M, MRd, "kNm", rel_tol=1e-7),
    ]


@benchmark("E03", "EC2 bending", "Doubly reinforced rectangle against the rectangular stress block",
           "EN 1992-1-1 3.1.7(3): λ = 0.8, η = 1.0; hand iteration on x (independent model, agreement within 1 %)")
def _e03():
    sec = _sec(top={"n": 3, "dia_mm": 16})
    rc = RCSection(sec, P)
    As, As2 = 4 * math.pi * 100, 3 * math.pi * 64
    d, d2 = H - 30 - 8 - 10, 30 + 8 + 8
    fcd, fyd, Es = 20.0, 500 / 1.15, 200_000.0
    lo, hi = 1.0, H
    for _ in range(200):
        x = 0.5 * (lo + hi)
        e2 = 0.0035 * (x - d2) / x
        s2 = max(min(Es * e2, fyd), -fyd)
        F = 0.8 * x * B * fcd + As2 * (s2 - (fcd if e2 > 0 else 0)) - As * fyd
        lo, hi = (lo, x) if F > 0 else (x, hi)
    M = (0.8 * x * B * fcd * (H / 2 - 0.4 * x) + As2 * (s2 - fcd) * (H / 2 - d2) + As * fyd * (d - H / 2)) * 1e-6
    st = rc.solve_state(0.0)
    return [Comparison("M_Rd (parabola-rectangle vs block)", st.M, M, "kNm", rel_tol=0.01)]


@benchmark("E04", "EC2 bending", "Resistance to pure axial force",
           "Strain ε_c2 = 2 ‰ over the section: N = b·h·f_cd + A_s·(E_s·ε_c2 − f_cd); tension: N = −A_s·f_yd")
def _e04():
    sec = _sec()
    rc = RCSection(sec, P)
    As = 4 * math.pi * 100 + 2 * math.pi * 36
    Nmax = (B * H * 20.0 + As * (min(200_000 * 0.002, 500 / 1.15) - 20.0)) * 1e-3
    Nmin = -As * 500 / 1.15 * 1e-3
    return [
        Comparison("N_Rd,max", rc.N_max, Nmax, "kN", rel_tol=1e-9),
        Comparison("N_Rd,min", rc.N_min, Nmin, "kN", rel_tol=1e-9),
    ]


@benchmark("E05", "EC2 shear", "Shear resistance without links",
           "EN 1992-1-1 6.2.2(1) Eq. 6.2; hand calculation for b = 300, d = 450, 4Ø20, C30/37: V_Rd,c = 81.9 kN")
def _e05():
    sec = _sec(h_mm=450 + 30 + 10, top=None, links=None)
    # cover 30 + bar radius 10 -> d = 450
    rc = RCSection(sec, P)
    r = ec2c.shear_resistance(rc, sec, P, np.array([[0.0]]), np.array([[1.0]]))
    return [
        Comparison("d", float(r["d"][0, 0]), 450.0, "mm"),
        Comparison("V_Rd,c", float(r["VRdc"][0, 0]), 81.90, "kN", abs_tol=0.05),
    ]


@benchmark("E06", "EC2 shear", "Shear resistance with links",
           "EN 1992-1-1 6.2.3 Eq. 6.8/6.9; hand calculation: Ø8 2 legs at 200 gives V_Rd,s = 221.3 kN at cot θ = 2.5; "
           "Ø12 4 legs at 100 gives V_Rd,max = 641.5 kN at cot θ = 1")
def _e06():
    out = []
    for links, exp, label in (({"dia_mm": 8, "legs": 2, "spacing_mm": 200}, 221.3, "V_Rd (light links)"),
                              ({"dia_mm": 12, "legs": 4, "spacing_mm": 100}, 641.5, "V_Rd (heavy links)")):
        # d = 450 regardless of the link diameter: explicit bar depth
        sec = _sec(h_mm=490, top=None, links=links, bottom={"n": 4, "dia_mm": 20, "depth_mm": 450})
        rc = RCSection(sec, P)
        r = ec2c.shear_resistance(rc, sec, P, np.array([[0.0]]), np.array([[1.0]]))
        out.append(Comparison(label, float(r["VRd"][0, 0]), exp, "kN", abs_tol=0.1))
    return out


@benchmark("E07", "EC2 serviceability", "Basic span/depth ratios",
           "EN 1992-1-1 Table 7.4N, C30/37, K = 1: 20 for ρ = 0.5 %, 14 for ρ = 1.5 % (tabulated values are rounded)")
def _e07():
    out = []
    for rho, exp in ((0.005, 20.0), (0.015, 14.0)):
        n = 2
        As = rho * B * 450
        dia = math.sqrt(As / n * 4 / math.pi)
        sec = _sec(h_mm=500, top=None, bottom={"n": n, "dia_mm": dia, "depth_mm": 450})
        rc = RCSection(sec, P)
        f = ec2c.MemberForces(np.array([0.0]), 5.0, *[np.zeros((1, 1))] * 3, ["x"], ["x"], np.zeros((1, 1)), ["x"], ["x"])
        member = parse_model_member()
        res = ec2c.check_span_depth(member, rc, sec, f, P)
        out.append(Comparison(f"(l/d)_lim at ρ = {rho * 100:.1f} %", res.capacity, exp, "", abs_tol=0.6))
    return out


def parse_model_member():
    from ..model import Member, MemberDesign
    return Member("M", "A", "B", "S", design=MemberDesign(type="beam", deflection_system="simply_supported"))


@benchmark("E08", "EC2 serviceability", "Deflection of an uncracked beam under sustained load",
           "EN 1992-1-1 7.4.3 with ζ = 0 (M < M_cr): δ = 5wL⁴/(384·E_c,eff·I_I)")
def _e08():
    sec = _sec()
    rc = RCSection(sec, P)
    L, w, phi = 5.0, 6.0, 2.0
    x = np.linspace(0, L, 201)
    M = (w * L / 2 * x - w * x ** 2 / 2)[None, :]
    Eeff = rc.c.Ecm / (1 + phi)
    props = rc.transformed(Eeff)
    Mcr = rc.c.fctm * props["I_I"] / (H - props["y_I"]) * 1e-6
    f = ec2c.MemberForces(x, L, *[np.zeros_like(M)] * 3, ["x"], ["x"], M, ["QP"], ["QP"])
    r = ec2c.deflection_curves(rc, sec, f, phi, P, cantilever=False)
    exact = 5 * w * L ** 4 / (384 * Eeff * 1e3 * props["I_I"] * 1e-12)
    return [
        Comparison("M_max below M_cr", float(w * L ** 2 / 8 < Mcr), 1.0),
        Comparison("midspan deflection", float(np.max(np.abs(r["w"]))) * 1e3, exact * 1e3, "mm", rel_tol=1e-4),
    ]


@benchmark("E09", "EC2 serviceability", "Cracked section properties, singly reinforced",
           "Elastic cracked section: x = d·(−αρ + √((αρ)² + 2αρ)), I_II = b·x³/3 + α·A_s·(d − x)²")
def _e09():
    sec = _sec(top=None)
    rc = RCSection(sec, P)
    Eeff = rc.c.Ecm / 3.0
    a = 200_000 / Eeff
    As = 4 * math.pi * 100
    d = H - 30 - 8 - 10
    rho = As / (B * d)
    x = d * (-a * rho + math.sqrt((a * rho) ** 2 + 2 * a * rho))
    I2 = B * x ** 3 / 3 + a * As * (d - x) ** 2
    p = rc.transformed(Eeff)
    return [
        Comparison("x_II", p["x_II"], x, "mm", rel_tol=1e-9),
        Comparison("I_II", p["I_II"], I2, "mm⁴", rel_tol=1e-9),
    ]


@benchmark("E10", "EC2 detailing", "Minimum tension reinforcement in a beam",
           "EN 1992-1-1 Eq. 9.1N: max(0.26·f_ctm/f_yk, 0.0013)·b·d; C30/37, b = 300, d = 450: 203.3 mm²")
def _e10():
    sec = _sec(h_mm=500, top=None, bottom={"n": 2, "dia_mm": 12, "depth_mm": 450})
    rc = RCSection(sec, P)
    f = ec2c.MemberForces(np.array([0.0]), 5.0, np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)),
                          ["x"], ["x"], np.zeros((1, 1)), ["x"], ["x"])
    res = {c.id: c for c in ec2c.check_detailing(parse_model_member(), rc, sec, f, P)}
    return [Comparison("A_s,min", res["ec2.as_min.bottom"].demand, 203.3, "mm²", abs_tol=0.1)]


def run_all() -> list[tuple[Benchmark, list[Comparison], str | None]]:
    out = []
    for bm in BENCHMARKS:
        try:
            out.append((bm, bm.run(), None))
        except Exception as exc:  # report, never hide
            out.append((bm, [], f"{type(exc).__name__}: {exc}"))
    return out
