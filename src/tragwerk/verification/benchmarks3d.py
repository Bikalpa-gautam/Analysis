"""3D verification benchmarks (registered in the same list as the 2D ones)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from ..building import BuildingSpec, _tributary, generate_building
from ..codes.ec2 import checks3d
from ..codes.ec2.params import EC2Params
from ..model import load_model
from ..model3d import parse_model3d
from ..project import run
from ..project3d import run3d
from ..solver3d import NU_CONCRETE, FrameAnalysis3D, analysis_modulus3d
from .benchmarks import Comparison, benchmark

FCK = 30.0
E = 22_000.0 * ((FCK + 8) / 10) ** 0.3 * 1000.0
G = E / (2 * (1 + NU_CONCRETE))
MAT = {"C": {"type": "concrete", "fck": FCK}, "R": {"type": "rebar", "fyk": 500}}
RECT = {"type": "rc_rect", "b_mm": 300, "h_mm": 500, "concrete": "C", "rebar": "R", "cover_mm": 30,
        "top": {"n": 3, "dia_mm": 20}, "bottom": {"n": 3, "dia_mm": 20}, "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200}}
FIX6 = {"ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True}


def _m3(nodes, members, supports, loads, sections=None, extra=None):
    return parse_model3d({
        "schema": "tragwerk3d/0.1", "materials": MAT, "sections": copy.deepcopy(sections or {"S": RECT}),
        "nodes": [{"id": k, "x": v[0], "y": v[1], "z": v[2]} for k, v in nodes.items()],
        "members": [dict({"id": k, "i": v[0], "j": v[1], "section": "S"}, **(v[2] if len(v) > 2 else {})) for k, v in members.items()],
        "supports": supports, "load_cases": [{"id": "G", "type": "permanent", "loads": loads}] + (extra or []),
        "design": {"stations_per_member": 20},
    })


def _solve(m):
    fa = FrameAnalysis3D(m)
    return fa, fa.solve_case(m.load_cases["G"])


def _eq(r):
    return [Comparison(f"equilibrium residual {k}", float(v), 0.0, abs_tol=1e-8) for k, v in r.equilibrium.items()]


B, H = 300.0, 500.0
Iy = B * H ** 3 / 12 * 1e-12
Iz = H * B ** 3 / 12 * 1e-12
J = H * B ** 3 * (1 / 3 - 0.21 * (B / H) * (1 - B ** 4 / (12 * H ** 4))) * 1e-12


@benchmark("S01", "3D analysis", "Cantilever beam in space: loads in both transverse directions and a torque",
           "Beam theory: δ_z = P_zL³/(3EI_y), δ_y = P_yL³/(3EI_z), twist = TL/(GJ); 300 × 500 section, h along global Z")
def _s01():
    L, Py, Pz, T = 3.0, 12.0, -30.0, 4.0
    m = _m3({"A": (0, 0, 0), "B": (L, 0, 0)}, {"M": ("A", "B")}, [{"node": "A", **FIX6}],
            [{"type": "nodal", "node": "B", "fy": Py, "fz": Pz, "mx": T}])
    fa, r = _solve(m)
    k = fa.nidx["B"]
    mr = r.members["M"]
    return [
        Comparison("δ_z at the tip", r.displacements[k, 2] * 1e3, Pz * L ** 3 / (3 * E * Iy) * 1e3, "mm"),
        Comparison("δ_y at the tip", r.displacements[k, 1] * 1e3, Py * L ** 3 / (3 * E * Iz) * 1e3, "mm"),
        Comparison("twist at the tip", r.displacements[k, 3], T * L / (G * J), "rad"),
        Comparison("M_y at the root (hogging)", float(mr.My[0]), Pz * L, "kNm"),
        Comparison("M_z at the root", float(mr.Mz[0]), Py * L, "kNm"),
        Comparison("torque in the member", float(mr.T[5]), T, "kNm"),
    ] + _eq(r)


@benchmark("S02", "3D analysis", "Vertical cantilever column, horizontal loads in X and Y",
           "Beam theory: for a vertical member local z is global X, so δ_x = P_xL³/(3EI_y) and δ_y = P_yL³/(3EI_z)")
def _s02():
    L, Px, Py, P = 4.0, 20.0, 15.0, -800.0
    m = _m3({"A": (0, 0, 0), "B": (0, 0, L)}, {"C": ("A", "B")}, [{"node": "A", **FIX6}],
            [{"type": "nodal", "node": "B", "fx": Px, "fy": Py, "fz": P}])
    fa, r = _solve(m)
    k = fa.nidx["B"]
    A = B * H * 1e-6
    return [
        Comparison("δ_x at the top", r.displacements[k, 0] * 1e3, Px * L ** 3 / (3 * E * Iy) * 1e3, "mm"),
        Comparison("δ_y at the top", r.displacements[k, 1] * 1e3, Py * L ** 3 / (3 * E * Iz) * 1e3, "mm"),
        Comparison("shortening", r.displacements[k, 2] * 1e3, P * L / (E * A) * 1e3, "mm"),
        Comparison("base reaction moment about Y", float(r.reactions["A"][4]), -Px * L, "kNm"),
        Comparison("base reaction moment about X", float(r.reactions["A"][3]), Py * L, "kNm"),
    ] + _eq(r)


@benchmark("S03", "3D analysis", "Torsion constant of a rectangle",
           "Timoshenko & Goodier: J = β·a·b³ with β = 0.229 for a/b = 2 and 0.196 for a/b = 1.5")
def _s03():
    from ..model3d import RCSection3D
    out = []
    for a, b, beta in ((600.0, 300.0, 0.229), (450.0, 300.0, 0.196)):
        sec = parse_model3d({"schema": "tragwerk3d/0.1", "materials": MAT,
                             "sections": {"S": dict(RECT, b_mm=b, h_mm=a)},
                             "nodes": [{"id": "A", "x": 0, "y": 0, "z": 0}, {"id": "B", "x": 1, "y": 0, "z": 0}],
                             "members": [{"id": "M", "i": "A", "j": "B", "section": "S"}],
                             "supports": [{"node": "A", **FIX6}], "load_cases": [{"id": "G", "type": "permanent", "loads": []}]}).sections["S"]
        out.append(Comparison(f"β for a/b = {a / b:g}", sec.J_m4 / (a * b ** 3 * 1e-12), beta, "", abs_tol=0.002))
    return out


@benchmark("S04", "3D analysis", "Grillage: two crossing beams sharing a midpoint load",
           "Compatibility at the crossing: the share carried by beam X is P·(L_y³/I_y,Y)/(L_x³/I_y,X + L_y³/I_y,Y) for simply supported beams")
def _s04():
    Lx, Ly, P = 8.0, 6.0, -100.0
    m = _m3({"X0": (0, Ly / 2, 0), "X1": (Lx, Ly / 2, 0), "Y0": (Lx / 2, 0, 0), "Y1": (Lx / 2, Ly, 0), "C": (Lx / 2, Ly / 2, 0)},
            {"BX1": ("X0", "C", {"releases": {"i": {"my": True, "mz": True, "t": False}}}), "BX2": ("C", "X1", {"releases": {"j": {"my": True, "mz": True}}}),
             "BY1": ("Y0", "C", {"releases": {"i": {"my": True, "mz": True}}}), "BY2": ("C", "Y1", {"releases": {"j": {"my": True, "mz": True}}})},
            [{"node": "X0", "ux": True, "uy": True, "uz": True, "rx": True}, {"node": "X1", "uy": True, "uz": True, "rx": True},
             {"node": "Y0", "ux": True, "uy": True, "uz": True, "rx": True}, {"node": "Y1", "ux": True, "uz": True, "rx": True}],
            [{"type": "nodal", "node": "C", "fz": P}])
    fa, r = _solve(m)
    fx = (Ly ** 3) / (Lx ** 3 + Ly ** 3)      # same section in both directions
    return [
        Comparison("share of beam X (sum of its support reactions)", float(r.reactions["X0"][2] + r.reactions["X1"][2]), -P * fx, "kN"),
        Comparison("deflection at the crossing", r.displacements[fa.nidx["C"], 2] * 1e3, P * fx * Lx ** 3 / (48 * E * Iy) * 1e3, "mm"),
    ] + _eq(r)


@benchmark("S05", "3D analysis", "A plane frame solved in 3D equals the 2D solver",
           "Consistency: the portal frame example rebuilt in the X-Z plane must give the 2D moments, shears and reactions")
def _s05():
    root = Path(__file__).resolve().parents[3] / "examples"
    d2 = json.loads((root / "portal_frame.json").read_text())
    m2 = load_model(root / "portal_frame.json")
    r2 = run(m2)
    d3 = {"schema": "tragwerk3d/0.1", "name": "portal 3D", "materials": d2["materials"], "sections": {},
          "nodes": [{"id": n["id"], "x": n["x"], "y": 0.0, "z": n["y"]} for n in d2["nodes"]],
          "members": d2["members"], "load_cases": [], "design": dict(d2["design"], stations_per_member=40)}
    for sid, s in d2["sections"].items():
        d3["sections"][sid] = {k: v for k, v in s.items()}
    d3["supports"] = [{"node": s["node"], "ux": s.get("ux", False), "uy": True, "uz": s.get("uy", False), "rx": True,
                       "ry": s.get("rz", False), "rz": True} for s in d2["supports"]]
    for lc in d2["load_cases"]:
        lc3 = {k: v for k, v in lc.items() if k != "loads"}
        lc3["loads"] = []
        for ld in lc["loads"]:
            ld3 = dict(ld)
            if ld["type"] == "nodal":
                ld3 = {"type": "nodal", "node": ld["node"], "fx": ld.get("fx", 0.0), "fz": ld.get("fy", 0.0), "my": -ld.get("mz", 0.0)}
            else:
                ld3["direction"] = {"global_x": "global_x", "global_y": "global_z"}[ld.get("direction", "global_y")]
            lc3["loads"].append(ld3)
        d3["load_cases"].append(lc3)
    m3 = parse_model3d(d3)
    fa3 = FrameAnalysis3D(m3)
    worst_M = worst_V = worst_R = 0.0
    for cid, lc in m3.load_cases.items():
        c3 = fa3.solve_case(lc)
        c2 = r2.cases[cid]
        for mid in m2.members:
            L2, cs, sn = m2.member_geometry(mid)
            sign = 1.0 if abs(sn) < 0.999 else -1.0     # vertical members: 2D local y = -X, 3D local z = +X
            worst_M = max(worst_M, float(np.max(np.abs(sign * c3.members[mid].My - c2.members[mid].M))))
            worst_V = max(worst_V, float(np.max(np.abs(sign * c3.members[mid].Vz - c2.members[mid].V))))
        for nid in m2.supports:
            R2, R3 = c2.reactions[nid], c3.reactions[nid]
            worst_R = max(worst_R, abs(R2[0] - R3[0]), abs(R2[1] - R3[2]), abs(R2[2] + R3[4]))
    return [Comparison("largest moment difference", worst_M, 0.0, "kNm", abs_tol=1e-8),
            Comparison("largest shear difference", worst_V, 0.0, "kN", abs_tol=1e-8),
            Comparison("largest reaction difference", worst_R, 0.0, "kN", abs_tol=1e-8)]


@benchmark("S06", "Building generator", "Load takedown of the generated office building",
           "Statics: every area load reaches the foundations, ΣR_z = q × floor area × number of floors; "
           "wind: ΣR_x = w × facade width × (H − h₁/2), the lower half of the ground storey goes straight to the foundation")
def _s06():
    spec = BuildingSpec()
    m = parse_model3d(generate_building(spec))
    fa = FrameAnalysis3D(m)
    area = sum(spec.spans_x) * sum(spec.spans_y)
    n_occ = len(spec.storeys) - 1
    out = []
    for cid, q, floors in (("Q", spec.imposed, n_occ), ("S", spec.snow, 1)):
        r = fa.solve_case(m.load_cases[cid])
        out.append(Comparison(f"ΣR_z, load case {cid}", sum(float(R[2]) for R in r.reactions.values()), q * area * floors, "kN"))
    r = fa.solve_case(m.load_cases["W+X"])
    H = spec.wind * (sum(spec.storeys) - spec.storeys[0] / 2) * sum(spec.spans_y)
    out.append(Comparison("ΣR_x, wind in +X", -sum(float(R[0]) for R in r.reactions.values()), H, "kN"))
    return out


@benchmark("S07", "Building generator", "Two-way slab load distribution (45° lines)",
           "Yield-line tributary areas: long beams take trapezoids, short beams triangles; the four shares sum to q·L_x·L_y")
def _s07():
    q, Lx, Ly = 5.0, 7.5, 5.0
    area = lambda segs: sum((b - a) * (w1 + w2) / 2 for a, b, w1, w2 in segs)  # noqa: E731
    long_ = area(_tributary(Lx, Ly, q, "two_way"))
    short = area(_tributary(Ly, Lx, q, "two_way"))
    return [
        Comparison("trapezoid on a long beam", long_, q * Ly / 2 * (Lx - Ly / 2), "kN"),
        Comparison("triangle on a short beam", short, q * Ly ** 2 / 4, "kN"),
        Comparison("total over the four beams", 2 * long_ + 2 * short, q * Lx * Ly, "kN"),
    ]


@benchmark("S08", "EC2 3D checks", "Biaxial bending exponent",
           "EN 1992-1-1 5.8.9(4): a = 1.0, 1.5, 2.0 at N_Ed/N_Rd = 0.1, 0.7, 1.0, linear in between")
def _s08():
    f = checks3d.biaxial_exponent
    return [Comparison("a at 0.1", f(0.1), 1.0), Comparison("a at 0.4", f(0.4), 1.25),
            Comparison("a at 0.7", f(0.7), 1.5), Comparison("a at 0.85", f(0.85), 1.75), Comparison("a at 1.0", f(1.0), 2.0)]


@benchmark("S09", "EC2 3D checks", "Torsion: strut resistance of a rectangle",
           "EN 1992-1-1 Eq. 6.30, hand calculation: 300 × 500, C30/37, cover 30, Ø8 links, Ø20 bars, cot θ = 1: T_Rd,max = 83.55 kNm")
def _s09():
    m = _m3({"A": (0, 0, 0), "B": (2.0, 0, 0)}, {"M": ("A", "B", {"design": {"type": "beam"}})}, [{"node": "A", **FIX6}],
            [{"type": "nodal", "node": "B", "mx": 30.0}])
    sec = m.sections["S"]
    p = EC2Params()
    S = checks3d.section_models(sec, p)
    fa, r = _solve(m)
    mr = r.members["M"]
    f = checks3d.MemberForces3D(mr.x, 2.0, *[getattr(mr, q)[None, :] for q in ("N", "Vy", "Vz", "T", "My", "Mz")],
                                ["U"], ["u"], mr.My[None, :], ["Q"], ["q"])
    # force cot(theta) = 1 by giving the links a very large capacity is not needed: compute directly
    c = S.rc_y.c
    tef = max(300 * 500 / 1600, 2 * (30 + 8 + 10))
    Ak = (300 - tef) * (500 - tef)
    TRdmax_cot1 = 2 * c.nu * c.fcd * Ak * tef * 0.5 * 1e-6
    res = checks3d.check_torsion(m.members["M"], S, f, p)
    tef_step = next(s.value for s in res.steps if s.symbol == "t_{ef}")
    # the check uses cot(theta) from the shear design (links Ø8 x 2 legs @ 200, d = 452 mm)
    z = 0.9 * 452
    a_ = 2 * math.pi * 16 / 200 * z * 500 / 1.15
    c_ = 300 * z * c.nu * c.fcd
    cot = min(max(math.sqrt(max(c_ / a_ - 1, 0)), 1.0), 2.5)
    TRdmax = 2 * c.nu * c.fcd * Ak * tef * cot / (1 + cot ** 2) * 1e-6
    return [Comparison("t_ef", tef_step, 96.0, "mm"),
            Comparison("T_Rd,max at cot θ = 1, hand value", TRdmax_cot1, 83.55, "kNm", abs_tol=0.01),
            Comparison("T_Rd,max used in the check", res.capacity, TRdmax, "kNm", rel_tol=1e-9),
            Comparison("torque in the check", res.demand, 30.0, "kNm")]


@benchmark("S10", "EC2 3D checks", "Nominal curvature: second-order moment of a slender column",
           "EN 1992-1-1 5.8.8, hand calculation: 300 × 300, 4Ø20, C30/37, l₀ = 5 m, N_Ed = 600 kN, φ_ef = 0: "
           "K_r = 1, 1/r = ε_yd/(0.45·d), e₂ = (1/r)·l₀²/10")
def _s10():
    sec = dict(RECT, b_mm=300, h_mm=300, top={"n": 2, "dia_mm": 20}, bottom={"n": 2, "dia_mm": 20}, cover_mm=30)
    L = 5.0
    m = _m3({"A": (0, 0, 0), "B": (0, 0, L)}, {"C": ("A", "B", {"design": {"type": "column", "creep_coefficient": 0.0}})},
            [{"node": "A", "ux": True, "uy": True, "uz": True, "rz": True},
             {"node": "B", "ux": True, "uy": True}],
            [{"type": "nodal", "node": "B", "fz": -600.0}], sections={"S": sec})
    p = EC2Params()
    S = checks3d.section_models(m.sections["S"], p)
    fa, r = _solve(m)
    mr = r.members["C"]
    f = checks3d.MemberForces3D(mr.x, L, *[getattr(mr, q)[None, :] for q in ("N", "Vy", "Vz", "T", "My", "Mz")],
                                ["U"], ["u"], mr.My[None, :], ["Q"], ["q"], qp_Mz=mr.Mz[None, :])
    res = checks3d.check_second_order(m.members["C"], S, f, p)
    # hand calculation, x-z plane
    z_bar = 150 - 30 - 8 - 10                   # 102 mm from the centroid
    i_s = z_bar                                  # all bars at +/- 102 mm about this axis
    d = 150 + i_s
    As = 4 * math.pi * 100
    omega = As * 500 / 1.15 / (300 * 300 * 20)
    n = 600e3 / (300 * 300 * 20)
    Kr = min((1 + omega - n) / (1 + omega - 0.4), 1.0)
    inv_r = Kr * (500 / 1.15 / 200000) / (0.45 * d)
    e2 = inv_r * 5000 ** 2 / 10
    M2 = 600 * e2 * 1e-3
    step = {s.symbol: s.value for s in res.steps}
    MEd = max(abs(step["M_{Ed,y}"]), abs(step["M_{Ed,z}"]))
    return [Comparison("K_r", step["K_{r,y}"], Kr, "", rel_tol=1e-9),
            Comparison("e₂", step["e_{2,y}"], e2, "mm", rel_tol=1e-9),
            Comparison("M₂", step["M_{2,y}"], M2, "kNm", rel_tol=1e-9),
            Comparison("M_Ed = N·e_i + M₂ (pinned ends, no first-order moment)", MEd,
                       max(600 * 5000 / 400 * 1e-3 + M2, 600 * 20 * 1e-3), "kNm", rel_tol=1e-9)]
