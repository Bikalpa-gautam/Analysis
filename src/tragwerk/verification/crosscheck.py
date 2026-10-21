"""Cross-checks against independent solvers (optional dependencies).

* PyNite (MIT licence): pip install PyNiteFEA
* OpenSeesPy: used only as a development-time oracle; check its licence
  before any other use.
"""
from __future__ import annotations

import numpy as np

from ..model import parse_model
from ..solver import FrameAnalysis, analysis_modulus
from .frames import random_frame

_DIR = {"global_y": "FY", "global_x": "FX", "local_y": "Fy", "local_x": "Fx"}


def pynite_model(model):
    from Pynite import FEModel3D
    fe = FEModel3D()
    for nid, n in model.nodes.items():
        fe.add_node(nid, n.x, n.y, 0.0)
        fe.def_support(nid, False, False, True, True, True, False)
    for sid, s in model.sections.items():
        E = analysis_modulus(s) * s.stiffness_factor
        fe.add_material(f"mat_{sid}", E, E / 2.4, 0.2, 0.0)
        fe.add_section(f"sec_{sid}", s.area_m2, s.inertia_m4, s.inertia_m4, 1.0)
    for mid, m in model.members.items():
        fe.add_member(mid, m.i, m.j, f"mat_{m.section}", f"sec_{m.section}")
        if m.release_i or m.release_j:
            fe.def_releases(mid, Rzi=m.release_i, Rzj=m.release_j)
    for nid, sp in model.supports.items():
        fe.def_support(nid, sp.ux, sp.uy, True, True, True, sp.rz)
    lc = model.load_cases["G"]
    for ld in lc.nodal:
        for d, v in (("FX", ld.fx), ("FY", ld.fy), ("MZ", ld.mz)):
            if v:
                fe.add_node_load(ld.node, d, v)
    for ld in lc.distributed:
        fe.add_member_dist_load(ld.member, _DIR[ld.direction], ld.w1, ld.w2, ld.a, ld.b)
    for ld in lc.point:
        fe.add_member_pt_load(ld.member, _DIR[ld.direction], ld.p, ld.a)
    fe.analyze_linear(check_statics=False)
    return fe


def compare_pynite(seed: int) -> dict:
    model = parse_model(random_frame(seed))
    fa = FrameAnalysis(model)
    r = fa.solve_case(model.load_cases["G"])
    fe = pynite_model(model)
    combo = next(iter(fe.load_combos))
    du = max(abs(r.displacements[k, q] - getattr(fe.nodes[nid], ("DX", "DY")[q])[combo])
             for k, nid in enumerate(fa.node_ids) for q in (0, 1))
    su = float(np.max(np.abs(r.displacements[:, :2])))
    dr = max(abs(R[q] - getattr(fe.nodes[nid], ("RxnFX", "RxnFY", "RxnMZ")[q])[combo])
             for nid, R in r.reactions.items() for q in range(3))
    sr = max(float(np.max(np.abs(R))) for R in r.reactions.values())
    dm, sm = 0.0, 0.0
    for mid, mr in r.members.items():
        mem = fe.members[mid]
        for x, side, M in zip(mr.x, mr.side, mr.M):
            if side < 0 and 0 < x < mr.x[-1]:
                continue
            theirs = mem.moment("Mz", float(min(max(x, 1e-9), mr.x[-1] - 1e-9)), combo)
            dm = max(dm, abs(abs(M) - abs(theirs)))
            sm = max(sm, abs(M))
    return {"solver": "PyNite", "seed": seed, "nodes": len(model.nodes), "members": len(model.members),
            "hinges": sum(m.release_i + m.release_j for m in model.members.values()),
            "displacement": du / su, "reaction": dr / sr, "moment": dm / sm}


def compare_opensees(seed: int) -> dict:
    import openseespy.opensees as ops
    model = parse_model(random_frame(seed, releases=False, partial_loads=False))
    fa = FrameAnalysis(model)
    r = fa.solve_case(model.load_cases["G"])
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    tags = {nid: k + 1 for k, nid in enumerate(fa.node_ids)}
    for nid, n in model.nodes.items():
        ops.node(tags[nid], n.x, n.y)
    for nid, sp in model.supports.items():
        ops.fix(tags[nid], int(sp.ux), int(sp.uy), int(sp.rz))
    ops.geomTransf("Linear", 1)
    etag = {}
    for k, (mid, m) in enumerate(model.members.items()):
        s = model.sections[m.section]
        ops.element("elasticBeamColumn", k + 1, tags[m.i], tags[m.j], s.area_m2, analysis_modulus(s), s.inertia_m4, 1)
        etag[mid] = k + 1
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    lc = model.load_cases["G"]
    for ld in lc.nodal:
        ops.load(tags[ld.node], ld.fx, ld.fy, ld.mz)
    for ld in lc.distributed:
        md = fa.members[ld.member]
        qx, qy = (md.s * ld.w1, md.c * ld.w1) if ld.direction == "global_y" else (md.c * ld.w1, -md.s * ld.w1)
        ops.eleLoad("-ele", etag[ld.member], "-type", "-beamUniform", qy, qx)
    for ld in lc.point:
        md = fa.members[ld.member]
        ops.eleLoad("-ele", etag[ld.member], "-type", "-beamPoint", md.c * ld.p, ld.a / md.L, md.s * ld.p)
    for cmd, args in (("system", ("BandGeneral",)), ("numberer", ("RCM",)), ("constraints", ("Plain",)),
                      ("integrator", ("LoadControl", 1.0)), ("algorithm", ("Linear",)), ("analysis", ("Static",))):
        getattr(ops, cmd)(*args)
    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees analysis failed")
    ops.reactions()
    du = max(abs(r.displacements[k, q] - ops.nodeDisp(tags[nid])[q]) for k, nid in enumerate(fa.node_ids) for q in range(2))
    su = float(np.max(np.abs(r.displacements[:, :2])))
    dr = max(abs(R[q] - ops.nodeReaction(tags[nid])[q]) for nid, R in r.reactions.items() for q in range(3))
    sr = max(float(np.max(np.abs(R))) for R in r.reactions.values())
    ops.wipe()
    return {"solver": "OpenSees", "seed": seed, "nodes": len(model.nodes), "members": len(model.members), "hinges": 0,
            "displacement": du / su, "reaction": dr / sr, "moment": None}


def run_crosschecks(seeds=range(12)) -> list[dict]:
    out = []
    for fn, mod in ((compare_pynite, "Pynite"), (compare_opensees, "openseespy")):
        try:
            __import__(mod)
        except ImportError:
            continue
        for s in seeds:
            out.append(fn(s))
    return out
