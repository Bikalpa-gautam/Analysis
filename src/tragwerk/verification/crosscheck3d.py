"""3D cross-checks against PyNite (MIT) and OpenSees (development-time oracle only)."""
from __future__ import annotations

import numpy as np

from ..model3d import parse_model3d
from ..solver3d import NU_CONCRETE, FrameAnalysis3D, analysis_modulus3d
from .frames3d import random_frame3d

_D = {"global_x": "FX", "global_y": "FY", "global_z": "FZ"}


def compare_pynite3d(seed: int) -> dict:
    from Pynite import FEModel3D
    model = parse_model3d(random_frame3d(seed))
    fa = FrameAnalysis3D(model)
    r = fa.solve_case(model.load_cases["G"])
    fe = FEModel3D()
    for nid, n in model.nodes.items():
        fe.add_node(nid, n.x, n.y, n.z)
    for sid, s in model.sections.items():
        E = analysis_modulus3d(s)
        fe.add_material(f"m{sid}", E, E / (2 * (1 + NU_CONCRETE)), NU_CONCRETE, 0.0)
        fe.add_section(f"s{sid}", s.area_m2, s.Iz_m4, s.Iy_m4, s.J_m4)
    for mid, m in model.members.items():
        fe.add_member(mid, m.i, m.j, f"m{m.section}", f"s{m.section}")
        if m.rel_i.my or m.rel_j.my:
            fe.def_releases(mid, Ryi=m.rel_i.my, Rzi=m.rel_i.mz, Ryj=m.rel_j.my, Rzj=m.rel_j.mz)
    for nid, sp in model.supports.items():
        fe.def_support(nid, *sp.fix)
    lc = model.load_cases["G"]
    for ld in lc.nodal:
        for d, v in zip(("FX", "FY", "FZ", "MX", "MY", "MZ"), ld.f):
            if v:
                fe.add_node_load(ld.node, d, v)
    for ld in lc.distributed:
        fe.add_member_dist_load(ld.member, _D[ld.direction], ld.w1, ld.w2, ld.a, ld.b)
    for ld in lc.point:
        fe.add_member_pt_load(ld.member, _D[ld.direction], ld.p, ld.a)
    fe.analyze_linear(check_statics=False)
    c = next(iter(fe.load_combos))
    su = float(np.max(np.abs(r.displacements[:, :3])))
    sr = float(np.max(np.abs(r.displacements[:, 3:])))
    du = 0.0
    for k, nid in enumerate(fa.node_ids):
        for q, key in enumerate(("DX", "DY", "DZ", "RX", "RY", "RZ")):
            if 6 * k + q in fa.auto_fixed:
                continue
            du = max(du, abs(r.displacements[k, q] - getattr(fe.nodes[nid], key)[c]) / (su if q < 3 else sr))
    sf = max(float(np.max(np.abs(R))) for R in r.reactions.values())
    dr = max(abs(R[q] - getattr(fe.nodes[nid], key)[c]) for nid, R in r.reactions.items()
             for q, key in enumerate(("RxnFX", "RxnFY", "RxnFZ", "RxnMX", "RxnMY", "RxnMZ"))) / sf
    return {"solver": "PyNite 3D", "seed": seed, "nodes": len(model.nodes), "members": len(model.members),
            "hinges": sum(m.rel_i.my + m.rel_j.my for m in model.members.values()), "displacement": du, "reaction": dr, "moment": None}


def rectangular_frame3d(seed: int) -> dict:
    """Random frame with rectangular sections and rolled members (orientation matters)."""
    import random
    d = random_frame3d(seed, releases=False, partial_loads=False)
    rnd = random.Random(1000 + seed)
    d["sections"]["COL"] = dict(d["sections"]["COL"], b_mm=300, h_mm=500)
    d["sections"]["BEAM"] = dict(d["sections"]["BEAM"], b_mm=300, h_mm=600)
    for m in d["members"]:
        if rnd.random() < 0.3:
            m["roll_deg"] = rnd.choice([30.0, 90.0, -45.0])
    return d


def compare_opensees3d(seed: int) -> dict:
    import openseespy.opensees as ops
    model = parse_model3d(rectangular_frame3d(seed))
    fa = FrameAnalysis3D(model)
    r = fa.solve_case(model.load_cases["G"])
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    tags = {nid: k + 1 for k, nid in enumerate(fa.node_ids)}
    for nid, n in model.nodes.items():
        ops.node(tags[nid], n.x, n.y, n.z)
    for nid, sp in model.supports.items():
        ops.fix(tags[nid], *[int(f) for f in sp.fix])
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    etag = {}
    for k, (mid, m) in enumerate(model.members.items()):
        md = fa.members[mid]
        s = model.sections[m.section]
        E = analysis_modulus3d(s)
        ops.geomTransf("Linear", k + 1, *md.lam[2])
        ops.element("elasticBeamColumn", k + 1, tags[m.i], tags[m.j], s.area_m2, E, E / (2 * (1 + NU_CONCRETE)),
                    s.J_m4, s.Iy_m4, s.Iz_m4, k + 1)
        etag[mid] = k + 1
    lc = model.load_cases["G"]
    for ld in lc.nodal:
        ops.load(tags[ld.node], *ld.f)
    for ld in lc.distributed:
        md = fa.members[ld.member]
        g = np.zeros(3)
        g["xyz".index(ld.direction[-1])] = ld.w1
        q = md.lam @ g
        ops.eleLoad("-ele", etag[ld.member], "-type", "-beamUniform", q[1], q[2], q[0])
    for ld in lc.point:
        md = fa.members[ld.member]
        g = np.zeros(3)
        g["xyz".index(ld.direction[-1])] = ld.p
        q = md.lam @ g
        ops.eleLoad("-ele", etag[ld.member], "-type", "-beamPoint", q[1], q[2], ld.a / md.L, q[0])
    for cmd, args in (("system", ("BandGeneral",)), ("numberer", ("RCM",)), ("constraints", ("Plain",)),
                      ("integrator", ("LoadControl", 1.0)), ("algorithm", ("Linear",)), ("analysis", ("Static",))):
        getattr(ops, cmd)(*args)
    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees analysis failed")
    ops.reactions()
    su = float(np.max(np.abs(r.displacements[:, :3])))
    du = max(abs(r.displacements[k, q] - ops.nodeDisp(tags[nid])[q]) for k, nid in enumerate(fa.node_ids) for q in range(3)) / su
    sf = max(float(np.max(np.abs(R))) for R in r.reactions.values())
    dr = max(abs(R[q] - ops.nodeReaction(tags[nid])[q]) for nid, R in r.reactions.items() for q in range(6)) / sf
    ops.wipe()
    return {"solver": "OpenSees 3D", "seed": seed, "nodes": len(model.nodes), "members": len(model.members), "hinges": 0,
            "displacement": du, "reaction": dr, "moment": None}


def run_crosschecks3d(seeds=range(8)) -> list[dict]:
    out = []
    for fn, mod in ((compare_pynite3d, "Pynite"), (compare_opensees3d, "openseespy")):
        try:
            __import__(mod)
        except ImportError:
            continue
        out += [fn(s) for s in seeds]
    return out
