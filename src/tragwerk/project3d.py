"""Run a 3D project: space-frame analysis, combinations, envelopes and EC2 checks."""
from __future__ import annotations

import math
import time

import numpy as np

from . import __version__
from .codes import sia262
from .codes.ec2 import checks3d
from .codes.ec2.params import make_params
from .combinations import generate
from .model3d import Model3D
from .project import Envelope, ProjectResult, _arr, _plain
from .solver3d import FrameAnalysis3D

QUANTITIES = ("N", "Vy", "Vz", "T", "My", "Mz")
COMBO_KEYS = ("gamma_G_sup", "gamma_G_inf", "gamma_Q", "xi", "psi", "vary_permanent_individually")


def _forces(res: ProjectResult, mid: str) -> checks3d.MemberForces3D:
    F, uls = res.combos.matrix("ULS")
    Fq, qp = res.combos.matrix("SLS_qp")
    st = lambda q: np.stack([getattr(res.cases[c].members[mid], q) for c in res.combos.case_ids])  # noqa: E731
    md = res.analysis.members[mid]
    m = res.model.members[mid]
    tip_at_i = False
    if m.design.deflection_system == "cantilever":
        def is_free(nid):
            others = [o for o in res.model.members.values() if o.id != mid and nid in (o.i, o.j)]
            return nid not in res.model.supports and not others
        tip_at_i = is_free(m.i) and not is_free(m.j)
    My = st("My")
    return checks3d.MemberForces3D(
        x=res.analysis.stations[mid][0], L=md.L,
        N=F @ st("N"), Vy=F @ st("Vy"), Vz=F @ st("Vz"), T=F @ st("T"), My=F @ My, Mz=F @ st("Mz"),
        uls_ids=[c.id for c in uls], uls_labels=[c.label for c in uls],
        qp_My=Fq @ My, qp_ids=[c.id for c in qp], qp_labels=[c.label for c in qp], tip_at_i=tip_at_i,
        qp_Mz=Fq @ st("Mz"))


def run3d(model: Model3D) -> ProjectResult:
    t0 = time.perf_counter()
    fa = FrameAnalysis3D(model)
    cases = {cid: fa.solve_case(lc) for cid, lc in model.load_cases.items()}
    combos = generate(model)
    params = make_params({k: v for k, v in model.settings.params.items() if k not in COMBO_KEYS})
    res = ProjectResult(model, fa, cases, combos, params, {}, {}, {})
    warnings = []
    names = ("X", "Y", "Z")
    for d in fa.auto_fixed:
        warnings.append(f"Node '{fa.node_ids[d // 6]}': rotation about {names[d % 6 - 3]} has no stiffness (released on every member) and was restrained automatically.")
    for cid, cr in cases.items():
        scale = max(1.0, *(abs(v) for v in cr.applied.values()))
        if max(abs(v) for v in cr.equilibrium.values()) > 1e-6 * scale:
            warnings.append(f"Load case {cid}: equilibrium residual larger than expected.")
    F, uls = combos.matrix("ULS")
    for mid in model.members:
        x, side = fa.stations[mid]
        mx, mn, mxc, mnc = {}, {}, {}, {}
        for q in QUANTITIES:
            C = F @ np.stack([getattr(cases[c].members[mid], q) for c in combos.case_ids])
            mx[q], mn[q] = C.max(axis=0), C.min(axis=0)
            mxc[q] = [uls[i].id for i in np.argmax(C, axis=0)]
            mnc[q] = [uls[i].id for i in np.argmin(C, axis=0)]
        res.envelopes[mid] = Envelope(x, side, mx, mn, mxc, mnc)
    for nid in model.supports:
        R = np.stack([cases[c].reactions[nid] for c in combos.case_ids])
        RU = F @ R
        res.reactions[nid] = {
            "per_case": {c: cases[c].reactions[nid].tolist() for c in combos.case_ids},
            "uls_max": RU.max(axis=0).tolist(), "uls_min": RU.min(axis=0).tolist(),
            "uls_max_combo": [uls[i].id for i in RU.argmax(axis=0)],
            "uls_min_combo": [uls[i].id for i in RU.argmin(axis=0)],
        }
    cache: dict[str, checks3d.SectionModels] = {}
    for mid, m in model.members.items():
        if m.section not in cache:
            cache[m.section] = checks3d.section_models(model.sections[m.section], params)
        if model.settings.design_code == "EC2":
            res.checks[mid] = checks3d.check_member3d(m, cache[m.section], _forces(res, mid), params)
        else:
            res.checks[mid] = sia262.check_member(m, None, None, None, params)
    res.warnings = warnings
    res.runtime_s = time.perf_counter() - t0
    return res


def to_dict3d(res: ProjectResult) -> dict:
    m = res.model
    gov = res.governing()
    out = {
        "tragwerk_version": __version__, "model": m.name, "dimension": 3, "design_code": m.settings.design_code,
        "summary": {
            "status": res.overall_status(),
            "max_utilisation": gov.utilisation if gov and gov.utilisation is not None and math.isfinite(gov.utilisation) else None,
            "governing": None if gov is None else {"member": gov.member, "check": gov.title, "clause": gov.clause},
            "n_combinations": {k: sum(1 for c in res.combos.combos if c.kind == k) for k in ("ULS", "SLS_char", "SLS_freq", "SLS_qp")},
            "runtime_s": round(res.runtime_s, 4),
        },
        "warnings": res.warnings,
        "load_cases": {cid: {
            "displacements": {nid: _arr(cr.displacements[k]) for k, nid in enumerate(res.analysis.node_ids)},
            "reactions": {nid: _arr(r) for nid, r in cr.reactions.items()},
            "equilibrium_residual": {k: float(v) for k, v in cr.equilibrium.items()},
        } for cid, cr in res.cases.items()},
        "combinations": [{"id": c.id, "kind": c.kind, "label": c.label, "leading": c.leading,
                          "factors": dict(zip(res.combos.case_ids, c.factors))} for c in res.combos.combos],
        "envelopes": {mid: {"x": _arr(e.x), **{f"{q}_max": _arr(e.max[q]) for q in QUANTITIES},
                            **{f"{q}_min": _arr(e.min[q]) for q in QUANTITIES}} for mid, e in res.envelopes.items()},
        "reactions": res.reactions,
        "checks": {mid: [c.to_dict() for c in lst] for mid, lst in res.checks.items()},
    }
    return _plain(out)
