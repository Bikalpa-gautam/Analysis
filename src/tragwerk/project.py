"""Run a whole project: analysis, combinations, envelopes and code checks."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .codes.base import FAIL, NOT_COVERED, PASS, WARN, CheckResult
from .codes.ec2 import checks as ec2_checks
from .codes.ec2.params import EC2Params, make_params
from .codes.ec2.section import RCSection
from .codes import sia262
from .combinations import CombinationSet, generate
from .model import Model
from .solver import CaseResult, FrameAnalysis


@dataclass
class Envelope:
    x: np.ndarray
    side: np.ndarray
    max: dict[str, np.ndarray]
    min: dict[str, np.ndarray]
    max_combo: dict[str, list[str]]
    min_combo: dict[str, list[str]]


@dataclass
class ProjectResult:
    model: Model
    analysis: FrameAnalysis
    cases: dict[str, CaseResult]
    combos: CombinationSet
    params: EC2Params
    checks: dict[str, list[CheckResult]]
    envelopes: dict[str, Envelope]
    reactions: dict[str, dict]
    warnings: list[str] = field(default_factory=list)
    runtime_s: float = 0.0

    # ---------------------------------------------------------- summaries
    def all_checks(self) -> list[CheckResult]:
        return [c for lst in self.checks.values() for c in lst]

    def governing(self) -> CheckResult | None:
        best, bu = None, -1.0
        for c in self.all_checks():
            u = c.utilisation if c.utilisation is not None else (math.inf if c.status == FAIL else -1)
            if c.status in (PASS, FAIL) and u > bu:
                best, bu = c, u
        return best

    def overall_status(self) -> str:
        sts = {c.status for c in self.all_checks()}
        if FAIL in sts:
            return FAIL
        if NOT_COVERED in sts or WARN in sts:
            return NOT_COVERED
        return PASS

    def combined(self, mid: str, quantity: str, kind: str):
        F, sel = self.combos.matrix(kind)
        A = np.stack([getattr(self.cases[c].members[mid], quantity) for c in self.combos.case_ids])
        return F @ A, sel


def _member_forces(res: ProjectResult, mid: str) -> ec2_checks.MemberForces:
    F, uls = res.combos.matrix("ULS")
    Fq, qp = res.combos.matrix("SLS_qp")
    st = lambda q: np.stack([getattr(res.cases[c].members[mid], q) for c in res.combos.case_ids])  # noqa: E731
    N, V, M = st("N"), st("V"), st("M")
    md = res.analysis.members[mid]
    m = res.model.members[mid]
    # cantilever: free end = node with no support and no other member
    tip_at_i = False
    if m.design.deflection_system == "cantilever":
        def is_free(nid):
            others = [o for o in res.model.members.values() if o.id != mid and nid in (o.i, o.j)]
            return nid not in res.model.supports and not others
        tip_at_i = is_free(m.i) and not is_free(m.j)
    return ec2_checks.MemberForces(
        x=res.analysis.stations[mid][0], L=md.L,
        uls_N=F @ N, uls_V=F @ V, uls_M=F @ M,
        uls_ids=[c.id for c in uls], uls_labels=[c.label for c in uls],
        qp_M=Fq @ M, qp_ids=[c.id for c in qp], qp_labels=[c.label for c in qp],
        tip_at_i=tip_at_i,
    )


def run(model: Model) -> ProjectResult:
    t0 = time.perf_counter()
    fa = FrameAnalysis(model)
    cases = {cid: fa.solve_case(lc) for cid, lc in model.load_cases.items()}
    combos = generate(model)
    params = make_params({k: v for k, v in model.settings.params.items()
                          if k not in ("gamma_G_sup", "gamma_G_inf", "gamma_Q", "xi", "psi", "vary_permanent_individually")})
    res = ProjectResult(model, fa, cases, combos, params, {}, {}, {})
    warnings = []
    for d in fa.auto_fixed:
        warnings.append(f"Node '{fa.node_ids[d // 3]}' is hinged on every member; its rotation was restrained automatically.")
    for cid, cr in cases.items():
        scale = max(1.0, *(abs(v) for v in cr.applied.values()))
        if max(abs(v) for v in cr.equilibrium.values()) > 1e-6 * scale:
            warnings.append(f"Load case {cid}: equilibrium residual larger than expected {cr.equilibrium}.")

    # envelopes (ULS) per member
    F, uls = combos.matrix("ULS")
    for mid in model.members:
        x, side = fa.stations[mid]
        mx, mn, mxc, mnc = {}, {}, {}, {}
        for q in ("N", "V", "M"):
            A = np.stack([getattr(cases[c].members[mid], q) for c in combos.case_ids])
            C = F @ A
            imax, imin = np.argmax(C, axis=0), np.argmin(C, axis=0)
            mx[q], mn[q] = C.max(axis=0), C.min(axis=0)
            mxc[q] = [uls[i].id for i in imax]
            mnc[q] = [uls[i].id for i in imin]
        res.envelopes[mid] = Envelope(x, side, mx, mn, mxc, mnc)

    # reactions: ULS envelope and characteristic per case
    for nid in model.supports:
        R = np.stack([cases[c].reactions[nid] for c in combos.case_ids])  # (n_cases, 3)
        RU = F @ R
        res.reactions[nid] = {
            "per_case": {c: cases[c].reactions[nid].tolist() for c in combos.case_ids},
            "uls_max": RU.max(axis=0).tolist(), "uls_min": RU.min(axis=0).tolist(),
            "uls_max_combo": [uls[i].id for i in RU.argmax(axis=0)],
            "uls_min_combo": [uls[i].id for i in RU.argmin(axis=0)],
        }

    # code checks
    rc_cache: dict[str, RCSection] = {}
    code = model.settings.design_code
    for mid, m in model.members.items():
        sec = model.sections[m.section]
        if m.section not in rc_cache:
            rc_cache[m.section] = RCSection(sec, params)
        forces = _member_forces(res, mid)
        if code == "EC2":
            res.checks[mid] = ec2_checks.check_member(m, sec, rc_cache[m.section], forces, params)
        else:
            res.checks[mid] = sia262.check_member(m, sec, rc_cache[m.section], forces, params)
    res.warnings = warnings
    res.runtime_s = time.perf_counter() - t0
    return res


# ------------------------------------------------------------------ JSON
def _arr(a) -> list:
    return [float(round(v, 10)) for v in np.asarray(a).ravel()]


def _plain(obj):
    """Recursively convert numpy scalars and arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def to_dict(res: ProjectResult) -> dict:
    return _plain(_to_dict(res))


def _to_dict(res: ProjectResult) -> dict:
    m = res.model
    gov = res.governing()
    out = {
        "tragwerk_version": __version__,
        "model": m.name,
        "design_code": m.settings.design_code,
        "basis": {
            "analysis": "Linear-elastic 2D frame, direct stiffness, Euler-Bernoulli members, gross concrete sections, E = Ecm",
            "combinations": f"EN 1990 Eq. ({m.settings.combination_rule}) with pattern loading; recommended psi factors",
            "design": "EN 1992-1-1:2004 with recommended values" if m.settings.design_code == "EC2" else "SIA 262 (not implemented)",
        },
        "summary": {
            "status": res.overall_status(),
            "max_utilisation": gov.utilisation if gov and gov.utilisation is not None and math.isfinite(gov.utilisation) else None,
            "governing": None if gov is None else {"member": gov.member, "check": gov.title, "clause": gov.clause},
            "n_combinations": {k: sum(1 for c in res.combos.combos if c.kind == k) for k in ("ULS", "SLS_char", "SLS_freq", "SLS_qp")},
            "runtime_s": round(res.runtime_s, 4),
        },
        "warnings": res.warnings,
        "load_cases": {
            cid: {
                "displacements": {nid: _arr(cr.displacements[k]) for k, nid in enumerate(res.analysis.node_ids)},
                "reactions": {nid: _arr(r) for nid, r in cr.reactions.items()},
                "equilibrium_residual": {k: float(v) for k, v in cr.equilibrium.items()},
                "members": {mid: {"x": _arr(mr.x), "N": _arr(mr.N), "V": _arr(mr.V), "M": _arr(mr.M),
                                  "v_rel_mm": _arr(mr.v_rel * 1000)} for mid, mr in cr.members.items()},
            } for cid, cr in res.cases.items()
        },
        "combinations": [{"id": c.id, "kind": c.kind, "label": c.label, "leading": c.leading,
                          "factors": dict(zip(res.combos.case_ids, c.factors))} for c in res.combos.combos],
        "envelopes": {mid: {"x": _arr(e.x), **{f"{q}_max": _arr(e.max[q]) for q in ("N", "V", "M")},
                            **{f"{q}_min": _arr(e.min[q]) for q in ("N", "V", "M")}} for mid, e in res.envelopes.items()},
        "reactions": res.reactions,
        "checks": {mid: [c.to_dict() for c in lst] for mid, lst in res.checks.items()},
    }
    return out
