"""EN 1998-1:2004 modal response spectrum analysis (4.3.3.3) for low-ductility (DCL) frames.

* Design spectrum for horizontal components, 3.2.2.5, Eqs. (3.13)-(3.16).
* Modal responses rᵢ for every result quantity from the modal displacements
  uᵢ = Γᵢ Sd(Tᵢ)/ωᵢ² φᵢ, combined by CQC (4.3.3.3.2(3)) with ζ = 5 %.
* Directions X and Y combined by the 30 % rule (4.3.3.5.2(4)), taking for every
  quantity the larger of E_X + 0.3 E_Y and 0.3 E_X + E_Y.
* The design action effect A_Ed enters the seismic design situation of EN 1990
  6.4.3.4: G + Σψ₂Q ± A_Ed. Because CQC loses signs, A_Ed is applied with both
  signs, and a second variant with the axial force reversed (A_Ed') covers the
  pairing of axial force and bending of opposite sign.
* Damage limitation (4.4.3.2): inter-storey drift d_r = q·d_r,e with ν d_r ≤ α h,
  checked for every column. Storey sensitivity θ (4.4.2.2(2)) is reported.

Not included: accidental torsional effects (4.3.3.3.3), capacity design rules of
DCM/DCH (so q is limited to DCL values in practice, ≤ 1.5), vertical component.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .codes.base import FAIL, PASS, CheckResult, Step, fmt, status_from_utilisation
from .combinations import PSI, Combination, CombinationSet
from .model3d import Model3D, SeismicSettings
from .modal import ModalResult, _release_transform
from .solver3d import CaseResult3D, MemberResult3D

# EN 1998-1 Tables 3.2 and 3.3: S, T_B, T_C, T_D
SPECTRUM = {
    1: {"A": (1.0, 0.15, 0.4, 2.0), "B": (1.2, 0.15, 0.5, 2.0), "C": (1.15, 0.20, 0.6, 2.0), "D": (1.35, 0.20, 0.8, 2.0), "E": (1.4, 0.15, 0.5, 2.0)},
    2: {"A": (1.0, 0.05, 0.25, 1.2), "B": (1.35, 0.05, 0.25, 1.2), "C": (1.5, 0.10, 0.25, 1.2), "D": (1.8, 0.10, 0.30, 1.2), "E": (1.6, 0.05, 0.25, 1.2)},
}
QUANT = ("N", "Vy", "Vz", "T", "My", "Mz")
CASE_E, CASE_E2 = "AEd", "AEd'"


def design_spectrum(T, s: SeismicSettings):
    """S_d(T) in m/s², EN 1998-1 Eqs. (3.13)-(3.16)."""
    S, TB, TC, TD = SPECTRUM[s.spectrum_type][s.ground]
    ag, q = s.agR * s.importance, s.q
    T = np.asarray(T, dtype=float)
    out = np.where(T <= TB, ag * S * (2 / 3 + T / TB * (2.5 / q - 2 / 3)),
          np.where(T <= TC, ag * S * 2.5 / q,
          np.where(T <= TD, np.maximum(ag * S * 2.5 / q * TC / np.maximum(T, 1e-12), s.beta * ag),
                   np.maximum(ag * S * 2.5 / q * TC * TD / np.maximum(T, 1e-12) ** 2, s.beta * ag))))
    return out


def cqc_rho(omega: np.ndarray, zeta: float) -> np.ndarray:
    """Der Kiureghian's correlation coefficients for equal modal damping."""
    r = omega[None, :] / omega[:, None]
    return 8 * zeta ** 2 * (1 + r) * r ** 1.5 / ((1 - r ** 2) ** 2 + 4 * zeta ** 2 * r * (1 + r) ** 2)


def cqc(R: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """R: (modes, ...) modal values -> combined magnitude."""
    v = np.einsum("i...,ij,j...->...", R, rho, R)
    return np.sqrt(np.maximum(v, 0.0))


@dataclass
class SeismicResult:
    settings: SeismicSettings
    modal: ModalResult
    Sd: np.ndarray                         # m/s² per mode
    base_shear: dict[str, float]           # kN per direction (CQC)
    modal_base_shear: dict[str, np.ndarray]
    members: dict[str, dict[str, np.ndarray]]    # A_Ed magnitudes at the static stations
    reactions: dict[str, np.ndarray]
    displacements: dict[str, np.ndarray]   # elastic d_e magnitudes per node (6)
    drift: dict[str, dict]                 # per column
    storeys: list[dict]
    gravity_factors: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def spectrum_curve(self, n: int = 121) -> tuple[np.ndarray, np.ndarray]:
        T = np.linspace(0.0, 4.0, n)
        return T, design_spectrum(T, self.settings)


def _modal_end_forces(mr: ModalResult, U: np.ndarray) -> dict[str, np.ndarray]:
    """Local end forces (12, modes) of every element of the refined model for modal displacements U (ndof, modes)."""
    fa = mr.analysis
    out = {}
    for eid, md in fa.members.items():
        ul = md.T @ U[fa._dofs(md)]
        if md.cidx:
            ul = _release_transform(md) @ ul
        out[eid] = md.k_full @ ul
    return out


def _stations_from_elements(mr: ModalResult, mid: str, xs: np.ndarray, F: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Modal internal forces (modes, stations) along an original member from element end forces."""
    subs = mr.submap[mid]
    k = next(iter(F.values())).shape[1]
    res = {q: np.zeros((k, len(xs))) for q in QUANT}
    for s, x in enumerate(xs):
        eid, x0, _ = next((e for e in subs if x <= e[2] + 1e-9), subs[-1])
        f, xl = F[eid], x - x0
        res["N"][:, s] = -f[0]
        res["Vy"][:, s] = f[1]
        res["Vz"][:, s] = f[2]
        res["T"][:, s] = -f[3]
        res["Mz"][:, s] = -f[5] + xl * f[1]
        res["My"][:, s] = f[4] + xl * f[2]
    return res


def _combine_dirs(per_dir: dict[str, np.ndarray]) -> np.ndarray:
    if len(per_dir) == 1:
        return next(iter(per_dir.values()))
    a, b = per_dir["X"], per_dir["Y"]
    return np.maximum(a + 0.3 * b, 0.3 * a + b)


def gravity_factors(model: Model3D) -> dict[str, float]:
    psi = dict(PSI)
    for cat, vals in (model.settings.params.get("psi") or {}).items():
        psi[cat] = tuple(float(v) for v in vals)
    return {cid: (1.0 if lc.type == "permanent" else psi[lc.category][2]) for cid, lc in model.load_cases.items()}


def rsa(model: Model3D, fa_static, cases: dict[str, CaseResult3D], mr: ModalResult) -> SeismicResult:
    s = model.seismic
    T = mr.period
    Sd = design_spectrum(T, s)
    rho = cqc_rho(mr.omega, s.damping)
    warnings = []
    for d in s.directions:
        cum = float(np.sum(mr.ratio(d)))
        if cum < 0.9:
            warnings.append(f"The {len(T)} modes capture {100 * cum:.0f} % of the mass in {d}; EN 1998-1 4.3.3.3.1(3) asks for 90 %. Increase the number of modes.")
    warnings.append("Accidental torsional effects (EN 1998-1 4.3.3.3.3) are not included.")
    fa = mr.analysis
    per_dir_members: dict[str, dict] = {}
    per_dir_react: dict[str, dict] = {}
    per_dir_disp: dict[str, dict] = {}
    modal_U, modal_F = {}, {}
    base_shear, modal_vb = {}, {}
    sup_nodes = list(model.supports)
    for d in s.directions:
        U = mr.phi * (mr.gamma[d] * Sd / mr.omega ** 2)[None, :]
        modal_U[d] = U
        F = _modal_end_forces(mr, U)
        modal_F[d] = F
        per_dir_members[d] = {}
        for mid in model.members:
            xs = fa_static.stations[mid][0]
            modal_vals = _stations_from_elements(mr, mid, xs, F)
            per_dir_members[d][mid] = {q: cqc(v, rho) for q, v in modal_vals.items()}
        KU = fa.K @ U
        per_dir_react[d] = {}
        q = "XY".index(d)
        vb = np.zeros(U.shape[1])
        for nid in sup_nodes:
            b = 6 * fa.nidx[nid]
            Rm = np.where(fa.fixed[b:b + 6, None], KU[b:b + 6], 0.0)    # (6, modes)
            per_dir_react[d][nid] = cqc(Rm.T, rho)
            vb += Rm[q]
        modal_vb[d] = vb
        base_shear[d] = float(cqc(vb, rho))
        per_dir_disp[d] = {nid: cqc(U[6 * fa.nidx[nid]:6 * fa.nidx[nid] + 6].T, rho) for nid in model.nodes}
    members = {mid: {q: _combine_dirs({d: per_dir_members[d][mid][q] for d in s.directions}) for q in QUANT} for mid in model.members}
    reactions = {nid: _combine_dirs({d: per_dir_react[d][nid] for d in s.directions}) for nid in sup_nodes}
    displacements = {nid: _combine_dirs({d: per_dir_disp[d][nid] for d in s.directions}) for nid in model.nodes}

    # ---- drift per column and storey sensitivity
    gf = gravity_factors(model)
    drift, storeys = {}, {}
    for mid, m in model.members.items():
        if m.design.type != "column":
            continue
        a, b = model.nodes[m.i], model.nodes[m.j]
        h = abs(b.z - a.z)
        if h < 1e-6:
            continue
        rel = {}
        for comp, q in (("x", 0), ("y", 1)):
            vals = {}
            for d in s.directions:
                U = modal_U[d]
                du = U[6 * fa.nidx[m.j] + q] - U[6 * fa.nidx[m.i] + q]
                vals[d] = cqc(du, rho)
            rel[comp] = float(_combine_dirs(vals))
        up = b.z >= a.z
        Ngrav = sum(f * float(cases[cid].members[mid].N[-1 if up else 0]) for cid, f in gf.items())
        shear = {}
        eid = mr.submap[mid][0][0] if up else mr.submap[mid][-1][0]
        lam = fa.members[eid].lam
        for d in s.directions:
            fb = modal_F[d][eid][0:3] if up else modal_F[d][eid][6:9]      # force on the column at its foot, local
            shear[d] = (lam.T @ fb)[:2]                                    # (2, modes): global X, Y
        drift[mid] = {"h": h, "dre_x": rel["x"], "dre_y": rel["y"], "N_grav": Ngrav, "shear": shear, "top_z": max(a.z, b.z), "bot_z": min(a.z, b.z)}
        key = (round(min(a.z, b.z), 3), round(max(a.z, b.z), 3))
        storeys.setdefault(key, []).append(mid)
    storey_list = []
    for (z0, z1), mids in sorted(storeys.items()):
        h = z1 - z0
        P = sum(max(-drift[m]["N_grav"], 0.0) for m in mids)
        rowV, rowD = {}, {}
        for comp, q in (("x", 0), ("y", 1)):
            vals = {}
            for d in s.directions:
                tot = sum(drift[m]["shear"][d][q] for m in mids)
                vals[d] = cqc(tot, rho)
            rowV[comp] = float(_combine_dirs(vals))
            rowD[comp] = s.q * float(np.mean([drift[m][f"dre_{comp}"] for m in mids]))
        theta = {c: (P * rowD[c] / (rowV[c] * h) if rowV[c] > 1e-9 else 0.0) for c in ("x", "y")}
        storey_list.append({"z0": z0, "z1": z1, "h": h, "columns": len(mids), "P": P, "V": rowV, "dr": rowD, "theta": theta})
        for m in mids:
            drift[m]["theta"] = max(theta.values())
        if max(theta.values()) > 0.1:
            warnings.append(f"Storey +{z0:.2f} to +{z1:.2f} m: θ = {max(theta.values()):.3f} > 0.10, second-order effects are not included "
                            "(EN 1998-1 4.4.2.2(3) asks for amplification by 1/(1 - θ)).")
    return SeismicResult(s, mr, Sd, base_shear, modal_vb, members, reactions, displacements, drift, storey_list, gf, warnings)


def drift_check(mid: str, sr: SeismicResult) -> CheckResult:
    s, d = sr.settings, sr.drift[mid]
    dre = max(d["dre_x"], d["dre_y"])
    comp = "x" if d["dre_x"] >= d["dre_y"] else "y"
    dr = s.q * dre
    lim = s.drift_limit * d["h"]
    util = s.nu * dr / lim
    th = d.get("theta", 0.0)
    steps = [
        Step("d_{r,e}", f"elastic inter-storey drift, global {comp} (CQC, 30 % rule)", "", dre * 1000, "mm", "4.3.3.3.2"),
        Step("d_{r}", "q·d_{r,e}", f"{fmt(s.q)}·{fmt(dre * 1000)}", dr * 1000, "mm", "4.3.4(1)"),
        Step("h", "storey height", "", d["h"] * 1000, "mm"),
        Step("ν", "reduction factor for the damage limitation seismic action", "", s.nu, "", "4.4.3.2(2)"),
        Step("α·h", "drift limit", f"{fmt(s.drift_limit)}·{fmt(d['h'] * 1000)}", lim * 1000, "mm", "4.4.3.2(1)"),
        Step("η", "ν·d_{r}/(α·h)", f"{fmt(s.nu)}·{fmt(dr * 1000)}/{fmt(lim * 1000)}", util, "", "Eq. 4.31-4.33"),
        Step("θ", "P_{tot}·d_{r}/(V_{tot}·h) of the storey", "", th, "", "4.4.2.2(2)"),
    ]
    notes = ["Limit α = 0.005 for brittle non-structural elements, 0.0075 for ductile ones, 0.010 where they do not interfere."]
    if th > 0.1:
        notes.append("θ > 0.10: second-order effects should be included by 1/(1 - θ); this check does not apply it.")
    st = status_from_utilisation(util)
    if th > 0.3:
        st = FAIL
        notes.append("θ > 0.30 is not permitted (4.4.2.2(4)).")
    return CheckResult(id="ec8.drift", title="Damage limitation, inter-storey drift", clause="EN 1998-1 4.4.3.2", member=mid,
                       status=st, utilisation=util, demand=dr * 1000, capacity=lim * 1000 / s.nu, unit="mm",
                       combination=CASE_E, combination_label="design seismic action (A_Ed)", steps=steps, notes=notes)


def pseudo_cases(fa_static, sr: SeismicResult, model: Model3D) -> dict[str, CaseResult3D]:
    """A_Ed and A_Ed' as case results at the static stations (magnitudes)."""
    out = {}
    nnodes = len(fa_static.node_ids)
    disp = np.array([sr.displacements[nid] for nid in fa_static.node_ids]).reshape(nnodes, 6)
    for cid, sgnN in ((CASE_E, 1.0), (CASE_E2, -1.0)):
        mem = {}
        for mid in model.members:
            xs, side = fa_static.stations[mid]
            e = sr.members[mid]
            z = np.zeros(len(xs))
            mem[mid] = MemberResult3D(xs, side, sgnN * e["N"], e["Vy"], e["Vz"], e["T"], e["My"], e["Mz"], z, z, z, z, z, np.zeros(12))
        zero = {k: 0.0 for k in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")}
        out[cid] = CaseResult3D(cid, disp.copy(), {nid: v.copy() for nid, v in sr.reactions.items()}, {}, mem, dict(zero), dict(zero))
    return out


def seismic_combinations(model: Model3D, combos: CombinationSet, gf: dict[str, float]) -> CombinationSet:
    """Append G + Σψ₂Q ± A_Ed (and ± A_Ed') as ULS combinations."""
    ids = combos.case_ids + [CASE_E, CASE_E2]
    base = [Combination(c.id, c.kind, tuple(c.factors) + (0.0, 0.0), c.leading, c.label) for c in combos.combos]
    groups: dict[str, list[str]] = {}
    free = []
    for cid, lc in model.load_cases.items():
        if lc.type == "variable" and gf[cid] > 0:
            (groups.setdefault(lc.group, []) if lc.group else free).append(cid)
    import itertools
    picks = list(itertools.product(*groups.values())) if groups else [()]
    out, n, seen = [], 0, set()
    for pick in picks:
        vec = [0.0] * len(ids)
        for cid, lc in model.load_cases.items():
            if lc.type == "permanent":
                vec[ids.index(cid)] = 1.0
        for cid in free + list(pick):
            vec[ids.index(cid)] = gf[cid]
        for e, sg in ((CASE_E, 1.0), (CASE_E, -1.0), (CASE_E2, 1.0), (CASE_E2, -1.0)):
            v = list(vec)
            v[ids.index(e)] = sg
            key = tuple(round(x, 9) for x in v)
            if key in seen:
                continue
            seen.add(key)
            n += 1
            parts = [f"{f:.2f}·{cid}" for cid, f in zip(ids[:-2], v[:-2]) if abs(f) > 1e-12]
            label = " + ".join(parts) + (" + " if sg > 0 else " − ") + e
            out.append(Combination(f"SEI{n:03d}", "ULS", tuple(v), e, label))
    return CombinationSet(ids, base + out)
