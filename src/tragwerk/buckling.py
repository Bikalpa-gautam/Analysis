"""Linear buckling analysis of 3D frames: elastic critical load factors α_cr and buckling modes.

    (K + α K_G(N_Ed)) φ = 0

* K is the elastic stiffness of the frame, K_G the geometric stiffness of the axial forces N_Ed of a ULS
  combination (first-order, linear analysis). Per element, with Hermitian (cubic) shape functions,
      K_G = N/L · [6/5, L/10, −6/5, L/10; L/10, 2L²/15, −L/10, −L²/30; …] in both bending planes,
  plus the torsional term N r²/L (r² = (I_y + I_z)/A) for torsional modes (McGuire, Gallagher & Ziemian,
  Matrix Structural Analysis, 9.2). N > 0 is tension.
* Members are divided into segments so that member buckling (not only storey sway) is captured; end releases
  are condensed with the same transformation as the stiffness.
* The eigenproblem is solved as −K_G φ = μ K φ with μ = 1/α: the largest μ give the smallest positive α_cr. K is
  factorised once; every ULS combination gets its α_cr, the governing one (smallest α_cr) also the next modes.
* EN 1993-1-1 5.2.1(3) and EN 1992-1-1 5.8.2(6) (via 5.8.7.3): if α_cr ≥ 10 the second-order effects are below
  about 10 % (amplification 1/(1 − 1/α_cr)) and a first-order analysis is sufficient; otherwise they must be taken
  into account (the amplification factor is reported).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .combinations import generate
from .model3d import Model3D
from .modal import _release_transform, member_shape, refine
from .solver3d import V_IDX, W_IDX, W_SIGN, FrameAnalysis3D


def _plane_g(L: float) -> np.ndarray:
    return np.array([[6 / 5, L / 10, -6 / 5, L / 10],
                     [L / 10, 2 * L * L / 15, -L / 10, -L * L / 30],
                     [-6 / 5, -L / 10, 6 / 5, -L / 10],
                     [L / 10, -L * L / 30, -L / 10, 2 * L * L / 15]]) / L


def geometric_stiffness_local(N: float, L: float, r2: float) -> np.ndarray:
    """12×12 geometric stiffness in local axes for the axial force N (tension positive)."""
    g = np.zeros((12, 12))
    pg = _plane_g(L)
    g[np.ix_(V_IDX, V_IDX)] = pg
    g[np.ix_(W_IDX, W_IDX)] = W_SIGN[:, None] * pg * W_SIGN[None, :]
    g[np.ix_([3, 9], [3, 9])] = r2 / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return N * g


@dataclass
class BucklingCombo:
    id: str
    label: str
    alpha: list[float]                 # α_cr; for the governing combination the first `modes` factors, ascending
    n_compression: float               # Σ |N| of compressed elements · L, a measure of the axial load (kN·m)

    @property
    def alpha_cr(self) -> float | None:
        return self.alpha[0] if self.alpha else None


@dataclass
class BucklingResult:
    model: Model3D                     # refined model
    analysis: FrameAnalysis3D
    submap: dict
    combos: list[BucklingCombo]
    governing: str | None              # combination with the smallest α_cr
    modes: list[np.ndarray] = field(default_factory=list)   # global DOF vectors of the governing combination
    alpha: list[float] = field(default_factory=list)
    axial: dict = field(default_factory=dict)               # element → N_Ed of the governing combination
    warnings: list[str] = field(default_factory=list)

    @property
    def alpha_cr(self) -> float | None:
        return self.alpha[0] if self.alpha else None


def _normalise(v: np.ndarray) -> np.ndarray:
    a = np.abs(v)
    k = int(np.argmax(a >= a.max() * (1 - 1e-6)))
    v = v / a.max()
    return -v if v[k] < 0 else v


def buckling(model: Model3D, modes: int = 4, segments: int = 4, combos=None) -> BucklingResult:
    ref, submap = refine(model, segments)
    fa = FrameAnalysis3D(ref)
    cset = combos or generate(model)
    F, uls = cset.matrix("ULS")
    cases = {cid: fa.solve_case(ref.load_cases[cid]) for cid in cset.case_ids}
    eids = list(fa.members)
    # mean axial force of every element and case (N varies along a column under its self weight)
    Nc = np.array([[float(np.mean(cases[cid].members[eid].N)) for eid in eids] for cid in cset.case_ids])
    free = fa.free
    Kff = csr_matrix(fa.K[np.ix_(free, free)])
    r2 = {eid: (ref.sections[ref.members[eid].section].Iy_m4 + ref.sections[ref.members[eid].section].Iz_m4) / ref.sections[ref.members[eid].section].area_m2
          for eid in eids}
    unit = {}
    for eid in eids:
        md = fa.members[eid]
        Tr = _release_transform(md)
        g = Tr.T @ geometric_stiffness_local(1.0, md.L, r2[eid]) @ Tr
        unit[eid] = (md.T.T @ g @ md.T, fa._dofs(md))
    pos = -np.ones(fa.ndof, dtype=int)
    pos[free] = np.arange(len(free))
    res = BucklingResult(ref, fa, submap, [], None)
    lu = splu(Kff.tocsc())
    Kinv = LinearOperator(Kff.shape, matvec=lu.solve, dtype=float)
    nf = len(free)

    # unit geometric stiffness of every element as COO triplets on the free DOFs; per combination −K_G = Σ (−N_e) g_e
    rows, cols, vals, owner = [], [], [], []
    for j, eid in enumerate(eids):
        g, dofs = unit[eid]
        p = pos[dofs]
        sel = np.flatnonzero(p >= 0)
        rr, cc = np.meshgrid(p[sel], p[sel], indexing="ij")
        gv = g[np.ix_(sel, sel)]
        nz = gv != 0
        rows.append(rr[nz]); cols.append(cc[nz]); vals.append(gv[nz]); owner.append(np.full(int(nz.sum()), j))
    rows, cols, vals, owner = (np.concatenate(a) for a in (rows, cols, vals, owner))

    def solve(Ne, k):
        G = csr_matrix((-np.asarray(Ne)[owner] * vals, (rows, cols)), shape=(nf, nf))     # −K_G: positive for compression
        mu, vec = eigsh(G, k=k, M=Kff, Minv=Kinv, which="LA", v0=np.ones(nf))
        order = np.argsort(-mu)
        mu, vec = mu[order], vec[:, order]
        keep = mu > 1e-12 * max(abs(mu).max(), 1e-300)
        return [float(1.0 / m) for m in mu[keep]], vec[:, keep]

    best = None
    for i, c in enumerate(uls):
        Ne = F[i] @ Nc
        ncomp = float(sum(-n * fa.members[eid].L for eid, n in zip(eids, Ne) if n < 0))
        if ncomp <= 1e-9:
            res.combos.append(BucklingCombo(c.id, c.label, [], ncomp))
            continue
        alpha, _ = solve(Ne, 1)
        res.combos.append(BucklingCombo(c.id, c.label, alpha, ncomp))
        if alpha and (best is None or alpha[0] < best[0]):
            best = (alpha[0], c.id, Ne, len(res.combos) - 1)
    if best is None:
        res.warnings.append("No ULS combination puts members in compression: there is nothing to buckle.")
        return res
    k = min(modes, nf - 1)
    alpha, vecs = solve(best[2], k)
    res.combos[best[3]].alpha = alpha
    res.governing, res.alpha, res.axial = best[1], alpha, {eid: float(n) for eid, n in zip(eids, best[2])}
    for j in range(vecs.shape[1]):
        v = np.zeros(fa.ndof)
        v[free] = vecs[:, j]
        res.modes.append(_normalise(v))
    a = res.alpha_cr
    if a < 1.0:
        res.warnings.append(f"α_cr = {a:.2f} < 1: the frame buckles under the design loads.")
    elif a < 10.0:
        res.warnings.append(f"α_cr = {a:.2f} < 10: second-order effects are significant; first-order results are amplified by "
                            f"about 1/(1 − 1/α_cr) = {1 / (1 - 1 / a):.2f} (EN 1993-1-1 5.2.1(3), EN 1992-1-1 5.8.2(6)).")
    return res


class _Shape:
    """Adapter so that modal.member_shape can draw buckling modes."""

    def __init__(self, br: BucklingResult):
        self.analysis, self.model, self.submap = br.analysis, br.model, br.submap


def buckling_member_shape(br: BucklingResult, mid: str, vec: np.ndarray, points_per_element: int = 6):
    return member_shape(_Shape(br), mid, vec, points_per_element)


def to_dict_buckling(br: BucklingResult, model: Model3D, points_per_element: int = 4) -> dict:
    return {"governing": br.governing, "alpha": br.alpha, "warnings": br.warnings,
            "combos": [{"id": c.id, "label": c.label, "alpha": c.alpha, "alpha_cr": c.alpha_cr} for c in br.combos],
            "modes": [{mid: [p.tolist() for p in buckling_member_shape(br, mid, v, points_per_element)] for mid in model.members} for v in br.modes]}


__all__ = ["buckling", "geometric_stiffness_local", "BucklingResult", "buckling_member_shape", "to_dict_buckling"]
