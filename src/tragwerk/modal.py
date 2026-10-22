"""Modal analysis of 3D frames: mass matrices, natural frequencies, mode shapes and
participating masses.

    (K - ω² M) φ = 0,      φᵢᵀ M φⱼ = δᵢⱼ

* Members are subdivided into ``segments`` beam elements for the dynamic model, so
  that member vibration (not only storey sway) is represented.
* Element mass matrices are consistent (Hermitian shape functions for bending,
  linear for axial and torsion; M_e = ∫ m̄(x) Nᵀ(x) N(x) dx) or lumped (lever rule,
  translational only plus torsional inertia).
* Mass comes from self weight and from the gravity loads of load cases, each case
  with a factor: permanent 1.0, variable ψ₂ (EN 1998-1 4.2.4 with φ = 1.0), unless
  given explicitly. A load counts as mass by its downward vertical component:
  m̄ = -q_z / g. Additional nodal masses are accepted.
* Member end releases condense the mass with the same transformation as the
  stiffness (u_c = -K_cc⁻¹ K_ca u_a).

Units: kN, m, s; masses in t (= kN·s²/m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular

from .combinations import PSI
from .model3d import DynamicsSettings, Model3D, parse_model3d
from .solver import _shape
from .solver3d import V_IDX, W_IDX, W_SIGN, FrameAnalysis3D

G = 9.81
_GX, _GWT = np.polynomial.legendre.leggauss(5)
GP5 = (_GX + 1.0) / 2.0          # on [0, 1]
GW5 = _GWT / 2.0
AXES = ("X", "Y", "Z")


# ===================================================================== refined model
def refine(model: Model3D, segments: int) -> tuple[Model3D, dict[str, list[tuple[str, float, float]]]]:
    """The model with every member split into ``segments`` elements.

    Returns the refined model and, for every original member, its elements as
    (element id, x0, x1) along the member.
    """
    raw = model.raw
    nodes = [{"id": n.id, "x": n.x, "y": n.y, "z": n.z} for n in model.nodes.values()]
    taken = set(model.nodes) | set(model.members)
    members, submap = [], {}

    def fresh(name):
        while name in taken:
            name += "'"
        taken.add(name)
        return name

    def rel(r):
        return {"t": r.t, "my": r.my, "mz": r.mz}

    for mid, m in model.members.items():
        L, _ = model.axes(mid)
        a, b = model.nodes[m.i], model.nodes[m.j]
        s = segments
        ids = [m.i] + [fresh(f"{mid}@{k}") for k in range(1, s)] + [m.j]
        for k in range(1, s):
            t = k / s
            nodes.append({"id": ids[k], "x": a.x + (b.x - a.x) * t, "y": a.y + (b.y - a.y) * t, "z": a.z + (b.z - a.z) * t})
        subs = []
        for k in range(s):
            eid = mid if s == 1 else fresh(f"{mid}#{k + 1}")
            releases = {}
            if k == 0:
                releases["i"] = rel(m.rel_i)
            if k == s - 1:
                releases["j"] = rel(m.rel_j)
            members.append({"id": eid, "i": ids[k], "j": ids[k + 1], "section": m.section, "roll_deg": m.roll_deg,
                            "releases": releases, "design": {"type": "none"}})
            subs.append((eid, L * k / s, L * (k + 1) / s))
        submap[mid] = subs

    def lmap(lc):
        out = []
        for ld in lc.nodal:
            out.append({"type": "nodal", "node": ld.node, **dict(zip(("fx", "fy", "fz", "mx", "my", "mz"), ld.f))})
        for ld in lc.distributed:
            for eid, x0, x1 in submap[ld.member]:
                lo, hi = max(ld.a, x0), min(ld.b, x1)
                if hi - lo > 1e-9:
                    w = lambda x: ld.w1 + (ld.w2 - ld.w1) * (x - ld.a) / (ld.b - ld.a)  # noqa: E731
                    out.append({"type": "udl", "member": eid, "w1": w(lo), "w2": w(hi), "a": lo - x0,
                                "b": min(hi - x0, x1 - x0), "direction": ld.direction})
        for ld in lc.point:
            for eid, x0, x1 in submap[ld.member]:
                if x0 - 1e-12 <= ld.a <= x1 + 1e-12:
                    out.append({"type": "point", "member": eid, "p": ld.p, "a": min(max(ld.a - x0, 0.0), x1 - x0), "direction": ld.direction})
                    break
        return out

    cases = [{"id": lc.id, "type": lc.type, "name": lc.name, "category": lc.category, "group": lc.group,
              "self_weight": lc.self_weight, "loads": lmap(lc)} for lc in model.load_cases.values()]
    supports = []
    keys = ("ux", "uy", "uz", "rx", "ry", "rz")
    skeys = ("kx", "ky", "kz", "krx", "kry", "krz")
    for sp in model.supports.values():
        d = {"node": sp.node, **dict(zip(keys, sp.fix))}
        d.update({k: v for k, v in zip(skeys, sp.springs) if v})
        supports.append(d)
    data = {"schema": raw["schema"], "name": model.name, "materials": raw["materials"], "sections": raw["sections"],
            "design": {"design_code": model.settings.design_code, "combination_rule": model.settings.combination_rule,
                       "params": model.settings.params, "stations_per_member": 1},
            "nodes": nodes, "supports": supports, "members": members, "load_cases": cases}
    return parse_model3d(data), submap


# ===================================================================== mass
def mass_factors(model: Model3D, settings: DynamicsSettings) -> dict[str, float]:
    """Factor on each load case's gravity loads (and self weight) that becomes mass."""
    if settings.mass_cases is not None:
        return {cid: float(settings.mass_cases.get(cid, 0.0)) for cid in model.load_cases}
    psi = dict(PSI)
    for cat, vals in (model.settings.params.get("psi") or {}).items():
        psi[cat] = tuple(float(v) for v in vals)
    return {cid: (1.0 if lc.type == "permanent" else psi[lc.category][2]) for cid, lc in model.load_cases.items()}


def _vertical(direction: str, lam: np.ndarray) -> float:
    """Global Z component of a unit load acting in ``direction``."""
    if direction.startswith("global_"):
        return 1.0 if direction == "global_z" else 0.0
    return float(lam["xyz".index(direction[-1]), 2])


@dataclass
class ElementMass:
    msw: float = 0.0                 # uniform mass per length from self weight, t/m
    r2: float = 0.0                  # polar radius of gyration squared (self weight torsional inertia), m²
    dist: list = field(default_factory=list)    # (a, b, m1, m2) t/m, may be negative (clamped at 0)
    point: list = field(default_factory=list)   # (a, m) t


def element_mass_local(L: float, em: ElementMass, kind: str) -> np.ndarray:
    """12x12 element mass matrix in local axes."""
    M = np.zeros((12, 12))

    def add_point(x, m, mr=0.0):
        xi = x / L
        N = _shape(np.array([xi]), L)[:, 0]
        if kind == "consistent":
            Nu = np.array([N[0], N[3]])
            H = np.array([N[1], N[2], N[4], N[5]])
            M[np.ix_([0, 6], [0, 6])] += m * np.outer(Nu, Nu)
            M[np.ix_(V_IDX, V_IDX)] += m * np.outer(H, H)
            Hw = W_SIGN * H
            M[np.ix_(W_IDX, W_IDX)] += m * np.outer(Hw, Hw)
            if mr:
                M[np.ix_([3, 9], [3, 9])] += mr * np.outer(Nu, Nu)
        else:
            for base, f in ((0, 1.0 - xi), (6, xi)):
                for q in range(3):
                    M[base + q, base + q] += m * f
                if mr:
                    M[base + 3, base + 3] += mr * f

    if em.msw > 0:
        for s, w in zip(GP5 * L, GW5 * L):
            add_point(s, em.msw * w, em.msw * em.r2 * w)
    for a, b, m1, m2 in em.dist:
        for t, w in zip(GP5, GW5 * (b - a)):
            m = m1 + (m2 - m1) * t
            if m > 0:
                add_point(a + (b - a) * t, m * w)
    for a, m in em.point:
        if m > 0:
            add_point(a, m)
    return M


def _release_transform(md) -> np.ndarray:
    """u_full = Tr u (released DOFs follow from static condensation)."""
    Tr = np.eye(12)
    if md.cidx:
        aidx = [i for i in range(12) if i not in md.cidx]
        k = md.k_full
        Tr[np.ix_(md.cidx, md.cidx)] = 0.0
        Tr[np.ix_(md.cidx, aidx)] = -np.linalg.solve(k[np.ix_(md.cidx, md.cidx)], k[np.ix_(md.cidx, aidx)])
    return Tr


def element_masses(ref: Model3D, fa: FrameAnalysis3D, factors: dict[str, float]) -> tuple[dict[str, ElementMass], np.ndarray]:
    """Mass description per element and nodal translational masses (per node index)."""
    ems = {mid: ElementMass() for mid in fa.members}
    nodal = np.zeros(len(fa.node_ids))
    for mid, md in fa.members.items():
        sec = ref.sections[ref.members[mid].section]
        ems[mid].r2 = (sec.Iy_m4 + sec.Iz_m4) / sec.area_m2
    for cid, f in factors.items():
        if f <= 0:
            continue
        lc = ref.load_cases[cid]
        if lc.self_weight:
            for mid in fa.members:
                sec = ref.sections[ref.members[mid].section]
                ems[mid].msw += f * sec.concrete.density * sec.area_m2 / G
        for ld in lc.distributed:
            vz = _vertical(ld.direction, fa.members[ld.member].lam)
            if vz:
                ems[ld.member].dist.append((ld.a, ld.b, -f * ld.w1 * vz / G, -f * ld.w2 * vz / G))
        for ld in lc.point:
            vz = _vertical(ld.direction, fa.members[ld.member].lam)
            if vz:
                ems[ld.member].point.append((ld.a, -f * ld.p * vz / G))
        for ld in lc.nodal:
            nodal[fa.nidx[ld.node]] += max(-f * ld.f[2] / G, 0.0)
    return ems, nodal


def assemble_mass(ref: Model3D, fa: FrameAnalysis3D, factors: dict[str, float], settings: DynamicsSettings) -> np.ndarray:
    ems, nodal = element_masses(ref, fa, factors)
    for nid, m in settings.masses:
        nodal[fa.nidx[nid]] += m
    M = np.zeros((fa.ndof, fa.ndof))
    for mid, md in fa.members.items():
        Me = element_mass_local(md.L, ems[mid], settings.mass)
        Tr = _release_transform(md)
        Mg = md.T.T @ (Tr.T @ Me @ Tr) @ md.T
        dofs = fa._dofs(md)
        M[np.ix_(dofs, dofs)] += Mg
    for k, m in enumerate(nodal):
        for q in range(3):
            M[6 * k + q, 6 * k + q] += m
    return M


# ===================================================================== eigen-solution
@dataclass
class ModalResult:
    settings: DynamicsSettings
    model: Model3D                    # refined model
    analysis: FrameAnalysis3D         # of the refined model
    submap: dict
    M: np.ndarray
    omega: np.ndarray                 # rad/s
    phi: np.ndarray                   # (ndof, modes), mass-normalised, zero at restrained DOFs
    gamma: dict[str, np.ndarray]      # participation factors per direction
    meff: dict[str, np.ndarray]       # effective modal masses, t
    mtot: dict[str, float]            # total mass moving in each direction, t
    factors: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def freq(self) -> np.ndarray:
        return self.omega / (2 * math.pi)

    @property
    def period(self) -> np.ndarray:
        return 2 * math.pi / self.omega

    def ratio(self, d: str) -> np.ndarray:
        return self.meff[d] / self.mtot[d] if self.mtot[d] > 0 else np.zeros_like(self.omega)

    def node_displacements(self, mode: int) -> np.ndarray:
        return self.phi[:, mode].reshape(-1, 6)


def _normalise_sign(phi: np.ndarray) -> np.ndarray:
    """Deterministic sign: the first component within 1e-6 of the largest magnitude is positive."""
    for i in range(phi.shape[1]):
        a = np.abs(phi[:, i])
        k = int(np.argmax(a >= a.max() * (1 - 1e-6)))
        if phi[k, i] < 0:
            phi[:, i] = -phi[:, i]
    return phi


def modal(model: Model3D, settings: DynamicsSettings | None = None) -> ModalResult:
    settings = settings or model.dynamics or DynamicsSettings()
    ref, submap = refine(model, settings.segments)
    fa = FrameAnalysis3D(ref)
    factors = mass_factors(model, settings)
    M = assemble_mass(ref, fa, factors, settings)
    free = fa.free
    Kff, Mff = fa.K[np.ix_(free, free)], M[np.ix_(free, free)]
    warnings = []
    nmass = int(np.sum(np.diag(Mff) > 1e-12 * max(float(np.max(np.diag(Mff))), 1e-300)))
    k = min(settings.modes, nmass)
    if k == 0:
        raise ValueError("the model has no mass: give load cases a mass factor or switch on self weight")
    if k < settings.modes:
        warnings.append(f"Only {k} modes exist with mass; {settings.modes} were requested.")
    n = len(free)
    L = cholesky(Kff, lower=True)
    A = solve_triangular(L, solve_triangular(L, Mff, lower=True).T, lower=True)
    A = (A + A.T) / 2
    lam, Y = eigh(A, subset_by_index=[n - k, n - 1])
    order = np.argsort(-lam)
    lam, Y = lam[order], Y[:, order]
    keep = lam > 1e-14 * lam[0]
    lam, Y = lam[keep], Y[:, keep]
    phi_f = solve_triangular(L.T, Y, lower=False)
    phi_f /= np.sqrt(np.einsum("ij,ij->j", phi_f, Mff @ phi_f))
    phi = np.zeros((fa.ndof, len(lam)))
    phi[free] = phi_f
    phi = _normalise_sign(phi)
    omega = 1.0 / np.sqrt(lam)
    gamma, meff, mtot = {}, {}, {}
    for q, d in enumerate(AXES):
        # rigid-body influence vector of the ground motion: supports move with the ground, so the
        # coupling of free and supported DOFs through a consistent mass matrix is included
        iota = np.zeros(fa.ndof)
        iota[q::6] = 1.0
        Mi = M @ iota
        gamma[d] = phi.T @ Mi
        meff[d] = gamma[d] ** 2
        mtot[d] = float(iota @ Mi)          # total mass of the structure, as in EN 1998-1 4.3.3.3.1(3)
    return ModalResult(settings, ref, fa, submap, M, omega, phi, gamma, meff, mtot, factors, warnings)


# ===================================================================== shapes along members
def member_shape(mr: ModalResult, mid: str, vec: np.ndarray, points_per_element: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Points along an original member and their global displacements for a DOF vector ``vec``.

    Hermitian interpolation of each element's end displacements in local axes,
    with released end rotations recovered by condensation.
    """
    fa = mr.analysis
    P, D = [], []
    for eid, x0, x1 in mr.submap[mid]:
        md = fa.members[eid]
        ul = md.T @ vec[fa._dofs(md)]
        ul = _release_transform(md) @ ul if md.cidx else ul
        ni = mr.model.nodes[mr.model.members[eid].i]
        for t in np.linspace(0.0, 1.0, points_per_element + 1)[(0 if not P else 1):]:
            N = _shape(np.array([t]), md.L)[:, 0]
            u = N[0] * ul[0] + N[3] * ul[6]
            v = N[1] * ul[1] + N[2] * ul[5] + N[4] * ul[7] + N[5] * ul[11]
            w = N[1] * ul[2] - N[2] * ul[4] + N[4] * ul[8] - N[5] * ul[10]
            P.append(np.array([ni.x, ni.y, ni.z]) + md.lam[0] * md.L * t)
            D.append(md.lam.T @ np.array([u, v, w]))
    return np.array(P), np.array(D)
