"""Linear-elastic 2D frame analysis by the direct stiffness method.

Elements are Euler-Bernoulli beams with axial stiffness (3 DOF per node:
ux, uy, rz). Member loads are converted to consistent nodal loads, which for a
cubic Hermite element equal the exact fixed-end forces. Moment releases are
handled by static condensation at element level. Internal forces and
deflections between nodes are recovered exactly from statics and from the
double integration of M/EI, so results along a member carry no
discretisation error for polynomial loads.

Sign conventions (member local axes, x from node i to node j, y = x + 90 deg)
  N > 0 tension; V = dM/dx; M > 0 sagging (tension on the local -y face).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import Model, LoadCase, RCRectSection

# 3-point Gauss-Legendre on [0, 1]: exact for polynomials up to degree 5
_GP = np.array([0.5 - math.sqrt(15) / 10, 0.5, 0.5 + math.sqrt(15) / 10])
_GW = np.array([5 / 18, 8 / 18, 5 / 18])


class StructureUnstable(RuntimeError):
    """The structure is a mechanism (singular stiffness matrix)."""


def analysis_modulus(section: RCRectSection) -> float:
    """Short-term secant modulus Ecm (EN 1992-1-1 Table 3.1) in kN/m2."""
    fcm = section.concrete.fck + 8.0
    return 22_000.0 * (fcm / 10.0) ** 0.3 * 1000.0


# ------------------------------------------------------------------ helpers
@dataclass
class MemberData:
    id: str
    ni: int
    nj: int
    L: float
    c: float
    s: float
    EA: float      # kN
    EI: float      # kNm2
    rel: tuple[bool, bool]
    k_local: np.ndarray = field(repr=False, default=None)   # condensed 6x6
    k_full: np.ndarray = field(repr=False, default=None)    # uncondensed 6x6
    cidx: list = field(repr=False, default_factory=list)     # released local DOFs
    T: np.ndarray = field(repr=False, default=None)         # 6x6 rotation


@dataclass
class LocalLoads:
    """Member loads of one load case in local coordinates."""
    dist: list = field(default_factory=list)   # (a, b, qx1, qx2, qy1, qy2) kN/m
    point: list = field(default_factory=list)  # (a, Px, Py) kN


def _local_components(direction: str, w: float, c: float, s: float) -> tuple[float, float]:
    if direction == "local_x":
        return w, 0.0
    if direction == "local_y":
        return 0.0, w
    if direction == "global_x":
        return c * w, -s * w
    return s * w, c * w  # global_y


def _beam_k(EA: float, EI: float, L: float) -> np.ndarray:
    a, b = EA / L, EI / L ** 3
    return np.array([
        [a, 0, 0, -a, 0, 0],
        [0, 12 * b, 6 * b * L, 0, -12 * b, 6 * b * L],
        [0, 6 * b * L, 4 * b * L * L, 0, -6 * b * L, 2 * b * L * L],
        [-a, 0, 0, a, 0, 0],
        [0, -12 * b, -6 * b * L, 0, 12 * b, -6 * b * L],
        [0, 6 * b * L, 2 * b * L * L, 0, -6 * b * L, 4 * b * L * L],
    ])


def _rotation(c: float, s: float) -> np.ndarray:
    r = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.zeros((6, 6))
    T[:3, :3] = r
    T[3:, 3:] = r
    return T


def _shape(xi: np.ndarray, L: float) -> np.ndarray:
    """Rows: axial i, v_i, th_i, axial j, v_j, th_j shape functions at xi."""
    return np.array([
        1 - xi,
        1 - 3 * xi ** 2 + 2 * xi ** 3,
        L * (xi - 2 * xi ** 2 + xi ** 3),
        xi,
        3 * xi ** 2 - 2 * xi ** 3,
        L * (-xi ** 2 + xi ** 3),
    ])


def equivalent_nodal_loads(L: float, loads: LocalLoads) -> np.ndarray:
    """Consistent nodal load vector (local) = minus the fixed-end forces."""
    p = np.zeros(6)
    for a, b, qx1, qx2, qy1, qy2 in loads.dist:
        s = a + (b - a) * _GP
        w = (b - a) * _GW
        t = (s - a) / (b - a)
        qx = qx1 + (qx2 - qx1) * t
        qy = qy1 + (qy2 - qy1) * t
        N = _shape(s / L, L)
        p[[0, 3]] += N[[0, 3]] @ (w * qx)
        p[[1, 2, 4, 5]] += N[[1, 2, 4, 5]] @ (w * qy)
    for a, Px, Py in loads.point:
        N = _shape(np.array([a / L]), L)[:, 0]
        p[[0, 3]] += N[[0, 3]] * Px
        p[[1, 2, 4, 5]] += N[[1, 2, 4, 5]] * Py
    return p


def _condense(k: np.ndarray, rel: tuple[bool, bool]):
    """Return (released local indices, condensed k)."""
    cidx = [i for i, r in zip((2, 5), rel) if r]
    if not cidx:
        return cidx, k
    aidx = [i for i in range(6) if i not in cidx]
    kcc = k[np.ix_(cidx, cidx)]
    kac = k[np.ix_(aidx, cidx)]
    kk = np.zeros((6, 6))
    kk[np.ix_(aidx, aidx)] = k[np.ix_(aidx, aidx)] - kac @ np.linalg.solve(kcc, kac.T)
    return cidx, kk


def _condense_load(k: np.ndarray, cidx: list[int], p: np.ndarray) -> np.ndarray:
    if not cidx:
        return p
    aidx = [i for i in range(6) if i not in cidx]
    kcc = k[np.ix_(cidx, cidx)]
    kac = k[np.ix_(aidx, cidx)]
    out = np.zeros(6)
    out[aidx] = p[aidx] - kac @ np.linalg.solve(kcc, p[cidx])
    return out


# ------------------------------------------------------------- integrals
def _cum_dist(x: float, loads: LocalLoads, side: int):
    """Integrals over [0, x] of the member loads (left part of the cut).

    Returns (int qx, int qy, int (x-s) qy ds), including point loads with
    a < x (side=-1, left limit) or a <= x (side=+1, right limit).
    """
    Sx = Sy = My = 0.0
    for a, b, qx1, qx2, qy1, qy2 in loads.dist:
        if x <= a:
            continue
        t = min(x, b)
        # linear q(s) = alpha + beta s on [a, b]
        bx = (qx2 - qx1) / (b - a)
        ax = qx1 - bx * a
        by = (qy2 - qy1) / (b - a)
        ay = qy1 - by * a
        Sx += ax * (t - a) + bx * (t * t - a * a) / 2
        Sy += ay * (t - a) + by * (t * t - a * a) / 2

        def F(s):  # antiderivative of (x - s)(ay + by s)
            return ay * x * s + (by * x - ay) * s * s / 2 - by * s ** 3 / 3

        My += F(t) - F(a)
    for a, Px, Py in loads.point:
        if a < x or (side > 0 and a <= x):
            Sx += Px
            Sy += Py
            My += (x - a) * Py
    return Sx, Sy, My


# ------------------------------------------------------------- results
@dataclass
class MemberResult:
    x: np.ndarray        # station positions (m)
    side: np.ndarray     # -1 left limit, +1 right limit
    N: np.ndarray        # kN, tension positive
    V: np.ndarray        # kN, V = dM/dx
    M: np.ndarray        # kNm, sagging positive
    u: np.ndarray        # local axial displacement (m)
    v: np.ndarray        # local transverse displacement (m)
    v_rel: np.ndarray    # transverse displacement relative to the chord (m)
    end_forces: np.ndarray  # local forces on the member [Fxi, Fyi, Mi, Fxj, Fyj, Mj]


@dataclass
class CaseResult:
    case_id: str
    displacements: np.ndarray            # (n_nodes, 3): ux m, uy m, rz rad
    reactions: dict[str, np.ndarray]     # node -> [Rx kN, Ry kN, Mz kNm]
    spring_forces: dict[str, np.ndarray]
    members: dict[str, MemberResult]
    equilibrium: dict[str, float]        # residual sums (should be ~0)
    applied: dict[str, float]            # resultant of applied loads


class FrameAnalysis:
    """Assembles and factorises the frame once; solves any number of load cases."""

    def __init__(self, model: Model):
        self.model = model
        self.node_ids = list(model.nodes)
        self.nidx = {nid: k for k, nid in enumerate(self.node_ids)}
        n = len(self.node_ids)
        self.ndof = 3 * n
        self.members: dict[str, MemberData] = {}
        K = np.zeros((self.ndof, self.ndof))
        connected = np.zeros(n, dtype=bool)
        for mid, m in model.members.items():
            sec = model.sections[m.section]
            L, c, s = model.member_geometry(mid)
            E = analysis_modulus(sec) * sec.stiffness_factor
            md = MemberData(mid, self.nidx[m.i], self.nidx[m.j], L, c, s,
                            EA=E * sec.area_m2, EI=E * sec.inertia_m4, rel=(m.release_i, m.release_j))
            k = _beam_k(md.EA, md.EI, L)
            md.cidx, md.k_local = _condense(k, md.rel)
            md.k_full = k
            md.T = _rotation(c, s)
            dofs = self._dofs(md)
            K[np.ix_(dofs, dofs)] += md.T.T @ md.k_local @ md.T
            connected[md.ni] = connected[md.nj] = True
            self.members[mid] = md
        for k, ok in enumerate(connected):
            if not ok:
                raise StructureUnstable(f"node '{self.node_ids[k]}' is not connected to any member")

        # supports and springs
        fixed = np.zeros(self.ndof, dtype=bool)
        self.springs: dict[int, float] = {}
        for nid, sp in model.supports.items():
            b = 3 * self.nidx[nid]
            for off, flag, kk in ((0, sp.ux, sp.kx), (1, sp.uy, sp.ky), (2, sp.rz, sp.kr)):
                if flag:
                    fixed[b + off] = True
                elif kk > 0:
                    K[b + off, b + off] += kk
                    self.springs[b + off] = kk
        # rotational DOFs with no stiffness (every member hinged at that node) are pin nodes
        scale = max(np.max(np.abs(np.diag(K))), 1.0)
        self.auto_fixed: list[int] = []
        for k in range(n):
            d = 3 * k + 2
            if not fixed[d] and abs(K[d, d]) < 1e-12 * scale:
                fixed[d] = True
                self.auto_fixed.append(d)
        self.K = K
        self.fixed = fixed
        self.free = np.flatnonzero(~fixed)
        Kff = K[np.ix_(self.free, self.free)]
        if len(self.free) == 0:  # every DOF restrained (e.g. a beam fixed at both ends)
            self._chol = np.zeros((0, 0))
        else:
            try:
                self._chol = np.linalg.cholesky(Kff)
            except np.linalg.LinAlgError:
                raise StructureUnstable(self._mechanism_message(Kff)) from None
            # guard against near-singular systems that still factorise
            d = np.diag(self._chol) ** 2
            if d.min() < 1e-13 * d.max():
                raise StructureUnstable(self._mechanism_message(Kff))
        self.stations = self._build_stations()

    # ----------------------------------------------------------- internals
    def _dofs(self, md: MemberData) -> list[int]:
        return [3 * md.ni, 3 * md.ni + 1, 3 * md.ni + 2, 3 * md.nj, 3 * md.nj + 1, 3 * md.nj + 2]

    def _mechanism_message(self, Kff: np.ndarray) -> str:
        w, v = np.linalg.eigh(Kff)
        mode = v[:, 0]
        k = int(np.argmax(np.abs(mode)))
        dof = self.free[k]
        names = ("move in X", "move in Y", "rotate")
        return (f"structure is unstable (mechanism): node '{self.node_ids[dof // 3]}' is free to "
                f"{names[dof % 3]}. Check supports and member releases.")

    def _solve(self, rhs: np.ndarray) -> np.ndarray:
        if rhs.size == 0:
            return rhs.copy()
        y = np.linalg.solve(self._chol, rhs)
        return np.linalg.solve(self._chol.T, y)

    def local_loads(self, case: LoadCase) -> dict[str, LocalLoads]:
        out = {mid: LocalLoads() for mid in self.members}
        for mid, md in self.members.items():
            if case.self_weight:
                sec = self.model.sections[self.model.members[mid].section]
                w = -sec.concrete.density * sec.area_m2
                qx, qy = _local_components("global_y", w, md.c, md.s)
                out[mid].dist.append((0.0, md.L, qx, qx, qy, qy))
        for ld in case.distributed:
            md = self.members[ld.member]
            qx1, qy1 = _local_components(ld.direction, ld.w1, md.c, md.s)
            qx2, qy2 = _local_components(ld.direction, ld.w2, md.c, md.s)
            out[ld.member].dist.append((ld.a, ld.b, qx1, qx2, qy1, qy2))
        for ld in case.point:
            md = self.members[ld.member]
            px, py = _local_components(ld.direction, ld.p, md.c, md.s)
            out[ld.member].point.append((ld.a, px, py))
        return out

    def _build_stations(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        nseg = self.model.settings.stations_per_member
        breaks: dict[str, set] = {mid: set() for mid in self.members}
        for case in self.model.load_cases.values():
            for ld in case.distributed:
                breaks[ld.member].update((ld.a, ld.b))
            for ld in case.point:
                breaks[ld.member].add(ld.a)
        out = {}
        for mid, md in self.members.items():
            inner: list[float] = []
            for b in sorted(breaks[mid]):
                if 1e-9 < b < md.L - 1e-9 and not any(abs(b - c) < 1e-9 for c in inner):
                    inner.append(float(b))
            grid = [float(x) for x in np.linspace(0.0, md.L, nseg + 1)
                    if not any(abs(x - b) < 1e-9 for b in inner)]
            pts = sorted([(x, +1) for x in grid] + [(b, -1) for b in inner] + [(b, +1) for b in inner])
            x = np.array([p[0] for p in pts])
            side = np.array([p[1] for p in pts], dtype=int)
            x[0], x[-1] = 0.0, md.L
            side[-1] = -1  # last station is the left limit at node j (inside the member)
            out[mid] = (x, side)
        return out

    # ------------------------------------------------------------- public
    def solve_case(self, case: LoadCase) -> CaseResult:
        n = len(self.node_ids)
        P = np.zeros(self.ndof)
        for ld in case.nodal:
            b = 3 * self.nidx[ld.node]
            P[b:b + 3] += (ld.fx, ld.fy, ld.mz)
        for d in self.auto_fixed:
            if abs(P[d]) > 1e-12:
                raise StructureUnstable(
                    f"a moment is applied at node '{self.node_ids[d // 3]}', where every member is hinged")
        loads = self.local_loads(case)
        p_eq_local = {}
        for mid, md in self.members.items():
            p = equivalent_nodal_loads(md.L, loads[mid])
            p_eq_local[mid] = p
            pc = _condense_load(md.k_full, md.cidx, p)
            P[self._dofs(md)] += md.T.T @ pc

        U = np.zeros(self.ndof)
        U[self.free] = self._solve(P[self.free])
        R_all = self.K @ U - P

        reactions, springs = {}, {}
        for nid, sp in self.model.supports.items():
            b = 3 * self.nidx[nid]
            r = np.where(self.fixed[b:b + 3], R_all[b:b + 3], 0.0)
            reactions[nid] = r
            sf = np.array([-self.springs.get(b + k, 0.0) * U[b + k] for k in range(3)])
            if np.any(sf):
                springs[nid] = sf

        members = {mid: self._member_result(md, U, loads[mid], p_eq_local[mid]) for mid, md in self.members.items()}
        eq, applied = self._equilibrium(case, loads, U, reactions, springs)
        return CaseResult(case.id, U.reshape(n, 3).copy(), reactions, springs, members, eq, applied)

    def _member_result(self, md: MemberData, U: np.ndarray, loads: LocalLoads, p_eq: np.ndarray) -> MemberResult:
        ul = md.T @ U[self._dofs(md)]
        k = md.k_full
        if md.cidx:
            aidx = [i for i in range(6) if i not in md.cidx]
            kcc = k[np.ix_(md.cidx, md.cidx)]
            kca = k[np.ix_(md.cidx, aidx)]
            ul = ul.copy()
            ul[md.cidx] = np.linalg.solve(kcc, p_eq[md.cidx] - kca @ ul[aidx])
        f = k @ ul - p_eq
        xs, sides = self.stations[md.id]
        N = np.empty(len(xs)); V = np.empty(len(xs)); M = np.empty(len(xs))
        for q, (x, sd) in enumerate(zip(xs, sides)):
            Sx, Sy, My = _cum_dist(x, loads, sd)
            N[q] = -f[0] - Sx
            V[q] = f[1] + Sy
            M[q] = -f[2] + x * f[1] + My
        # exact double integration of curvature and axial strain between stations
        ux, ui = np.unique(xs, return_index=True)
        F1 = np.zeros(len(ux)); G = np.zeros(len(ux)); E = np.zeros(len(ux))
        for q in range(1, len(ux)):
            x0, x1 = ux[q - 1], ux[q]
            sg = x0 + (x1 - x0) * _GP
            kap = np.empty(3); eps = np.empty(3)
            for g, sgp in enumerate(sg):
                Sx, Sy, My = _cum_dist(sgp, loads, +1)
                kap[g] = (-f[2] + sgp * f[1] + My) / md.EI
                eps[g] = (-f[0] - Sx) / md.EA
            w = (x1 - x0) * _GW
            F1[q] = F1[q - 1] + w @ kap
            G[q] = G[q - 1] + w @ (sg * kap)
            E[q] = E[q - 1] + w @ eps
        L = md.L
        theta = (ul[4] - ul[1] - (L * F1[-1] - G[-1])) / L
        v_u = ul[1] + theta * ux + ux * F1 - G
        u_u = ul[0] + E
        v = np.interp(xs, ux, v_u)
        u = np.interp(xs, ux, u_u)
        v_rel = v - (ul[1] + (ul[4] - ul[1]) * xs / L)
        return MemberResult(xs, sides, N, V, M, u, v, v_rel, f)

    def _equilibrium(self, case, loads, U, reactions, springs):
        Fx = Fy = Mo = 0.0
        for ld in case.nodal:
            nd = self.model.nodes[ld.node]
            Fx += ld.fx; Fy += ld.fy
            Mo += ld.mz + nd.x * ld.fy - nd.y * ld.fx
        for mid, md in self.members.items():
            xi, yi = self.model.nodes[self.model.members[mid].i].x, self.model.nodes[self.model.members[mid].i].y

            def add(px_l, py_l, s):
                gx = md.c * px_l - md.s * py_l
                gy = md.s * px_l + md.c * py_l
                X, Y = xi + md.c * s, yi + md.s * s
                return gx, gy, X * gy - Y * gx

            for a, b, qx1, qx2, qy1, qy2 in loads[mid].dist:
                for wgt, s in zip((b - a) * _GW, a + (b - a) * _GP):
                    t = (s - a) / (b - a)
                    gx, gy, mo = add(qx1 + (qx2 - qx1) * t, qy1 + (qy2 - qy1) * t, s)
                    Fx += wgt * gx; Fy += wgt * gy; Mo += wgt * mo
            for a, px, py in loads[mid].point:
                gx, gy, mo = add(px, py, a)
                Fx += gx; Fy += gy; Mo += mo
        applied = {"Fx": Fx, "Fy": Fy, "Mz_origin": Mo}
        rx = ry = rm = 0.0
        for nid, r in list(reactions.items()) + list(springs.items()):
            nd = self.model.nodes[nid]
            rx += r[0]; ry += r[1]
            rm += r[2] + nd.x * r[1] - nd.y * r[0]
        return {"Fx": Fx + rx, "Fy": Fy + ry, "Mz_origin": Mo + rm}, applied


def analyse(model: Model) -> dict[str, CaseResult]:
    fa = FrameAnalysis(model)
    return {cid: fa.solve_case(lc) for cid, lc in model.load_cases.items()}
