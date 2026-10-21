"""Linear-elastic 3D frame analysis (space frames), direct stiffness method.

Members are Euler-Bernoulli beams with axial, St Venant torsion and biaxial
bending stiffness (6 DOF per node: ux uy uz rx ry rz). Member loads are applied
as consistent nodal loads (exact fixed-end forces); end releases are condensed
at element level. Internal forces and deflections along members are recovered
exactly from statics and the integration of curvature in both bending planes.

Result conventions (member local axes, see model3d):
  N > 0 tension; T torque about local x;
  My: bending in the x-z plane, > 0 with compression on the +z face, Vz = dMy/dx;
  Mz: bending in the x-y plane, > 0 with compression on the +y face, Vy = dMz/dx.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model3d import Model3D, LoadCase3D, RCSection3D
from .solver import StructureUnstable, _GP, _GW, _shape

NU_CONCRETE = 0.2   # EN 1992-1-1 3.1.3(4), uncracked


def analysis_modulus3d(sec: RCSection3D) -> float:
    fcm = sec.concrete.fck + 8.0
    return 22_000.0 * (fcm / 10.0) ** 0.3 * 1000.0   # kN/m2


def _plane_k(EI: float, L: float) -> np.ndarray:
    b = EI / L ** 3
    return b * np.array([[12, 6 * L, -12, 6 * L],
                         [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                         [-12, -6 * L, 12, -6 * L],
                         [6 * L, 2 * L * L, -6 * L, 4 * L * L]])


V_IDX = [1, 5, 7, 11]      # (v, theta_z) dofs: same sense as the 2D element
W_IDX = [2, 4, 8, 10]      # (w, theta_y) dofs: theta_y = -dw/dx
W_SIGN = np.array([1.0, -1.0, 1.0, -1.0])


def beam_k3d(EA, GJ, EIy, EIz, L) -> np.ndarray:
    k = np.zeros((12, 12))
    a, t = EA / L, GJ / L
    k[np.ix_([0, 6], [0, 6])] = [[a, -a], [-a, a]]
    k[np.ix_([3, 9], [3, 9])] = [[t, -t], [-t, t]]
    k[np.ix_(V_IDX, V_IDX)] = _plane_k(EIz, L)
    k[np.ix_(W_IDX, W_IDX)] = W_SIGN[:, None] * _plane_k(EIy, L) * W_SIGN[None, :]
    return k


def _loc(direction: str, w: float, lam: np.ndarray) -> np.ndarray:
    """Load vector (qx, qy, qz) in local axes for a load w in the given direction."""
    if direction.startswith("local_"):
        v = np.zeros(3)
        v["xyz".index(direction[-1])] = w
        return v
    g = np.zeros(3)
    g["xyz".index(direction[-1])] = w
    return lam @ g


@dataclass
class LocalLoads3D:
    dist: list = field(default_factory=list)   # (a, b, q1[3], q2[3])
    point: list = field(default_factory=list)  # (a, P[3])


def equivalent_nodal_loads3d(L: float, loads: LocalLoads3D) -> np.ndarray:
    p = np.zeros(12)
    for a, b, q1, q2 in loads.dist:
        s = a + (b - a) * _GP
        w = (b - a) * _GW
        t = (s - a) / (b - a)
        q = q1[None, :] + (q2 - q1)[None, :] * t[:, None]      # (3, 3): gauss x component
        N = _shape(s / L, L)                                     # rows: ax_i, v_i, th_i, ax_j, v_j, th_j
        p[[0, 6]] += N[[0, 3]] @ (w * q[:, 0])
        pv = N[[1, 2, 4, 5]] @ (w * q[:, 1])
        pw = N[[1, 2, 4, 5]] @ (w * q[:, 2])
        p[V_IDX] += pv
        p[W_IDX] += W_SIGN * pw
    for a, P in loads.point:
        N = _shape(np.array([a / L]), L)[:, 0]
        p[[0, 6]] += N[[0, 3]] * P[0]
        p[V_IDX] += N[[1, 2, 4, 5]] * P[1]
        p[W_IDX] += W_SIGN * N[[1, 2, 4, 5]] * P[2]
    return p


def _cum3d(x: float, loads: LocalLoads3D, side: int):
    """Integrals over [0, x]: (S[3] = int q, Mv = int (x-s) qy, Mw = int (x-s) qz)."""
    S = np.zeros(3)
    My = Mz = 0.0   # moment integrals of qz and qy
    for a, b, q1, q2 in loads.dist:
        if x <= a:
            continue
        t = min(x, b)
        beta = (q2 - q1) / (b - a)
        alpha = q1 - beta * a
        S += alpha * (t - a) + beta * (t * t - a * a) / 2

        def F(s, al, be):
            return al * x * s + (be * x - al) * s * s / 2 - be * s ** 3 / 3

        Mz += F(t, alpha[1], beta[1]) - F(a, alpha[1], beta[1])
        My += F(t, alpha[2], beta[2]) - F(a, alpha[2], beta[2])
    for a, P in loads.point:
        if a < x or (side > 0 and a <= x):
            S += P
            Mz += (x - a) * P[1]
            My += (x - a) * P[2]
    return S, Mz, My


@dataclass
class MemberData3D:
    id: str
    ni: int
    nj: int
    L: float
    lam: np.ndarray
    EA: float
    GJ: float
    EIy: float
    EIz: float
    cidx: list
    k_full: np.ndarray = field(repr=False)
    k_local: np.ndarray = field(repr=False)
    T: np.ndarray = field(repr=False)


@dataclass
class MemberResult3D:
    x: np.ndarray
    side: np.ndarray
    N: np.ndarray
    Vy: np.ndarray
    Vz: np.ndarray
    T: np.ndarray
    My: np.ndarray
    Mz: np.ndarray
    u: np.ndarray       # local displacements along the member (m)
    v: np.ndarray
    w: np.ndarray
    v_rel: np.ndarray
    w_rel: np.ndarray
    end_forces: np.ndarray


@dataclass
class CaseResult3D:
    case_id: str
    displacements: np.ndarray            # (n_nodes, 6)
    reactions: dict[str, np.ndarray]     # node -> 6 components
    spring_forces: dict[str, np.ndarray]
    members: dict[str, MemberResult3D]
    equilibrium: dict[str, float]
    applied: dict[str, float]


def _condense(k, cidx):
    if not cidx:
        return k.copy()
    aidx = [i for i in range(12) if i not in cidx]
    kcc = k[np.ix_(cidx, cidx)]
    kac = k[np.ix_(aidx, cidx)]
    kk = np.zeros((12, 12))
    kk[np.ix_(aidx, aidx)] = k[np.ix_(aidx, aidx)] - kac @ np.linalg.solve(kcc, kac.T)
    return kk


def _condense_load(k, cidx, p):
    if not cidx:
        return p
    aidx = [i for i in range(12) if i not in cidx]
    out = np.zeros(12)
    out[aidx] = p[aidx] - k[np.ix_(aidx, cidx)] @ np.linalg.solve(k[np.ix_(cidx, cidx)], p[cidx])
    return out


class FrameAnalysis3D:
    def __init__(self, model: Model3D):
        self.model = model
        self.node_ids = list(model.nodes)
        self.nidx = {nid: k for k, nid in enumerate(self.node_ids)}
        n = len(self.node_ids)
        self.ndof = 6 * n
        K = np.zeros((self.ndof, self.ndof))
        connected = np.zeros(n, dtype=bool)
        self.members: dict[str, MemberData3D] = {}
        for mid, m in model.members.items():
            sec = model.sections[m.section]
            L, lam = model.axes(mid)
            E = analysis_modulus3d(sec) * sec.stiffness_factor
            G = E / (2.0 * (1.0 + NU_CONCRETE))
            cidx = []
            for base, rel in ((0, m.rel_i), (6, m.rel_j)):
                if rel.t:
                    cidx.append(base + 3)
                if rel.my:
                    cidx.append(base + 4)
                if rel.mz:
                    cidx.append(base + 5)
            k = beam_k3d(E * sec.area_m2, G * sec.J_m4 * sec.torsion_factor, E * sec.Iy_m4, E * sec.Iz_m4, L)
            T = np.zeros((12, 12))
            for b in range(4):
                T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = lam
            md = MemberData3D(mid, self.nidx[m.i], self.nidx[m.j], L, lam, E * sec.area_m2, G * sec.J_m4 * sec.torsion_factor,
                              E * sec.Iy_m4, E * sec.Iz_m4, cidx, k, _condense(k, cidx), T)
            dofs = self._dofs(md)
            K[np.ix_(dofs, dofs)] += T.T @ md.k_local @ T
            connected[md.ni] = connected[md.nj] = True
            self.members[mid] = md
        for k, ok in enumerate(connected):
            if not ok:
                raise StructureUnstable(f"node '{self.node_ids[k]}' is not connected to any member")
        fixed = np.zeros(self.ndof, dtype=bool)
        self.springs: dict[int, float] = {}
        for nid, sp in model.supports.items():
            b = 6 * self.nidx[nid]
            for off in range(6):
                if sp.fix[off]:
                    fixed[b + off] = True
                elif sp.springs[off] > 0:
                    K[b + off, b + off] += sp.springs[off]
                    self.springs[b + off] = sp.springs[off]
        scale = max(float(np.max(np.abs(np.diag(K)))), 1.0)
        self.auto_fixed: list[int] = []
        for k in range(n):
            for off in (3, 4, 5):
                d = 6 * k + off
                if not fixed[d] and abs(K[d, d]) < 1e-12 * scale:
                    fixed[d] = True
                    self.auto_fixed.append(d)
        self.K, self.fixed = K, fixed
        self.free = np.flatnonzero(~fixed)
        Kff = K[np.ix_(self.free, self.free)]
        if len(self.free) == 0:
            self._chol = np.zeros((0, 0))
        else:
            try:
                self._chol = np.linalg.cholesky(Kff)
            except np.linalg.LinAlgError:
                raise StructureUnstable(self._mechanism_message(Kff)) from None
            d = np.diag(self._chol) ** 2
            if d.min() < 1e-13 * d.max():
                raise StructureUnstable(self._mechanism_message(Kff))
        self.stations = self._build_stations()

    def _dofs(self, md):
        return list(range(6 * md.ni, 6 * md.ni + 6)) + list(range(6 * md.nj, 6 * md.nj + 6))

    def _mechanism_message(self, Kff):
        w, v = np.linalg.eigh(Kff)
        dof = self.free[int(np.argmax(np.abs(v[:, 0])))]
        names = ("move in X", "move in Y", "move in Z", "rotate about X", "rotate about Y", "rotate about Z")
        return (f"structure is unstable (mechanism): node '{self.node_ids[dof // 6]}' is free to {names[dof % 6]}. "
                "Check supports and member releases.")

    def _solve(self, rhs):
        if rhs.size == 0:
            return rhs.copy()
        y = np.linalg.solve(self._chol, rhs)
        return np.linalg.solve(self._chol.T, y)

    def local_loads(self, case: LoadCase3D) -> dict[str, LocalLoads3D]:
        out = {mid: LocalLoads3D() for mid in self.members}
        for mid, md in self.members.items():
            if case.self_weight:
                sec = self.model.sections[self.model.members[mid].section]
                q = _loc("global_z", -sec.concrete.density * sec.area_m2, md.lam)
                out[mid].dist.append((0.0, md.L, q, q.copy()))
        for ld in case.distributed:
            md = self.members[ld.member]
            out[ld.member].dist.append((ld.a, ld.b, _loc(ld.direction, ld.w1, md.lam), _loc(ld.direction, ld.w2, md.lam)))
        for ld in case.point:
            md = self.members[ld.member]
            out[ld.member].point.append((ld.a, _loc(ld.direction, ld.p, md.lam)))
        return out

    def _build_stations(self):
        nseg = self.model.settings.stations_per_member
        breaks = {mid: set() for mid in self.members}
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
            grid = [float(x) for x in np.linspace(0.0, md.L, nseg + 1) if not any(abs(x - b) < 1e-9 for b in inner)]
            pts = sorted([(x, +1) for x in grid] + [(b, -1) for b in inner] + [(b, +1) for b in inner])
            x = np.array([p[0] for p in pts])
            side = np.array([p[1] for p in pts], dtype=int)
            x[0], x[-1] = 0.0, md.L
            side[-1] = -1
            out[mid] = (x, side)
        return out

    def solve_case(self, case: LoadCase3D) -> CaseResult3D:
        n = len(self.node_ids)
        P = np.zeros(self.ndof)
        for ld in case.nodal:
            b = 6 * self.nidx[ld.node]
            P[b:b + 6] += ld.f
        for d in self.auto_fixed:
            if abs(P[d]) > 1e-12:
                raise StructureUnstable(f"a moment is applied at node '{self.node_ids[d // 6]}' about a released axis")
        loads = self.local_loads(case)
        peq = {}
        for mid, md in self.members.items():
            p = equivalent_nodal_loads3d(md.L, loads[mid])
            peq[mid] = p
            P[self._dofs(md)] += md.T.T @ _condense_load(md.k_full, md.cidx, p)
        U = np.zeros(self.ndof)
        U[self.free] = self._solve(P[self.free])
        R_all = self.K @ U - P
        reactions, springs = {}, {}
        for nid in self.model.supports:
            b = 6 * self.nidx[nid]
            reactions[nid] = np.where(self.fixed[b:b + 6], R_all[b:b + 6], 0.0)
            sf = np.array([-self.springs.get(b + k, 0.0) * U[b + k] for k in range(6)])
            if np.any(sf):
                springs[nid] = sf
        members = {mid: self._member_result(md, U, loads[mid], peq[mid]) for mid, md in self.members.items()}
        eq, applied = self._equilibrium(case, loads, reactions, springs)
        return CaseResult3D(case.id, U.reshape(n, 6).copy(), reactions, springs, members, eq, applied)

    def _member_result(self, md: MemberData3D, U, loads: LocalLoads3D, p_eq) -> MemberResult3D:
        ul = md.T @ U[self._dofs(md)]
        k = md.k_full
        if md.cidx:
            aidx = [i for i in range(12) if i not in md.cidx]
            ul = ul.copy()
            ul[md.cidx] = np.linalg.solve(k[np.ix_(md.cidx, md.cidx)], p_eq[md.cidx] - k[np.ix_(md.cidx, aidx)] @ ul[aidx])
        f = k @ ul - p_eq
        xs, sides = self.stations[md.id]
        m = len(xs)
        N = np.empty(m); Vy = np.empty(m); Vz = np.empty(m); My = np.empty(m); Mz = np.empty(m)
        for q, (x, sd) in enumerate(zip(xs, sides)):
            S, Iz_, Iy_ = _cum3d(x, loads, sd)
            N[q] = -f[0] - S[0]
            Vy[q] = f[1] + S[1]
            Vz[q] = f[2] + S[2]
            Mz[q] = -f[5] + x * f[1] + Iz_
            My[q] = f[4] + x * f[2] + Iy_
        Tq = np.full(m, -f[3])
        ux, ui = np.unique(xs, return_index=True)
        n = len(ux)
        F1v = np.zeros(n); Gv = np.zeros(n); F1w = np.zeros(n); Gw = np.zeros(n); E = np.zeros(n)
        for q in range(1, n):
            x0, x1 = ux[q - 1], ux[q]
            sg = x0 + (x1 - x0) * _GP
            kv = np.empty(3); kw = np.empty(3); ea = np.empty(3)
            for g, s in enumerate(sg):
                S, Iz_, Iy_ = _cum3d(s, loads, +1)
                kv[g] = (-f[5] + s * f[1] + Iz_) / md.EIz
                kw[g] = (f[4] + s * f[2] + Iy_) / md.EIy
                ea[g] = (-f[0] - S[0]) / md.EA
            w = (x1 - x0) * _GW
            F1v[q] = F1v[q - 1] + w @ kv
            Gv[q] = Gv[q - 1] + w @ (sg * kv)
            F1w[q] = F1w[q - 1] + w @ kw
            Gw[q] = Gw[q - 1] + w @ (sg * kw)
            E[q] = E[q - 1] + w @ ea
        L = md.L
        th_v = (ul[7] - ul[1] - (L * F1v[-1] - Gv[-1])) / L
        th_w = (ul[8] - ul[2] - (L * F1w[-1] - Gw[-1])) / L
        v_u = ul[1] + th_v * ux + ux * F1v - Gv
        w_u = ul[2] + th_w * ux + ux * F1w - Gw
        u_u = ul[0] + E
        v = np.interp(xs, ux, v_u)
        w = np.interp(xs, ux, w_u)
        u = np.interp(xs, ux, u_u)
        v_rel = v - (ul[1] + (ul[7] - ul[1]) * xs / L)
        w_rel = w - (ul[2] + (ul[8] - ul[2]) * xs / L)
        return MemberResult3D(xs, sides, N, Vy, Vz, Tq, My, Mz, u, v, w, v_rel, w_rel, f)

    def _equilibrium(self, case, loads, reactions, springs):
        F = np.zeros(3)
        M = np.zeros(3)
        for ld in case.nodal:
            nd = self.model.nodes[ld.node]
            r = np.array([nd.x, nd.y, nd.z])
            f = np.array(ld.f[:3])
            F += f
            M += np.cross(r, f) + np.array(ld.f[3:])
        for mid, md in self.members.items():
            m = self.model.members[mid]
            ri = np.array([self.model.nodes[m.i].x, self.model.nodes[m.i].y, self.model.nodes[m.i].z])
            for a, b, q1, q2 in loads[mid].dist:
                for wg, s in zip((b - a) * _GW, a + (b - a) * _GP):
                    t = (s - a) / (b - a)
                    g = md.lam.T @ (q1 + (q2 - q1) * t)
                    F += wg * g
                    M += wg * np.cross(ri + md.lam[0] * s, g)
            for a, Pl in loads[mid].point:
                g = md.lam.T @ Pl
                F += g
                M += np.cross(ri + md.lam[0] * a, g)
        applied = {"Fx": F[0], "Fy": F[1], "Fz": F[2], "Mx": M[0], "My": M[1], "Mz": M[2]}
        for nid, r in list(reactions.items()) + list(springs.items()):
            nd = self.model.nodes[nid]
            pos = np.array([nd.x, nd.y, nd.z])
            F += r[:3]
            M += np.cross(pos, r[:3]) + r[3:]
        return {"Fx": F[0], "Fy": F[1], "Fz": F[2], "Mx": M[0], "My": M[1], "Mz": M[2]}, applied
