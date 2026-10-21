"""Resistance of a reinforced-concrete rectangle to bending with axial force.

EN 1992-1-1:2004, 6.1. Strain compatibility with
* the parabola-rectangle diagram for concrete (3.1.7(1), Eq. 3.17/3.18),
* a bilinear steel diagram with horizontal top branch (3.2.7(2) b),
* the ultimate strain distributions of Fig. 6.1, generated with the pivots
  A (steel strain eps_ud), B (concrete eps_cu2) and C (eps_c2 at depth
  (1 - eps_c2/eps_cu2) h). Pivot A only parameterises the planes; the steel
  stress stays at f_yd beyond yield, which is conservative against option b
  without a strain limit.

Internal sign convention: compressive strains and forces are positive,
depth y is measured from the top face (local +y face of the member),
moments are taken about mid-depth and are positive when the top is in
compression (sagging in the analysis convention).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...model import RCRectSection
from .materials import ConcreteProps, SteelProps, concrete_props, steel_props
from .params import EC2Params

_GL_X, _GL_W = np.polynomial.legendre.leggauss(5)


@dataclass
class StrainState:
    s: float            # boundary parameter in [0, 4]
    eps_top: float
    eps_bot: float
    N: float            # kN, compression +
    M: float            # kNm, top compression +
    x: float | None     # neutral axis depth from the compressed face (mm), None if no zero crossing
    steel: list         # (label, depth mm, strain, stress MPa)


class RCSection:
    """EC2 resistance model of an RCRectSection."""

    def __init__(self, sec: RCRectSection, p: EC2Params, displaced_concrete: bool = True):
        self.sec = sec
        self.p = p
        self.c: ConcreteProps = concrete_props(sec.concrete, p)
        self.st: SteelProps = steel_props(sec.rebar, p)
        self.b = sec.b_mm
        self.h = sec.h_mm
        lay = sec.layers()
        self.As = np.array([a for a, _, _ in lay], dtype=float)
        self.ys = np.array([y for _, y, _ in lay], dtype=float)
        self.labels = [l for _, _, l in lay]
        self.displaced = displaced_concrete
        self._curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------ basics
    @property
    def Ac(self) -> float:
        return self.b * self.h

    def resultants(self, eps_top: float, eps_bot: float, flip: bool = False) -> tuple[float, float]:
        """(N kN, M kNm) for a linear strain plane. flip=True mirrors the section (top <-> bottom)."""
        h, c = self.h, self.c
        ys = self.h - self.ys if flip else self.ys

        def eps(y):
            return eps_top + (eps_bot - eps_top) * y / h

        # concrete: split [0, h] where the stress law changes, Gauss on each piece
        cuts = {0.0, h}
        de = eps_bot - eps_top
        if abs(de) > 1e-15:
            for target in (0.0, c.eps_c2):
                y = (target - eps_top) * h / de
                if 0.0 < y < h:
                    cuts.add(y)
        cuts = sorted(cuts)
        Nc = Mc = 0.0
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            yg = 0.5 * (y1 - y0) * _GL_X + 0.5 * (y1 + y0)
            wg = 0.5 * (y1 - y0) * _GL_W
            sg = c.sigma(eps(yg))
            Nc += self.b * (wg @ sg)
            Mc += self.b * (wg @ (sg * (h / 2.0 - yg)))
        es = eps(ys)
        ss = self.st.sigma(es)
        if self.displaced:
            ss = ss - c.sigma(es)
        Ns = self.As @ ss
        Ms = self.As @ (ss * (h / 2.0 - ys))
        N = (Nc + Ns) * 1e-3
        M = (Mc + Ms) * 1e-6
        return N, (-M if flip else M)

    def _d_max(self, flip: bool) -> float:
        if len(self.ys) == 0:
            return self.h
        return float(np.max(self.h - self.ys if flip else self.ys))

    def plane(self, s: float, flip: bool = False) -> tuple[float, float]:
        """Ultimate strain plane (eps_top, eps_bot) on the boundary, s in [0, 4]."""
        c, h = self.c, self.h
        eud = self.st.eps_ud
        dm = self._d_max(flip)
        s = min(max(s, 0.0), 4.0)
        if s <= 2.0:  # pivot A: eps(d_max) = -eps_ud
            et = -eud * (1.0 - s) if s <= 1.0 else (s - 1.0) * c.eps_cu2
            eb = et + (-eud - et) * h / dm
        elif s <= 3.0:  # pivot B: eps_top = eps_cu2, x from x_AB to h
            xab = c.eps_cu2 / (c.eps_cu2 + eud) * dm
            x = xab + (s - 2.0) * (h - xab)
            et, eb = c.eps_cu2, c.eps_cu2 * (1.0 - h / x)
        else:  # pivot C
            yc = (1.0 - c.eps_c2 / c.eps_cu2) * h
            eb = (s - 3.0) * c.eps_c2
            m = (eb - c.eps_c2) / (h - yc)
            et = c.eps_c2 - m * yc
        return (eb, et) if flip else (et, eb)

    def boundary_point(self, s: float, flip: bool = False) -> tuple[float, float]:
        et, eb = self.plane(s, flip)
        return self.resultants(et, eb)

    # ---------------------------------------------------------- capacity
    @property
    def N_min(self) -> float:
        return self.boundary_point(0.0)[0]

    @property
    def N_max(self) -> float:
        return self.boundary_point(4.0)[0]

    def curve(self, flip: bool, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        key = int(flip)
        if key not in self._curves:
            ss = np.concatenate([np.linspace(k, k + 1, n, endpoint=(k == 3)) for k in range(4)])
            pts = np.array([self.boundary_point(s, flip) for s in ss])
            self._curves[key] = (pts[:, 0], pts[:, 1])
        return self._curves[key]

    def m_bounds(self, N) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated (M_hi, M_lo) at axial force N (kN, compression +). Fast, approximate."""
        Np, Mp = self.curve(False)
        Nn, Mn = self.curve(True)
        return np.interp(N, Np, Mp), np.interp(N, Nn, Mn)

    def solve_state(self, N: float, flip: bool = False, tol: float = 1e-9) -> StrainState:
        """Exact boundary state with axial force N on the positive (flip=False) or negative side."""
        lo, hi = 0.0, 4.0
        Nlo = self.boundary_point(lo, flip)[0]
        Nhi = self.boundary_point(hi, flip)[0]
        if not (Nlo - 1e-9 <= N <= Nhi + 1e-9):
            raise ValueError("axial force outside the resistance range")
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.boundary_point(mid, flip)[0] < N:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        s = 0.5 * (lo + hi)
        return self.state(s, flip)

    def state(self, s: float, flip: bool = False) -> StrainState:
        et, eb = self.plane(s, flip)
        N, M = self.resultants(et, eb)
        x = None
        ec, et_ = (eb, et) if flip else (et, eb)   # compressed face first
        if ec > 0 and et_ < 0:
            x = ec / (ec - et_) * self.h
        elif ec > 0 and et_ >= 0:
            x = math.inf
        steel = []
        for lab, y in zip(self.labels, self.ys):
            e = et + (eb - et) * y / self.h
            steel.append((lab, float(y), float(e), float(self.st.sigma(e))))
        return StrainState(s, et, eb, N, M, x, steel)

    def utilisation(self, N, M):
        """Vectorised utilisation for bending with axial force (constant-N approach).

        N, M arrays in kN (compression +) and kNm (top compression +).
        Returns u >= 0; values > 1 mean the point lies outside the interaction diagram.
        """
        N = np.asarray(N, dtype=float)
        M = np.asarray(M, dtype=float)
        u = np.zeros(np.broadcast(N, M).shape)
        Nmin, Nmax = self.curve(False)[0][0], self.curve(False)[0][-1]
        hi, lo = self.m_bounds(N)
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = M >= 0
            u_pos = np.where(hi > 0, M / hi, np.where(M > hi, np.inf, 0.0))
            u_pos = np.where((lo > 0) & (M > 0), np.maximum(u_pos, lo / M), u_pos)
            u_pos = np.where((lo > 0) & (M <= 0), np.inf, u_pos)
            u_neg = np.where(lo < 0, M / lo, np.where(M < lo, np.inf, 0.0))
            u_neg = np.where((hi < 0) & (M < 0), np.maximum(u_neg, hi / M), u_neg)
            u_neg = np.where((hi < 0) & (M >= 0), np.inf, u_neg)
            u = np.where(pos, u_pos, u_neg)
            u = np.where(N > Nmax, np.maximum(u, N / Nmax), u)
            u = np.where(N < Nmin, np.maximum(u, np.where(Nmin < 0, N / Nmin, np.inf)), u)
        return np.nan_to_num(u, nan=0.0, posinf=np.inf)

    # ---------------------------------------------- cracked / uncracked
    def transformed(self, Ec_eff: float, sagging: bool = True) -> dict:
        """Uncracked and cracked transformed-section properties (mm, mm4).

        alpha_e = Es / Ec_eff. Depths measured from the compressed face.
        Bars in the compression zone use (alpha_e - 1) to allow for displaced concrete.
        """
        ae = self.st.Es / Ec_eff
        b, h = self.b, self.h
        ys = self.ys if sagging else self.h - self.ys
        As = self.As
        # uncracked
        A1 = b * h + (ae - 1.0) * As.sum()
        y1 = (b * h * h / 2.0 + (ae - 1.0) * (As @ ys)) / A1
        I1 = b * h ** 3 / 12.0 + b * h * (h / 2.0 - y1) ** 2 + (ae - 1.0) * (As @ (ys - y1) ** 2)
        # cracked: solve b x^2/2 + sum(a_i As_i (x - y_i)) = 0 with a_i = ae-1 above x, ae below
        x = self._cracked_na(ae, ys)
        coef = np.where(ys < x, ae - 1.0, ae)
        I2 = b * x ** 3 / 3.0 + coef @ (As * (ys - x) ** 2)
        return {"alpha_e": ae, "A_I": A1, "y_I": y1, "I_I": I1, "x_II": x, "I_II": I2}

    def _cracked_na(self, ae: float, ys: np.ndarray) -> float:
        b, As = self.b, self.As

        def f(x):
            coef = np.where(ys < x, ae - 1.0, ae)
            return b * x * x / 2.0 + coef @ (As * (x - ys))

        lo, hi = 1e-9, self.h
        if f(hi) <= 0:
            return self.h
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
