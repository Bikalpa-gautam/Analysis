"""Material properties to EN 1992-1-1:2004, sections 3.1 and 3.2."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ...model import Concrete, Rebar
from .params import EC2Params


@dataclass(frozen=True)
class ConcreteProps:
    fck: float      # MPa
    fcm: float      # MPa, Table 3.1
    fctm: float     # MPa, Table 3.1
    Ecm: float      # MPa, Table 3.1
    fcd: float      # MPa, 3.1.6(1) Eq. (3.15)
    eps_c2: float   # Table 3.1, parabola-rectangle
    eps_cu2: float
    n: float
    lam: float      # 3.1.7(3) Eq. (3.19)/(3.20)
    eta: float      # 3.1.7(3) Eq. (3.21)/(3.22)
    nu: float       # 6.2.2(6) Eq. (6.6N)

    def sigma(self, eps):
        """Parabola-rectangle design stress, Eq. (3.17)/(3.18). Compression strain positive."""
        import numpy as np
        e = np.asarray(eps, dtype=float)
        out = np.zeros_like(e)
        para = (e > 0) & (e < self.eps_c2)
        out[para] = self.fcd * (1.0 - (1.0 - e[para] / self.eps_c2) ** self.n)
        out[e >= self.eps_c2] = self.fcd
        return out


@dataclass(frozen=True)
class SteelProps:
    fyk: float
    fyd: float      # 3.2.7, Eq. fyd = fyk / gamma_s
    Es: float
    eps_yd: float
    eps_ud: float   # 3.2.7(2) Note 1: eps_ud = 0.9 eps_uk

    def sigma(self, eps):
        """Bilinear design stress with horizontal top branch (3.2.7(2) b). Compression positive."""
        import numpy as np
        return np.clip(self.Es * np.asarray(eps, dtype=float), -self.fyd, self.fyd)


def concrete_props(c: Concrete, p: EC2Params) -> ConcreteProps:
    fck = c.fck
    fcm = fck + 8.0
    if fck <= 50:
        fctm = 0.30 * fck ** (2.0 / 3.0)
        eps_c2, eps_cu2, n = 2.0e-3, 3.5e-3, 2.0
        lam, eta = 0.8, 1.0
    else:
        fctm = 2.12 * math.log(1.0 + fcm / 10.0)
        eps_c2 = (2.0 + 0.085 * (fck - 50.0) ** 0.53) * 1e-3
        eps_cu2 = (2.6 + 35.0 * ((90.0 - fck) / 100.0) ** 4) * 1e-3
        n = 1.4 + 23.4 * ((90.0 - fck) / 100.0) ** 4
        lam = 0.8 - (fck - 50.0) / 400.0
        eta = 1.0 - (fck - 50.0) / 200.0
    Ecm = 22_000.0 * (fcm / 10.0) ** 0.3
    return ConcreteProps(
        fck=fck, fcm=fcm, fctm=fctm, Ecm=Ecm,
        fcd=p.alpha_cc * fck / p.gamma_c,
        eps_c2=eps_c2, eps_cu2=eps_cu2, n=n, lam=lam, eta=eta,
        nu=0.6 * (1.0 - fck / 250.0),
    )


def steel_props(r: Rebar, p: EC2Params) -> SteelProps:
    fyd = r.fyk / p.gamma_s
    return SteelProps(fyk=r.fyk, fyd=fyd, Es=r.Es, eps_yd=fyd / r.Es, eps_ud=p.eps_ud_factor * r.eps_uk)
