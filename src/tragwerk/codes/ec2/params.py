"""Nationally determined parameters for EN 1992-1-1:2004.

Every value below is the *recommended* value of EN 1992-1-1:2004 (+AC:2010).
A National Annex may change them; pass overrides through the model's
``design.params`` block, e.g. ``{"gamma_c": 1.5, "alpha_cc": 0.85}``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class EC2Params:
    # 2.4.2.4 partial factors for materials (persistent and transient)
    gamma_c: float = 1.5
    gamma_s: float = 1.15
    # 3.1.6 coefficients for long-term effects / unfavourable effects
    alpha_cc: float = 1.0
    alpha_ct: float = 1.0
    # 3.2.7 design strain limit of reinforcement: eps_ud = factor * eps_uk
    eps_ud_factor: float = 0.9
    # 6.2.2 members without design shear reinforcement
    C_Rd_c_numerator: float = 0.18      # C_Rd,c = 0.18 / gamma_c
    k1_shear: float = 0.15
    vmin_coefficient: float = 0.035     # v_min = 0.035 k^1.5 fck^0.5
    # 6.2.3 members with shear reinforcement
    cot_theta_min: float = 1.0
    cot_theta_max: float = 2.5
    # 6.1(4) minimum eccentricity e0 = max(h/30, 20 mm)
    e0_min_mm: float = 20.0
    e0_h_divisor: float = 30.0
    # 7.4.1(4) deflection limit span / 250 under the quasi-permanent combination
    deflection_limit_divisor: float = 250.0
    beta_sustained: float = 0.5         # 7.4.3(3) Eq. (7.19), sustained / repeated loading
    # 8.2(2) clear distance between bars: max(k1*phi, dg + k2, 20 mm)
    k1_spacing: float = 1.0
    k2_spacing_mm: float = 5.0
    # 9.2.1.1 beams, longitudinal reinforcement
    As_min_fctm_factor: float = 0.26
    As_min_absolute: float = 0.0013
    As_max_ratio: float = 0.04
    # 9.2.2 beams, shear reinforcement
    rho_w_min_factor: float = 0.08      # rho_w,min = 0.08 sqrt(fck) / fyk
    sl_max_factor: float = 0.75         # s_l,max = 0.75 d (1 + cot alpha)
    # 9.5.2 / 9.5.3 columns
    column_As_min_N_factor: float = 0.10
    column_As_min_Ac_ratio: float = 0.002
    column_phi_min_mm: float = 8.0
    column_link_spacing_phi_factor: float = 20.0
    column_link_spacing_max_mm: float = 400.0

    @property
    def C_Rd_c(self) -> float:
        return self.C_Rd_c_numerator / self.gamma_c


def make_params(overrides: dict | None = None) -> EC2Params:
    if not overrides:
        return EC2Params()
    known = {f.name for f in fields(EC2Params)}
    unknown = sorted(set(overrides) - known - {"gamma_G_sup", "gamma_G_inf", "gamma_Q", "gamma_Q_traffic", "xi", "psi"})
    if unknown:
        raise ValueError(f"unknown design parameter(s): {', '.join(unknown)}")
    return replace(EC2Params(), **{k: float(v) for k, v in overrides.items() if k in known})
