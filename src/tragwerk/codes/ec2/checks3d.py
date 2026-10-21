"""EN 1992-1-1 checks for members of 3D frames.

Builds on the plane checks: the strong-axis (x-z plane) behaviour reuses the 2D
functions through ``RCSection3D.as_2d()``; the weak axis uses the section's
x-y view. New here: biaxial bending with axial force (5.8.9, Eq. 5.39),
shear in both directions, torsion (6.3) and slenderness about both axes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ...model3d import Member3D, RCSection3D
from ..base import FAIL, INFO, NOT_COVERED, PASS, CheckResult, Step, fmt, status_from_utilisation
from . import checks as c2
from .params import EC2Params
from .section import RCSection


@dataclass
class MemberForces3D:
    x: np.ndarray
    L: float
    N: np.ndarray
    Vy: np.ndarray
    Vz: np.ndarray
    T: np.ndarray
    My: np.ndarray
    Mz: np.ndarray
    uls_ids: list[str]
    uls_labels: list[str]
    qp_My: np.ndarray
    qp_ids: list[str]
    qp_labels: list[str]
    tip_at_i: bool = False
    qp_Mz: np.ndarray | None = None

    def plane(self, axis: str) -> c2.MemberForces:
        """2D view: axis 'y' = x-z plane (My, Vz), axis 'z' = x-y plane (Mz, Vy)."""
        M, V = (self.My, self.Vz) if axis == "y" else (self.Mz, self.Vy)
        return c2.MemberForces(self.x, self.L, self.N, V, M, self.uls_ids, self.uls_labels,
                               self.qp_My, self.qp_ids, self.qp_labels, self.tip_at_i)


@dataclass
class SectionModels:
    sec: RCSection3D
    sec2d: object
    view_y: object
    view_z: object
    rc_y: RCSection
    rc_z: RCSection


def section_models(sec: RCSection3D, p: EC2Params) -> SectionModels:
    vy, vz = sec.view("y"), sec.view("z")
    return SectionModels(sec, sec.as_2d(), vy, vz, RCSection(vy, p), RCSection(vz, p))


def biaxial_exponent(ratio: float) -> float:
    """EN 1992-1-1 5.8.9(4), rectangular sections: a = 1.0 / 1.5 / 2.0 at N_Ed/N_Rd = 0.1 / 0.7 / 1.0."""
    if ratio <= 0.1:
        return 1.0
    if ratio <= 0.7:
        return 1.0 + (ratio - 0.1) / 0.6 * 0.5
    if ratio <= 1.0:
        return 1.5 + (ratio - 0.7) / 0.3 * 0.5
    return 2.0


# ================================================================ biaxial
def check_biaxial(member: Member3D, S: SectionModels, f: MemberForces3D, p: EC2Params) -> CheckResult:
    sec = S.sec
    is_column = member.design.type == "column"
    Nc = -f.N
    c, st = S.rc_y.c, S.rc_y.st
    As_tot = float(S.rc_y.As.sum())
    NRd = (sec.b_mm * sec.h_mm * c.fcd + As_tot * st.fyd) * 1e-3
    ratio = np.maximum(Nc, 0.0) / NRd
    a = np.vectorize(biaxial_exponent)(ratio)
    e0y = max(sec.h_mm / p.e0_h_divisor, p.e0_min_mm)
    e0z = max(sec.b_mm / p.e0_h_divisor, p.e0_min_mm)

    def eta(My, Mz):
        uy = S.rc_y.utilisation(Nc, My)
        uz = S.rc_z.utilisation(Nc, Mz)
        with np.errstate(over="ignore", invalid="ignore"):
            e = np.where(np.isinf(uy) | np.isinf(uz), np.inf, (uy ** a + uz ** a) ** (1.0 / a))
        return e, uy, uz

    My, Mz = f.My, f.Mz
    if is_column:
        # 6.1(4) / 5.8.9(2): the minimum eccentricity acts about one axis at a time, the worse one governs
        cands = []
        for e0, which in ((e0y, "y"), (e0z, "z")):
            Mmin = np.where(Nc > 0, Nc * e0 * 1e-3, 0.0)
            base = My if which == "y" else Mz
            small = np.abs(base) < Mmin
            for sgn in (1.0, -1.0):
                mod = np.where(small, sgn * Mmin, base)
                My_c, Mz_c = (mod, Mz) if which == "y" else (My, mod)
                e, uy, uz = eta(My_c, Mz_c)
                cands.append((e, uy, uz, My_c, Mz_c))
        stack = np.stack([cd[0] for cd in cands])
        best = np.argmax(stack, axis=0)
        e = np.max(stack, axis=0)
        pick = lambda k: np.choose(best, [cd[k] for cd in cands])  # noqa: E731
        uy, uz, Myd, Mzd = pick(1), pick(2), pick(3), pick(4)
    else:
        e, uy, uz = eta(My, Mz)
        Myd, Mzd = My, Mz
    ci, si = (int(v) for v in np.unravel_index(int(np.argmax(e)), e.shape))
    NEd, MyEd, MzEd = float(Nc[ci, si]), float(Myd[ci, si]), float(Mzd[ci, si])
    av = float(a[ci, si])
    steps = [
        Step("N_{Ed}", "axial force (compression +)", fmt(NEd), NEd, "kN"),
        Step("M_{Edy}", "moment in the x-z plane (compression on +z face +)" + (", at least N_{Ed}·e_{0}" if is_column else ""), fmt(MyEd), MyEd, "kNm"),
        Step("M_{Edz}", "moment in the x-y plane (compression on +y face +)" + (", at least N_{Ed}·e_{0}" if is_column else ""), fmt(MzEd), MzEd, "kNm"),
    ]
    notes = []
    Nmin, Nmax = S.rc_y.N_min, S.rc_y.N_max
    if NEd > Nmax or NEd < Nmin:
        cap = Nmax if NEd > Nmax else Nmin
        util = NEd / cap if cap else math.inf
        steps.append(Step("N_{Rd}", "resistance to pure axial force", "", cap, "kN", "6.1"))
        notes.append("The axial force alone exceeds the section resistance.")
        capacity = cap
    else:
        MRdy = S.rc_y.solve_state(NEd, flip=MyEd < 0).M
        MRdz = S.rc_z.solve_state(NEd, flip=MzEd < 0).M
        uyv = abs(MyEd / MRdy) if abs(MyEd) > 1e-12 else 0.0
        uzv = abs(MzEd / MRdz) if abs(MzEd) > 1e-12 else 0.0
        util_exact = (uyv ** av + uzv ** av) ** (1.0 / av)
        steps += [
            Step("M_{Rdy}", "uniaxial resistance at N_{Ed}, x-z plane (strain compatibility, Fig. 6.1)", "", MRdy, "kNm", "6.1"),
            Step("M_{Rdz}", "uniaxial resistance at N_{Ed}, x-y plane", "", MRdz, "kNm", "6.1"),
            Step("N_{Rd}", "A_{c}·f_{cd} + A_{s}·f_{yd}", f"{fmt(sec.b_mm * sec.h_mm)}·{fmt(c.fcd)} + {fmt(As_tot)}·{fmt(st.fyd)}", NRd, "kN", "5.8.9(4)"),
            Step("a", "1.0 / 1.5 / 2.0 at N_{Ed}/N_{Rd} = 0.1 / 0.7 / 1.0, interpolated", f"N_{{Ed}}/N_{{Rd}} = {fmt(max(NEd, 0) / NRd)}", av, "", "5.8.9(4)"),
            Step("η", "[(M_{Edy}/M_{Rdy})^{a} + (M_{Edz}/M_{Rdz})^{a}]^{1/a}", f"[{fmt(uyv)}^{{{av:.2f}}} + {fmt(uzv)}^{{{av:.2f}}}]^{{1/{av:.2f}}}",
                 util_exact, "", "Eq. 5.39"),
        ]
        util = util_exact
        capacity = None
    notes.append("Eq. 5.39 is applied to every member; with bending about one axis only it reduces to the uniaxial check of 6.1.")
    return CheckResult(id="ec2.biaxial", title="Bending with axial force, both axes", clause="EN 1992-1-1 6.1, 5.8.9",
                       member=member.id, status=status_from_utilisation(util) if math.isfinite(util) else FAIL,
                       utilisation=util, demand=None, capacity=capacity, x=float(f.x[si]),
                       combination=f.uls_ids[ci], combination_label=f.uls_labels[ci], steps=steps, notes=notes)


# ================================================================ torsion
def check_torsion(member: Member3D, S: SectionModels, f: MemberForces3D, p: EC2Params) -> CheckResult | None:
    sec = S.sec
    c, st = S.rc_y.c, S.rc_y.st
    b, h = sec.b_mm, sec.h_mm
    A, u = b * h, 2 * (b + h)
    phil = max([lay.dia_mm for lay in (sec.top, sec.bottom, sec.side) if lay and lay.n > 0] or [0.0])
    edge = sec.cover_mm + (sec.links.dia_mm if sec.links else 0.0) + phil / 2
    tef = max(A / u, 2 * edge)
    Ak = (b - tef) * (h - tef)
    uk = 2 * ((b - tef) + (h - tef))
    fctd = p.alpha_ct * 0.7 * c.fctm / p.gamma_c
    TRdc = 2 * Ak * tef * fctd * 1e-6                       # kNm
    if float(np.max(np.abs(f.T))) < 0.1 * TRdc:
        return None   # small compatibility torsion, 6.3.1(2): minimum reinforcement suffices
    rv = c2.shear_resistance(S.rc_y, S.view_y, p, -f.N, f.My)
    cot = rv["cot"] if "cot" in rv else np.full_like(f.N, 1.0)
    sin_cos = cot / (1 + cot ** 2)
    TRdmax = 2 * c.nu * 1.0 * c.fcd * Ak * tef * sin_cos * 1e-6
    VRdmax = rv.get("VRdmax", 0.5 * S.rc_y.b * 0.9 * rv["d"] * c.nu * c.fcd * 1e-3)
    T, V = np.abs(f.T), np.abs(f.Vz)
    e631 = T / TRdc + V / rv["VRdc"]
    e629 = T / TRdmax + V / VRdmax
    if sec.links:
        A1 = math.pi * sec.links.dia_mm ** 2 / 4
        fywd = (sec.links.fyk or sec.rebar.fyk) / p.gamma_s
        TRds = 2 * Ak * A1 / sec.links.spacing_mm * fywd * cot * 1e-6
        elinks = np.where(e631 > 1.0, T / TRds + V / rv["VRds"], 0.0)
    else:
        TRds = np.zeros_like(T)
        elinks = np.where(e631 > 1.0, np.inf, 0.0)
    e = np.maximum(e629, elinks)
    ci, si = (int(v) for v in np.unravel_index(int(np.argmax(np.where(np.isfinite(e), e, 1e9))), e.shape))
    g = lambda arr: float(arr[ci, si]) if isinstance(arr, np.ndarray) and arr.ndim == 2 else float(arr)  # noqa: E731
    TEd, VEd = g(T), g(V)
    steps = [
        Step("T_{Ed}", "design torsional moment", fmt(TEd), TEd, "kNm"),
        Step("V_{Ed}", "concurrent shear force in the x-z plane", fmt(VEd), VEd, "kN"),
        Step("t_{ef}", "A/u, not less than twice the edge-to-bar-centre distance", f"max({fmt(A)}/{fmt(u)}, 2·{fmt(edge)})", tef, "mm", "6.3.2(1)"),
        Step("A_{k}", "(b − t_{ef})·(h − t_{ef})", f"({fmt(b)} − {fmt(tef)})·({fmt(h)} − {fmt(tef)})", Ak, "mm²", "6.3.2(1)"),
        Step("T_{Rd,c}", "2·A_{k}·t_{ef}·f_{ctd} (cracking torque)", f"2·{fmt(Ak)}·{fmt(tef)}·{fmt(fctd)}", TRdc, "kNm", "6.3.2(5)"),
        Step("check", "T_{Ed}/T_{Rd,c} + V_{Ed}/V_{Rd,c} ≤ 1: only minimum reinforcement needed", "", g(e631), "", "Eq. 6.31"),
        Step("T_{Rd,max}", "2·ν·α_{cw}·f_{cd}·A_{k}·t_{ef}·sin θ·cos θ", f"cot θ = {fmt(g(cot))}", g(TRdmax), "kNm", "Eq. 6.30"),
        Step("η_{struts}", "T_{Ed}/T_{Rd,max} + V_{Ed}/V_{Rd,max} ≤ 1", "", g(e629), "", "Eq. 6.29"),
    ]
    notes = ["Shown because T_{Ed} ≥ 0.1·T_{Rd,c}. Where torsion arises from compatibility only, 6.3.1(2) allows it to be neglected at ULS."]
    if g(e631) > 1.0:
        steps.append(Step("T_{Rd,s}", "2·A_{k}·(A_{sw,1}/s)·f_{ywd}·cot θ (outer leg)", "", g(TRds), "kNm", "Eq. 6.28"))
        steps.append(Step("η_{links}", "T_{Ed}/T_{Rd,s} + V_{Ed}/V_{Rd,s} ≤ 1 (links shared by shear and torsion)", "", g(elinks), "", "6.3.2(2)"))
        Asl = TEd * 1e6 * uk * g(cot) / (2 * Ak * st.fyd)
        steps.append(Step("ΣA_{sl}", "T_{Ed}·u_{k}·cot θ/(2·A_{k}·f_{yd}), additional longitudinal steel", "", Asl, "mm²", "Eq. 6.28"))
        notes.append("Distribute the additional longitudinal torsion steel around the perimeter; it is reported, not checked against the bars.")
    util = g(e)
    return CheckResult(id="ec2.torsion", title="Torsion with shear", clause="EN 1992-1-1 6.3", member=member.id,
                       status=status_from_utilisation(util) if math.isfinite(util) else FAIL, utilisation=util,
                       demand=TEd, capacity=g(TRdmax), unit="kNm", x=float(f.x[si]), combination=f.uls_ids[ci],
                       combination_label=f.uls_labels[ci], steps=steps, notes=notes)


# ======================================================== second order
def _slenderness(member: Member3D, S: SectionModels, L: float, NEd: float, axis: str):
    """(l0 mm, i mm, lambda, lambda_lim, n, omega, B) for one axis, 5.8.3."""
    c, st = S.rc_y.c, S.rc_y.st
    Ac = S.sec.b_mm * S.sec.h_mm
    depth = S.sec.h_mm if axis == "y" else S.sec.b_mm
    factor = member.design.l0_factor_y if axis == "y" else member.design.l0_factor_z
    l0 = factor * L * 1000.0
    i = depth / math.sqrt(12.0)
    lam = l0 / i
    n = max(NEd, 1e-9) * 1e3 / (Ac * c.fcd)
    omega = float(S.rc_y.As.sum()) * st.fyd / (Ac * c.fcd)
    B = math.sqrt(1.0 + 2.0 * omega)
    lam_lim = 20.0 * 0.7 * B * 0.7 / math.sqrt(n)
    return l0, i, lam, lam_lim, n, omega, B


def check_second_order(member: Member3D, S: SectionModels, f: MemberForces3D, p: EC2Params) -> CheckResult | None:
    """Nominal curvature method, EN 1992-1-1 5.8.8, with biaxial interaction 5.8.9."""
    sec = S.sec
    Nc_all = -f.N
    NEd_max = float(np.max(Nc_all))
    if NEd_max <= 0:
        return None
    c, st = S.rc_y.c, S.rc_y.st
    Ac = sec.b_mm * sec.h_mm
    info = {ax: _slenderness(member, S, f.L, NEd_max, ax) for ax in ("y", "z")}
    if all(v[2] <= v[3] for v in info.values()):
        return None
    bars = sec.bars()
    As_tot = sum(b[0] for b in bars)
    NRd = (Ac * c.fcd + As_tot * st.fyd) * 1e-3
    phi = member.design.creep_coefficient
    rows = []
    for k in range(f.N.shape[0]):
        jmax = int(np.argmax(Nc_all[k]))
        NEd = float(Nc_all[k, jmax])
        if NEd <= 0:
            continue
        per = {}
        for ax in ("y", "z"):
            l0, i, lam, lam_lim, n, omega, B = _slenderness(member, S, f.L, NEd, ax)
            M = (f.My if ax == "y" else f.Mz)[k]
            Ma, Mb = float(M[0]), float(M[-1])
            M02, M01 = (Ma, Mb) if abs(Ma) >= abs(Mb) else (Mb, Ma)
            M0e = 0.6 * M02 + 0.4 * M01
            if abs(M0e) < 0.4 * abs(M02):
                M0e = math.copysign(0.4 * abs(M02), M02)
            sgn = 1.0 if M0e >= 0 else -1.0
            ei = l0 / 400.0                                    # 5.2(7), simplified for isolated members
            M2 = 0.0
            detail = {}
            if lam > lam_lim:
                coord = [b[2] for b in bars] if ax == "y" else [b[1] for b in bars]
                i_s = math.sqrt(sum(b[0] * z * z for b, z in zip(bars, coord)) / As_tot) if As_tot else 0.0
                depth = sec.h_mm if ax == "y" else sec.b_mm
                d = depth / 2.0 + i_s
                nu_ = 1.0 + omega
                Kr = min(max((nu_ - n) / (nu_ - 0.4), 0.0), 1.0)
                qpM = f.qp_My if ax == "y" else (f.qp_Mz if f.qp_Mz is not None else f.qp_My)
                M0qp = float(np.max(np.abs(qpM))) if qpM.size else 0.0
                M0Ed = max(abs(M02), 1e-9)
                phi_ef = phi * min(1.0, M0qp / M0Ed)
                beta = 0.35 + c.fck / 200.0 - lam / 150.0
                Kphi = max(1.0, 1.0 + beta * phi_ef)
                inv_r = Kr * Kphi * st.eps_yd / (0.45 * d)
                e2 = inv_r * l0 ** 2 / 10.0
                M2 = NEd * e2 * 1e-3
                detail = dict(i_s=i_s, d=d, Kr=Kr, phi_ef=phi_ef, beta=beta, Kphi=Kphi, inv_r=inv_r, e2=e2)
            e0 = max((sec.h_mm if ax == "y" else sec.b_mm) / p.e0_h_divisor, p.e0_min_mm)
            per[ax] = dict(lam=lam, lam_lim=lam_lim, l0=l0, M01=M01, M02=M02, M0e=M0e, ei=ei, M2=M2, sgn=sgn,
                           e0=e0, n=n, omega=omega, B=B, **detail)
        a = biaxial_exponent(max(NEd, 0.0) / NRd)
        best = None
        for imp in ("y", "z"):                                   # imperfection about one axis at a time
            Md = {}
            for ax in ("y", "z"):
                q = per[ax]
                add = NEd * q["ei"] * 1e-3 if ax == imp else 0.0
                MEd = max(abs(q["M02"]), abs(q["M0e"]) + add + q["M2"], abs(q["M01"]) + 0.5 * q["M2"])
                if ax == imp:
                    MEd = max(MEd, NEd * q["e0"] * 1e-3)
                Md[ax] = q["sgn"] * MEd
            uy = float(S.rc_y.utilisation(np.array([NEd]), np.array([Md["y"]]))[0])
            uz = float(S.rc_z.utilisation(np.array([NEd]), np.array([Md["z"]]))[0])
            e = math.inf if math.isinf(uy) or math.isinf(uz) else (uy ** a + uz ** a) ** (1 / a)
            if best is None or e > best[0]:
                best = (e, imp, Md, uy, uz)
        rows.append((best[0], k, NEd, a, per, best))
    if not rows:
        return None
    eta, k, NEd, a, per, (e, imp, Md, uy, uz) = max(rows, key=lambda r: r[0])
    steps = [Step("N_{Ed}", "maximum compression in the governing combination", fmt(NEd), NEd, "kN")]
    for ax in ("y", "z"):
        q = per[ax]
        plane = "x-z" if ax == "y" else "x-y"
        steps.append(Step(f"λ_{{{ax}}}", f"slenderness, {plane} plane; λ_{{lim}} = 20·A·B·C/√n with A = C = 0.7, B = √(1 + 2ω) = {fmt(q['B'])}",
                          f"λ_{{lim}} = {fmt(q['lam_lim'])}", q["lam"], "", "5.8.3.1"))
        if q["M2"] > 0:
            steps += [
                Step(f"K_{{r,{ax}}}", "(n_{u} − n)/(n_{u} − n_{bal}) ≤ 1, n_{u} = 1 + ω, n_{bal} = 0.4", f"n = {fmt(q['n'])}, ω = {fmt(q['omega'])}", q["Kr"], "", "Eq. 5.36"),
                Step(f"K_{{φ,{ax}}}", "1 + β·φ_{ef} ≥ 1, β = 0.35 + f_{ck}/200 − λ/150", f"β = {fmt(q['beta'])}, φ_{{ef}} = {fmt(q['phi_ef'])}", q["Kphi"], "", "Eq. 5.37"),
                Step(f"1/r_{{{ax}}}", "K_{r}·K_{φ}·ε_{yd}/(0.45·d), d = h/2 + i_{s}", f"d = {fmt(q['d'])} mm", q["inv_r"] * 1e3, "1/m", "Eq. 5.34, 5.35"),
                Step(f"e_{{2,{ax}}}", "(1/r)·l_{0}²/c, c = 10", f"l_{{0}} = {fmt(q['l0'])} mm", q["e2"], "mm", "Eq. 5.33"),
                Step(f"M_{{2,{ax}}}", "N_{Ed}·e_{2}", "", q["M2"], "kNm", "5.8.8.2(3)"),
            ]
        steps.append(Step(f"M_{{0e,{ax}}}", "0.6·M_{02} + 0.4·M_{01} ≥ 0.4·M_{02}", f"M_{{02}} = {fmt(q['M02'])}, M_{{01}} = {fmt(q['M01'])}", q["M0e"], "kNm", "Eq. 5.32"))
        steps.append(Step(f"M_{{Ed,{ax}}}", "max(M_{02}, M_{0e} + N·e_{i} + M_{2}, M_{01} + M_{2}/2), e_{i} = l_{0}/400" + (" (imperfection about this axis)" if ax == imp else ""),
                          f"e_{{i}} = {fmt(q['ei'])} mm", Md[ax], "kNm", "5.8.8.2, 5.2(7)"))
    steps += [
        Step("η_{y}, η_{z}", "M_{Ed}/M_{Rd} about each axis at N_{Ed}", f"{fmt(uy)}, {fmt(uz)}", max(uy, uz), "", "6.1"),
        Step("η", "[(M_{Edy}/M_{Rdy})^{a} + (M_{Edz}/M_{Rdz})^{a}]^{1/a}", f"a = {a:.2f}", eta, "", "Eq. 5.39"),
    ]
    notes = ["Nominal curvature method for an isolated member; l_{0} from the member's effective length factors. "
             "For an unbraced (sway) frame, set the factors accordingly.",
             "φ_{ef} = φ(∞,t_{0})·M_{0Eqp}/M_{0Ed} with M_{0Eqp} the largest quasi-permanent moment."]
    return CheckResult(id="ec2.second_order", title="Slender column, second-order effects", clause="EN 1992-1-1 5.8.8, 5.8.9",
                       member=member.id, status=status_from_utilisation(eta) if math.isfinite(eta) else FAIL, utilisation=eta,
                       combination=f.uls_ids[k], combination_label=f.uls_labels[k], steps=steps, notes=notes)


# ============================================================ slenderness
def check_slenderness3d(member: Member3D, S: SectionModels, f: MemberForces3D, p: EC2Params) -> list[CheckResult]:
    out = []
    NEd = float(np.max(-f.N))
    for axis in ("y", "z"):
        l0, i, lam, lam_lim, n, omega, B = _slenderness(member, S, f.L, NEd, axis)
        depth = S.sec.h_mm if axis == "y" else S.sec.b_mm
        steps = [
            Step("l_{0}", f"effective length for buckling in the x-{'z' if axis == 'y' else 'y'} plane", "", l0, "mm", "5.8.3.2"),
            Step("i", f"{'h' if axis == 'y' else 'b'}/√12", f"{fmt(depth)}/√12", i, "mm"),
            Step("λ", "l_{0}/i", f"{fmt(l0)}/{fmt(i)}", lam, "", "5.8.3.2(1)"),
            Step("n", "N_{Ed}/(A_{c}·f_{cd})", "", n, "", "5.8.3.1(1)"),
            Step("B", "√(1 + 2ω), ω = A_{s}·f_{yd}/(A_{c}·f_{cd})", f"ω = {fmt(omega)}", B, "", "5.8.3.1(1)"),
            Step("λ_{lim}", "20·A·B·C/√n with A = 0.7, C = 0.7", f"20·0.7·{fmt(B)}·0.7/√{fmt(n)}", lam_lim, "", "Eq. 5.13N"),
        ]
        ok = lam <= lam_lim
        out.append(CheckResult(
            id=f"ec2.slenderness_{axis}", title=f"Slenderness, x-{'z' if axis == 'y' else 'y'} plane", clause="EN 1992-1-1 5.8.3.1",
            member=member.id, status=PASS if ok else INFO, utilisation=None, demand=lam, capacity=lam_lim,
            steps=steps, notes=["λ ≤ λ_lim: second-order effects may be ignored."] if ok else
            ["λ > λ_lim: second-order effects are included in the check 'Slender column, second-order effects' (5.8.8)."]))
    return out


# =============================================================== driver
def check_member3d(member: Member3D, S: SectionModels, f: MemberForces3D, p: EC2Params) -> list[CheckResult]:
    if member.design.type == "none":
        return []
    out = [check_biaxial(member, S, f, p)]
    sz = c2.check_shear(member, S.rc_y, S.view_y, f.plane("y"), p)
    sz.id, sz.title = "ec2.shear_z", "Shear, x-z plane (V_{z})"
    sy = c2.check_shear(member, S.rc_z, S.view_z, f.plane("z"), p)
    sy.id, sy.title = "ec2.shear_y", "Shear, x-y plane (V_{y})"
    out += [sz, sy]
    tors = check_torsion(member, S, f, p)
    if tors:
        out.append(tors)
    m2d = _Member2DView(member)
    out += c2.check_detailing(m2d, RCSection(S.sec2d, p), S.sec2d, f.plane("y"), p)
    if member.design.type == "column":
        so = check_second_order(member, S, f, p)
        if so:
            out.append(so)
        out += check_slenderness3d(member, S, f, p)
    else:
        rc2 = RCSection(S.sec2d, p)
        ld = c2.check_span_depth(m2d, rc2, S.sec2d, f.plane("y"), p)
        defl = c2.check_deflection(m2d, rc2, S.sec2d, f.plane("y"), p)
        if ld.status == FAIL and defl.status == PASS:
            ld.status = INFO
            ld.notes.append("Limit exceeded, so the calculated deflection (7.4.3) governs; it passes.")
        out += [ld, defl]
    return out


class _Member2DView:
    """Exposes a Member3D through the attributes the plane checks use."""

    def __init__(self, m: Member3D):
        self.id = m.id
        self.design = _Design2D(m.design)


class _Design2D:
    def __init__(self, d):
        self.type = d.type
        self.deflection_system = d.deflection_system
        self.creep_coefficient = d.creep_coefficient
        self.l0_factor = d.l0_factor_y
