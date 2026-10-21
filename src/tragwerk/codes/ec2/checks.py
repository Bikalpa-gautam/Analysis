"""Member checks to EN 1992-1-1:2004 with recommended values.

Every check works on the forces of *all* combinations at *all* stations of a
member, picks the governing one, and records the calculation steps for it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...model import Member, RCRectSection
from ..base import FAIL, INFO, NOT_COVERED, PASS, WARN, CheckResult, Step, fmt, status_from_utilisation
from .params import EC2Params
from .section import RCSection

K_SYSTEM = {  # Table 7.4N, structural system factor K
    "simply_supported": 1.0,
    "end_span": 1.3,
    "interior_span": 1.5,
    "cantilever": 0.4,
}


@dataclass
class MemberForces:
    """Combined forces along one member (rows = combinations, columns = stations)."""
    x: np.ndarray
    L: float
    uls_N: np.ndarray       # analysis convention: tension +
    uls_V: np.ndarray
    uls_M: np.ndarray       # sagging +
    uls_ids: list[str]
    uls_labels: list[str]
    qp_M: np.ndarray
    qp_ids: list[str]
    qp_labels: list[str]
    tip_at_i: bool = False  # cantilevers: the free end is node i


# ======================================================== bending + axial
def check_bending_axial(member: Member, rc: RCSection, f: MemberForces, p: EC2Params) -> CheckResult:
    is_column = member.design.type == "column"
    Nc = -f.uls_N                       # compression positive
    M = f.uls_M.copy()
    e0 = max(rc.h / p.e0_h_divisor, p.e0_min_mm)  # mm
    if is_column:
        # 6.1(4): at least N_Ed * e0, applied in the more unfavourable direction
        Mmin = np.where(Nc > 0, Nc * e0 * 1e-3, 0.0)
        small = np.abs(M) < Mmin
        Mp = np.where(small, Mmin, M)
        Mn = np.where(small, -Mmin, M)
        up, un = rc.utilisation(Nc, Mp), rc.utilisation(Nc, Mn)
        M = np.where(un > up, Mn, Mp)
        u = np.maximum(up, un)
    else:
        u = rc.utilisation(Nc, M)
    k = np.unravel_index(int(np.argmax(u)), u.shape)
    ci, si = int(k[0]), int(k[1])
    NEd, MEd, MEd_raw = float(Nc[ci, si]), float(M[ci, si]), float(f.uls_M[ci, si])

    steps: list[Step] = []
    c, st = rc.c, rc.st
    steps.append(Step("f_{cd}", "α_{cc}·f_{ck}/γ_{c}", f"{fmt(p.alpha_cc)}·{fmt(c.fck)}/{fmt(p.gamma_c)}", c.fcd, "MPa", "3.1.6(1)"))
    steps.append(Step("f_{yd}", "f_{yk}/γ_{s}", f"{fmt(st.fyk)}/{fmt(p.gamma_s)}", st.fyd, "MPa", "3.2.7"))
    steps.append(Step("ε_{c2}, ε_{cu2}", "Table 3.1", f"{c.eps_c2 * 1e3:.2f} ‰, {c.eps_cu2 * 1e3:.2f} ‰", c.eps_cu2 * 1e3, "‰", "3.1.7"))
    steps.append(Step("N_{Ed}", "axial force (compression +)", fmt(NEd), NEd, "kN"))
    if is_column:
        steps.append(Step("e_{0}", "max(h/30, 20 mm)", f"max({fmt(rc.h)}/30, 20)", e0, "mm", "6.1(4)"))
        steps.append(Step("M_{Ed}", "max(|M_{Ed,analysis}|, N_{Ed}·e_{0}) with the sign of M", f"max({fmt(abs(MEd_raw))}, {fmt(max(NEd, 0) * e0 * 1e-3)})", MEd, "kNm", "6.1(4)"))
    else:
        steps.append(Step("M_{Ed}", "bending moment (sagging +)", fmt(MEd), MEd, "kNm"))

    Nmin, Nmax = rc.N_min, rc.N_max
    notes = []
    status = PASS
    capacity = None
    if NEd > Nmax or NEd < Nmin:
        capacity = Nmax if NEd > Nmax else Nmin
        util = NEd / capacity if capacity != 0 else math.inf
        steps.append(Step("N_{Rd}", "resistance of the section to pure axial force", "", capacity, "kN", "6.1"))
        notes.append("The axial force alone exceeds the section resistance.")
        status = FAIL
    else:
        flip = MEd < 0
        state = rc.solve_state(NEd, flip=flip)
        MRd = state.M
        capacity = MRd
        if state.x is not None and math.isfinite(state.x):
            steps.append(Step("x", "neutral axis depth from the compressed face (strain compatibility)", "", state.x, "mm", "6.1(2)"))
        face = "bottom" if flip else "top"
        steps.append(Step(f"ε_{{c,{face}}}", "strain at the compressed face", "", (state.eps_bot if flip else state.eps_top) * 1e3, "‰", "Fig. 6.1"))
        for lab, y, e, s in state.steel:
            steps.append(Step(f"σ_{{s,{lab}}}", f"E_{{s}}·ε_{{s}} ≤ f_{{yd}} with ε_{{s}} = {e * 1e3:.2f} ‰ (bars at depth {y:.0f} mm)", "", s, "MPa", "3.2.7"))
        steps.append(Step("M_{Rd}", "resistance moment at N_{Ed} (parabola-rectangle, Fig. 6.1)", "", MRd, "kNm", "6.1"))
        if MRd * MEd > 0:
            util = MEd / MRd
        elif abs(MEd) < 1e-12:
            util = 0.0
        else:
            util = math.inf
        status = status_from_utilisation(util)
        if not is_column and state.x is not None and math.isfinite(state.x) and not flip:
            d = max(rc.ys) if len(rc.ys) else rc.h
            xd = state.x / d
            if xd > 0.45 and c.fck <= 50:
                notes.append(f"x/d = {xd:.2f} > 0.45: low ductility; avoid moment redistribution (5.6.3).")
    steps.append(Step("η", "M_{Ed}/M_{Rd}", f"{fmt(MEd)}/{fmt(capacity)}" if capacity else "", util, "", "6.1"))
    if is_column:
        notes.append("Second-order effects are checked separately (slenderness, 5.8.3.1).")
    return CheckResult(
        id="ec2.bending_axial", title="Bending with axial force", clause="EN 1992-1-1 6.1",
        member=member.id, status=status, utilisation=util, demand=MEd, capacity=capacity, unit="kNm",
        x=float(f.x[si]), combination=f.uls_ids[ci], combination_label=f.uls_labels[ci], steps=steps, notes=notes,
    )


# ================================================================= shear
def _tension_steel(rc: RCSection, sagging: np.ndarray):
    """Area (mm2) and effective depth (mm) of the tension reinforcement for each station."""
    h = rc.h
    ys, As = rc.ys, rc.As
    bot = ys > h / 2.0
    top = ys < h / 2.0
    A_b, A_t = As[bot].sum(), As[top].sum()
    d_b = (As[bot] @ ys[bot]) / A_b if A_b > 0 else 0.9 * h
    d_t = h - ((As[top] @ ys[top]) / A_t) if A_t > 0 else 0.9 * h
    Asl = np.where(sagging, A_b, A_t)
    d = np.where(sagging, d_b, d_t)
    return Asl, d


def shear_resistance(rc: RCSection, sec: RCRectSection, p: EC2Params, Nc, M) -> dict:
    """Vectorised shear resistance. Nc in kN (compression +), M in kNm."""
    c, st = rc.c, rc.st
    b = rc.b
    Asl, d = _tension_steel(rc, np.asarray(M) >= -1e-9)   # |M| ~ 0: treat as sagging (deterministic)
    k = np.minimum(1.0 + np.sqrt(200.0 / d), 2.0)
    rho = np.minimum(Asl / (b * d), 0.02)
    sig = np.minimum(np.asarray(Nc) * 1e3 / rc.Ac, 0.2 * c.fcd)
    vmin = p.vmin_coefficient * k ** 1.5 * math.sqrt(c.fck)
    v1 = p.C_Rd_c * k * np.cbrt(100.0 * rho * c.fck) + p.k1_shear * sig
    v2 = vmin + p.k1_shear * sig
    VRdc = np.maximum(np.maximum(v1, v2), 0.0) * b * d * 1e-3
    Vlim = 0.5 * b * d * c.nu * c.fcd * 1e-3     # 6.2.2(6)
    out = {"k": k, "rho": rho, "sigma_cp": sig, "vmin": vmin, "VRdc": VRdc, "d": d, "Asl": Asl, "Vlim": Vlim}
    if sec.links:
        z = 0.9 * d
        fywd = (sec.links.fyk or sec.rebar.fyk) / p.gamma_s
        a = sec.links.area_mm2 / sec.links.spacing_mm * z * fywd   # N per unit cot(theta)
        cc = 1.0 * b * z * c.nu * c.fcd                            # alpha_cw = 1, nu_1 = nu
        with np.errstate(invalid="ignore"):
            cot = np.sqrt(np.maximum(cc / a - 1.0, 0.0))
        cot = np.clip(cot, p.cot_theta_min, p.cot_theta_max)
        VRds = a * cot * 1e-3
        VRdmax = cc / (cot + 1.0 / cot) * 1e-3
        VRd_links = np.minimum(VRds, VRdmax)
        VRd = np.maximum(VRdc, VRd_links)
        out.update(z=z, fywd=fywd, cot=cot, VRds=VRds, VRdmax=VRdmax, VRd_links=VRd_links)
    else:
        VRd = np.minimum(VRdc, Vlim)
    out["VRd"] = VRd
    return out


def check_shear(member: Member, rc: RCSection, sec: RCRectSection, f: MemberForces, p: EC2Params) -> CheckResult:
    Nc = -f.uls_N
    r = shear_resistance(rc, sec, p, Nc, f.uls_M)
    V = np.abs(f.uls_V)
    with np.errstate(divide="ignore"):
        u = np.where(r["VRd"] > 0, V / r["VRd"], np.where(V > 0, np.inf, 0.0))
    ci, si = (int(v) for v in np.unravel_index(int(np.argmax(u)), u.shape))
    g = {k: (v[ci, si] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in r.items()}
    c = rc.c
    VEd = float(V[ci, si])
    face = "bottom" if f.uls_M[ci, si] >= -1e-9 else "top"
    steps = [
        Step("V_{Ed}", "design shear force", fmt(VEd), VEd, "kN"),
        Step("d", f"effective depth to the {face} (tension) reinforcement", "", float(g["d"]), "mm"),
        Step("k", "1 + √(200/d) ≤ 2.0", f"1 + √(200/{fmt(float(g['d']))})", float(g["k"]), "", "6.2.2(1)"),
        Step("ρ_{l}", "A_{sl}/(b_{w}·d) ≤ 0.02", f"{fmt(float(g['Asl']))}/({fmt(rc.b)}·{fmt(float(g['d']))})", float(g["rho"]), "", "6.2.2(1)"),
        Step("σ_{cp}", "N_{Ed}/A_{c} < 0.2·f_{cd}", f"{fmt(float(Nc[ci, si]) * 1e3)}/{fmt(rc.Ac)}", float(g["sigma_cp"]), "MPa", "6.2.2(1)"),
        Step("v_{min}", "0.035·k^{3/2}·f_{ck}^{1/2}", f"0.035·{fmt(float(g['k']))}^{{1.5}}·{fmt(c.fck)}^{{0.5}}", float(g["vmin"]), "MPa", "6.2.2(1) Eq. 6.3N"),
        Step("V_{Rd,c}", "[C_{Rd,c}·k·(100·ρ_{l}·f_{ck})^{1/3} + k_{1}·σ_{cp}]·b_{w}·d ≥ (v_{min} + k_{1}·σ_{cp})·b_{w}·d",
             f"[{fmt(p.C_Rd_c)}·{fmt(float(g['k']))}·(100·{fmt(float(g['rho']))}·{fmt(c.fck)})^{{1/3}} + {fmt(p.k1_shear)}·{fmt(float(g['sigma_cp']))}]·{fmt(rc.b)}·{fmt(float(g['d']))}",
             float(g["VRdc"]), "kN", "6.2.2(1) Eq. 6.2"),
    ]
    notes = []
    if sec.links:
        L = sec.links
        steps += [
            Step("A_{sw}/s", f"{L.legs} legs Ø{L.dia_mm:g} at {L.spacing_mm:g} mm", f"{fmt(L.area_mm2)}/{fmt(L.spacing_mm)}", L.area_mm2 / L.spacing_mm, "mm²/mm"),
            Step("z", "0.9·d", f"0.9·{fmt(float(g['d']))}", float(g["z"]), "mm", "6.2.3(1)"),
            Step("cot θ", "chosen in [1, 2.5] to maximise min(V_{Rd,s}, V_{Rd,max})", "", float(g["cot"]), "", "6.2.3(2) Eq. 6.7N"),
            Step("V_{Rd,s}", "(A_{sw}/s)·z·f_{ywd}·cot θ", f"{fmt(L.area_mm2 / L.spacing_mm)}·{fmt(float(g['z']))}·{fmt(float(g['fywd']))}·{fmt(float(g['cot']))}", float(g["VRds"]), "kN", "6.2.3(3) Eq. 6.8"),
            Step("V_{Rd,max}", "α_{cw}·b_{w}·z·ν_{1}·f_{cd}/(cot θ + tan θ)", f"1.0·{fmt(rc.b)}·{fmt(float(g['z']))}·{fmt(c.nu)}·{fmt(c.fcd)}/({fmt(float(g['cot']))} + {fmt(1 / float(g['cot']))})", float(g["VRdmax"]), "kN", "6.2.3(3) Eq. 6.9"),
            Step("V_{Rd}", "max(V_{Rd,c}, min(V_{Rd,s}, V_{Rd,max}))", "", float(g["VRd"]), "kN", "6.2.1"),
        ]
    else:
        steps.append(Step("V_{Rd}", "min(V_{Rd,c}, 0.5·b_{w}·d·ν·f_{cd})", f"min({fmt(float(g['VRdc']))}, {fmt(float(g['Vlim']))})", float(g["VRd"]), "kN", "6.2.2(6)"))
        notes.append("No shear links defined: resistance from concrete alone (6.2.2).")
    util = float(u[ci, si])
    steps.append(Step("η", "V_{Ed}/V_{Rd}", f"{fmt(VEd)}/{fmt(float(g['VRd']))}", util, "", "6.2.1"))
    notes.append("Checked at every station, including at the support axis (conservative; 6.2.1(8) allows d from the face).")
    return CheckResult(
        id="ec2.shear", title="Shear", clause="EN 1992-1-1 6.2", member=member.id,
        status=status_from_utilisation(util), utilisation=util, demand=VEd, capacity=float(g["VRd"]), unit="kN",
        x=float(f.x[si]), combination=f.uls_ids[ci], combination_label=f.uls_labels[ci], steps=steps, notes=notes,
    )


# ============================================================= detailing
def check_detailing(member: Member, rc: RCSection, sec: RCRectSection, f: MemberForces, p: EC2Params) -> list[CheckResult]:
    out: list[CheckResult] = []
    c, st = rc.c, rc.st
    b, h = rc.b, rc.h
    Ac = rc.Ac
    As_tot = float(rc.As.sum())
    is_column = member.design.type == "column"

    # maximum reinforcement 9.2.1.1(3) / 9.5.2(3)
    As_max = p.As_max_ratio * Ac
    u = As_tot / As_max
    out.append(CheckResult(
        id="ec2.as_max", title="Maximum longitudinal reinforcement", clause="EN 1992-1-1 9.2.1.1(3), 9.5.2(3)",
        member=member.id, status=status_from_utilisation(u), utilisation=u, demand=As_tot, capacity=As_max, unit="mm²",
        steps=[Step("A_{s,max}", "0.04·A_{c}", f"0.04·{fmt(Ac)}", As_max, "mm²"),
               Step("A_{s,prov}", "all longitudinal bars", "", As_tot, "mm²")],
        notes=["Limit applies outside lap locations."]))

    if not is_column:
        # minimum tension reinforcement 9.2.1.1(1) for each face that sees tension
        faces = []
        if np.max(f.uls_M) > 1e-9:
            faces.append(("bottom", sec.bottom))
        if np.min(f.uls_M) < -1e-9:
            faces.append(("top", sec.top))
        for face, layer in faces:
            prov = layer.area_mm2 if layer and layer.n > 0 else 0.0
            d = (sec.layer_depth(layer, face) if face == "bottom" else h - sec.layer_depth(layer, face)) if prov > 0 else 0.9 * h
            req = max(p.As_min_fctm_factor * c.fctm / st.fyk, p.As_min_absolute) * b * d
            uu = req / prov if prov > 0 else math.inf
            out.append(CheckResult(
                id=f"ec2.as_min.{face}", title=f"Minimum tension reinforcement, {face}", clause="EN 1992-1-1 9.2.1.1(1)",
                member=member.id, status=status_from_utilisation(uu), utilisation=uu, demand=req, capacity=prov, unit="mm²",
                steps=[Step("f_{ctm}", "Table 3.1", "", c.fctm, "MPa", "3.1.2"),
                       Step("A_{s,min}", "max(0.26·f_{ctm}/f_{yk}, 0.0013)·b_{t}·d",
                            f"max(0.26·{fmt(c.fctm)}/{fmt(st.fyk)}, 0.0013)·{fmt(b)}·{fmt(d)}", req, "mm²", "Eq. 9.1N"),
                       Step("A_{s,prov}", f"{face} bars", "", prov, "mm²")],
                notes=["Ratio shown is A_{s,min}/A_{s,prov}."]))
        # shear reinforcement 9.2.2(5), (6)
        if sec.links:
            L = sec.links
            rho_w = L.area_mm2 / (L.spacing_mm * b)
            fyk_w = L.fyk or st.fyk
            rho_min = p.rho_w_min_factor * math.sqrt(c.fck) / fyk_w
            uu = rho_min / rho_w
            out.append(CheckResult(
                id="ec2.links_min", title="Minimum shear reinforcement", clause="EN 1992-1-1 9.2.2(5)",
                member=member.id, status=status_from_utilisation(uu), utilisation=uu, demand=rho_min, capacity=rho_w,
                steps=[Step("ρ_{w}", "A_{sw}/(s·b_{w})", f"{fmt(L.area_mm2)}/({fmt(L.spacing_mm)}·{fmt(b)})", rho_w, "", "Eq. 9.4"),
                       Step("ρ_{w,min}", "0.08·√f_{ck}/f_{yk}", f"0.08·√{fmt(c.fck)}/{fmt(fyk_w)}", rho_min, "", "Eq. 9.5N")]))
            d = float(np.max(rc.ys)) if len(rc.ys) else 0.9 * h
            s_max = p.sl_max_factor * d
            uu = L.spacing_mm / s_max
            out.append(CheckResult(
                id="ec2.links_spacing", title="Maximum link spacing", clause="EN 1992-1-1 9.2.2(6)",
                member=member.id, status=status_from_utilisation(uu), utilisation=uu, demand=L.spacing_mm, capacity=s_max, unit="mm",
                steps=[Step("s_{l,max}", "0.75·d·(1 + cot α), α = 90°", f"0.75·{fmt(d)}", s_max, "mm", "Eq. 9.6N")]))
        else:
            out.append(CheckResult(
                id="ec2.links_min", title="Minimum shear reinforcement", clause="EN 1992-1-1 9.2.2(5)",
                member=member.id, status=FAIL, notes=["Beams need minimum shear reinforcement; no links are defined."]))
    else:
        NEd = float(np.max(-f.uls_N))
        req = max(p.column_As_min_N_factor * max(NEd, 0.0) * 1e3 / st.fyd, p.column_As_min_Ac_ratio * Ac)
        uu = req / As_tot if As_tot > 0 else math.inf
        out.append(CheckResult(
            id="ec2.column_as_min", title="Minimum column reinforcement", clause="EN 1992-1-1 9.5.2(2)",
            member=member.id, status=status_from_utilisation(uu), utilisation=uu, demand=req, capacity=As_tot, unit="mm²",
            steps=[Step("N_{Ed}", "maximum compression", "", NEd, "kN"),
                   Step("A_{s,min}", "max(0.10·N_{Ed}/f_{yd}, 0.002·A_{c})", f"max(0.10·{fmt(NEd * 1e3)}/{fmt(st.fyd)}, 0.002·{fmt(Ac)})", req, "mm²", "Eq. 9.12N")]))
        dias = [lay.dia_mm for lay in (sec.top, sec.bottom, *sec.extra_layers) if lay and lay.n > 0]
        if dias:
            phi = min(dias)
            uu = p.column_phi_min_mm / phi
            out.append(CheckResult(
                id="ec2.column_phi_min", title="Minimum bar diameter in columns", clause="EN 1992-1-1 9.5.2(1)",
                member=member.id, status=status_from_utilisation(uu), utilisation=uu, demand=p.column_phi_min_mm, capacity=phi, unit="mm",
                steps=[Step("φ_{min}", "recommended value", "", p.column_phi_min_mm, "mm")]))
            if sec.links:
                s_max = min(p.column_link_spacing_phi_factor * phi, min(b, h), p.column_link_spacing_max_mm)
                uu = sec.links.spacing_mm / s_max
                out.append(CheckResult(
                    id="ec2.column_links", title="Maximum link spacing in columns", clause="EN 1992-1-1 9.5.3(3)",
                    member=member.id, status=status_from_utilisation(uu), utilisation=uu, demand=sec.links.spacing_mm, capacity=s_max, unit="mm",
                    steps=[Step("s_{cl,tmax}", "min(20·φ_{l,min}, b, 400 mm)", f"min(20·{fmt(phi)}, {fmt(min(b, h))}, 400)", s_max, "mm")]))
            else:
                out.append(CheckResult(id="ec2.column_links", title="Column links", clause="EN 1992-1-1 9.5.3",
                                       member=member.id, status=FAIL, notes=["Columns need transverse reinforcement; no links are defined."]))

    # clear bar spacing 8.2(2) for top and bottom layers
    dg = sec.concrete.dmax_mm
    link = sec.links.dia_mm if sec.links else 0.0
    for face, layer in (("bottom", sec.bottom), ("top", sec.top)):
        if not layer or layer.n < 2:
            continue
        clear = (b - 2 * sec.cover_mm - 2 * link - layer.n * layer.dia_mm) / (layer.n - 1)
        req = max(p.k1_spacing * layer.dia_mm, dg + p.k2_spacing_mm, 20.0)
        uu = req / clear if clear > 0 else math.inf
        out.append(CheckResult(
            id=f"ec2.bar_spacing.{face}", title=f"Clear bar spacing, {face}", clause="EN 1992-1-1 8.2(2)",
            member=member.id, status=status_from_utilisation(uu), utilisation=uu, demand=req, capacity=clear, unit="mm",
            steps=[Step("a", "(b − 2·c − 2·φ_{w} − n·φ)/(n − 1)",
                        f"({fmt(b)} − 2·{fmt(sec.cover_mm)} − 2·{fmt(link)} − {layer.n}·{fmt(layer.dia_mm)})/{layer.n - 1}", clear, "mm"),
                   Step("a_{min}", "max(k_{1}·φ, d_{g} + k_{2}, 20 mm)", f"max({fmt(layer.dia_mm)}, {fmt(dg)} + 5, 20)", req, "mm")],
            notes=["Assumes one layer of bars across the width."]))
    return out


# ================================================================ columns
def check_slenderness(member: Member, rc: RCSection, f: MemberForces, p: EC2Params) -> CheckResult:
    l0 = member.design.l0_factor * f.L * 1000.0
    i = rc.h / math.sqrt(12.0)
    lam = l0 / i
    NEd = float(np.max(-f.uls_N))
    n = max(NEd, 1e-9) * 1e3 / (rc.Ac * rc.c.fcd)
    lam_lim = 20.0 * 0.7 * 1.1 * 0.7 / math.sqrt(n)
    steps = [
        Step("l_{0}", "effective length = factor·L", f"{fmt(member.design.l0_factor)}·{fmt(f.L * 1000)}", l0, "mm", "5.8.3.2"),
        Step("i", "h/√12 (bending in the frame plane)", f"{fmt(rc.h)}/√12", i, "mm"),
        Step("λ", "l_{0}/i", f"{fmt(l0)}/{fmt(i)}", lam, "", "5.8.3.2(1)"),
        Step("n", "N_{Ed}/(A_{c}·f_{cd})", f"{fmt(NEd * 1e3)}/({fmt(rc.Ac)}·{fmt(rc.c.fcd)})", n, "", "5.8.3.1(1)"),
        Step("λ_{lim}", "20·A·B·C/√n with A = 0.7, B = 1.1, C = 0.7", f"20·0.7·1.1·0.7/√{fmt(n)}", lam_lim, "", "Eq. 5.13N"),
    ]
    if lam <= lam_lim:
        return CheckResult(id="ec2.slenderness", title="Slenderness (second-order effects)", clause="EN 1992-1-1 5.8.3.1",
                           member=member.id, status=PASS, utilisation=lam / lam_lim, demand=lam, capacity=lam_lim, steps=steps,
                           notes=["λ ≤ λ_lim: second-order effects may be ignored."])
    return CheckResult(id="ec2.slenderness", title="Slenderness (second-order effects)", clause="EN 1992-1-1 5.8.3.1",
                       member=member.id, status=NOT_COVERED, utilisation=lam / lam_lim, demand=lam, capacity=lam_lim, steps=steps,
                       notes=["λ > λ_lim: second-order effects must be considered (5.8.5). This version does not compute them; "
                              "verify this column by another method."])


# ============================================================ deflection
def _rho_for_ld(rc: RCSection, sec: RCRectSection, system: str):
    tension_face = "top" if system == "cantilever" else "bottom"
    comp_face = "bottom" if tension_face == "top" else "top"
    t = getattr(sec, tension_face)
    cmp_ = getattr(sec, comp_face)
    if not t or t.n == 0:
        return None
    d = sec.layer_depth(t, tension_face) if tension_face == "bottom" else rc.h - sec.layer_depth(t, tension_face)
    rho = t.area_mm2 / (rc.b * d)
    rho_c = (cmp_.area_mm2 if cmp_ else 0.0) / (rc.b * d)
    return d, rho, rho_c, tension_face


def check_span_depth(member: Member, rc: RCSection, sec: RCRectSection, f: MemberForces, p: EC2Params) -> CheckResult:
    system = member.design.deflection_system
    K = K_SYSTEM[system]
    fck = rc.c.fck
    got = _rho_for_ld(rc, sec, system)
    if got is None:
        return CheckResult(id="ec2.span_depth", title="Span/depth ratio", clause="EN 1992-1-1 7.4.2", member=member.id,
                           status=FAIL, notes=["No tension reinforcement on the relevant face."])
    d, rho, rho_c, face = got
    rho0 = math.sqrt(fck) * 1e-3
    sf = math.sqrt(fck)
    if rho <= rho0:
        lim = K * (11 + 1.5 * sf * rho0 / rho + 3.2 * sf * (rho0 / rho - 1) ** 1.5)
        expr = "K·[11 + 1.5·√f_{ck}·ρ_{0}/ρ + 3.2·√f_{ck}·(ρ_{0}/ρ − 1)^{3/2}]"
        eq = "Eq. 7.16a"
    else:
        rho_c_used = rho_c if rho_c < 0.5 * rho else 0.0   # Eq. 7.16b is singular as rho' -> rho
        lim = K * (11 + 1.5 * sf * rho0 / (rho - rho_c_used) + sf / 12 * math.sqrt(rho_c_used / rho0))
        expr = "K·[11 + 1.5·√f_{ck}·ρ_{0}/(ρ − ρ') + (1/12)·√f_{ck}·√(ρ'/ρ_{0})]"
        eq = "Eq. 7.16b"
        if rho_c_used != rho_c:
            rho_c = 0.0
            expr += ", ρ' taken as 0 (conservative; the equation is singular as ρ' → ρ)"
    actual = f.L * 1000.0 / d
    u = actual / lim
    steps = [
        Step("K", f"structural system: {system.replace('_', ' ')}", "", K, "", "Table 7.4N"),
        Step("ρ_{0}", "√f_{ck}·10^{-3}", f"√{fmt(fck)}·10^{{-3}}", rho0, "", "7.4.2(2)"),
        Step("ρ", f"A_{{s}}/(b·d), {face} bars as provided (conservative)", "", rho, ""),
        Step("ρ'", "compression reinforcement ratio", "", rho_c, ""),
        Step("(l/d)_{lim}", expr, "", lim, "", eq),
        Step("l/d", "span/effective depth", f"{fmt(f.L * 1000)}/{fmt(d)}", actual, ""),
    ]
    return CheckResult(id="ec2.span_depth", title="Span/depth ratio (deemed to satisfy)", clause="EN 1992-1-1 7.4.2",
                       member=member.id, status=status_from_utilisation(u), utilisation=u, demand=actual, capacity=lim,
                       steps=steps, notes=["Uses the provided reinforcement and omits the 310/σ_s factor, both conservative."])


def deflection_curves(rc: RCSection, sec: RCRectSection, f: MemberForces, phi: float, p: EC2Params,
                      cantilever: bool) -> dict:
    """Long-term deflection along the member for every quasi-permanent combination (7.4.3).

    Curvature interpolated between the uncracked and fully cracked states,
    Eq. (7.18)/(7.19) with zeta = 1 - beta (M_cr/M)^2. Creep through
    E_c,eff = E_cm/(1 + phi), Eq. (7.20). Shrinkage curvature is not included.
    """
    c = rc.c
    Eeff = c.Ecm / (1.0 + phi)
    sag = rc.transformed(Eeff, sagging=True)
    hog = rc.transformed(Eeff, sagging=False)
    Mcr_s = c.fctm * sag["I_I"] / (rc.h - sag["y_I"]) * 1e-6     # kNm
    Mcr_h = c.fctm * hog["I_I"] / (rc.h - hog["y_I"]) * 1e-6
    M = f.qp_M
    pos = M >= 0
    I1 = np.where(pos, sag["I_I"], hog["I_I"])
    I2 = np.where(pos, sag["I_II"], hog["I_II"])
    Mcr = np.where(pos, Mcr_s, Mcr_h)
    absM = np.abs(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(absM > Mcr, 1.0 - p.beta_sustained * (Mcr / absM) ** 2, 0.0)
    k1 = M * 1e6 / (Eeff * I1) * 1e3     # 1/m
    k2 = M * 1e6 / (Eeff * I2) * 1e3
    kap = zeta * k2 + (1.0 - zeta) * k1
    x = f.x
    if cantilever and f.tip_at_i:
        x_from_root = f.L - x
    else:
        x_from_root = x
    order = np.argsort(x_from_root, kind="stable")
    xs = x_from_root[order]
    kk = kap[:, order]
    dx = np.diff(xs)
    F1 = np.concatenate([np.zeros((kk.shape[0], 1)), np.cumsum(0.5 * dx * (kk[:, 1:] + kk[:, :-1]), axis=1)], axis=1)
    sk = kk * xs
    G = np.concatenate([np.zeros((kk.shape[0], 1)), np.cumsum(0.5 * dx * (sk[:, 1:] + sk[:, :-1]), axis=1)], axis=1)
    if cantilever:
        w = xs * F1 - G                    # relative to the tangent at the root
    else:
        theta = -(xs[-1] * F1[:, -1] - G[:, -1]) / xs[-1]
        w = theta[:, None] * xs + xs * F1 - G   # relative to the chord
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return {"w": w[:, inv], "Eeff": Eeff, "sag": sag, "hog": hog, "Mcr_sag": Mcr_s, "Mcr_hog": Mcr_h, "zeta": zeta}


def check_deflection(member: Member, rc: RCSection, sec: RCRectSection, f: MemberForces, p: EC2Params) -> CheckResult:
    phi = member.design.creep_coefficient
    cant = member.design.deflection_system == "cantilever"
    r = deflection_curves(rc, sec, f, phi, p, cant)
    w = np.abs(r["w"]) * 1000.0  # mm
    ci, si = (int(v) for v in np.unravel_index(int(np.argmax(w)), w.shape))
    wmax = float(w[ci, si])
    lim = f.L * 1000.0 / p.deflection_limit_divisor
    u = wmax / lim
    sag = r["sag"]
    steps = [
        Step("E_{c,eff}", "E_{cm}/(1 + φ(∞,t_{0}))", f"{fmt(rc.c.Ecm)}/(1 + {fmt(phi)})", r["Eeff"], "MPa", "Eq. 7.20"),
        Step("α_{e}", "E_{s}/E_{c,eff}", "", sag["alpha_e"], ""),
        Step("I_{I}", "uncracked transformed section, sagging", "", sag["I_I"] * 1e-6, "×10⁶ mm⁴"),
        Step("I_{II}", "fully cracked transformed section, sagging", "", sag["I_II"] * 1e-6, "×10⁶ mm⁴"),
        Step("M_{cr}", "f_{ctm}·I_{I}/(h − y_{I}), sagging", "", r["Mcr_sag"], "kNm", "7.4.3(3)"),
        Step("ζ", "1 − β·(M_{cr}/M_{qp})^{2}, β = 0.5, at the governing station", "", float(r["zeta"][ci, si]), "", "Eq. 7.19"),
        Step("w_{max}", "∫ κ·m̄ dx with κ = ζ·κ_{II} + (1 − ζ)·κ_{I}" + (" (tip, relative to the root tangent)" if cant else " (relative to the member chord)"),
             "", wmax, "mm", "Eq. 7.18"),
        Step("w_{lim}", "span/250", f"{fmt(f.L * 1000)}/250", lim, "mm", "7.4.1(4)"),
        Step("η", "w_{max}/w_{lim}", f"{fmt(wmax)}/{fmt(lim)}", u, ""),
    ]
    notes = ["Moments from the linear-elastic analysis (no redistribution from cracking).",
             "Shrinkage curvature (7.4.3(6)) is not included; add a margin or check it separately."]
    return CheckResult(id="ec2.deflection", title="Long-term deflection (quasi-permanent)", clause="EN 1992-1-1 7.4.3",
                       member=member.id, status=status_from_utilisation(u), utilisation=u, demand=wmax, capacity=lim, unit="mm",
                       x=float(f.x[si]), combination=f.qp_ids[ci], combination_label=f.qp_labels[ci], steps=steps, notes=notes)


# =============================================================== driver
def check_member(member: Member, sec: RCRectSection, rc: RCSection, f: MemberForces, p: EC2Params) -> list[CheckResult]:
    if member.design.type == "none":
        return []
    out = [check_bending_axial(member, rc, f, p), check_shear(member, rc, sec, f, p)]
    out += check_detailing(member, rc, sec, f, p)
    if member.design.type == "column":
        out.append(check_slenderness(member, rc, f, p))
    else:
        ld = check_span_depth(member, rc, sec, f, p)
        defl = check_deflection(member, rc, sec, f, p)
        if ld.status == FAIL and defl.status == PASS:
            ld.status = INFO
            ld.notes.append("Limit exceeded, so the calculated deflection (7.4.3) governs; it passes.")
        out += [ld, defl]
    return out
