"""Clause-by-clause calculation report as a self-contained HTML page."""
from __future__ import annotations

import datetime as _dt
import html
import math
import re

import numpy as np

from .. import __version__
from ..codes.base import FAIL, INFO, NOT_COVERED, PASS, WARN, CheckResult
from ..combinations import CATEGORY_LABEL, PSI
from . import svg

STATUS_TEXT = {PASS: "Pass", FAIL: "Fail", WARN: "Warning", INFO: "Info", NOT_COVERED: "Not covered"}

CSS = """
:root{--ink:#1d2430;--muted:#5b6472;--rule:#d9dde3;--soft:#f4f5f7;--accent:#1f4e8c;
--pass:#157a3c;--pass-bg:#e6f4ea;--fail:#b42318;--fail-bg:#fdecea;--nc:#a15c07;--nc-bg:#fdf3e1;--info:#475467;--info-bg:#eef0f3}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;background:#e9ebef;color:var(--ink);font:13px/1.45 "Source Sans 3","Inter","Helvetica Neue",Arial,"Liberation Sans",sans-serif;font-variant-numeric:tabular-nums}
.page{max-width:860px;margin:24px auto;background:#fff;padding:44px 52px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
header.top{display:grid;grid-template-columns:1fr auto;gap:24px;border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:22px}
.kicker{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}
h1{font-size:24px;line-height:1.2;margin:0 0 6px}
.meta{color:var(--muted);font-size:12px}
.verdict{text-align:right;min-width:190px}
.verdict .big{font-size:30px;font-weight:700;line-height:1}
.verdict .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
h2{font-size:16px;margin:30px 0 10px;padding-top:6px;border-top:1px solid var(--rule);break-after:avoid}
h2 .n{color:var(--accent);margin-right:8px}
h3{font-size:14px;margin:22px 0 8px;break-after:avoid}
h4{font-size:13px;margin:0}
p{margin:6px 0}
.muted{color:var(--muted)}
table{width:100%;border-collapse:collapse;margin:8px 0 14px;font-size:12px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--ink)}
td.num,th.num{text-align:right;white-space:nowrap}
tr{break-inside:avoid}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}
.pill.pass{color:var(--pass);background:var(--pass-bg)}.pill.fail{color:var(--fail);background:var(--fail-bg)}
.pill.not_covered,.pill.warn{color:var(--nc);background:var(--nc-bg)}.pill.info{color:var(--info);background:var(--info-bg)}
.bar{display:inline-block;width:90px;height:7px;background:var(--soft);border-radius:4px;vertical-align:middle;margin-right:6px;position:relative;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
.bar i.pass{background:var(--pass)}.bar i.fail{background:var(--fail)}.bar i.not_covered,.bar i.info,.bar i.warn{background:var(--nc)}
.fig{margin:10px 0 18px;break-inside:avoid}
.fig svg{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:4px;background:#fff}
.fig .cap{font-size:11px;color:var(--muted);margin-top:4px}
.check{border:1px solid var(--rule);border-radius:6px;margin:12px 0;break-inside:avoid-page}
.check .hd{display:flex;gap:10px;align-items:baseline;justify-content:space-between;padding:8px 12px;background:var(--soft);border-bottom:1px solid var(--rule);border-radius:6px 6px 0 0}
.check .where{font-size:11px;color:var(--muted);padding:6px 12px 0}
.check table{margin:4px 0 6px}
.check td.sym{white-space:nowrap;font-weight:600;width:110px}
.check td.expr{color:var(--ink)}
.check td.sub{color:var(--muted);font-size:11px}
.check td.cl{color:var(--muted);font-size:11px;white-space:nowrap}
.check ul{margin:2px 0 10px;padding-left:28px;color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
ul.plain{padding-left:18px;margin:6px 0}
.callout{border-left:3px solid var(--nc);background:var(--nc-bg);padding:8px 12px;margin:10px 0;font-size:12px}
.callout.ok{border-color:var(--pass);background:var(--pass-bg)}
sub,sup{font-size:75%;line-height:0}
footer{margin-top:28px;padding-top:10px;border-top:1px solid var(--rule);font-size:11px;color:var(--muted)}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:none;padding:0}h2{break-before:auto}}
@page{size:A4;margin:16mm 14mm 18mm}
"""

_SUB = re.compile(r"_\{([^}]*)\}")
_SUP = re.compile(r"\^\{([^}]*)\}")


def mk(text: str) -> str:
    """Escape and convert the light symbol markup (x_{a}, x^{b}) to HTML."""
    t = html.escape(text or "")
    t = _SUB.sub(r"<sub>\1</sub>", t)
    t = _SUP.sub(r"<sup>\1</sup>", t)
    return t


def num(v, digits=3) -> str:
    if v is None:
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if not math.isfinite(v):
        return "∞"
    a = abs(v)
    if a == 0:
        return "0"
    if 1e5 <= a < 1e12:
        return f"{v:,.0f}".replace(",", " ")
    if a >= 1e12 or a < 1e-3:
        return f"{v:.3e}"
    if a >= 100:
        return f"{v:,.1f}".replace(",", " ")
    if a >= 10:
        return f"{v:.2f}"
    if a >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def pill(status: str) -> str:
    return f'<span class="pill {status}">{STATUS_TEXT.get(status, status)}</span>'


def ubar(u, status) -> str:
    if u is None:
        return "–"
    w = 100 if not math.isfinite(u) else min(u, 1.0) * 100
    return f'<span class="bar"><i class="{status}" style="width:{w:.0f}%"></i></span>{num(u) if math.isfinite(u) else "∞"}'


def _table(head: list[str], rows: list[list[str]], num_cols: set[int] = frozenset()) -> str:
    th = "".join(f'<th class="num">{h}</th>' if k in num_cols else f"<th>{h}</th>" for k, h in enumerate(head))
    trs = []
    for r in rows:
        trs.append("<tr>" + "".join(f'<td class="num">{c}</td>' if k in num_cols else f"<td>{c}</td>" for k, c in enumerate(r)) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _load_text(ld, kind) -> str:
    dirs = {"global_x": "global X", "global_y": "global Y", "local_x": "local x", "local_y": "local y"}
    if kind == "nodal":
        parts = [f"{k} = {num(v)}" for k, v in (("F<sub>x</sub>", ld.fx), ("F<sub>y</sub>", ld.fy), ("M<sub>z</sub>", ld.mz)) if v]
        return f"Node {html.escape(ld.node)}: " + ", ".join(parts) + " (kN, kNm)"
    if kind == "dist":
        w = f"{num(ld.w1)} kN/m" if ld.w1 == ld.w2 else f"{num(ld.w1)} → {num(ld.w2)} kN/m"
        return f"Member {html.escape(ld.member)}: line load {w}, {dirs[ld.direction]}, from {num(ld.a)} to {num(ld.b)} m"
    return f"Member {html.escape(ld.member)}: point load {num(ld.p)} kN, {dirs[ld.direction]}, at {num(ld.a)} m"


def _check_block(c: CheckResult) -> str:
    where = []
    if c.x is not None:
        where.append(f"at x = {c.x:.3f} m from node i")
    if c.combination:
        where.append(f"combination {html.escape(c.combination)}: {html.escape(c.combination_label or '')}")
    rows = []
    for s in c.steps:
        rows.append(f'<tr><td class="sym">{mk(s.symbol)}</td><td class="expr">{mk(s.expression)}'
                    + (f'<br><span class="sub">= {mk(s.substituted)}</span>' if s.substituted else "")
                    + f'</td><td class="num"><b>{num(s.value)}</b> {mk(s.unit)}</td><td class="cl">{html.escape(s.clause)}</td></tr>')
    body = f'<table><tbody>{"".join(rows)}</tbody></table>' if rows else ""
    notes = "".join(f"<li>{mk(n)}</li>" for n in c.notes)
    util = f" · η = {num(c.utilisation)}" if c.utilisation is not None and math.isfinite(c.utilisation) else ""
    return (f'<div class="check"><div class="hd"><h4>{html.escape(c.title)} <span class="muted">· {html.escape(c.clause)}{util}</span></h4>{pill(c.status)}</div>'
            + (f'<div class="where">Governing {"; ".join(where)}</div>' if where else "")
            + body + (f"<ul>{notes}</ul>" if notes else "") + "</div>")


def render(res, verification=None, project_info: dict | None = None) -> str:
    m = res.model
    info = project_info or {}
    gov = res.governing()
    status = res.overall_status()
    p = res.params
    today = info.get("date") or _dt.date.today().isoformat()
    st_word = {PASS: "All checks pass", FAIL: "Checks fail", NOT_COVERED: "Incomplete: some checks not covered"}[status]
    gov_u = gov.utilisation if gov and gov.utilisation is not None and math.isfinite(gov.utilisation) else None
    color = {PASS: "var(--pass)", FAIL: "var(--fail)", NOT_COVERED: "var(--nc)"}[status]
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(m.name)} – calculation</title><style>{CSS}</style></head><body><div class='page'>"]
    out.append(
        f"<header class='top'><div><p class='kicker'>Structural calculation · {html.escape(info.get('project', 'Project'))}</p>"
        f"<h1>{html.escape(m.name)}</h1><div class='meta'>{today} · Tragwerk {__version__} · "
        f"{'EN 1992-1-1:2004 with recommended values' if m.settings.design_code == 'EC2' else 'SIA 262 (not implemented)'} · "
        f"{html.escape(info.get('author', ''))}</div></div>"
        f"<div class='verdict'><div class='lab'>Maximum utilisation</div><div class='big' style='color:{color}'>{num(gov_u) if gov_u is not None else '–'}</div>"
        f"<div style='margin-top:6px'>{pill(status)}</div><div class='meta' style='margin-top:4px'>{st_word}"
        + (f"<br>governing: {html.escape(gov.member)}, {html.escape(gov.title.lower())}" if gov else "") + "</div></div></header>")
    if res.warnings:
        out.append("<div class='callout'><b>Analysis warnings</b><ul class='plain'>" + "".join(f"<li>{html.escape(w)}</li>" for w in res.warnings) + "</ul></div>")

    # 1 basis
    out.append("<h2><span class='n'>1</span>Basis of design</h2>")
    out.append(_table(["Item", "Basis"], [
        ["Analysis", "Linear-elastic 2D frame analysis (direct stiffness method), Euler–Bernoulli members with axial stiffness. "
                     "Gross concrete sections, E = E<sub>cm</sub> (EN 1992-1-1 Table 3.1). Exact internal forces between nodes."],
        ["Combinations", f"EN 1990 Eq. ({html.escape(m.settings.combination_rule)}) for ULS (STR), Eq. 6.14b/6.15b/6.16b for SLS; "
                         "every variable case on or off (pattern loading); recommended ψ factors (Table A1.1)."],
        ["Design", "EN 1992-1-1:2004 + AC:2010, recommended values of the nationally determined parameters." if m.settings.design_code == "EC2"
         else "SIA 262:2013 requested; checks not implemented in this version."],
    ]))
    out.append("<div class='grid2'><div>" + _table(["Parameter", "Value", "Clause"], [
        ["γ<sub>c</sub> / γ<sub>s</sub>", f"{p.gamma_c} / {p.gamma_s}", "2.4.2.4"],
        ["α<sub>cc</sub>", f"{p.alpha_cc}", "3.1.6(1)"],
        ["C<sub>Rd,c</sub> / k<sub>1</sub>", f"{num(p.C_Rd_c)} / {p.k1_shear}", "6.2.2(1)"],
        ["cot θ range", f"{p.cot_theta_min} – {p.cot_theta_max}", "6.2.3(2)"],
        ["ε<sub>ud</sub>", f"{p.eps_ud_factor} · ε<sub>uk</sub>", "3.2.7(2)"],
        ["Deflection limit", f"span/{p.deflection_limit_divisor:.0f}", "7.4.1(4)"],
    ]) + "</div><div>" + _table(["Load factor", "Value"], [
        ["γ<sub>G,sup</sub> / γ<sub>G,inf</sub>", f"{m.settings.params.get('gamma_G_sup', 1.35)} / {m.settings.params.get('gamma_G_inf', 1.0)}"],
        ["γ<sub>Q</sub>", f"{m.settings.params.get('gamma_Q', 1.5)}"],
        ["ξ (6.10b only)", f"{m.settings.params.get('xi', 0.85)}"],
    ]) + "</div></div>")

    # 2 model
    out.append("<h2><span class='n'>2</span>Structural model</h2>")
    out.append(f"<div class='fig'>{svg.model_svg(m)}<div class='cap'>Geometry, supports and member names. Open circles mark moment releases.</div></div>")
    rows = []
    for mid, mem in m.members.items():
        L, c, s = m.member_geometry(mid)
        rel = ", ".join(e for e, r in (("i", mem.release_i), ("j", mem.release_j)) if r) or "–"
        rows.append([html.escape(mid), f"{html.escape(mem.i)} → {html.escape(mem.j)}", num(L), html.escape(mem.section),
                     html.escape(mem.design.type), rel])
    out.append(_table(["Member", "Nodes", "Length [m]", "Section", "Design as", "Hinges"], rows, {2}))
    sup_rows = []
    for nid, sp in m.supports.items():
        fix = ", ".join(n for n, f in (("u<sub>x</sub>", sp.ux), ("u<sub>y</sub>", sp.uy), ("φ<sub>z</sub>", sp.rz)) if f) or "–"
        spr = ", ".join(f"{n} = {num(v)}" for n, v in (("k<sub>x</sub>", sp.kx), ("k<sub>y</sub>", sp.ky), ("k<sub>φ</sub>", sp.kr)) if v) or "–"
        sup_rows.append([html.escape(nid), fix, spr])
    out.append(_table(["Support node", "Restrained", "Springs [kN/m, kNm/rad]"], sup_rows))
    sec_rows = []
    from ..codes.ec2.materials import concrete_props, steel_props
    for sid, sec in m.sections.items():
        c = concrete_props(sec.concrete, p)
        st = steel_props(sec.rebar, p)
        lay = "; ".join(f"{l}: {a:.0f} mm² at {y:.0f} mm" for a, y, l in sec.layers()) or "none"
        links = f"Ø{sec.links.dia_mm:g} × {sec.links.legs} legs @ {sec.links.spacing_mm:g}" if sec.links else "none"
        sec_rows.append([html.escape(sid), f"{sec.b_mm:g} × {sec.h_mm:g}", f"{html.escape(sec.concrete.name)}: f<sub>cd</sub> = {num(c.fcd)}, f<sub>ctm</sub> = {num(c.fctm)}, E<sub>cm</sub> = {c.Ecm / 1000:.1f} GPa",
                         f"{html.escape(sec.rebar.name)}: f<sub>yd</sub> = {num(st.fyd)}", f"{sec.cover_mm:g}", lay, links])
    out.append(_table(["Section", "b × h [mm]", "Concrete [MPa]", "Steel [MPa]", "Cover [mm]", "Bars (depth from top)", "Links"], sec_rows))

    # 3 loads
    out.append("<h2><span class='n'>3</span>Actions</h2>")
    for cid, lc in m.load_cases.items():
        if lc.type == "variable":
            ps = PSI[lc.category]
            kind = f"variable, {CATEGORY_LABEL[lc.category]}, ψ<sub>0</sub> = {ps[0]}, ψ<sub>1</sub> = {ps[1]}, ψ<sub>2</sub> = {ps[2]}"
            if lc.group:
                kind += f", exclusive group “{html.escape(lc.group)}”"
        else:
            kind = "permanent"
        items = []
        if lc.self_weight:
            items.append("Self weight of all members (25 kN/m³ unless set per concrete)")
        items += [_load_text(ld, "nodal") for ld in lc.nodal] + [_load_text(ld, "dist") for ld in lc.distributed] + [_load_text(ld, "point") for ld in lc.point]
        out.append(f"<h3>{html.escape(cid)} – {html.escape(lc.name)} <span class='muted' style='font-weight:400'>({kind})</span></h3>"
                   + "<ul class='plain'>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>")

    # 4 combinations
    counts = {k: sum(1 for c in res.combos.combos if c.kind == k) for k in ("ULS", "SLS_char", "SLS_freq", "SLS_qp")}
    out.append("<h2><span class='n'>4</span>Combinations of actions</h2>")
    out.append(f"<p>{counts['ULS']} ULS, {counts['SLS_char']} characteristic, {counts['SLS_freq']} frequent and {counts['SLS_qp']} quasi-permanent combinations were generated and all were checked. Combinations that are referenced in this report:</p>")
    used = {c.combination for c in res.all_checks() if c.combination}
    rows = [[html.escape(c.id), html.escape(c.kind.replace("_", " ")), html.escape(c.label)] for c in res.combos.combos if c.id in used]
    out.append(_table(["Id", "Kind", "Combination"], rows))

    # 5 analysis
    out.append("<h2><span class='n'>5</span>Analysis results</h2>")
    eq_rows = []
    for cid, cr in res.cases.items():
        eq_rows.append([html.escape(cid), num(cr.applied["Fx"]), num(cr.applied["Fy"]),
                        f"{max(abs(v) for v in cr.equilibrium.values()):.1e}"])
    out.append("<p>Self-check: for every load case the support reactions balance the applied loads.</p>")
    out.append(_table(["Load case", "ΣF<sub>x</sub> applied [kN]", "ΣF<sub>y</sub> applied [kN]", "Largest residual [kN, kNm]"], eq_rows, {1, 2, 3}))
    r_rows = []
    for nid, r in res.reactions.items():
        for q, lab in enumerate(("R<sub>x</sub> [kN]", "R<sub>y</sub> [kN]", "M<sub>z</sub> [kNm]")):
            vals = [r["per_case"][c][q] for c in res.combos.case_ids]
            if all(abs(v) < 1e-9 for v in vals) and abs(r["uls_max"][q]) < 1e-9 and abs(r["uls_min"][q]) < 1e-9:
                continue
            r_rows.append([html.escape(nid), lab] + [num(v) for v in vals] + [num(r["uls_max"][q]), num(r["uls_min"][q])])
    ncase = len(res.combos.case_ids)
    out.append("<h3>Support reactions</h3>")
    out.append(_table(["Node", "Component"] + [f"{html.escape(c)} (char.)" for c in res.combos.case_ids] + ["ULS max", "ULS min"],
                      r_rows, set(range(2, 4 + ncase))))
    for q, title, unit in (("M", "Bending moment, ULS envelope", "kNm"), ("V", "Shear force, ULS envelope", "kN"), ("N", "Axial force, ULS envelope", "kN")):
        peak = max(float(np.max(np.abs(np.concatenate([e.max[q], e.min[q]])))) for e in res.envelopes.values())
        if peak < 1e-6:
            out.append(f"<p class='muted'>{title.split(',')[0]}: zero in every combination.</p>")
            continue
        out.append(f"<div class='fig'>{svg.diagram_svg(res, q, title, unit)}</div>")
    dsvg, dlabel = svg.deflection_svg(res)
    out.append(f"<div class='fig'>{dsvg}<div class='cap'>Short-term elastic deflections, gross sections, {html.escape(dlabel)}. "
               "Long-term deflections with cracking and creep are checked in section 7.</div></div>")

    # 6 summary
    out.append("<h2><span class='n'>6</span>Summary of design checks</h2>")
    rows = []
    for mid, lst in res.checks.items():
        for c in lst:
            rows.append([html.escape(mid), html.escape(c.title), html.escape(c.clause.replace("EN 1992-1-1 ", "")), ubar(c.utilisation, c.status), pill(c.status)])
    out.append(_table(["Member", "Check", "Clause", "Utilisation η", "Result"], rows))

    # 7 details
    out.append("<h2><span class='n'>7</span>Design checks in detail</h2>")
    out.append("<p class='muted'>Each check shows the governing location and combination over all ULS (or quasi-permanent) combinations and all stations along the member, with the formula, the substituted values and the clause.</p>")
    for mid, lst in res.checks.items():
        mem = m.members[mid]
        out.append(f"<h3>Member {html.escape(mid)} <span class='muted' style='font-weight:400'>· section {html.escape(mem.section)} · designed as {html.escape(mem.design.type)}</span></h3>")
        out += [_check_block(c) for c in lst]

    # 8 limitations
    out.append("<h2><span class='n'>8</span>Assumptions and limitations</h2><ul class='plain'>" + "".join(f"<li>{t}</li>" for t in [
        "Plane frame; out-of-plane behaviour, torsion and lateral-torsional effects are not considered.",
        "Linear-elastic analysis with gross sections; no moment redistribution, no second-order (P-Δ) effects and no geometric imperfections (5.2).",
        "Reinforcement is assumed uniform along each member (no curtailment); anchorage and laps (8.4–8.7) are not checked.",
        "Shear is checked at every station including the support axis; torsion and punching are not covered.",
        "Columns: slender columns (λ &gt; λ<sub>lim</sub>) are flagged as not covered; biaxial bending is not considered.",
        "Deflection: curvature interpolation (7.4.3) using elastic moments; shrinkage curvature is excluded. Crack widths (7.3) and stress limits (7.2) are not yet checked.",
        "Durability, cover requirements (4.4) and fire resistance are the responsibility of the designer.",
        "A licensed engineer remains responsible for the model, its inputs and the use of these results.",
    ]) + "</ul>")

    # 9 verification
    if verification is not None:
        n_b = len(verification)
        n_c = sum(len(cs) for _, cs, _ in verification)
        ok = all(err is None and all(c.passed for c in cs) for _, cs, err in verification)
        out.append("<h2><span class='n'>9</span>Software verification</h2>")
        out.append(f"<div class='callout {'ok' if ok else ''}'>This build of Tragwerk {__version__} ran its verification suite when producing this report: "
                   f"{n_b} benchmarks with {n_c} comparisons against closed-form solutions and hand calculations, "
                   f"<b>{'all passed' if ok else 'NOT all passed – do not rely on these results'}</b>. "
                   "The solver is also cross-checked against two independent programs in the test suite.</div>")
    out.append(f"<footer>Generated by Tragwerk {__version__} on {today}. Results are reproducible from the model file; "
               f"analysis time {res.runtime_s:.2f} s.</footer></div></body></html>")
    return "".join(out)
