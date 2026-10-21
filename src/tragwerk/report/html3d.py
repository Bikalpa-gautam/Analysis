"""Calculation report for 3D models: axonometric drawings, summaries and clause-by-clause checks."""
from __future__ import annotations

import datetime as _dt
import html
import math

import numpy as np

from .. import __version__
from ..codes.base import FAIL, NOT_COVERED, PASS
from ..combinations import CATEGORY_LABEL, PSI
from .html import CSS, _check_block, _table, mk, num, pill, ubar

AZ, EL = math.radians(-35.0), math.radians(28.0)


def _proj(x, y, z):
    X = x * math.cos(AZ) - y * math.sin(AZ)
    Y = z * math.cos(EL) + (x * math.sin(AZ) + y * math.cos(AZ)) * math.sin(EL)
    return X, -Y


class _Axo:
    def __init__(self, model, width=720, pad=40):
        pts = [_proj(n.x, n.y, n.z) for n in model.nodes.values()]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        self.x0, self.x1, self.y0, self.y1 = min(xs), max(xs), min(ys), max(ys)
        dx, dy = max(self.x1 - self.x0, 1e-9), max(self.y1 - self.y0, 1e-9)
        self.s = (width - 2 * pad) / dx
        self.w = width
        self.h = int(min(dy * self.s, 620) + 2 * pad)
        if dy * self.s > 620:
            self.s = 620 / dy
        self.ox = (width - dx * self.s) / 2 - self.x0 * self.s
        self.oy = (self.h - dy * self.s) / 2 - self.y0 * self.s
        self.size = max(dx, dy)

    def p(self, x, y, z):
        X, Y = _proj(x, y, z)
        return self.ox + X * self.s, self.oy + Y * self.s


def _color(u, st):
    if st == FAIL:
        return "#b42318"
    if st == NOT_COVERED:
        return "#a15c07"
    t = min(max(u, 0.0), 1.0)
    # green to amber as utilisation rises
    r = int(23 + (200 - 23) * t ** 2)
    g = int(128 + (140 - 128) * t)
    b = int(61 * (1 - t))
    return f"rgb({r},{g},{b})"


def member_max(res, mid):
    u, st = 0.0, PASS
    for c in res.checks.get(mid, []):
        if c.status == FAIL:
            st = FAIL
        elif c.status == NOT_COVERED and st != FAIL:
            st = NOT_COVERED
        if c.utilisation is not None and c.status in (PASS, FAIL):
            u = max(u, c.utilisation if math.isfinite(c.utilisation) else 9.99)
    return u, st


def axo_svg(res, mode="model") -> str:
    m = res.model
    ax = _Axo(m)
    body = []
    slabs = (m.raw.get("meta") or {}).get("slabs", [])
    for sl in slabs:
        pts = [ax.p(sl["x0"], sl["y0"], sl["z"]), ax.p(sl["x1"], sl["y0"], sl["z"]), ax.p(sl["x1"], sl["y1"], sl["z"]), ax.p(sl["x0"], sl["y1"], sl["z"])]
        body.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="#8ca0b8" fill-opacity="0.08" stroke="none"/>')
    order = sorted(m.members.values(), key=lambda mm: -(m.nodes[mm.i].x + m.nodes[mm.j].x) * math.sin(-AZ) - (m.nodes[mm.i].y + m.nodes[mm.j].y) * math.cos(AZ))
    for mem in order:
        a, b = m.nodes[mem.i], m.nodes[mem.j]
        (x1, y1), (x2, y2) = ax.p(a.x, a.y, a.z), ax.p(b.x, b.y, b.z)
        if mode == "util":
            u, st = member_max(res, mem.id)
            col, w = _color(u, st), 3.2
        else:
            col = "#1d2430" if mem.design.type == "column" else "#2f5f98" if mem.design.type == "beam" else "#7a8695"
            w = 2.0
        body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>')
    if mode == "def":
        F, sel = res.combos.matrix("SLS_char")
        U = np.stack([res.cases[c].displacements for c in res.combos.case_ids])
        disp = np.einsum("kc,cnq->knq", F, U)
        k = int(np.argmax(np.max(np.abs(disp[:, :, :3]), axis=(1, 2))))
        combo = sel[k]
        dmax = 0.0
        shapes = []
        for mid in m.members:
            L, lam = m.axes(mid)
            a = m.nodes[m.members[mid].i]
            g = lambda q: sum(f * getattr(res.cases[cid].members[mid], q) for f, cid in zip(combo.factors, res.combos.case_ids))  # noqa: E731
            u, v, w = g("u"), g("v"), g("w")
            x = res.cases[res.combos.case_ids[0]].members[mid].x
            d = np.outer(u, lam[0]) + np.outer(v, lam[1]) + np.outer(w, lam[2])
            dmax = max(dmax, float(np.max(np.linalg.norm(d, axis=1))))
            shapes.append((a, lam, x, d))
        mag = 0.05 * (max(n.x for n in m.nodes.values()) - min(n.x for n in m.nodes.values()) + 1.0) / dmax if dmax > 0 else 1.0
        for a, lam, x, d in shapes:
            pts = [ax.p(a.x + lam[0][0] * xx + mag * dd[0], a.y + lam[0][1] * xx + mag * dd[1], a.z + lam[0][2] * xx + mag * dd[2]) for xx, dd in zip(x, d)]
            body.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in pts)}" fill="none" stroke="#7c3aed" stroke-width="1.6"/>')
        body.append(f'<text x="12" y="18" font-weight="600" fill="#1d2430">Deflected shape, {html.escape(combo.id)}</text>'
                    f'<text x="12" y="33" fill="#6b7280">max. displacement {dmax * 1000:.1f} mm, drawn ×{mag:.0f}</text>')
    for nid, sp in m.supports.items():
        n = m.nodes[nid]
        x, y = ax.p(n.x, n.y, n.z)
        if all(sp.fix[:3]) and all(sp.fix[3:]):
            body.append(f'<rect x="{x - 6:.1f}" y="{y:.1f}" width="12" height="4" fill="#1d2430"/>')
        else:
            body.append(f'<path d="M{x:.1f},{y:.1f} l-5,8 h10 z" fill="#fff" stroke="#1d2430" stroke-width="1"/>')
    if mode == "util":
        body.append('<text x="12" y="18" font-weight="600" fill="#1d2430">Largest utilisation per member</text>'
                    '<text x="12" y="33" fill="#6b7280">green: low, amber: close to 1, red: fails</text>')
    grid = (m.raw.get("meta") or {}).get("grid")
    if grid and mode == "model":
        for name, gx in grid["x"].items():
            x, y = ax.p(gx, min(grid["y"].values()) - 1.2, 0)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#fff" stroke="#6b7280"/><text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="10" fill="#1d2430">{html.escape(name)}</text>')
        for name, gy in grid["y"].items():
            x, y = ax.p(min(grid["x"].values()) - 1.2, gy, 0)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#fff" stroke="#6b7280"/><text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="10" fill="#1d2430">{html.escape(name)}</text>')
    return (f'<svg viewBox="0 0 {ax.w} {ax.h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="3D model" '
            f'font-family="inherit" font-size="11">{"".join(body)}</svg>')


def render3d(res, verification=None, project_info: dict | None = None, detail_members: int = 12, full: bool = False) -> str:
    m = res.model
    info = project_info or {}
    gov = res.governing()
    status = res.overall_status()
    p = res.params
    today = info.get("date") or _dt.date.today().isoformat()
    gov_u = gov.utilisation if gov and gov.utilisation is not None and math.isfinite(gov.utilisation) else None
    color = {PASS: "var(--pass)", FAIL: "var(--fail)", NOT_COVERED: "var(--nc)"}[status]
    word = {PASS: "All checks pass", FAIL: "Checks fail", NOT_COVERED: "Incomplete: some checks not covered"}[status]
    n_col = sum(1 for mm in m.members.values() if mm.design.type == "column")
    n_beam = sum(1 for mm in m.members.values() if mm.design.type == "beam")
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(m.name)} – calculation</title><style>{CSS}</style></head><body><div class='page'>"]
    out.append(
        f"<header class='top'><div><p class='kicker'>Structural calculation · 3D frame · {html.escape(info.get('project', 'Project'))}</p>"
        f"<h1>{html.escape(m.name)}</h1><div class='meta'>{today} · Tragwerk {__version__} · EN 1992-1-1:2004 with recommended values · "
        f"{len(m.nodes)} nodes, {n_col} columns, {n_beam} beams · {html.escape(info.get('author', ''))}</div></div>"
        f"<div class='verdict'><div class='lab'>Maximum utilisation</div><div class='big' style='color:{color}'>{num(gov_u) if gov_u is not None else '–'}</div>"
        f"<div style='margin-top:6px'>{pill(status)}</div><div class='meta' style='margin-top:4px'>{word}"
        + (f"<br>governing: {html.escape(gov.member)}, {html.escape(gov.title.lower())}" if gov else "") + "</div></div></header>")
    if res.warnings:
        out.append("<div class='callout'><b>Analysis warnings</b><ul class='plain'>" + "".join(f"<li>{html.escape(w)}</li>" for w in res.warnings) + "</ul></div>")

    out.append("<h2><span class='n'>1</span>Basis of design</h2>")
    out.append(_table(["Item", "Basis"], [
        ["Analysis", "Linear-elastic 3D frame analysis (direct stiffness method, 6 DOF per node). Members with axial, St Venant torsion and biaxial bending stiffness; gross concrete sections, E = E<sub>cm</sub>, G = E/(2(1 + ν)), ν = 0.2. Exact member diagrams between nodes."],
        ["Slabs", "Slabs are not modelled as elements. Their loads reach the beams by tributary areas; no rigid-diaphragm constraint is applied." if (m.raw.get("meta") or {}).get("slabs") else "No slabs."],
        ["Combinations", f"EN 1990 Eq. ({html.escape(m.settings.combination_rule)}), pattern loading of variable cases, exclusive groups, recommended ψ factors."],
        ["Design", "EN 1992-1-1:2004 + AC:2010, recommended NDPs: bending with axial force about both axes (6.1, 5.8.9 Eq. 5.39), shear in both planes (6.2), torsion (6.3), slender columns by nominal curvature (5.8.8), detailing (8.2, 9.2, 9.5), deflection (7.4)."],
    ]))
    out.append("<h2><span class='n'>2</span>Structural model</h2>")
    out.append(f"<div class='fig'>{axo_svg(res, 'model')}<div class='cap'>Axonometric view. Dark lines: columns; blue: beams; tinted areas: slab panels (loads only).</div></div>")
    grid = (m.raw.get("meta") or {}).get("grid")
    if grid:
        out.append(_table(["Grid", "Positions [m]"], [
            ["Axes along X", ", ".join(f"{k} = {v:g}" for k, v in grid["x"].items())],
            ["Axes along Y", ", ".join(f"{k} = {v:g}" for k, v in grid["y"].items())],
            ["Levels Z", ", ".join(f"L{k} = {v:g}" for k, v in enumerate(grid["levels"]))],
        ]))
    from ..codes.ec2.materials import concrete_props, steel_props
    rows = []
    for sid, s in m.sections.items():
        c = concrete_props(s.concrete, p)
        As = sum(b[0] for b in s.bars())
        lay = []
        for lab, l in (("top", s.top), ("bottom", s.bottom), ("each side", s.side)):
            if l and l.n:
                lay.append(f"{lab} {l.n}Ø{l.dia_mm:g}")
        rows.append([html.escape(sid), f"{s.b_mm:g} × {s.h_mm:g}", f"{html.escape(s.concrete.name)}, f<sub>cd</sub> = {num(c.fcd)}",
                     ", ".join(lay) + f" (A<sub>s</sub> = {As:.0f} mm²)",
                     f"Ø{s.links.dia_mm:g} @ {s.links.spacing_mm:g}" if s.links else "none",
                     str(sum(1 for mm in m.members.values() if mm.section == sid))])
    out.append(_table(["Section", "b × h [mm]", "Concrete [MPa]", "Longitudinal bars", "Links", "Members"], rows))
    fixed = sum(1 for sp in m.supports.values() if all(sp.fix))
    out.append(f"<p>{len(m.supports)} supports, {fixed} of them fully fixed. Member local axes: x from start to end node; for beams y is horizontal and z points up; for columns z is global X.</p>")

    out.append("<h2><span class='n'>3</span>Actions</h2>")
    rows = []
    for cid, lc in m.load_cases.items():
        cr = res.cases[cid]
        kind = "permanent" if lc.type == "permanent" else f"variable, {CATEGORY_LABEL[lc.category]}, ψ = {', '.join(f'{v:g}' for v in PSI[lc.category])}"
        if lc.group:
            kind += f", group “{html.escape(lc.group)}”"
        n_loads = len(lc.nodal) + len(lc.distributed) + len(lc.point)
        rows.append([html.escape(cid), html.escape(lc.name), kind + (" + self weight" if lc.self_weight else ""), str(n_loads),
                     num(cr.applied["Fx"]), num(cr.applied["Fy"]), num(cr.applied["Fz"])])
    out.append(_table(["Case", "Name", "Type", "Loads", "ΣF<sub>x</sub> [kN]", "ΣF<sub>y</sub> [kN]", "ΣF<sub>z</sub> [kN]"], rows, {3, 4, 5, 6}))

    counts = {k: sum(1 for c in res.combos.combos if c.kind == k) for k in ("ULS", "SLS_char", "SLS_freq", "SLS_qp")}
    out.append("<h2><span class='n'>4</span>Combinations of actions</h2>")
    out.append(f"<p>{counts['ULS']} ULS, {counts['SLS_char']} characteristic, {counts['SLS_freq']} frequent and {counts['SLS_qp']} quasi-permanent combinations, all checked.</p>")

    out.append("<h2><span class='n'>5</span>Analysis results</h2>")
    eq = max(max(abs(v) for v in cr.equilibrium.values()) for cr in res.cases.values())
    out.append(f"<p>Equilibrium self-check: the largest residual between applied loads and reactions over all load cases is {eq:.1e} kN (or kNm).</p>")
    tot = np.zeros(6)
    out.append("<h3>Support reactions, ULS envelope (min … max)</h3>")
    head = "".join(f"<th class='num'>{h}</th>" for h in ("R<sub>x</sub> [kN]", "R<sub>y</sub> [kN]", "R<sub>z</sub> [kN]", "M<sub>x</sub> [kNm]", "M<sub>y</sub> [kNm]", "M<sub>z</sub> [kNm]"))
    body = "".join("<tr><td style='white-space:nowrap'>" + html.escape(nid) + "</td>" + "".join(
        f"<td class='num'>{r['uls_min'][q]:.1f} … {r['uls_max'][q]:.1f}</td>" for q in range(6)) + "</tr>" for nid, r in res.reactions.items())
    out.append(f"<table style='font-size:10.5px'><thead><tr><th>Node</th>{head}</tr></thead><tbody>{body}</tbody></table>")
    out.append(f"<div class='fig'>{axo_svg(res, 'def')}<div class='cap'>Short-term elastic deflections, gross sections.</div></div>")
    out.append(f"<div class='fig'>{axo_svg(res, 'util')}</div>")

    out.append("<h2><span class='n'>6</span>Summary of design checks</h2>")
    ranked = []
    for mid, lst in res.checks.items():
        u, st = member_max(res, mid)
        worst = max((c for c in lst if c.utilisation is not None and c.status in (PASS, FAIL)), key=lambda c: c.utilisation, default=None)
        ranked.append((st == FAIL, u, mid, st, worst))
    ranked.sort(key=lambda t: (not t[0], -t[1]))
    rows = [[html.escape(mid), html.escape(m.members[mid].design.type), html.escape(m.members[mid].section),
             html.escape(w.title if w else "–"), ubar(u, st), pill(st)] for _, u, mid, st, w in ranked]
    out.append(_table(["Member", "Type", "Section", "Governing check", "Utilisation η", "Result"], rows))

    out.append("<h2><span class='n'>7</span>Design checks in detail</h2>")
    if full:
        chosen = [t[2] for t in ranked]
        out.append("<p class='muted'>All members.</p>")
    else:
        chosen = [t[2] for t in ranked if t[3] == FAIL] + [t[2] for t in ranked if t[3] != FAIL][:detail_members]
        out.append(f"<p class='muted'>Shown for every member that fails and the {detail_members} most utilised members. "
                   "Run with the full option for every member; all members appear in the summary above.</p>")
    for mid in dict.fromkeys(chosen):
        mem = m.members[mid]
        out.append(f"<h3>Member {html.escape(mid)} <span class='muted' style='font-weight:400'>· {html.escape(mem.section)} · {html.escape(mem.design.type)}</span></h3>")
        out += [_check_block(c) for c in res.checks[mid]]

    out.append("<h2><span class='n'>8</span>Assumptions and limitations</h2><ul class='plain'>" + "".join(f"<li>{t}</li>" for t in [
        "Linear-elastic first-order analysis with gross sections; slender columns get second-order moments by the nominal curvature method on the isolated member (5.8.8) with the effective lengths set in the model.",
        "The generator assumes a braced frame (effective length = storey height). For an unbraced sway frame, increase the column effective length factors.",
        "Slabs are not designed and not modelled as plates; there is no rigid-diaphragm constraint. Wind is applied at the windward facade nodes.",
        "Biaxial bending by EN 1992-1-1 Eq. 5.39 using exact uniaxial resistances; shear checked in each plane separately; torsion shown where T<sub>Ed</sub> ≥ 0.1·T<sub>Rd,c</sub>.",
        "Reinforcement uniform along each member; anchorage, laps, crack widths and punching are not checked.",
        "A licensed engineer remains responsible for the model, its inputs and the use of these results.",
    ]) + "</ul>")
    if verification is not None:
        n_c = sum(len(cs) for _, cs, _ in verification)
        ok = all(err is None and all(c.passed for c in cs) for _, cs, err in verification)
        out.append("<h2><span class='n'>9</span>Software verification</h2>")
        out.append(f"<div class='callout {'ok' if ok else ''}'>This build ran its verification suite when producing this report: {len(verification)} benchmarks with {n_c} comparisons, "
                   f"<b>{'all passed' if ok else 'NOT all passed – do not rely on these results'}</b>.</div>")
    out.append(f"<footer>Generated by Tragwerk {__version__} on {today}; analysis time {res.runtime_s:.2f} s.</footer></div></body></html>")
    return "".join(out)
