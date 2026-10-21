"""SVG drawings for the calculation report (no external libraries)."""
from __future__ import annotations

import html
import math

import numpy as np

INK = "#1d2430"
MUTED = "#6b7280"
GRID = "#d7dbe2"
POS = "#2f6db5"    # positive values
NEG = "#c2410c"    # negative values
DEF = "#7c3aed"


class Frame:
    """Maps model coordinates (m, Y up) to SVG pixels."""

    def __init__(self, model, width=720, max_height=420, pad=70, extra=0.0):
        xs = [n.x for n in model.nodes.values()]
        ys = [n.y for n in model.nodes.values()]
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)
        size = max(self.x1 - self.x0, self.y1 - self.y0, 1.0)
        self.size = size
        dx = max(self.x1 - self.x0, 1e-9)
        dy = max(self.y1 - self.y0, 0.0)
        e = extra * size
        self.s = min((width - 2 * pad) / (dx + 2 * e), (max_height - 2 * pad) / max(dy + 2 * e, 1e-9))
        self.width = width
        self.height = int(max(dy * self.s + 2 * pad + 2 * e * self.s, 2 * pad + 2 * e * self.s, 160))
        self.ox = (width - dx * self.s) / 2 - self.x0 * self.s
        self.oy = self.height / 2 + (self.y0 + self.y1) / 2 * self.s

    def p(self, x, y):
        return self.ox + x * self.s, self.oy - y * self.s


def _svg(fr: Frame, body: str, title: str) -> str:
    return (f'<svg viewBox="0 0 {fr.width} {fr.height}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{html.escape(title)}" font-family="inherit" font-size="11">{body}</svg>')


def _members(model, fr, color=INK, width=2.2):
    out = []
    for m in model.members.values():
        a, b = model.nodes[m.i], model.nodes[m.j]
        (x1, y1), (x2, y2) = fr.p(a.x, a.y), fr.p(b.x, b.y)
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')
        for end, rel in (("i", m.release_i), ("j", m.release_j)):
            if rel:
                L = math.hypot(x2 - x1, y2 - y1)
                t = 9 / L if end == "i" else 1 - 9 / L
                cx, cy = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#fff" stroke="{color}" stroke-width="1.5"/>')
    return "".join(out)


def _supports(model, fr):
    out = []
    for nid, sp in model.supports.items():
        n = model.nodes[nid]
        x, y = fr.p(n.x, n.y)
        if sp.rz and sp.ux and sp.uy:
            out.append(f'<rect x="{x - 14:.1f}" y="{y + 2:.1f}" width="28" height="5" fill="{INK}"/>')
            for k in range(5):
                xx = x - 12 + k * 6
                out.append(f'<line x1="{xx:.1f}" y1="{y + 7:.1f}" x2="{xx - 5:.1f}" y2="{y + 13:.1f}" stroke="{INK}" stroke-width="1"/>')
        elif sp.uy or sp.ux:
            out.append(f'<path d="M{x:.1f},{y + 2:.1f} l-9,14 h18 z" fill="#fff" stroke="{INK}" stroke-width="1.4"/>')
            if not sp.ux:
                out.append(f'<line x1="{x - 11:.1f}" y1="{y + 20:.1f}" x2="{x + 11:.1f}" y2="{y + 20:.1f}" stroke="{INK}" stroke-width="1.4"/>')
            else:
                for k in range(4):
                    xx = x - 9 + k * 6
                    out.append(f'<line x1="{xx:.1f}" y1="{y + 16:.1f}" x2="{xx - 4:.1f}" y2="{y + 21:.1f}" stroke="{INK}" stroke-width="1"/>')
        if sp.kx or sp.ky or sp.kr:
            out.append(f'<text x="{x + 12:.1f}" y="{y + 22:.1f}" fill="{MUTED}">spring</text>')
    return "".join(out)


def model_svg(model) -> str:
    fr = Frame(model)
    body = [_members(model, fr), _supports(model, fr)]
    for nid, n in model.nodes.items():
        x, y = fr.p(n.x, n.y)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{INK}"/>')
        body.append(f'<text x="{x - 6:.1f}" y="{y - 8:.1f}" text-anchor="end" fill="{MUTED}">{html.escape(nid)}</text>')
    for m in model.members.values():
        a, b = model.nodes[m.i], model.nodes[m.j]
        (x1, y1), (x2, y2) = fr.p(a.x, a.y), fr.p(b.x, b.y)
        L = math.hypot(b.x - a.x, b.y - a.y)
        nx, ny = -(y2 - y1), (x2 - x1)
        k = math.hypot(nx, ny)
        mx, my = (x1 + x2) / 2 + nx / k * 14, (y1 + y2) / 2 + ny / k * 14
        if abs(y2 - y1) < 1e-6:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 10
        body.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" fill="{INK}" font-weight="600">{html.escape(m.id)}</text>')
        body.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 + 18:.1f}" text-anchor="middle" fill="{MUTED}">{L:.2f} m</text>')
    return _svg(fr, "".join(body), "Structural model")


def diagram_svg(res, quantity: str, title: str, unit: str) -> str:
    """ULS envelope (max and min) of N, V or M drawn on the members."""
    model = res.model
    fr = Frame(model, extra=0.12)
    vals = [np.max(np.abs(np.concatenate([e.max[quantity], e.min[quantity]]))) for e in res.envelopes.values()]
    vmax = max(vals) if vals else 0.0
    amp = 0.11 * fr.size * fr.s
    body = [_members(model, fr, color="#9aa3af", width=1.6), _supports(model, fr)]
    sign = -1.0 if quantity == "M" else 1.0    # moments drawn on the tension side
    labels = []
    for mid, env in res.envelopes.items():
        m = model.members[mid]
        a = model.nodes[m.i]
        L, c, s = model.member_geometry(mid)
        for arr, col in ((env.max[quantity], POS), (env.min[quantity], NEG)):
            if vmax <= 0 or np.max(np.abs(arr)) < 1e-9 * max(vmax, 1):
                continue
            pts = []
            for x, v in zip(env.x, arr):
                bx, by = a.x + c * x, a.y + s * x
                off = sign * v / vmax * amp / fr.s
                px, py = fr.p(bx - s * off, by + c * off)
                pts.append((px, py))
            (sx, sy), (ex, ey) = fr.p(a.x, a.y), fr.p(a.x + c * L, a.y + s * L)
            poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in [(sx, sy)] + pts + [(ex, ey)])
            body.append(f'<polygon points="{poly}" fill="{col}" fill-opacity="0.10" stroke="{col}" stroke-width="1.3"/>')
            k = int(np.argmax(np.abs(arr)))
            if abs(arr[k]) > 0.02 * vmax:
                labels.append((pts[k], arr[k], col))
    # label the extreme values, avoiding pile-ups
    placed = []
    for (px, py), v, col in sorted(labels, key=lambda t: -abs(t[1])):
        if any(abs(px - qx) < 46 and abs(py - qy) < 14 for qx, qy in placed):
            continue
        placed.append((px, py))
        body.append(f'<text x="{px:.1f}" y="{py - 5:.1f}" text-anchor="middle" fill="{col}" font-weight="600">{v:.1f}</text>')
    body.append(f'<text x="12" y="18" fill="{INK}" font-weight="600">{html.escape(title)} [{unit}]</text>')
    body.append(f'<text x="12" y="33" fill="{MUTED}">blue: maximum, orange: minimum over all ULS combinations</text>')
    return _svg(fr, "".join(body), title)


def deflection_svg(res) -> tuple[str, str]:
    """Deflected shape under the characteristic combination with the largest displacement."""
    model = res.model
    F, sel = res.combos.matrix("SLS_char")
    U = np.stack([res.cases[c].displacements for c in res.combos.case_ids])   # (cases, nodes, 3)
    disp = np.einsum("kc,cnq->knq", F, U)
    k = int(np.argmax(np.max(np.abs(disp[:, :, :2]), axis=(1, 2))))
    combo = sel[k]
    fr = Frame(model, extra=0.08)
    body = [_members(model, fr, color="#c4c9d1", width=1.6), _supports(model, fr)]
    shapes = []
    dmax = 0.0
    for mid in model.members:
        L, c, s = model.member_geometry(mid)
        a = model.nodes[model.members[mid].i]
        u = sum(f * res.cases[cid].members[mid].u for f, cid in zip(combo.factors, res.combos.case_ids))
        v = sum(f * res.cases[cid].members[mid].v for f, cid in zip(combo.factors, res.combos.case_ids))
        x = res.cases[res.combos.case_ids[0]].members[mid].x
        gx, gy = c * u - s * v, s * u + c * v
        dmax = max(dmax, float(np.max(np.hypot(gx, gy))))
        shapes.append((a, c, s, x, gx, gy))
    mag = 0.07 * fr.size / dmax if dmax > 0 else 1.0
    for a, c, s, x, gx, gy in shapes:
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in (fr.p(a.x + c * xx + mag * dx, a.y + s * xx + mag * dy)
                                                           for xx, dx, dy in zip(x, gx, gy)))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{DEF}" stroke-width="2"/>')
    body.append(f'<text x="12" y="18" fill="{INK}" font-weight="600">Deflected shape, {combo.id}</text>')
    body.append(f'<text x="12" y="33" fill="{MUTED}">max. displacement {dmax * 1000:.2f} mm, drawn ×{mag:.0f}</text>')
    return _svg(fr, "".join(body), "Deflected shape"), combo.label
