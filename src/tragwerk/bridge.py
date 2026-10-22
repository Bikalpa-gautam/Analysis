"""Parametric generator: a continuous reinforced-concrete girder bridge.

Gives a complete ``tragwerk3d/0.1`` model of a beam-and-slab road bridge:
longitudinal girders continuous over one or more spans, crossbeams at the
supports and at midspan, bearings at the abutments and piers built into the
crossbeams (integral piers, fixed at the foundation).

Loads (EN 1991-2, Load Model 1, adjustment factors α = 1):

* G1  self weight of girders, crossbeams, piers and the deck slab,
* G2  surfacing on the carriageway and parapets on both edges,
* UDL one case per span (pattern loading): 9 kN/m² on notional lane 1,
  2.5 kN/m² on the other lanes and the remaining area,
* TS  the tandem system (axles 300 / 200 / 100 kN in lanes 1 / 2 / 3, 1.2 m
  apart, wheels 2.0 m apart) at a few stations of every span; the positions are
  mutually exclusive load cases (one group), so the combinations move it.

Deck loads reach the girders by the lever rule across the deck (a slab simply
supported between the girders; outside the outer girders the load goes to the
edge girder). Notional lanes are placed from the left kerb, which is the
worst case for girder 1. Traffic actions use EN 1990 Annex A2: γ_Q = 1.35,
ψ = 0.75 / 0.75 / 0 for TS and 0.40 / 0.40 / 0 for UDL (group gr1a).

Not included: braking and acceleration, wind, temperature, fatigue, the
EN 1992-2 specific rules and torsion of a single spine girder from eccentric
traffic. The calculation report states these limits.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

from .building import CONCRETE_NAMES, _section
from .model3d import SCHEMA3D

TS_AXLES = (300.0, 200.0, 100.0)        # EN 1991-2 Table 4.2, Q_ik per axle, lanes 1-3
UDL_LANE = (9.0, 2.5, 2.5)              # q_ik, lanes 1-3; further lanes and remaining area 2.5
UDL_REST = 2.5


def _cracked(sec: dict) -> dict:
    """Girders and crossbeams carry compatibility torsion only: GJ of the cracked section, about 10 % (EN 1992-1-1 6.3.1(2))."""
    return dict(sec, torsion_factor=0.1)


@dataclass
class BridgeSpec:
    name: str = "Road bridge, 3 spans, 2 girders"
    spans: list = field(default_factory=lambda: [16.0, 20.0, 16.0])
    width: float = 10.0                       # deck width, m
    carriageway: float = 8.0                  # between the kerbs, centred on the deck, m
    girders: int = 2
    pier_heights: list = field(default_factory=lambda: [6.0, 6.0])   # one per interior support (the last repeats)
    pier_columns: str = "under_girders"       # under_girders: one column under each girder | single: one column on the axis
    divisions: int = 8                        # girder elements per span (even)
    fck: float = 35.0
    girder: dict = field(default_factory=lambda: _cracked(_section(1000, 1600, (10, 40), (10, 40), side=(4, 20), links=(14, 4, 200), cover=50)))
    crossbeam: dict = field(default_factory=lambda: _cracked(_section(500, 1200, (4, 25), (4, 25), side=(2, 16), links=(12, 2, 200), cover=45)))
    pier: dict = field(default_factory=lambda: _section(1000, 1000, (5, 32), (5, 32), side=(3, 32), links=(12, 2, 200), cover=50))
    pier_cap: dict = field(default_factory=lambda: _cracked(_section(1600, 1800, (18, 40), (8, 32), side=(4, 25), links=(16, 4, 150), cover=50)))
    slab_mm: float = 250.0                    # deck slab thickness for self weight
    surfacing: float = 2.4                    # kN/m² on the carriageway
    parapet: float = 1.5                      # kN/m on each edge
    traffic: bool = True                      # EN 1991-2 Load Model 1
    ts_stations: list = field(default_factory=lambda: [0.05, 0.45, 0.95])   # tandem positions, fraction of each span
    pier_l0_factor: float = 1.0
    combination_rule: str = "6.10"
    modes: int = 0
    seismic: dict | None = None


def _check(spec: BridgeSpec):
    if not 1 <= len(spec.spans) <= 8 or any(not 5.0 <= float(s) <= 200.0 for s in spec.spans):
        raise ValueError("spans: 1 to 8 spans of 5 to 200 m")
    if not 1 <= int(spec.girders) <= 6:
        raise ValueError("girders: 1 to 6")
    if not 3.0 <= spec.width <= 40.0 or not 2.5 <= spec.carriageway <= spec.width:
        raise ValueError("width 3 to 40 m and a carriageway of 2.5 m up to the deck width")
    if len(spec.spans) > 1 and (not spec.pier_heights or any(not 1.0 <= float(h) <= 100.0 for h in spec.pier_heights)):
        raise ValueError("pier_heights: 1 to 100 m")
    if spec.pier_columns not in ("under_girders", "single"):
        raise ValueError("pier_columns: under_girders or single")
    if int(spec.divisions) < 2 or int(spec.divisions) > 40 or int(spec.divisions) % 2:
        raise ValueError("divisions: an even number from 2 to 40")
    if any(not 0.0 <= float(f) <= 1.0 for f in spec.ts_stations) or len(spec.ts_stations) > 8:
        raise ValueError("ts_stations: up to 8 fractions between 0 and 1")
    if len(spec.spans) * len(spec.ts_stations) > 60:
        raise ValueError("too many tandem positions: reduce spans × stations to 60")


def notional_lanes(w: float) -> tuple[int, float]:
    """Number and width of notional lanes, EN 1991-2 Table 4.1."""
    if w < 5.4:
        return 1, 3.0
    if w < 6.0:
        return 2, w / 2
    return int(math.floor(w / 3.0 + 1e-9)), 3.0


def lever(y: float, ys: list[float]) -> list[float]:
    """Share of a load at transverse position y carried by each girder (lever rule)."""
    n = len(ys)
    w = [0.0] * n
    if n == 1 or y <= ys[0]:
        w[0] = 1.0
        return w
    if y >= ys[-1]:
        w[-1] = 1.0
        return w
    for g in range(n - 1):
        if ys[g] <= y <= ys[g + 1]:
            t = (y - ys[g]) / (ys[g + 1] - ys[g])
            w[g], w[g + 1] = 1.0 - t, t
            return w
    return w


def lever_strip(y0: float, y1: float, ys: list[float]) -> list[float]:
    """Load per girder (per unit area load) of a strip y0 ≤ y ≤ y1: exact integral of the lever rule."""
    pts = sorted({y0, y1, *[y for y in ys if y0 < y < y1]})
    out = [0.0] * len(ys)
    for a, b in zip(pts[:-1], pts[1:]):
        wa, wb = lever(a, ys), lever(b, ys)
        for g in range(len(ys)):
            out[g] += (b - a) * (wa[g] + wb[g]) / 2
    return out


def generate_bridge(spec: BridgeSpec | dict | None = None) -> dict:
    if spec is None:
        spec = BridgeSpec()
    elif isinstance(spec, dict):
        base = asdict(BridgeSpec())
        base.update(spec)
        spec = BridgeSpec(**base)
    _check(spec)
    nsp, ng, nd = len(spec.spans), int(spec.girders), int(spec.divisions)
    xs_sup = [0.0]
    for s in spec.spans:
        xs_sup.append(round(xs_sup[-1] + float(s), 6))
    heights = [float(spec.pier_heights[min(k, len(spec.pier_heights) - 1)]) for k in range(nsp - 1)] if nsp > 1 else []
    zd = max(heights) if heights else 0.0                               # deck level
    s_g = spec.width / ng
    ys = [round(-spec.width / 2 + s_g / 2 + g * s_g, 6) for g in range(ng)]
    st = []                                                              # stations along the girders
    for s in range(nsp):
        for e in range(nd + (1 if s == nsp - 1 else 0)):
            st.append(round(xs_sup[s] + float(spec.spans[s]) * e / nd, 6))
    sup_idx = [s * nd for s in range(nsp + 1)]
    mid_idx = [s * nd + nd // 2 for s in range(nsp)]
    gname = lambda g, i: f"G{g + 1}-{i}"          # noqa: E731
    sup_names = ["A0"] + [f"P{k}" for k in range(1, nsp)] + [f"A{nsp}"]

    nodes = [{"id": gname(g, i), "x": x, "y": ys[g], "z": zd} for g in range(ng) for i, x in enumerate(st)]
    members, supports = [], []

    def system(s):
        return "simply_supported" if nsp == 1 else ("end_span" if s in (0, nsp - 1) else "interior_span")

    for g in range(ng):
        for s in range(nsp):
            for e in range(nd):
                i = s * nd + e
                members.append({"id": f"G{g + 1}-S{s + 1}-{e + 1}", "i": gname(g, i), "j": gname(g, i + 1), "section": "Girder",
                                "design": {"type": "beam", "deflection_system": system(s)}})
    pier_top = {}
    for k in range(1, nsp):
        i = sup_idx[k]
        zb = round(zd - heights[k - 1], 6)
        if spec.pier_columns == "under_girders" and ng > 1:
            for g in range(ng):
                bot = f"P{k}-{g + 1}-B"
                nodes.append({"id": bot, "x": st[i], "y": ys[g], "z": zb})
                members.append({"id": f"P{k}-{g + 1}", "i": bot, "j": gname(g, i), "section": "Pier",
                                "design": {"type": "column", "l0_factor": spec.pier_l0_factor}})
                supports.append({"node": bot, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True})
            continue
        mid = [g for g in range(ng) if abs(ys[g]) < 1e-9]
        if mid:
            top = gname(mid[0], i)
        else:
            top = f"P{k}-T"
            nodes.append({"id": top, "x": st[i], "y": 0.0, "z": zd})
        bot = f"P{k}-B"
        nodes.append({"id": bot, "x": st[i], "y": 0.0, "z": zb})
        pier_top[i] = top
        members.append({"id": f"P{k}", "i": bot, "j": top, "section": "Pier", "design": {"type": "column", "l0_factor": spec.pier_l0_factor}})
        supports.append({"node": bot, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True})
    if ng > 1 or pier_top:
        for i in sorted(set(sup_idx + mid_idx)):
            row = [(ys[g], gname(g, i)) for g in range(ng)]
            if i in pier_top and pier_top[i].endswith("-T"):
                row.append((0.0, pier_top[i]))
            row.sort()
            label = sup_names[sup_idx.index(i)] if i in sup_idx else f"M{mid_idx.index(i) + 1}"
            sec = "Pier cap" if i in pier_top else "Crossbeam"          # a single column carries the girders through a pier cap
            for q, ((_, a), (_, b)) in enumerate(zip(row[:-1], row[1:])):
                members.append({"id": f"X-{label}-{q + 1}", "i": a, "j": b, "section": sec, "design": {"type": "beam", "deflection_system": "simply_supported"}})
    for end, i in (("A0", 0), ("An", len(st) - 1)):
        for g in range(ng):
            s = {"node": gname(g, i), "ux": end == "A0", "uy": g == 0, "uz": True, "rx": ng == 1, "ry": False, "rz": False}
            supports.append(s)

    # ---- deck loads to the girders (lever rule), kN/m per girder
    girder_members = {g: [f"G{g + 1}-S{s + 1}-{e + 1}" for s in range(nsp) for e in range(nd)] for g in range(ng)}
    span_members = {(g, s): [f"G{g + 1}-S{s + 1}-{e + 1}" for e in range(nd)] for g in range(ng) for s in range(nsp)}

    def line_loads(per_girder: list[float], mids) -> list[dict]:
        out = []
        for g, w in enumerate(per_girder):
            if abs(w) > 1e-12:
                out += [{"type": "udl", "member": m, "w": round(-w, 6), "direction": "global_z"} for m in mids(g)]
        return out

    slab = lever_strip(-spec.width / 2, spec.width / 2, ys)
    g1 = line_loads([25.0 * spec.slab_mm / 1000.0 * v for v in slab], lambda g: girder_members[g])
    c = spec.carriageway
    surf = lever_strip(-c / 2, c / 2, ys)
    par_l, par_r = lever(-spec.width / 2, ys), lever(spec.width / 2, ys)
    g2 = line_loads([spec.surfacing * surf[g] + spec.parapet * (par_l[g] + par_r[g]) for g in range(ng)], lambda g: girder_members[g])
    cases = [{"id": "G1", "name": "Self weight of the structure and deck slab", "type": "permanent", "self_weight": True, "loads": g1},
             {"id": "G2", "name": "Surfacing and parapets", "type": "permanent", "loads": g2}]

    if spec.traffic:
        nl, wl = notional_lanes(c)
        y0 = -c / 2
        lanes = [(y0 + k * wl, y0 + (k + 1) * wl) for k in range(nl)]
        udl = [0.0] * ng
        for k, (a, b) in enumerate(lanes):
            q = UDL_LANE[k] if k < 3 else UDL_REST
            sh = lever_strip(a, b, ys)
            udl = [u + q * v for u, v in zip(udl, sh)]
        if y0 + nl * wl < c / 2 - 1e-9:
            sh = lever_strip(y0 + nl * wl, c / 2, ys)
            udl = [u + UDL_REST * v for u, v in zip(udl, sh)]
        for s in range(nsp):
            cases.append({"id": f"UDL{s + 1}", "name": f"LM1 UDL on span {s + 1}", "type": "variable", "category": "traffic_UDL",
                          "loads": line_loads(udl, lambda g, s=s: span_members[(g, s)])})
        axle = [0.0] * ng                       # one axle line, all lanes, kN per girder
        for k, (a, b) in enumerate(lanes[:3]):
            yc = (a + b) / 2
            for yw in (yc - 1.0, yc + 1.0):
                sh = lever(yw, ys)
                axle = [p + TS_AXLES[k] / 2 * v for p, v in zip(axle, sh)]
        L_tot = xs_sup[-1]
        for s in range(nsp):
            for q, f in enumerate(spec.ts_stations):
                x0 = xs_sup[s] + float(spec.spans[s]) * float(f)
                xa = [x0, x0 + 1.2] if x0 + 1.2 <= L_tot + 1e-9 else [x0 - 1.2, x0]
                loads = []
                for g in range(ng):
                    if abs(axle[g]) < 1e-12:
                        continue
                    for x in xa:
                        x = min(max(x, 0.0), L_tot)
                        i = max(0, min(len(st) - 2, max(k for k in range(len(st) - 1) if st[k] <= x + 1e-9)))
                        si, e = i // nd, i % nd
                        if si >= nsp:
                            si, e = nsp - 1, nd - 1
                        loads.append({"type": "point", "member": f"G{g + 1}-S{si + 1}-{e + 1}", "p": round(-axle[g], 6),
                                      "a": round(x - st[si * nd + e], 6), "direction": "global_z"})
                pct = int(math.floor(100 * float(f) + 0.5))
                cases.append({"id": f"TS{s + 1}.{q + 1}", "name": f"LM1 tandem, span {s + 1} at {pct} %", "type": "variable",
                              "category": "traffic_TS", "group": "TS", "loads": loads})

    cname = CONCRETE_NAMES.get(int(spec.fck), f"C{spec.fck:g}")
    sections = {"Girder": copy.deepcopy(spec.girder), "Crossbeam": copy.deepcopy(spec.crossbeam), "Pier": copy.deepcopy(spec.pier)}
    if pier_top and ng > 1:
        sections["Pier cap"] = copy.deepcopy(spec.pier_cap)
    for sec in sections.values():
        sec["concrete"] = cname
    extra = {}
    if spec.modes or spec.seismic:
        extra["dynamics"] = {"modes": int(spec.modes or 12)}
    if spec.seismic:
        extra["seismic"] = copy.deepcopy(spec.seismic)
    zs = sorted({round(zd - h, 6) for h in heights} | {zd})
    return {
        "schema": SCHEMA3D,
        "name": spec.name,
        "design": {"design_code": "EC2", "combination_rule": spec.combination_rule},
        "materials": {cname: {"type": "concrete", "fck": spec.fck}, "B500B": {"type": "rebar", "fyk": 500}},
        "sections": sections,
        "nodes": nodes, "supports": supports, "members": members, "load_cases": cases,
        **extra,
        "meta": {
            "generator": "tragwerk.bridge",
            "spec": asdict(spec),
            "grid": {"x": dict(zip(sup_names, xs_sup)), "y": {str(g + 1): ys[g] for g in range(ng)}, "levels": zs},
            "slabs": [{"level": 1, "x0": xs_sup[s], "x1": xs_sup[s + 1], "y0": -spec.width / 2, "y1": spec.width / 2, "z": zd,
                       "t_mm": spec.slab_mm} for s in range(nsp)],
            "deck_z": zd,
        },
    }
