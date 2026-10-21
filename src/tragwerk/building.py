"""Parametric generator: a reinforced-concrete building frame from a few numbers.

Gives a complete ``tragwerk3d/0.1`` model: grid of columns and beams, supports,
slab self weight, finishes, imposed and snow loads carried to the beams by
tributary areas (45° two-way yield lines, or one-way), facade loads and wind
in four directions as one exclusive group.

Grid naming follows drawing practice: grid lines A, B, C ... along X,
1, 2, 3 ... along Y, levels L0 (foundation) to Ln (roof).
"""
from __future__ import annotations

import copy
import string
from dataclasses import asdict, dataclass, field

from .model3d import SCHEMA3D


CONCRETE_NAMES = {20: "C20/25", 25: "C25/30", 30: "C30/37", 35: "C35/45", 40: "C40/50", 45: "C45/55", 50: "C50/60",
                  55: "C55/67", 60: "C60/75", 70: "C70/85", 80: "C80/95", 90: "C90/105"}


def _section(b, h, top, bottom, side=None, links=(8, 2, 200), cover=30):
    s = {"type": "rc_rect", "b_mm": b, "h_mm": h, "concrete": "concrete", "rebar": "B500B", "cover_mm": cover,
         "top": {"n": top[0], "dia_mm": top[1]}, "bottom": {"n": bottom[0], "dia_mm": bottom[1]},
         "links": {"dia_mm": links[0], "legs": links[1], "spacing_mm": links[2]}}
    if side:
        s["side"] = {"n": side[0], "dia_mm": side[1]}
    return s


@dataclass
class BuildingSpec:
    name: str = "Office building, 3 × 2 bays, 4 storeys"
    spans_x: list = field(default_factory=lambda: [6.0, 6.0, 6.0])
    spans_y: list = field(default_factory=lambda: [5.5, 5.5])
    storeys: list = field(default_factory=lambda: [3.6, 3.2, 3.2, 3.2])   # bottom to top, m
    base: str = "fixed"                      # fixed | pinned
    fck: float = 30.0
    column: dict = field(default_factory=lambda: _section(400, 400, (3, 20), (3, 20), side=(1, 20), links=(8, 2, 240), cover=35))
    beam_x: dict = field(default_factory=lambda: _section(300, 550, (4, 20), (3, 20)))
    beam_y: dict = field(default_factory=lambda: _section(300, 500, (4, 20), (3, 16)))
    slab_mm: float = 220.0                   # slab thickness for self weight
    finishes: float = 1.5                    # kN/m2, floors and roof
    imposed: float = 3.0                     # kN/m2, floors
    imposed_category: str = "B"
    snow: float = 1.0                        # kN/m2 on the roof (category snow)
    facade: float = 6.0                      # kN/m on perimeter beams of every floor
    wind: float = 0.9                        # kN/m2 net pressure on the facade
    distribution: str = "two_way"            # two_way | one_way_x | one_way_y
    pattern_by_storey: bool = False          # one imposed load case per storey
    column_l0_factor: float = 1.0            # braced frame; use about 2.0 for an unbraced sway frame
    combination_rule: str = "6.10"


def grid_names(n: int) -> list[str]:
    letters = string.ascii_uppercase
    return [letters[k] if k < 26 else letters[k // 26 - 1] + letters[k % 26] for k in range(n)]


def _tributary(L: float, S: float, q: float, mode: str) -> list[tuple[float, float, float, float]]:
    """Line-load segments (a, b, w1, w2) on a beam of length L from one panel of width S with load q (kN/m2).

    mode 'two_way': 45° lines; 'full': q·S/2 uniform (one-way support); 'none': nothing.
    """
    if mode == "none" or q == 0:
        return []
    if mode == "full":
        return [(0.0, L, q * S / 2, q * S / 2)]
    c = min(L, S) / 2
    p = q * c
    segs = [(0.0, c, 0.0, p)]
    if L - 2 * c > 1e-9:
        segs.append((c, L - c, p, p))
    segs.append((L - c, L, p, 0.0))
    return segs


def generate_building(spec: BuildingSpec | dict | None = None) -> dict:
    if spec is None:
        spec = BuildingSpec()
    elif isinstance(spec, dict):
        base = asdict(BuildingSpec())
        base.update(spec)
        spec = BuildingSpec(**base)
    gx, gy = grid_names(len(spec.spans_x) + 1), [str(k + 1) for k in range(len(spec.spans_y) + 1)]
    xs = [0.0]
    for s in spec.spans_x:
        xs.append(round(xs[-1] + float(s), 6))
    ys = [0.0]
    for s in spec.spans_y:
        ys.append(round(ys[-1] + float(s), 6))
    zs = [0.0]
    for h in spec.storeys:
        zs.append(round(zs[-1] + float(h), 6))
    nlev = len(zs) - 1
    nid = lambda i, j, k: f"{gx[i]}{gy[j]}-L{k}"  # noqa: E731

    nodes = [{"id": nid(i, j, k), "x": x, "y": y, "z": z}
             for k, z in enumerate(zs) for j, y in enumerate(ys) for i, x in enumerate(xs)]
    fix_rot = spec.base == "fixed"
    supports = [{"node": nid(i, j, 0), "ux": True, "uy": True, "uz": True, "rx": fix_rot, "ry": fix_rot, "rz": True}
                for j in range(len(ys)) for i in range(len(xs))]

    def system(idx, n):
        if n == 1:
            return "simply_supported"
        return "end_span" if idx in (0, n - 1) else "interior_span"

    members = []
    for k in range(1, nlev + 1):
        for j in range(len(ys)):
            for i in range(len(xs)):
                members.append({"id": f"C-{gx[i]}{gy[j]}-L{k}", "i": nid(i, j, k - 1), "j": nid(i, j, k), "section": "Column",
                                "design": {"type": "column", "l0_factor": spec.column_l0_factor}})
        for j in range(len(ys)):
            for i in range(len(xs) - 1):
                members.append({"id": f"BX-{gx[i]}{gy[j]}-L{k}", "i": nid(i, j, k), "j": nid(i + 1, j, k), "section": "Beam X",
                                "design": {"type": "beam", "deflection_system": system(i, len(xs) - 1)}})
        for i in range(len(xs)):
            for j in range(len(ys) - 1):
                members.append({"id": f"BY-{gx[i]}{gy[j]}-L{k}", "i": nid(i, j, k), "j": nid(i, j + 1, k), "section": "Beam Y",
                                "design": {"type": "beam", "deflection_system": system(j, len(ys) - 1)}})

    def area_loads(q: float, levels) -> list[dict]:
        """Distribute an area load on the given levels to the beams."""
        loads = []
        if q == 0:
            return loads
        mx = {"two_way": "two_way", "one_way_x": "none", "one_way_y": "full"}[spec.distribution]
        my = {"two_way": "two_way", "one_way_x": "full", "one_way_y": "none"}[spec.distribution]
        for k in levels:
            for j in range(len(ys) - 1):
                for i in range(len(xs) - 1):
                    Lx, Ly = xs[i + 1] - xs[i], ys[j + 1] - ys[j]
                    for jj in (j, j + 1):       # beams along X on both edges of the panel
                        for a, b, w1, w2 in _tributary(Lx, Ly, q, mx):
                            loads.append({"type": "udl", "member": f"BX-{gx[i]}{gy[jj]}-L{k}", "w1": -w1, "w2": -w2,
                                          "a": round(a, 6), "b": round(b, 6), "direction": "global_z"})
                    for ii in (i, i + 1):       # beams along Y
                        for a, b, w1, w2 in _tributary(Ly, Lx, q, my):
                            loads.append({"type": "udl", "member": f"BY-{gx[ii]}{gy[j]}-L{k}", "w1": -w1, "w2": -w2,
                                          "a": round(a, 6), "b": round(b, 6), "direction": "global_z"})
        return loads

    floors = list(range(1, nlev + 1))
    occupied = floors[:-1] if nlev > 1 else floors
    roof = [nlev]
    g_slab = area_loads(25.0 * spec.slab_mm / 1000.0, floors)
    g_fin = area_loads(spec.finishes, floors)
    facade = []
    if spec.facade:
        for k in occupied if nlev > 1 else floors:
            for j in (0, len(ys) - 1):
                for i in range(len(xs) - 1):
                    facade.append({"type": "udl", "member": f"BX-{gx[i]}{gy[j]}-L{k}", "w": -spec.facade, "direction": "global_z"})
            for i in (0, len(xs) - 1):
                for j in range(len(ys) - 1):
                    facade.append({"type": "udl", "member": f"BY-{gx[i]}{gy[j]}-L{k}", "w": -spec.facade, "direction": "global_z"})
    cases = [
        {"id": "G1", "name": "Self weight of frame and slabs", "type": "permanent", "self_weight": True, "loads": g_slab},
        {"id": "G2", "name": "Finishes and facade", "type": "permanent", "loads": g_fin + facade},
    ]
    if spec.imposed:
        if spec.pattern_by_storey:
            for k in occupied:
                cases.append({"id": f"Q-L{k}", "name": f"Imposed load, level {k}", "type": "variable",
                              "category": spec.imposed_category, "loads": area_loads(spec.imposed, [k])})
        else:
            cases.append({"id": "Q", "name": "Imposed load, all floors", "type": "variable",
                          "category": spec.imposed_category, "loads": area_loads(spec.imposed, occupied)})
    if spec.snow and nlev > 1:
        cases.append({"id": "S", "name": "Snow on the roof", "type": "variable", "category": "snow", "loads": area_loads(spec.snow, roof)})
    if spec.wind:
        def htrib(k):
            below = zs[k] - zs[k - 1]
            above = zs[k + 1] - zs[k] if k < nlev else 0.0
            return below / 2 + above / 2

        for cid, axis, sgn, label in (("W+X", 0, 1, "Wind in +X"), ("W-X", 0, -1, "Wind in -X"),
                                      ("W+Y", 1, 1, "Wind in +Y"), ("W-Y", 1, -1, "Wind in -Y")):
            loads = []
            for k in floors:
                if axis == 0:
                    i = 0 if sgn > 0 else len(xs) - 1
                    for j in range(len(ys)):
                        bt = ((ys[j] - ys[j - 1]) if j > 0 else 0) / 2 + ((ys[j + 1] - ys[j]) if j < len(ys) - 1 else 0) / 2
                        loads.append({"type": "nodal", "node": nid(i, j, k), "fx": round(sgn * spec.wind * htrib(k) * bt, 6)})
                else:
                    j = 0 if sgn > 0 else len(ys) - 1
                    for i in range(len(xs)):
                        bt = ((xs[i] - xs[i - 1]) if i > 0 else 0) / 2 + ((xs[i + 1] - xs[i]) if i < len(xs) - 1 else 0) / 2
                        loads.append({"type": "nodal", "node": nid(i, j, k), "fy": round(sgn * spec.wind * htrib(k) * bt, 6)})
            cases.append({"id": cid, "name": label, "type": "variable", "category": "wind", "group": "wind", "loads": loads})

    cname = CONCRETE_NAMES.get(int(spec.fck), f"C{spec.fck:g}")
    sections = {"Column": copy.deepcopy(spec.column), "Beam X": copy.deepcopy(spec.beam_x), "Beam Y": copy.deepcopy(spec.beam_y)}
    for sec in sections.values():
        sec["concrete"] = cname
    return {
        "schema": SCHEMA3D,
        "name": spec.name,
        "design": {"design_code": "EC2", "combination_rule": spec.combination_rule},
        "materials": {cname: {"type": "concrete", "fck": spec.fck}, "B500B": {"type": "rebar", "fyk": 500}},
        "sections": sections,
        "nodes": nodes, "supports": supports, "members": members, "load_cases": cases,
        "meta": {
            "generator": "tragwerk.building",
            "spec": asdict(spec),
            "grid": {"x": dict(zip(gx, xs)), "y": dict(zip(gy, ys)), "levels": zs},
            "slabs": [{"level": k, "x0": xs[i], "x1": xs[i + 1], "y0": ys[j], "y1": ys[j + 1], "z": zs[k], "t_mm": spec.slab_mm}
                      for k in floors for j in range(len(ys) - 1) for i in range(len(xs) - 1)],
        },
    }
