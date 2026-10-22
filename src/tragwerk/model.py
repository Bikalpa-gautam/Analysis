"""Structural model and its open JSON format ("tragwerk/0.1").

Units and conventions
---------------------
* Global geometry in metres, forces in kN, moments in kNm.
* Section dimensions in millimetres (every such key ends in ``_mm``).
* Material strengths in MPa (N/mm2).
* Global axes: X to the right, Y up. Gravity loads are therefore negative Y.
* Member local axes: x from node i to node j, y = x rotated +90 degrees.
  "bottom" reinforcement is on the local -y face, i.e. on the right-hand side
  when walking from node i to node j. For a beam drawn left to right that is
  the physical bottom.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "tragwerk/0.1"

LOAD_DIRECTIONS = ("global_x", "global_y", "local_x", "local_y")
VARIABLE_CATEGORIES = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "snow", "snow_high", "wind", "temperature", "traffic_TS", "traffic_UDL",
)
DESIGN_TYPES = ("beam", "column", "none")
DEFLECTION_SYSTEMS = ("simply_supported", "end_span", "interior_span", "cantilever")


class ModelError(ValueError):
    """Raised when a model is invalid. The message names the offending field."""


# ---------------------------------------------------------------- materials
@dataclass(frozen=True)
class Concrete:
    name: str
    fck: float                 # MPa, characteristic cylinder strength
    density: float = 25.0      # kN/m3, reinforced concrete
    dmax_mm: float = 16.0      # maximum aggregate size


@dataclass(frozen=True)
class Rebar:
    name: str
    fyk: float = 500.0         # MPa
    Es: float = 200_000.0      # MPa
    eps_uk: float = 0.05       # characteristic strain at maximum force (class B)


# ---------------------------------------------------------------- sections
@dataclass(frozen=True)
class BarLayer:
    n: int
    dia_mm: float
    depth_mm: float | None = None  # explicit depth from the top (+y) face; None = derived

    @property
    def area_mm2(self) -> float:
        return self.n * math.pi * self.dia_mm ** 2 / 4.0


@dataclass(frozen=True)
class Links:
    dia_mm: float
    legs: int
    spacing_mm: float
    fyk: float | None = None   # defaults to the section's rebar fyk

    @property
    def area_mm2(self) -> float:
        return self.legs * math.pi * self.dia_mm ** 2 / 4.0


@dataclass(frozen=True)
class RCRectSection:
    name: str
    b_mm: float
    h_mm: float
    concrete: Concrete
    rebar: Rebar
    cover_mm: float
    bottom: BarLayer | None = None
    top: BarLayer | None = None
    links: Links | None = None
    extra_layers: tuple[BarLayer, ...] = ()
    stiffness_factor: float = 1.0  # multiplies EI and EA in the analysis (e.g. 0.5 for cracked)

    # --- derived geometry (mm) ---
    @property
    def link_dia(self) -> float:
        return self.links.dia_mm if self.links else 0.0

    def layer_depth(self, layer: BarLayer, face: str) -> float:
        """Depth of a layer's centroid measured from the top (+y) face, in mm."""
        if layer.depth_mm is not None:
            return layer.depth_mm
        offset = self.cover_mm + self.link_dia + layer.dia_mm / 2.0
        return offset if face == "top" else self.h_mm - offset

    def layers(self) -> list[tuple[float, float, str]]:
        """All bar layers as (area mm2, depth from top mm, label)."""
        out: list[tuple[float, float, str]] = []
        if self.top and self.top.n > 0:
            out.append((self.top.area_mm2, self.layer_depth(self.top, "top"), "top"))
        if self.bottom and self.bottom.n > 0:
            out.append((self.bottom.area_mm2, self.layer_depth(self.bottom, "bottom"), "bottom"))
        for k, lay in enumerate(self.extra_layers):
            if lay.n > 0:
                if lay.depth_mm is None:
                    raise ModelError(f"section {self.name}: extra layer {k} needs depth_mm")
                out.append((lay.area_mm2, lay.depth_mm, f"extra {k + 1}"))
        return out

    @property
    def area_m2(self) -> float:
        return self.b_mm * self.h_mm * 1e-6

    @property
    def inertia_m4(self) -> float:
        return self.b_mm * self.h_mm ** 3 / 12.0 * 1e-12


# ---------------------------------------------------------------- topology
@dataclass(frozen=True)
class Node:
    id: str
    x: float
    y: float


@dataclass(frozen=True)
class Support:
    node: str
    ux: bool = False
    uy: bool = False
    rz: bool = False
    kx: float = 0.0   # kN/m spring (only used when the direction is not fixed)
    ky: float = 0.0
    kr: float = 0.0   # kNm/rad


@dataclass(frozen=True)
class MemberDesign:
    type: str = "beam"                     # beam | column | none
    deflection_system: str = "simply_supported"
    l0_factor: float = 1.0                 # effective length / member length (columns)
    creep_coefficient: float = 2.0         # phi(inf, t0) for the deflection estimate


@dataclass(frozen=True)
class Member:
    id: str
    i: str
    j: str
    section: str
    release_i: bool = False   # moment release (hinge) at node i
    release_j: bool = False
    design: MemberDesign = field(default_factory=MemberDesign)


# ---------------------------------------------------------------- loads
@dataclass(frozen=True)
class NodalLoad:
    node: str
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0


@dataclass(frozen=True)
class DistributedLoad:
    """Linearly varying line load over [a, b] (metres from node i), per metre of member."""
    member: str
    w1: float
    w2: float
    a: float | None = None
    b: float | None = None
    direction: str = "global_y"


@dataclass(frozen=True)
class PointLoad:
    member: str
    p: float
    a: float
    direction: str = "global_y"


@dataclass(frozen=True)
class LoadCase:
    id: str
    type: str                       # permanent | variable
    name: str = ""
    category: str | None = None     # variable actions: EN 1990 Table A1.1 category
    group: str | None = None        # cases in one group never act together
    self_weight: bool = False
    nodal: tuple[NodalLoad, ...] = ()
    distributed: tuple[DistributedLoad, ...] = ()
    point: tuple[PointLoad, ...] = ()


@dataclass(frozen=True)
class DesignSettings:
    design_code: str = "EC2"
    combination_rule: str = "6.10"   # "6.10" or "6.10ab"
    params: dict = field(default_factory=dict)
    stations_per_member: int = 40


@dataclass
class Model:
    name: str
    nodes: dict[str, Node]
    members: dict[str, Member]
    sections: dict[str, RCRectSection]
    supports: dict[str, Support]
    load_cases: dict[str, LoadCase]
    settings: DesignSettings = field(default_factory=DesignSettings)
    raw: dict = field(default_factory=dict, repr=False)

    # convenience geometry
    def member_geometry(self, mid: str) -> tuple[float, float, float]:
        """(length m, cos, sin) of a member."""
        m = self.members[mid]
        ni, nj = self.nodes[m.i], self.nodes[m.j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        L = math.hypot(dx, dy)
        return L, dx / L, dy / L


# ================================================================ parsing
def _req(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise ModelError(f"{where}: missing required field '{key}'")
    return d[key]


def _num(v: Any, where: str, positive: bool = False, nonneg: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ModelError(f"{where}: expected a number, got {v!r}")
    if positive and v <= 0:
        raise ModelError(f"{where}: must be > 0, got {v}")
    if nonneg and v < 0:
        raise ModelError(f"{where}: must be >= 0, got {v}")
    return float(v)


def _layer(d: dict | None, where: str) -> BarLayer | None:
    if d is None:
        return None
    n = d.get("n", 0)
    if not isinstance(n, int) or n < 0:
        raise ModelError(f"{where}.n: expected a whole number of bars >= 0")
    depth = d.get("depth_mm")
    return BarLayer(
        n=n,
        dia_mm=_num(_req(d, "dia_mm", where), where + ".dia_mm", positive=True),
        depth_mm=None if depth is None else _num(depth, where + ".depth_mm", positive=True),
    )


def parse_model(data: dict) -> Model:
    """Validate a model dictionary and build a :class:`Model`."""
    if not isinstance(data, dict):
        raise ModelError("model must be a JSON object")
    schema = data.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise ModelError(f"unsupported schema '{schema}', expected '{SCHEMA}'")

    # materials
    materials: dict[str, Concrete | Rebar] = {}
    for name, md in _req(data, "materials", "model").items():
        where = f"materials.{name}"
        kind = _req(md, "type", where)
        if kind == "concrete":
            fck = _num(_req(md, "fck", where), where + ".fck", positive=True)
            if not 12 <= fck <= 90:
                raise ModelError(f"{where}.fck: EN 1992-1-1 covers 12 to 90 MPa, got {fck}")
            materials[name] = Concrete(
                name, fck,
                density=_num(md.get("density", 25.0), where + ".density", positive=True),
                dmax_mm=_num(md.get("dmax_mm", 16.0), where + ".dmax_mm", positive=True),
            )
        elif kind == "rebar":
            fyk = _num(md.get("fyk", 500.0), where + ".fyk", positive=True)
            if not 400 <= fyk <= 600:
                raise ModelError(f"{where}.fyk: EN 1992-1-1 covers 400 to 600 MPa, got {fyk}")
            materials[name] = Rebar(
                name, fyk,
                Es=_num(md.get("Es", 200_000.0), where + ".Es", positive=True),
                eps_uk=_num(md.get("eps_uk", 0.05), where + ".eps_uk", positive=True),
            )
        else:
            raise ModelError(f"{where}.type: expected 'concrete' or 'rebar', got {kind!r}")

    # sections
    sections: dict[str, RCRectSection] = {}
    for name, sd in _req(data, "sections", "model").items():
        where = f"sections.{name}"
        if sd.get("type", "rc_rect") != "rc_rect":
            raise ModelError(f"{where}.type: only 'rc_rect' is supported in this version")
        cname, rname = _req(sd, "concrete", where), _req(sd, "rebar", where)
        conc, reb = materials.get(cname), materials.get(rname)
        if not isinstance(conc, Concrete):
            raise ModelError(f"{where}.concrete: '{cname}' is not a concrete material")
        if not isinstance(reb, Rebar):
            raise ModelError(f"{where}.rebar: '{rname}' is not a rebar material")
        links = None
        if sd.get("links"):
            ld = sd["links"]
            links = Links(
                dia_mm=_num(_req(ld, "dia_mm", where + ".links"), where + ".links.dia_mm", positive=True),
                legs=int(_num(ld.get("legs", 2), where + ".links.legs", positive=True)),
                spacing_mm=_num(_req(ld, "spacing_mm", where + ".links"), where + ".links.spacing_mm", positive=True),
                fyk=None if ld.get("fyk") is None else _num(ld["fyk"], where + ".links.fyk", positive=True),
            )
        sec = RCRectSection(
            name=name,
            b_mm=_num(_req(sd, "b_mm", where), where + ".b_mm", positive=True),
            h_mm=_num(_req(sd, "h_mm", where), where + ".h_mm", positive=True),
            concrete=conc, rebar=reb,
            cover_mm=_num(_req(sd, "cover_mm", where), where + ".cover_mm", nonneg=True),
            bottom=_layer(sd.get("bottom"), where + ".bottom"),
            top=_layer(sd.get("top"), where + ".top"),
            links=links,
            extra_layers=tuple(_layer(x, f"{where}.extra_layers[{k}]") for k, x in enumerate(sd.get("extra_layers", []))),
            stiffness_factor=_num(sd.get("stiffness_factor", 1.0), where + ".stiffness_factor", positive=True),
        )
        for area, depth, label in sec.layers():
            if not 0 < depth < sec.h_mm:
                raise ModelError(f"{where}: {label} bars at depth {depth:.1f} mm lie outside the section")
        sections[name] = sec

    # nodes
    nodes: dict[str, Node] = {}
    for k, nd in enumerate(_req(data, "nodes", "model")):
        where = f"nodes[{k}]"
        nid = str(_req(nd, "id", where))
        if nid in nodes:
            raise ModelError(f"{where}: duplicate node id '{nid}'")
        nodes[nid] = Node(nid, _num(_req(nd, "x", where), where + ".x"), _num(_req(nd, "y", where), where + ".y"))
    if len(nodes) < 2:
        raise ModelError("model needs at least two nodes")

    # members
    members: dict[str, Member] = {}
    for k, md in enumerate(_req(data, "members", "model")):
        where = f"members[{k}]"
        mid = str(_req(md, "id", where))
        if mid in members:
            raise ModelError(f"{where}: duplicate member id '{mid}'")
        i, j = str(_req(md, "i", where)), str(_req(md, "j", where))
        for end, nid in (("i", i), ("j", j)):
            if nid not in nodes:
                raise ModelError(f"{where}.{end}: unknown node '{nid}'")
        if i == j:
            raise ModelError(f"{where}: start and end node are the same")
        if math.hypot(nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y) < 1e-6:
            raise ModelError(f"{where}: member has zero length")
        sec = str(_req(md, "section", where))
        if sec not in sections:
            raise ModelError(f"{where}.section: unknown section '{sec}'")
        rel = md.get("releases", {}) or {}
        dd = md.get("design", {}) or {}
        dtype = dd.get("type")
        if dtype is None:  # infer from inclination
            dx, dy = nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y
            dtype = "beam" if abs(dy) <= abs(dx) * math.tan(math.radians(30)) else "column"
        if dtype not in DESIGN_TYPES:
            raise ModelError(f"{where}.design.type: expected one of {DESIGN_TYPES}")
        system = dd.get("deflection_system", "simply_supported")
        if system not in DEFLECTION_SYSTEMS:
            raise ModelError(f"{where}.design.deflection_system: expected one of {DEFLECTION_SYSTEMS}")
        members[mid] = Member(
            id=mid, i=i, j=j, section=sec,
            release_i=bool(rel.get("i", False)), release_j=bool(rel.get("j", False)),
            design=MemberDesign(
                type=dtype, deflection_system=system,
                l0_factor=_num(dd.get("l0_factor", 1.0), where + ".design.l0_factor", positive=True),
                creep_coefficient=_num(dd.get("creep_coefficient", 2.0), where + ".design.creep_coefficient", nonneg=True),
            ),
        )
    if not members:
        raise ModelError("model needs at least one member")

    # supports
    supports: dict[str, Support] = {}
    for k, sd in enumerate(data.get("supports", [])):
        where = f"supports[{k}]"
        nid = str(_req(sd, "node", where))
        if nid not in nodes:
            raise ModelError(f"{where}.node: unknown node '{nid}'")
        if nid in supports:
            raise ModelError(f"{where}: node '{nid}' already has a support")
        supports[nid] = Support(
            node=nid, ux=bool(sd.get("ux", False)), uy=bool(sd.get("uy", False)), rz=bool(sd.get("rz", False)),
            kx=_num(sd.get("kx", 0.0), where + ".kx", nonneg=True),
            ky=_num(sd.get("ky", 0.0), where + ".ky", nonneg=True),
            kr=_num(sd.get("kr", 0.0), where + ".kr", nonneg=True),
        )
    if not supports:
        raise ModelError("model has no supports")

    # load cases
    load_cases: dict[str, LoadCase] = {}
    for k, cd in enumerate(_req(data, "load_cases", "model")):
        where = f"load_cases[{k}]"
        cid = str(_req(cd, "id", where))
        if cid in load_cases:
            raise ModelError(f"{where}: duplicate load case id '{cid}'")
        ctype = _req(cd, "type", where)
        if ctype not in ("permanent", "variable"):
            raise ModelError(f"{where}.type: expected 'permanent' or 'variable'")
        cat = cd.get("category")
        if ctype == "variable":
            if cat not in VARIABLE_CATEGORIES:
                raise ModelError(f"{where}.category: variable actions need one of {VARIABLE_CATEGORIES}")
        nodal, dist, point = [], [], []
        for q, ld in enumerate(cd.get("loads", [])):
            lw = f"{where}.loads[{q}]"
            ltype = _req(ld, "type", lw)
            if ltype == "nodal":
                nid = str(_req(ld, "node", lw))
                if nid not in nodes:
                    raise ModelError(f"{lw}.node: unknown node '{nid}'")
                nodal.append(NodalLoad(nid, _num(ld.get("fx", 0.0), lw + ".fx"),
                                       _num(ld.get("fy", 0.0), lw + ".fy"), _num(ld.get("mz", 0.0), lw + ".mz")))
                continue
            mid = str(_req(ld, "member", lw))
            if mid not in members:
                raise ModelError(f"{lw}.member: unknown member '{mid}'")
            L = math.hypot(nodes[members[mid].j].x - nodes[members[mid].i].x,
                           nodes[members[mid].j].y - nodes[members[mid].i].y)
            direction = ld.get("direction", "global_y")
            if direction not in LOAD_DIRECTIONS:
                raise ModelError(f"{lw}.direction: expected one of {LOAD_DIRECTIONS}")
            if ltype == "udl":
                if "w" in ld:
                    w1 = w2 = _num(ld["w"], lw + ".w")
                else:
                    w1 = _num(_req(ld, "w1", lw), lw + ".w1")
                    w2 = _num(_req(ld, "w2", lw), lw + ".w2")
                a = _num(ld.get("a", 0.0), lw + ".a", nonneg=True)
                b = _num(ld.get("b", L), lw + ".b", nonneg=True)
                if not (0 <= a < b <= L + 1e-9):
                    raise ModelError(f"{lw}: need 0 <= a < b <= member length ({L:.3f} m)")
                dist.append(DistributedLoad(mid, w1, w2, a, min(b, L), direction))
            elif ltype == "point":
                a = _num(_req(ld, "a", lw), lw + ".a", nonneg=True)
                if a > L + 1e-9:
                    raise ModelError(f"{lw}.a: position {a} m is beyond the member length ({L:.3f} m)")
                point.append(PointLoad(mid, _num(_req(ld, "p", lw), lw + ".p"), min(a, L), direction))
            else:
                raise ModelError(f"{lw}.type: expected 'nodal', 'udl' or 'point', got {ltype!r}")
        load_cases[cid] = LoadCase(
            id=cid, type=ctype, name=str(cd.get("name", cid)), category=cat,
            group=cd.get("group"), self_weight=bool(cd.get("self_weight", False)),
            nodal=tuple(nodal), distributed=tuple(dist), point=tuple(point),
        )
    if not load_cases:
        raise ModelError("model has no load cases")
    if not any(lc.type == "permanent" for lc in load_cases.values()):
        raise ModelError("model needs at least one permanent load case (self weight counts)")

    sd = data.get("design", {}) or {}
    rule = sd.get("combination_rule", "6.10")
    if rule not in ("6.10", "6.10ab"):
        raise ModelError("design.combination_rule: expected '6.10' or '6.10ab'")
    code = sd.get("design_code", "EC2")
    if code not in ("EC2", "SIA262"):
        raise ModelError("design.design_code: expected 'EC2' or 'SIA262'")
    settings = DesignSettings(
        design_code=code, combination_rule=rule, params=dict(sd.get("params", {}) or {}),
        stations_per_member=int(_num(sd.get("stations_per_member", 40), "design.stations_per_member", positive=True)),
    )
    return Model(str(data.get("name", "Untitled model")), nodes, members, sections, supports, load_cases, settings, raw=data)


def load_model(path: str | Path) -> Model:
    with open(path, encoding="utf-8") as fh:
        return parse_model(json.load(fh))
