"""3D structural model and its open JSON format ("tragwerk3d/0.1").

Units and conventions
---------------------
* Global geometry in metres, forces in kN, moments in kNm, sections in mm, strengths in MPa.
* Global axes: X and Y horizontal, Z up. Gravity loads are negative Z.
* Member local axes: x from node i to node j.
  - Non-vertical members: y = Z × x (horizontal), z = x × y (points upward).
  - Vertical members: z = global X, y = z × x.
  - ``roll_deg`` then rotates y and z about x.
* Section: width b along local y, depth h along local z. "top" bars lie on the +z face,
  "bottom" bars on the -z face, "side" bars on both ±y faces between them.
* Moments in results: My bends in the x-z plane and is positive with compression on the
  +z (top) face; Mz bends in the x-y plane and is positive with compression on the +y face.
  So My > 0 is sagging for a beam.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .model import (BarLayer, Concrete, DesignSettings, Links, ModelError, Rebar, _layer, _num, _req)

SCHEMA3D = "tragwerk3d/0.1"
DIRECTIONS3D = ("global_x", "global_y", "global_z", "local_x", "local_y", "local_z")
VARIABLE_CATEGORIES = ("A", "B", "C", "D", "E", "F", "G", "H", "snow", "snow_high", "wind", "temperature")
DESIGN_TYPES = ("beam", "column", "none")
DEFLECTION_SYSTEMS = ("simply_supported", "end_span", "interior_span", "cantilever")


# ------------------------------------------------------------------ section
@dataclass(frozen=True)
class RCSection3D:
    name: str
    b_mm: float            # along local y
    h_mm: float            # along local z
    concrete: Concrete
    rebar: Rebar
    cover_mm: float
    top: BarLayer | None = None
    bottom: BarLayer | None = None
    side: BarLayer | None = None      # n bars on EACH ±y face, between top and bottom
    links: Links | None = None
    links_legs_y: int | None = None   # legs resisting Vy (default: same as links.legs)
    stiffness_factor: float = 1.0
    torsion_factor: float = 1.0       # multiplies GJ (cracked members: often 0.1 - 0.3)

    @property
    def area_m2(self) -> float:
        return self.b_mm * self.h_mm * 1e-6

    @property
    def Iy_m4(self) -> float:   # bending in the x-z plane (about local y)
        return self.b_mm * self.h_mm ** 3 / 12.0 * 1e-12

    @property
    def Iz_m4(self) -> float:   # bending in the x-y plane (about local z)
        return self.h_mm * self.b_mm ** 3 / 12.0 * 1e-12

    @property
    def J_m4(self) -> float:
        """St Venant torsion constant of a rectangle (Roark, a >= b)."""
        a, b = max(self.b_mm, self.h_mm), min(self.b_mm, self.h_mm)
        return a * b ** 3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b ** 4 / (12.0 * a ** 4))) * 1e-12

    def bars(self) -> list[tuple[float, float, float, str]]:
        """Every bar as (area mm2, y mm, z mm, group label), origin at the section centre."""
        wl = self.links.dia_mm if self.links else 0.0
        out = []

        def ylim(d):
            return self.b_mm / 2 - self.cover_mm - wl - d / 2

        def zlim(d):
            return self.h_mm / 2 - self.cover_mm - wl - d / 2

        def row(lay: BarLayer, z: float, label: str):
            a = math.pi * lay.dia_mm ** 2 / 4
            yl = ylim(lay.dia_mm)
            for k in range(lay.n):
                y = 0.0 if lay.n == 1 else -yl + 2 * yl * k / (lay.n - 1)
                out.append((a, y, z, label))

        zt = zb = None
        if self.top and self.top.n > 0:
            zt = zlim(self.top.dia_mm)
            row(self.top, zt, "top")
        if self.bottom and self.bottom.n > 0:
            zb = -zlim(self.bottom.dia_mm)
            row(self.bottom, zb, "bottom")
        if self.side and self.side.n > 0:
            a = math.pi * self.side.dia_mm ** 2 / 4
            z1 = zt if zt is not None else zlim(self.side.dia_mm)
            z0 = zb if zb is not None else -zlim(self.side.dia_mm)
            ys = ylim(self.side.dia_mm)
            for k in range(1, self.side.n + 1):
                z = z0 + (z1 - z0) * k / (self.side.n + 1)
                out.append((a, ys, z, "side"))
                out.append((a, -ys, z, "side"))
        return out

    def view(self, axis: str) -> "SectionView":
        """Uniaxial view for bending in the x-z plane (axis 'y') or the x-y plane (axis 'z')."""
        groups: dict[float, float] = {}
        for a, y, z, _ in self.bars():
            if axis == "y":
                depth = self.h_mm / 2 - z
            else:
                depth = self.b_mm / 2 - y
            key = round(depth, 6)
            groups[key] = groups.get(key, 0.0) + a
        depths = sorted(groups)
        layers = []
        for k, d in enumerate(depths):
            lab = f"row {k + 1}"
            layers.append((groups[d], d, lab))
        if axis == "y":
            return SectionView(self.name, self.b_mm, self.h_mm, self.concrete, self.rebar, layers, self.links,
                               self.links.legs if self.links else 0, self.cover_mm)
        legs = (self.links_legs_y or self.links.legs) if self.links else 0
        links = replace(self.links, legs=legs) if self.links else None
        return SectionView(self.name, self.h_mm, self.b_mm, self.concrete, self.rebar, layers, links, legs, self.cover_mm)

    def as_2d(self):
        """The section as a 2D RCRectSection for bending in the x-z plane (side bars become extra layers)."""
        from .model import RCRectSection
        extra = []
        for a, y, z, lab in self.bars():
            if lab == "side" and y > 0:
                extra.append(BarLayer(n=2, dia_mm=self.side.dia_mm, depth_mm=self.h_mm / 2 - z))
        return RCRectSection(self.name, self.b_mm, self.h_mm, self.concrete, self.rebar, self.cover_mm,
                             bottom=self.bottom, top=self.top, links=self.links, extra_layers=tuple(extra),
                             stiffness_factor=self.stiffness_factor)


@dataclass(frozen=True)
class SectionView:
    """A section seen in one bending plane: width b, depth h, bar layers by depth from the compressed face."""
    name: str
    b_mm: float
    h_mm: float
    concrete: Concrete
    rebar: Rebar
    _layers: list
    links: Links | None
    legs: int
    cover_mm: float

    def layers(self):
        return list(self._layers)

    @property
    def link_area_mm2(self) -> float:
        return self.legs * math.pi * self.links.dia_mm ** 2 / 4 if self.links else 0.0


# ------------------------------------------------------------------ topology
@dataclass(frozen=True)
class Node3D:
    id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Support3D:
    node: str
    fix: tuple[bool, bool, bool, bool, bool, bool]          # ux uy uz rx ry rz
    springs: tuple[float, float, float, float, float, float]  # kN/m, kNm/rad


@dataclass(frozen=True)
class Releases:
    t: bool = False
    my: bool = False
    mz: bool = False


@dataclass(frozen=True)
class MemberDesign3D:
    type: str = "beam"
    deflection_system: str = "simply_supported"
    l0_factor_y: float = 1.0   # buckling in the x-z plane
    l0_factor_z: float = 1.0   # buckling in the x-y plane
    creep_coefficient: float = 2.0


@dataclass(frozen=True)
class Member3D:
    id: str
    i: str
    j: str
    section: str
    roll_deg: float = 0.0
    rel_i: Releases = field(default_factory=Releases)
    rel_j: Releases = field(default_factory=Releases)
    design: MemberDesign3D = field(default_factory=MemberDesign3D)


@dataclass(frozen=True)
class NodalLoad3D:
    node: str
    f: tuple[float, float, float, float, float, float]   # fx fy fz mx my mz


@dataclass(frozen=True)
class DistLoad3D:
    member: str
    w1: float
    w2: float
    a: float
    b: float
    direction: str = "global_z"


@dataclass(frozen=True)
class PointLoad3D:
    member: str
    p: float
    a: float
    direction: str = "global_z"


@dataclass(frozen=True)
class LoadCase3D:
    id: str
    type: str
    name: str = ""
    category: str | None = None
    group: str | None = None
    self_weight: bool = False
    nodal: tuple[NodalLoad3D, ...] = ()
    distributed: tuple[DistLoad3D, ...] = ()
    point: tuple[PointLoad3D, ...] = ()


@dataclass
class Model3D:
    name: str
    nodes: dict[str, Node3D]
    members: dict[str, Member3D]
    sections: dict[str, RCSection3D]
    supports: dict[str, Support3D]
    load_cases: dict[str, LoadCase3D]
    settings: DesignSettings = field(default_factory=DesignSettings)
    raw: dict = field(default_factory=dict, repr=False)
    is3d: bool = True

    def axes(self, mid: str) -> tuple[float, np.ndarray]:
        """(length, 3x3 matrix whose rows are the local x, y, z unit vectors in global axes)."""
        m = self.members[mid]
        a, b = self.nodes[m.i], self.nodes[m.j]
        d = np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)
        L = float(np.linalg.norm(d))
        return L, local_axes(d / L, m.roll_deg)


def local_axes(ex: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    Z = np.array([0.0, 0.0, 1.0])
    c = np.cross(Z, ex)
    if np.linalg.norm(c) < 1e-6:            # vertical member
        ez = np.array([1.0, 0.0, 0.0])
        ey = np.cross(ez, ex)
    else:
        ey = c / np.linalg.norm(c)
        ez = np.cross(ex, ey)
    if roll_deg:
        r = math.radians(roll_deg)
        ey, ez = math.cos(r) * ey + math.sin(r) * ez, -math.sin(r) * ey + math.cos(r) * ez
    return np.vstack([ex, ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)])


# ================================================================== parsing
def _releases(v, where) -> Releases:
    if v is None or v is False:
        return Releases()
    if v is True:
        return Releases(my=True, mz=True)
    if not isinstance(v, dict):
        raise ModelError(f"{where}: expected true/false or an object with 't', 'my', 'mz'")
    return Releases(t=bool(v.get("t", False)), my=bool(v.get("my", False)), mz=bool(v.get("mz", False)))


def parse_model3d(data: dict) -> Model3D:
    if not isinstance(data, dict):
        raise ModelError("model must be a JSON object")
    if data.get("schema") != SCHEMA3D:
        raise ModelError(f"expected schema '{SCHEMA3D}'")
    materials: dict = {}
    for name, md in _req(data, "materials", "model").items():
        where = f"materials.{name}"
        kind = _req(md, "type", where)
        if kind == "concrete":
            fck = _num(_req(md, "fck", where), where + ".fck", positive=True)
            if not 12 <= fck <= 90:
                raise ModelError(f"{where}.fck: EN 1992-1-1 covers 12 to 90 MPa, got {fck}")
            materials[name] = Concrete(name, fck, density=_num(md.get("density", 25.0), where + ".density", positive=True),
                                       dmax_mm=_num(md.get("dmax_mm", 16.0), where + ".dmax_mm", positive=True))
        elif kind == "rebar":
            fyk = _num(md.get("fyk", 500.0), where + ".fyk", positive=True)
            if not 400 <= fyk <= 600:
                raise ModelError(f"{where}.fyk: EN 1992-1-1 covers 400 to 600 MPa, got {fyk}")
            materials[name] = Rebar(name, fyk, Es=_num(md.get("Es", 200_000.0), where + ".Es", positive=True),
                                    eps_uk=_num(md.get("eps_uk", 0.05), where + ".eps_uk", positive=True))
        else:
            raise ModelError(f"{where}.type: expected 'concrete' or 'rebar', got {kind!r}")

    sections: dict[str, RCSection3D] = {}
    for name, sd in _req(data, "sections", "model").items():
        where = f"sections.{name}"
        if sd.get("type", "rc_rect") != "rc_rect":
            raise ModelError(f"{where}.type: only 'rc_rect' is supported in this version")
        conc, reb = materials.get(sd.get("concrete")), materials.get(sd.get("rebar"))
        if not isinstance(conc, Concrete):
            raise ModelError(f"{where}.concrete: '{sd.get('concrete')}' is not a concrete material")
        if not isinstance(reb, Rebar):
            raise ModelError(f"{where}.rebar: '{sd.get('rebar')}' is not a rebar material")
        links, legs_y = None, None
        if sd.get("links"):
            ld = sd["links"]
            links = Links(dia_mm=_num(_req(ld, "dia_mm", where + ".links"), where + ".links.dia_mm", positive=True),
                          legs=int(_num(ld.get("legs", 2), where + ".links.legs", positive=True)),
                          spacing_mm=_num(_req(ld, "spacing_mm", where + ".links"), where + ".links.spacing_mm", positive=True),
                          fyk=None if ld.get("fyk") is None else _num(ld["fyk"], where + ".links.fyk", positive=True))
            if ld.get("legs_y") is not None:
                legs_y = int(_num(ld["legs_y"], where + ".links.legs_y", positive=True))
        sec = RCSection3D(
            name=name, b_mm=_num(_req(sd, "b_mm", where), where + ".b_mm", positive=True),
            h_mm=_num(_req(sd, "h_mm", where), where + ".h_mm", positive=True), concrete=conc, rebar=reb,
            cover_mm=_num(_req(sd, "cover_mm", where), where + ".cover_mm", nonneg=True),
            top=_layer(sd.get("top"), where + ".top"), bottom=_layer(sd.get("bottom"), where + ".bottom"),
            side=_layer(sd.get("side"), where + ".side"), links=links, links_legs_y=legs_y,
            stiffness_factor=_num(sd.get("stiffness_factor", 1.0), where + ".stiffness_factor", positive=True),
            torsion_factor=_num(sd.get("torsion_factor", 1.0), where + ".torsion_factor", positive=True),
        )
        wl = links.dia_mm if links else 0.0
        for lab, lay in (("top", sec.top), ("bottom", sec.bottom), ("side", sec.side)):
            if lay and lay.n > 0:
                zl = sec.h_mm / 2 - sec.cover_mm - wl - lay.dia_mm / 2
                yl = sec.b_mm / 2 - sec.cover_mm - wl - lay.dia_mm / 2
                if zl <= 0 or yl < 0 or (yl == 0 and (lay.n > 1 or lab == "side")):
                    raise ModelError(f"{where}: {lab} bars lie outside the section; check cover and bar diameters")
        sections[name] = sec

    nodes: dict[str, Node3D] = {}
    for k, nd in enumerate(_req(data, "nodes", "model")):
        where = f"nodes[{k}]"
        nid = str(_req(nd, "id", where))
        if nid in nodes:
            raise ModelError(f"{where}: duplicate node id '{nid}'")
        nodes[nid] = Node3D(nid, _num(_req(nd, "x", where), where + ".x"), _num(_req(nd, "y", where), where + ".y"),
                            _num(nd.get("z", 0.0), where + ".z"))
    if len(nodes) < 2:
        raise ModelError("model needs at least two nodes")

    members: dict[str, Member3D] = {}
    for k, md in enumerate(_req(data, "members", "model")):
        where = f"members[{k}]"
        mid = str(_req(md, "id", where))
        if mid in members:
            raise ModelError(f"{where}: duplicate member id '{mid}'")
        i, j = str(_req(md, "i", where)), str(_req(md, "j", where))
        for end, nid in (("i", i), ("j", j)):
            if nid not in nodes:
                raise ModelError(f"{where}.{end}: unknown node '{nid}'")
        a, b = nodes[i], nodes[j]
        dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
        if math.sqrt(dx * dx + dy * dy + dz * dz) < 1e-6:
            raise ModelError(f"{where}: member has zero length")
        sec = str(_req(md, "section", where))
        if sec not in sections:
            raise ModelError(f"{where}.section: unknown section '{sec}'")
        rel = md.get("releases", {}) or {}
        dd = md.get("design", {}) or {}
        dtype = dd.get("type")
        if dtype is None:
            dtype = "column" if abs(dz) > math.hypot(dx, dy) * math.tan(math.radians(60)) else "beam"
        if dtype not in DESIGN_TYPES:
            raise ModelError(f"{where}.design.type: expected one of {DESIGN_TYPES}")
        system = dd.get("deflection_system", "simply_supported")
        if system not in DEFLECTION_SYSTEMS:
            raise ModelError(f"{where}.design.deflection_system: expected one of {DEFLECTION_SYSTEMS}")
        l0 = dd.get("l0_factor", 1.0)
        members[mid] = Member3D(
            id=mid, i=i, j=j, section=sec, roll_deg=_num(md.get("roll_deg", 0.0), where + ".roll_deg"),
            rel_i=_releases(rel.get("i"), where + ".releases.i"), rel_j=_releases(rel.get("j"), where + ".releases.j"),
            design=MemberDesign3D(
                type=dtype, deflection_system=system,
                l0_factor_y=_num(dd.get("l0_factor_y", l0), where + ".design.l0_factor_y", positive=True),
                l0_factor_z=_num(dd.get("l0_factor_z", l0), where + ".design.l0_factor_z", positive=True),
                creep_coefficient=_num(dd.get("creep_coefficient", 2.0), where + ".design.creep_coefficient", nonneg=True)),
        )
    if not members:
        raise ModelError("model needs at least one member")

    supports: dict[str, Support3D] = {}
    for k, sd in enumerate(data.get("supports", [])):
        where = f"supports[{k}]"
        nid = str(_req(sd, "node", where))
        if nid not in nodes:
            raise ModelError(f"{where}.node: unknown node '{nid}'")
        if nid in supports:
            raise ModelError(f"{where}: node '{nid}' already has a support")
        fix = tuple(bool(sd.get(key, False)) for key in ("ux", "uy", "uz", "rx", "ry", "rz"))
        spr = tuple(_num(sd.get(key, 0.0), f"{where}.{key}", nonneg=True) for key in ("kx", "ky", "kz", "krx", "kry", "krz"))
        supports[nid] = Support3D(nid, fix, spr)
    if not supports:
        raise ModelError("model has no supports")

    load_cases: dict[str, LoadCase3D] = {}
    for k, cd in enumerate(_req(data, "load_cases", "model")):
        where = f"load_cases[{k}]"
        cid = str(_req(cd, "id", where))
        if cid in load_cases:
            raise ModelError(f"{where}: duplicate load case id '{cid}'")
        ctype = _req(cd, "type", where)
        if ctype not in ("permanent", "variable"):
            raise ModelError(f"{where}.type: expected 'permanent' or 'variable'")
        cat = cd.get("category")
        if ctype == "variable" and cat not in VARIABLE_CATEGORIES:
            raise ModelError(f"{where}.category: variable actions need one of {VARIABLE_CATEGORIES}")
        nodal, dist, point = [], [], []
        for q, ld in enumerate(cd.get("loads", [])):
            lw = f"{where}.loads[{q}]"
            lt = _req(ld, "type", lw)
            if lt == "nodal":
                nid = str(_req(ld, "node", lw))
                if nid not in nodes:
                    raise ModelError(f"{lw}.node: unknown node '{nid}'")
                nodal.append(NodalLoad3D(nid, tuple(_num(ld.get(key, 0.0), f"{lw}.{key}") for key in ("fx", "fy", "fz", "mx", "my", "mz"))))
                continue
            mid = str(_req(ld, "member", lw))
            if mid not in members:
                raise ModelError(f"{lw}.member: unknown member '{mid}'")
            m = members[mid]
            L = math.dist((nodes[m.i].x, nodes[m.i].y, nodes[m.i].z), (nodes[m.j].x, nodes[m.j].y, nodes[m.j].z))
            direction = ld.get("direction", "global_z")
            if direction not in DIRECTIONS3D:
                raise ModelError(f"{lw}.direction: expected one of {DIRECTIONS3D}")
            if lt == "udl":
                if "w" in ld:
                    w1 = w2 = _num(ld["w"], lw + ".w")
                else:
                    w1, w2 = _num(_req(ld, "w1", lw), lw + ".w1"), _num(_req(ld, "w2", lw), lw + ".w2")
                a = _num(ld.get("a", 0.0), lw + ".a", nonneg=True)
                b = _num(ld.get("b", L), lw + ".b", nonneg=True)
                if not (0 <= a < b <= L + 1e-9):
                    raise ModelError(f"{lw}: need 0 <= a < b <= member length ({L:.3f} m)")
                dist.append(DistLoad3D(mid, w1, w2, a, min(b, L), direction))
            elif lt == "point":
                a = _num(_req(ld, "a", lw), lw + ".a", nonneg=True)
                if a > L + 1e-9:
                    raise ModelError(f"{lw}.a: position {a} m is beyond the member length ({L:.3f} m)")
                point.append(PointLoad3D(mid, _num(_req(ld, "p", lw), lw + ".p"), min(a, L), direction))
            else:
                raise ModelError(f"{lw}.type: expected 'nodal', 'udl' or 'point', got {lt!r}")
        load_cases[cid] = LoadCase3D(cid, ctype, str(cd.get("name", cid)), cat, cd.get("group"), bool(cd.get("self_weight", False)),
                                     tuple(nodal), tuple(dist), tuple(point))
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
    settings = DesignSettings(design_code=code, combination_rule=rule, params=dict(sd.get("params", {}) or {}),
                              stations_per_member=int(_num(sd.get("stations_per_member", 20), "design.stations_per_member", positive=True)))
    return Model3D(str(data.get("name", "Untitled model")), nodes, members, sections, supports, load_cases, settings, raw=data)


def load_any(path_or_data):
    """Parse a 2D or 3D model from a path or a dict."""
    from .model import parse_model
    data = path_or_data
    if not isinstance(data, dict):
        with open(path_or_data, encoding="utf-8") as fh:
            data = json.load(fh)
    return parse_model3d(data) if data.get("schema") == SCHEMA3D else parse_model(data)
