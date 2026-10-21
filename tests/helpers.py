"""Small builders for test models."""
from __future__ import annotations

import copy

from tragwerk.model import parse_model

MATERIALS = {
    "C30/37": {"type": "concrete", "fck": 30},
    "B500B": {"type": "rebar", "fyk": 500},
}
SECTION = {
    "type": "rc_rect", "b_mm": 300, "h_mm": 500, "concrete": "C30/37", "rebar": "B500B",
    "cover_mm": 30, "bottom": {"n": 4, "dia_mm": 20}, "top": {"n": 2, "dia_mm": 12},
    "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200},
}


def model_dict(nodes, members, supports, load_cases, sections=None, **design):
    """nodes: {id: (x, y)}; members: {id: (i, j)} or {id: (i, j, extra_dict)}."""
    secs = sections or {"S": SECTION}
    mlist = []
    for mid, spec in members.items():
        d = {"id": mid, "i": spec[0], "j": spec[1], "section": "S"}
        if len(spec) > 2:
            d.update(spec[2])
        mlist.append(d)
    return {
        "schema": "tragwerk/0.1",
        "name": "test",
        "design": design,
        "materials": copy.deepcopy(MATERIALS),
        "sections": copy.deepcopy(secs),
        "nodes": [{"id": k, "x": v[0], "y": v[1]} for k, v in nodes.items()],
        "members": mlist,
        "supports": supports,
        "load_cases": load_cases,
    }


def build(*args, **kw):
    return parse_model(model_dict(*args, **kw))


def single_case(loads, self_weight=False):
    return [{"id": "G", "type": "permanent", "self_weight": self_weight, "loads": loads}]
