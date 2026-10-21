"""Random 2D frames for property tests and cross-checks against other solvers."""
from __future__ import annotations

import random

MATERIALS = {
    "C30/37": {"type": "concrete", "fck": 30},
    "B500B": {"type": "rebar", "fyk": 500},
}
SECTION = {
    "type": "rc_rect", "b_mm": 300, "h_mm": 500, "concrete": "C30/37", "rebar": "B500B",
    "cover_mm": 30, "bottom": {"n": 4, "dia_mm": 20}, "top": {"n": 2, "dia_mm": 12},
    "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200},
}

SECTIONS = {
    "S": SECTION,
    "C": dict(SECTION, b_mm=400, h_mm=400),
}


def random_frame(seed: int, releases: bool = True, partial_loads: bool = True) -> dict:
    rnd = random.Random(seed)
    bays = rnd.randint(1, 3)
    stories = rnd.randint(1, 3)
    xs = [0.0]
    for _ in range(bays):
        xs.append(xs[-1] + rnd.choice([4.0, 5.0, 6.5, 7.2]))
    ys = [0.0]
    for _ in range(stories):
        ys.append(ys[-1] + rnd.choice([3.0, 3.5, 4.2]))
    nodes, members, supports, loads = [], [], [], []
    nid = lambda i, j: f"N{i}_{j}"  # noqa: E731
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            nodes.append({"id": nid(i, j), "x": x, "y": y})
    for i in range(len(xs)):
        base = rnd.choice(["fix", "fix", "pin"])
        supports.append({"node": nid(i, 0), "ux": True, "uy": True, "rz": base == "fix"})
        for j in range(stories):
            members.append({"id": f"C{i}_{j}", "i": nid(i, j), "j": nid(i, j + 1), "section": "C",
                            "design": {"type": "column"}})
    for j in range(1, len(ys)):
        for i in range(bays):
            m = {"id": f"B{i}_{j}", "i": nid(i, j), "j": nid(i + 1, j), "section": "S", "design": {"type": "beam"}}
            if releases and rnd.random() < 0.25:
                m["releases"] = {rnd.choice(["i", "j"]): True}
            members.append(m)
            L = xs[i + 1] - xs[i]
            loads.append({"type": "udl", "member": m["id"], "w": -rnd.uniform(5, 30)})
            if rnd.random() < 0.5:
                loads.append({"type": "point", "member": m["id"], "p": -rnd.uniform(10, 80), "a": round(rnd.uniform(0.2, 0.8) * L, 3)})
            if partial_loads and rnd.random() < 0.4:
                a = round(rnd.uniform(0.0, 0.4) * L, 3)
                b = round(rnd.uniform(0.6, 1.0) * L, 3)
                loads.append({"type": "udl", "member": m["id"], "w1": -rnd.uniform(0, 20), "w2": -rnd.uniform(0, 20), "a": a, "b": b})
        loads.append({"type": "nodal", "node": nid(0, j), "fx": rnd.uniform(5, 40)})
    if rnd.random() < 0.5:
        loads.append({"type": "udl", "member": "C0_0", "w": rnd.uniform(1, 6), "direction": "global_x"})
    return {
        "schema": "tragwerk/0.1", "name": f"random frame {seed}",
        "materials": MATERIALS, "sections": SECTIONS,
        "nodes": nodes, "members": members, "supports": supports,
        "load_cases": [{"id": "G", "type": "permanent", "loads": loads}],
    }
