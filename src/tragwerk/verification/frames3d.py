"""Random 3D frames for property tests and cross-checks against other solvers.

Sections are square so that member orientation conventions (which differ
between programs) cannot influence the comparison; releases are full pins.
"""
from __future__ import annotations

import random

MATERIALS = {"C30/37": {"type": "concrete", "fck": 30}, "B500B": {"type": "rebar", "fyk": 500}}
SQUARE = {"type": "rc_rect", "b_mm": 400, "h_mm": 400, "concrete": "C30/37", "rebar": "B500B", "cover_mm": 35,
          "top": {"n": 3, "dia_mm": 20}, "bottom": {"n": 3, "dia_mm": 20}, "side": {"n": 1, "dia_mm": 20},
          "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200}}
SQUARE_BEAM = dict(SQUARE, b_mm=350, h_mm=350)


def random_frame3d(seed: int, releases: bool = True, partial_loads: bool = True) -> dict:
    rnd = random.Random(seed)
    nx, ny, nz = rnd.randint(1, 3), rnd.randint(1, 2), rnd.randint(1, 3)
    xs = [0.0]
    for _ in range(nx):
        xs.append(xs[-1] + rnd.choice([4.0, 5.5, 6.0]))
    ys = [0.0]
    for _ in range(ny):
        ys.append(ys[-1] + rnd.choice([4.5, 5.0, 7.0]))
    zs = [0.0]
    for _ in range(nz):
        zs.append(zs[-1] + rnd.choice([3.0, 3.5, 4.0]))
    nid = lambda i, j, k: f"N{i}{j}{k}"  # noqa: E731
    nodes, members, supports, loads = [], [], [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                nodes.append({"id": nid(i, j, k), "x": x, "y": y, "z": z})
            fixed = rnd.random() < 0.7
            supports.append({"node": nid(i, j, 0), "ux": True, "uy": True, "uz": True,
                             "rx": fixed, "ry": fixed, "rz": True if not fixed else True})
            for k in range(nz):
                members.append({"id": f"C{i}{j}{k}", "i": nid(i, j, k), "j": nid(i, j, k + 1), "section": "COL"})
    for k in range(1, nz + 1):
        for j in range(ny + 1):
            for i in range(nx):
                m = {"id": f"BX{i}{j}{k}", "i": nid(i, j, k), "j": nid(i + 1, j, k), "section": "BEAM"}
                if releases and rnd.random() < 0.2:
                    m["releases"] = {rnd.choice(["i", "j"]): True}
                members.append(m)
                L = xs[i + 1] - xs[i]
                loads.append({"type": "udl", "member": m["id"], "w": -rnd.uniform(5, 30)})
                if rnd.random() < 0.4:
                    loads.append({"type": "point", "member": m["id"], "p": -rnd.uniform(10, 60), "a": round(rnd.uniform(0.2, 0.8) * L, 3)})
                if partial_loads and rnd.random() < 0.3:
                    loads.append({"type": "udl", "member": m["id"], "w1": -rnd.uniform(0, 15), "w2": -rnd.uniform(0, 15),
                                  "a": round(0.1 * L, 3), "b": round(0.7 * L, 3)})
        for i in range(nx + 1):
            for j in range(ny):
                m = {"id": f"BY{i}{j}{k}", "i": nid(i, j, k), "j": nid(i, j + 1, k), "section": "BEAM"}
                members.append(m)
                loads.append({"type": "udl", "member": m["id"], "w": -rnd.uniform(5, 25)})
                if rnd.random() < 0.3:
                    loads.append({"type": "udl", "member": m["id"], "w": rnd.uniform(1, 4), "direction": "global_x"})
        loads.append({"type": "nodal", "node": nid(0, 0, k), "fx": rnd.uniform(5, 30), "fy": rnd.uniform(-20, 20)})
        loads.append({"type": "nodal", "node": nid(nx, ny, k), "fy": rnd.uniform(5, 30), "mz": rnd.uniform(-10, 10)})
    return {
        "schema": "tragwerk3d/0.1", "name": f"random 3D frame {seed}",
        "materials": MATERIALS, "sections": {"COL": SQUARE, "BEAM": SQUARE_BEAM},
        "nodes": nodes, "members": members, "supports": supports,
        "load_cases": [{"id": "G", "type": "permanent", "loads": loads}],
    }
