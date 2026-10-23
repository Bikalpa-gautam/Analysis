"""Plane continuum finite elements (tragwerk.fe2d) and the browser port (web/fe2d.js)."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tragwerk.fe2d import (ELEMENT_TYPES, NNODES, SCHEMA_FE, FEError, constitutive, element_details, element_mass,
                           element_stiffness, gauss, parse_fe, run_fe, shape, solve_fe, timoshenko_cantilever, to_dict_fe)

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _wall(et, **kw):
    d = {"schema": SCHEMA_FE, "name": "Wall", "element": et, "analysis": "plane_stress",
         "material": {"E_MPa": 30000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25},
         "geometry": {"type": "rectangle", "Lx": 6.0, "Ly": 3.0, "h": 0.5, "openings": [{"x0": 2.0, "y0": 0.0, "x1": 3.0, "y1": 2.2}]},
         "supports": [{"edge": "bottom", "ux": True, "uy": True}], "loads": [{"edge": "top", "qx": 20, "qy": -80}, {"at": [6, 3], "fx": -15, "fy": 0}],
         "self_weight": True}
    d.update(kw)
    return d


@pytest.mark.parametrize("et", ELEMENT_TYPES)
def test_shape_functions(et):
    """Partition of unity, Kronecker property at the nodes, derivatives against finite differences."""
    nodes = {"T3": [(0, 0), (1, 0), (0, 1)], "T6": [(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)],
             "Q4": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
             "Q8": [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0)]}
    nodes["Q8R"] = nodes["Q8"]
    nodes = nodes[et]
    for i, (a, b) in enumerate(nodes):
        N, _ = shape(et, a, b)
        assert np.allclose(N, np.eye(len(nodes))[i], atol=1e-14)
    for xi, eta in [(0.2, 0.3), (0.1, 0.05), (-0.4, 0.7) if et[0] == "Q" else (0.3, 0.6)]:
        N, dN = shape(et, xi, eta)
        assert math.isclose(N.sum(), 1.0, rel_tol=1e-14)
        h = 1e-6
        fd = np.array([(shape(et, xi + h, eta)[0] - shape(et, xi - h, eta)[0]) / (2 * h),
                       (shape(et, xi, eta + h)[0] - shape(et, xi, eta - h)[0]) / (2 * h)])
        assert np.allclose(dN, fd, atol=1e-8)


@pytest.mark.parametrize("et", ELEMENT_TYPES)
def test_element_matrices(et):
    """k_e symmetric, positive semi-definite with exactly three rigid-body modes; m_e sums to the mass."""
    xy = {"T3": [[0, 0], [2, 0.3], [0.4, 1.5]], "Q4": [[0, 0], [2, 0.2], [2.2, 1.4], [-0.1, 1.2]]}
    xy["T6"] = xy["T3"] + [list((np.array(xy["T3"][a]) + xy["T3"][b]) / 2) for a, b in ((0, 1), (1, 2), (2, 0))]
    xy["Q8"] = xy["Q4"] + [list((np.array(xy["Q4"][a]) + xy["Q4"][b]) / 2) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))]
    xy["Q8R"] = xy["Q8"]
    X = np.array(xy[et], dtype=float)
    k, _ = element_stiffness(et, X, constitutive(30e6, 0.2, "plane_stress"), 0.3)
    assert np.allclose(k, k.T, rtol=1e-12, atol=1e-6)
    lam = np.linalg.eigvalsh(k)
    # Q8 with reduced 2×2 integration has one spurious zero-energy mode on its own (not communicable in a mesh)
    assert np.sum(np.abs(lam) < 1e-9 * lam.max()) == (4 if et == "Q8R" else 3) and lam.min() > -1e-9 * lam.max()
    for mode in (np.tile([1.0, 0.0], len(X)), np.tile([0.0, 1.0], len(X)), np.column_stack([-X[:, 1], X[:, 0]]).ravel()):
        assert np.max(np.abs(k @ mode)) < 1e-8 * lam.max()
    m = element_mass(et, X, 2.5, 0.3)
    area = sum(w * abs(np.linalg.det(shape(et, a, b)[1] @ X)) for a, b, w in gauss(et, "mass"))
    assert math.isclose(m[0::2, 0::2].sum(), 2.5 * 0.3 * area, rel_tol=1e-12)


def test_cst_closed_form():
    """CST: k = t A Bᵀ C B with B from the nodal coordinates (b_i = y_j − y_k, c_i = x_k − x_j)."""
    X = np.array([[0.0, 0.0], [3.0, 0.5], [1.0, 2.0]])
    C = constitutive(30e6, 0.25, "plane_strain")
    A = 0.5 * ((X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1]) - (X[2, 0] - X[0, 0]) * (X[1, 1] - X[0, 1]))
    B = np.zeros((3, 6))
    for i in range(3):
        j, k_ = (i + 1) % 3, (i + 2) % 3
        b, c = X[j, 1] - X[k_, 1], X[k_, 0] - X[j, 0]
        B[:, 2 * i:2 * i + 2] = np.array([[b, 0], [0, c], [c, b]]) / (2 * A)
    k, _ = element_stiffness("T3", X, C, 0.2)
    assert np.allclose(k, 0.2 * A * B.T @ C @ B, rtol=1e-12)


@pytest.mark.parametrize("et", ELEMENT_TYPES)
def test_equilibrium_wall_with_opening(et):
    r = run_fe(_wall(et))
    applied = np.array([r.f[0::2].sum(), r.f[1::2].sum()])
    W = 25 * 0.25 * (6 * 3 - 1 * 2.2)
    assert math.isclose(applied[1], -80 * 6 - W, rel_tol=1e-10)
    assert np.allclose(r.reactions.sum(axis=0) + applied, 0, atol=1e-8)
    out = to_dict_fe(r)
    json.dumps(out, allow_nan=False)
    assert out["summary"]["max_mises_MPa"] > 0


@pytest.mark.parametrize("et,tol", [("Q8", 5e-5), ("T6", 2e-4)])
def test_timoshenko_cantilever(et, tol):
    P, L, D, E, nu, t = 100.0, 8.0, 1.0, 3e7, 0.25, 0.3
    d = {"schema": SCHEMA_FE, "element": et, "material": {"E_MPa": E / 1000, "nu": nu, "thickness_m": t},
         "geometry": {"type": "rectangle", "Lx": L, "Ly": D, "h": 0.25}, "supports": [{"edge": "left", "ux": True, "uy": True}],
         "loads": [{"edge": "right", "profile": "parabolic", "P": P, "dir": "y"}]}
    m = parse_fe(d)
    m.nodes[:, 1] -= D / 2
    pres = {}
    for i, (x, y) in enumerate(m.nodes):
        if abs(x) < 1e-12:
            ux, uy, *_ = timoshenko_cantilever(x, y, P, L, D, E, nu, t)
            pres[2 * i], pres[2 * i + 1] = ux, uy
    m.fixed = {}
    r = solve_fe(m, pres)
    for i, (x, y) in enumerate(m.nodes):
        ux, uy, *_ = timoshenko_cantilever(x, y, P, L, D, E, nu, t)
        assert abs(r.u[i, 1] - uy) < tol * 0.023 and abs(r.u[i, 0] - ux) < tol * 0.023


def test_validation():
    for bad in ({"element": "Q9"}, {"material": {"nu": 0.5}}, {"supports": []},
                {"geometry": {"type": "rectangle", "Lx": 1, "Ly": 1, "h": 0.001}},
                {"geometry": {"type": "rectangle", "Lx": 1, "Ly": 1, "openings": [{"x0": -1, "y0": -1, "x1": 2, "y1": 2}]}},
                {"loads": [{"edge": "middle", "qx": 1}]}):
        with pytest.raises(FEError):
            run_fe(_wall("Q4", **bad))
    with pytest.raises(FEError):                       # only a roller: a mechanism
        run_fe(_wall("Q4", supports=[{"at": [0, 0], "uy": True}]))


def test_element_details():
    d = _wall("Q8")
    r = run_fe(d)
    det = element_details(r.model, 3, r.u)
    assert len(det["gauss"]) == 9 and len(det["k"]) == 16
    assert np.allclose(np.array(det["k"]) @ np.array(det["u_e"]), det["f_e"])


@pytest.mark.skipif(NODE is None, reason="node.js not available")
def test_js_parity(tmp_path):
    models = [_wall(et) for et in ELEMENT_TYPES]
    models += [_wall(et, analysis="plane_strain", geometry={"type": "rectangle", "Lx": 4, "Ly": 4, "h": 0.4}) for et in ("T6", "Q4")]
    models += [{"schema": SCHEMA_FE, "element": et, "material": {"E_MPa": 200000, "nu": 0.3, "thickness_m": 0.02},
                "geometry": {"type": "plate_hole", "W": 5.0, "a": 0.5, "n_theta": 4, "n_r": 5},
                "supports": [{"edge": "left", "ux": True}, {"edge": "bottom", "uy": True}],
                "loads": [{"edge": "right", "qx": 100.0}, {"edge": "top", "qy": -30.0}]} for et in ELEMENT_TYPES]
    models.append({"schema": SCHEMA_FE, "element": "Q8", "material": {"E_MPa": 30000, "nu": 0.25, "thickness_m": 0.3},
                   "geometry": {"type": "rectangle", "Lx": 8, "Ly": 1, "h": 0.25}, "supports": [{"edge": "left", "ux": True, "uy": True}],
                   "loads": [{"edge": "right", "profile": "parabolic", "P": -100, "dir": "y"}]})
    models.append(_wall("Q8", geometry={"type": "rectangle", "Lx": 6, "Ly": 3, "h": 0.3}, self_weight=True,
                        supports=[{"edge": "bottom", "from": 0, "to": 0.4, "ux": True, "uy": True}, {"edge": "bottom", "from": 5.6, "to": 6, "uy": True}],
                        loads=[{"edge": "top", "from": 2.7, "to": 3.3, "qy": -2000}, {"edge": "left", "from": 1, "qx": 10}]))
    models.append(_wall("Q4", supports=[{"at": [0, 0], "uy": True}]))     # a mechanism: both must refuse it
    p = tmp_path / "fe.json"
    p.write_text(json.dumps(models))
    out = json.loads(subprocess.run([NODE, str(ROOT / "web" / "parity_fe.js"), str(p)], capture_output=True, text=True, check=True).stdout)
    for d, js in zip(models, out):
        try:
            r = run_fe(d)
        except FEError:
            assert not js["ok"]
            continue
        assert js["ok"], js.get("error")
        assert np.array_equal(np.array(js["elements"]), r.model.elements)
        assert np.allclose(js["nodes"], r.model.nodes, rtol=0, atol=1e-13)
        scale = np.max(np.abs(r.u))
        assert np.allclose(js["u"], r.u, rtol=0, atol=1e-9 * scale)
        assert np.allclose(js["reactions"], r.reactions, rtol=0, atol=1e-8 * max(np.max(np.abs(r.reactions)), 1))
        s = np.max(np.abs(r.nodal))
        assert np.allclose(js["nodal"], r.nodal, rtol=0, atol=1e-8 * s)
        assert np.allclose(js["elem_nodal"], r.elem_nodal, rtol=0, atol=1e-8 * s)
        assert np.allclose(js["gp"], np.array(r.gp), rtol=0, atol=1e-8 * s)
        assert np.allclose(js["mises"], r.field("mises"), rtol=0, atol=1e-8 * s)
        assert math.isclose(js["energy"], r.energy, rel_tol=1e-9)
        det = element_details(r.model, 0, r.u)
        assert np.allclose(js["k0"], det["k"], rtol=1e-12, atol=1e-9 * np.max(np.abs(det["k"])))
        assert np.allclose(js["m0"], det["m"], rtol=1e-12, atol=1e-15)
        assert {k: len(v) for k, v in r.model.edges.items()} == js["edges"]
