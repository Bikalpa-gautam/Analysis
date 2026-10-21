"""Physical properties every correct linear frame solver must satisfy."""
import math

import numpy as np
import pytest

from tragwerk.verification.frames import random_frame
from helpers import build, single_case
from tragwerk.model import parse_model
from tragwerk.solver import FrameAnalysis, StructureUnstable

PIN, ROLLER, FIX = {"ux": True, "uy": True}, {"uy": True}, {"ux": True, "uy": True, "rz": True}


@pytest.mark.parametrize("seed", range(8))
def test_stiffness_symmetric_and_equilibrium(seed):
    m = parse_model(random_frame(seed))
    fa = FrameAnalysis(m)
    assert np.allclose(fa.K, fa.K.T, atol=1e-6 * np.max(np.abs(fa.K)))
    r = fa.solve_case(m.load_cases["G"])
    scale = max(abs(v) for v in r.applied.values())
    assert all(abs(v) < 1e-9 * scale for v in r.equilibrium.values())


@pytest.mark.parametrize("seed", range(6))
def test_maxwell_betti_reciprocity(seed):
    data = random_frame(seed, releases=False)
    nodes = [n["id"] for n in data["nodes"] if n["y"] > 0]
    a, b = nodes[0], nodes[-1]
    out = {}
    for src, dst in ((a, b), (b, a)):
        d = dict(data, load_cases=[{"id": "G", "type": "permanent",
                                    "loads": [{"type": "nodal", "node": src, "fx": 1.0}]}])
        m = parse_model(d)
        fa = FrameAnalysis(m)
        r = fa.solve_case(m.load_cases["G"])
        out[(src, dst)] = r.displacements[fa.nidx[dst], 0]
    assert out[(a, b)] == pytest.approx(out[(b, a)], rel=1e-9)


@pytest.mark.parametrize("angle", [17.0, 90.0, 213.0])
def test_rotation_invariance(angle):
    """Rotating the whole structure and its loads leaves internal forces unchanged."""
    def make(theta):
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        pts = {"A": (0, 0), "B": (0, 4), "C": (6, 4), "D": (6, 0)}
        rot = {k: (c * x - s * y, s * x + c * y) for k, (x, y) in pts.items()}
        loads = [{"type": "udl", "member": "bm", "w": -20, "direction": "local_y"},
                 {"type": "point", "member": "c1", "p": 15, "a": 2.0, "direction": "local_y"},
                 {"type": "nodal", "node": "B", "fx": 30 * c, "fy": 30 * s}]
        return build(rot, {"c1": ("A", "B"), "bm": ("B", "C"), "c2": ("D", "C")},
                     [{"node": "A", **FIX}, {"node": "D", **FIX}] if theta == 0 else
                     [{"node": "A", **FIX}, {"node": "D", **FIX}], single_case(loads))
    ref = FrameAnalysis(make(0)).solve_case(make(0).load_cases["G"])
    m = make(angle)
    rot = FrameAnalysis(m).solve_case(m.load_cases["G"])
    for mid in ("c1", "bm", "c2"):
        for q in ("N", "V", "M"):
            assert np.allclose(getattr(rot.members[mid], q), getattr(ref.members[mid], q), atol=1e-8)


def test_mechanism_detected():
    m = build({"A": (0, 0), "B": (5, 0)}, {"M": ("A", "B")}, [{"node": "A", **ROLLER}, {"node": "B", **ROLLER}],
              single_case([{"type": "udl", "member": "M", "w": -1}]))
    with pytest.raises(StructureUnstable, match="move in X"):
        FrameAnalysis(m)


def test_hinged_chain_is_a_mechanism():
    m = build({"A": (0, 0), "B": (4, 0), "C": (8, 0)},
              {"M1": ("A", "B", {"releases": {"j": True}}), "M2": ("B", "C", {"releases": {"i": True}})},
              [{"node": "A", **PIN}, {"node": "C", **ROLLER}], single_case([]))
    with pytest.raises(StructureUnstable):
        FrameAnalysis(m)


def test_pin_node_restrained_automatically():
    # three-hinged frame: every member hinged at the apex node C
    m = build({"A": (0, 0), "C": (4, 3), "B": (8, 0)},
              {"L": ("A", "C", {"releases": {"j": True}}), "R": ("C", "B", {"releases": {"i": True}})},
              [{"node": "A", **PIN}, {"node": "B", **PIN}],
              single_case([{"type": "nodal", "node": "C", "fy": -100}]))
    fa = FrameAnalysis(m)
    r = fa.solve_case(m.load_cases["G"])
    assert fa.auto_fixed == [3 * fa.nidx["C"] + 2]
    # symmetric three-hinged arch: V = 50 each, H = 50*4/3
    assert r.reactions["A"][1] == pytest.approx(50.0)
    assert abs(r.reactions["A"][0]) == pytest.approx(50.0 * 4 / 3)
    assert np.allclose(r.members["L"].M, 0.0, atol=1e-9)


def test_moment_on_pin_node_rejected():
    m = build({"A": (0, 0), "C": (4, 3), "B": (8, 0)},
              {"L": ("A", "C", {"releases": {"j": True}}), "R": ("C", "B", {"releases": {"i": True}})},
              [{"node": "A", **PIN}, {"node": "B", **PIN}], single_case([{"type": "nodal", "node": "C", "mz": 5}]))
    fa = FrameAnalysis(m)
    with pytest.raises(StructureUnstable, match="hinged"):
        fa.solve_case(m.load_cases["G"])


def test_self_weight():
    m = build({"A": (0, 0), "B": (6, 0)}, {"M": ("A", "B")}, [{"node": "A", **PIN}, {"node": "B", **ROLLER}],
              single_case([], self_weight=True))
    r = FrameAnalysis(m).solve_case(m.load_cases["G"])
    w = 25.0 * 0.3 * 0.5
    assert r.reactions["A"][1] == pytest.approx(w * 3.0)


def test_point_load_at_member_end_goes_to_node():
    m = build({"A": (0, 0), "B": (3, 0)}, {"M": ("A", "B")}, [{"node": "A", **FIX}],
              single_case([{"type": "point", "member": "M", "p": -10, "a": 3.0}]))
    r = FrameAnalysis(m).solve_case(m.load_cases["G"])
    mr = r.members["M"]
    assert mr.V[0] == pytest.approx(10.0)
    assert mr.V[-1] == pytest.approx(10.0)   # left limit at the tip, inside the member
    assert mr.M[0] == pytest.approx(-30.0)


def test_shear_jump_at_point_load():
    m = build({"A": (0, 0), "B": (4, 0)}, {"M": ("A", "B")}, [{"node": "A", **PIN}, {"node": "B", **ROLLER}],
              single_case([{"type": "point", "member": "M", "p": -20, "a": 1.0}]))
    mr = FrameAnalysis(m).solve_case(m.load_cases["G"]).members["M"]
    k = int(np.flatnonzero((np.abs(mr.x - 1.0) < 1e-12) & (mr.side < 0))[0])
    assert mr.V[k] - mr.V[k + 1] == pytest.approx(20.0)
    assert mr.M[k] == pytest.approx(mr.M[k + 1])


def test_axial_compatibility_along_member():
    m = parse_model(random_frame(3))
    fa = FrameAnalysis(m)
    r = fa.solve_case(m.load_cases["G"])
    for mid, md in fa.members.items():
        ul = md.T @ r.displacements[[md.ni, md.nj]].ravel()
        mr = r.members[mid]
        assert mr.u[-1] == pytest.approx(ul[3], abs=1e-12)
        assert mr.v[-1] == pytest.approx(ul[4], abs=1e-12)
