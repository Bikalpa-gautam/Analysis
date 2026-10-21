"""3D model, solver properties, building generator and 3D checks."""
import copy
import json
import math

import numpy as np
import pytest

from tragwerk.building import BuildingSpec, generate_building
from tragwerk.model import ModelError
from tragwerk.model3d import load_any, parse_model3d
from tragwerk.project3d import run3d, to_dict3d
from tragwerk.solver import StructureUnstable
from tragwerk.solver3d import FrameAnalysis3D
from tragwerk.verification.frames3d import random_frame3d


def test_schema_and_validation():
    d = random_frame3d(0)
    m = parse_model3d(d)
    assert m.is3d and len(m.nodes) == 18
    bad = copy.deepcopy(d)
    bad["sections"]["COL"]["cover_mm"] = 250
    with pytest.raises(ModelError, match="outside the section"):
        parse_model3d(bad)
    bad = copy.deepcopy(d)
    bad["load_cases"][0]["loads"].append({"type": "udl", "member": "C000", "w": 1, "direction": "up"})
    with pytest.raises(ModelError, match="direction"):
        parse_model3d(bad)
    assert load_any(d).is3d


@pytest.mark.parametrize("seed", range(5))
def test_symmetry_and_equilibrium(seed):
    m = parse_model3d(random_frame3d(seed))
    fa = FrameAnalysis3D(m)
    assert np.allclose(fa.K, fa.K.T, atol=1e-6 * np.max(np.abs(fa.K)))
    r = fa.solve_case(m.load_cases["G"])
    scale = max(abs(v) for v in r.applied.values())
    assert all(abs(v) < 1e-9 * scale for v in r.equilibrium.values())


def test_reciprocity_3d():
    d = random_frame3d(4, releases=False)
    top = [n["id"] for n in d["nodes"] if n["z"] > 0]
    a, b = top[0], top[-1]
    vals = {}
    for src, dst, comp, key in ((a, b, 0, "fy"), (b, a, 1, "fx")):
        dd = dict(d, load_cases=[{"id": "G", "type": "permanent", "loads": [{"type": "nodal", "node": src, key: 1.0}]}])
        m = parse_model3d(dd)
        fa = FrameAnalysis3D(m)
        vals[(src, dst)] = fa.solve_case(m.load_cases["G"]).displacements[fa.nidx[dst], comp]
    assert vals[(a, b)] == pytest.approx(vals[(b, a)], rel=1e-9)


@pytest.mark.parametrize("angle", [30.0, 90.0, 200.0])
def test_rotation_about_vertical_axis(angle):
    """Rotating the whole building about Z leaves member forces unchanged."""
    d = random_frame3d(1, releases=False, partial_loads=False)
    # only gravity loads, which are invariant under rotation about Z
    for lc in d["load_cases"]:
        lc["loads"] = [ld for ld in lc["loads"] if ld["type"] != "nodal" and ld.get("direction", "global_z") == "global_z"]
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rot = copy.deepcopy(d)
    for n in rot["nodes"]:
        n["x"], n["y"] = c * n["x"] - s * n["y"], s * n["x"] + c * n["y"]
    r0 = FrameAnalysis3D(parse_model3d(d)).solve_case(parse_model3d(d).load_cases["G"])
    m1 = parse_model3d(rot)
    r1 = FrameAnalysis3D(m1).solve_case(m1.load_cases["G"])
    for mid in r0.members:
        a, b = r0.members[mid], r1.members[mid]
        # column axes stay tied to global X, so compare invariants: N, T and resultant moment and shear
        assert np.allclose(a.N, b.N, atol=1e-8)
        assert np.allclose(a.T, b.T, atol=1e-8)
        assert np.allclose(np.hypot(a.My, a.Mz), np.hypot(b.My, b.Mz), atol=1e-8), mid
        assert np.allclose(np.hypot(a.Vy, a.Vz), np.hypot(b.Vy, b.Vz), atol=1e-8), mid


def test_mechanism_detected_3d():
    d = random_frame3d(2)
    for s in d["supports"]:
        s.update(ux=False)
    with pytest.raises(StructureUnstable, match="move in X"):
        FrameAnalysis3D(parse_model3d(d))


@pytest.mark.parametrize("dist", ["two_way", "one_way_x", "one_way_y"])
def test_building_load_totals(dist):
    spec = BuildingSpec(spans_x=[5.0, 7.0], spans_y=[6.0], storeys=[3.0, 3.0], distribution=dist, wind=0.0, snow=0.0)
    m = parse_model3d(generate_building(spec))
    fa = FrameAnalysis3D(m)
    r = fa.solve_case(m.load_cases["Q"])
    assert sum(R[2] for R in r.reactions.values()) == pytest.approx(spec.imposed * 12 * 6 * 1)


def test_building_variants():
    d = generate_building({"storeys": [3.5, 3.0, 3.0], "pattern_by_storey": True, "base": "pinned"})
    ids = [c["id"] for c in d["load_cases"]]
    assert {"Q-L1", "Q-L2", "S", "W+X", "W-Y"} <= set(ids)
    assert not d["supports"][0]["rx"]
    assert len(d["meta"]["slabs"]) == 3 * 2 * 3


def test_default_building_runs_and_serialises():
    res = run3d(parse_model3d(generate_building()))
    data = to_dict3d(res)
    json.dumps(data, allow_nan=False)
    titles = {c.title for lst in res.checks.values() for c in lst}
    assert "Bending with axial force, both axes" in titles
    assert "Slender column, second-order effects" in titles
    for lst in res.checks.values():
        for c in lst:
            assert c.steps or c.notes


def test_biaxial_reduces_to_uniaxial():
    """A beam bent about one axis only: Eq. 5.39 equals M_Ed/M_Rd."""
    from tragwerk.codes.ec2 import checks3d
    from tragwerk.codes.ec2.params import EC2Params
    d = random_frame3d(0)
    m = parse_model3d(d)
    S = checks3d.section_models(m.sections["BEAM"], EC2Params())
    x = np.linspace(0, 5, 11)
    My = (40 * x * (5 - x))[None, :]
    z = np.zeros_like(My)
    f = checks3d.MemberForces3D(x, 5.0, z, z, z, z, My, z, ["U"], ["u"], My, ["Q"], ["q"])
    res = checks3d.check_biaxial(m.members["BX001"], S, f, EC2Params())
    MRd = S.rc_y.solve_state(0.0).M
    assert res.utilisation == pytest.approx(float(np.max(My)) / MRd, rel=1e-9)
