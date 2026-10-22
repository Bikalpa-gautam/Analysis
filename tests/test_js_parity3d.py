"""The browser 3D engine (web/engine3d.js) must reproduce the Python 3D core.

Compares the building generator output exactly, and for every model every
displacement, reaction, internal force, envelope, combination and check result
(including each calculation step) to tight tolerances.
"""
import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tragwerk.building import generate_building
from tragwerk.model3d import parse_model3d
from tragwerk.project3d import run3d
from tragwerk.verification.frames3d import random_frame3d

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node.js not available")

SPECS = [
    {},
    {"spans_x": [7.5], "spans_y": [5.0, 4.0], "storeys": [4.0, 3.0], "base": "pinned", "distribution": "one_way_x",
     "combination_rule": "6.10ab", "fck": 55.0},
    {"spans_x": [5.0, 6.5], "spans_y": [6.0], "storeys": [3.5, 3.0, 3.0], "pattern_by_storey": True, "distribution": "one_way_y",
     "column_l0_factor": 2.0, "wind": 1.4, "snow": 0.0},
    {"spans_x": [6.0], "spans_y": [6.0], "storeys": [5.5], "column": {"type": "rc_rect", "b_mm": 250, "h_mm": 300, "concrete": "x",
     "rebar": "B500B", "cover_mm": 30, "top": {"n": 2, "dia_mm": 16}, "bottom": {"n": 2, "dia_mm": 16},
     "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200}}, "column_l0_factor": 2.0, "imposed": 5.0, "imposed_category": "C"},
]


def _torsion_model():
    """Cantilever balconies framing into edge beams: equilibrium torsion, rolled member, local loads, releases."""
    m = generate_building({"spans_x": [6.0, 6.0], "spans_y": [5.0], "storeys": [3.5, 3.2], "wind": 0.0})
    m["nodes"] += [{"id": "T1", "x": 3.0, "y": -2.0, "z": 3.5}, {"id": "T2", "x": 3.0, "y": 0.0, "z": 3.5}]
    # split edge beam BX-A1-L1 at mid-span so the cantilever frames into it
    beam = next(b for b in m["members"] if b["id"] == "BX-A1-L1")
    m["members"].remove(beam)
    m["members"] += [
        dict(beam, id="BX-A1-L1a", j="T2"), dict(beam, id="BX-A1-L1b", i="T2"),
        {"id": "CANT", "i": "T2", "j": "T1", "section": "Beam Y", "design": {"type": "beam", "deflection_system": "cantilever"}},
        {"id": "ROLL", "i": "A2-L2", "j": "B1-L2", "section": "Beam X", "roll_deg": 30.0, "releases": {"i": {"t": False, "my": True, "mz": True}}},
    ]
    for lc in m["load_cases"]:
        lc["loads"] = [ld for ld in lc["loads"] if ld.get("member") != "BX-A1-L1"]
    m["load_cases"][1]["loads"] += [
        {"type": "udl", "member": "CANT", "w": -12.0},
        {"type": "point", "member": "CANT", "p": -15.0, "a": 2.0},
        {"type": "udl", "member": "ROLL", "w": -4.0, "direction": "local_z"},
        {"type": "udl", "member": "ROLL", "w1": 1.0, "w2": 3.0, "a": 1.0, "b": 5.0, "direction": "local_y"},
        {"type": "point", "member": "BX-A1-L1a", "p": 8.0, "a": 1.5, "direction": "global_y"},
        {"type": "nodal", "node": "B2-L2", "mx": 5.0, "my": -3.0, "mz": 2.0},
    ]
    m["load_cases"].append({"id": "QB", "type": "variable", "category": "A", "loads": [{"type": "udl", "member": "CANT", "w": -6.0}]})
    return m


def _random(seed):
    m = random_frame3d(seed)
    loads = m["load_cases"][0]["loads"]
    g = [ld for k, ld in enumerate(loads) if k % 3 != 1 or ld["type"] == "nodal"]
    q = [ld for k, ld in enumerate(loads) if k % 3 == 1 and ld["type"] != "nodal"]
    m["load_cases"] = [{"id": "G", "type": "permanent", "self_weight": True, "loads": g},
                       {"id": "Q", "type": "variable", "category": "B", "loads": q}]
    return m


BRIDGES = [{}, {"girders": 3, "width": 12.0, "carriageway": 10.0, "pier_columns": "single"},
           {"girders": 1, "spans": [18.0], "width": 8.0, "carriageway": 6.5},
           {"spans": [15.0, 22.5, 22.5, 15.0], "pier_heights": [5.0, 8.5, 5.0], "girders": 4, "width": 13.0, "carriageway": 11.5,
            "ts_stations": [0.1, 0.5, 0.9, 0.333]}]


def _models():
    from tragwerk.bridge import generate_bridge
    return ([generate_building(s) for s in SPECS] + [_torsion_model()] + [_random(s) for s in range(6)]
            + [generate_bridge(b) for b in BRIDGES[:2]])


def test_bridge_generator_identical(tmp_path):
    from tragwerk.bridge import generate_bridge
    js = _node([{"bridge_spec": s} for s in BRIDGES], tmp_path)
    for spec, j in zip(BRIDGES, js):
        assert j["ok"], j.get("error")
        assert j["model"] == json.loads(json.dumps(generate_bridge(spec)))


def _node(items, tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps(items))
    out = subprocess.run([NODE, str(ROOT / "web" / "parity3d.js"), str(path)], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_generator_identical(tmp_path):
    js = _node([{"building_spec": s} for s in SPECS], tmp_path)
    for spec, j in zip(SPECS, js):
        assert j["ok"], j.get("error")
        assert j["model"] == json.loads(json.dumps(generate_building(spec)))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    models = _models()
    js = _node(models, tmp_path_factory.mktemp("parity3d"))
    return list(zip(models, js))


def _close(a, b, scale, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.allclose(a, b, rtol=0, atol=tol * max(scale, 1e-12))


def test_engine_runs_every_model(results):
    for _, js in results:
        assert js["ok"], js.get("error")


def test_torsion_and_second_order_are_exercised(results):
    ids = {c["id"] for _, js in results for lst in js["checks"].values() for c in lst}
    assert {"ec2.torsion", "ec2.second_order", "ec2.biaxial", "ec2.shear_y", "ec2.deflection"} <= ids


def test_parity(results):
    for data, js in results:
        res = run3d(parse_model3d(copy.deepcopy(data)))
        py_combos = [(c.id, c.kind, tuple(round(f, 12) for f in c.factors), c.label) for c in res.combos.combos]
        assert py_combos == [(c["id"], c["kind"], tuple(round(f, 12) for f in c["factors"]), c["label"]) for c in js["combos"]]
        for cid, cr in res.cases.items():
            jc = js["cases"][cid]
            sd = float(np.max(np.abs(cr.displacements))) or 1.0
            assert _close(cr.displacements, jc["displacements"], sd), cid
            sr = max(float(np.max(np.abs(v))) for v in cr.reactions.values()) or 1.0
            for nid, R in cr.reactions.items():
                assert _close(R, jc["reactions"][nid], sr), (cid, nid)
            for mid, mr in cr.members.items():
                jm = jc["members"][mid]
                assert _close(mr.x, jm["x"], mr.x[-1])
                for q in ("N", "Vy", "Vz", "T", "My", "Mz"):
                    s = max(float(np.max(np.abs(getattr(mr, q)))), sr * 1e-3, 1e-6)
                    assert _close(getattr(mr, q), jm[q], s), (cid, mid, q)
                for q in ("u", "v", "w", "v_rel", "w_rel"):
                    assert _close(getattr(mr, q), jm[q], sd), (cid, mid, q)
        for mid, env in res.envelopes.items():
            for q in ("N", "Vy", "Vz", "T", "My", "Mz"):
                s = max(float(np.max(np.abs(env.max[q]))), float(np.max(np.abs(env.min[q]))), 1e-3)
                assert _close(env.max[q], js["envelopes"][mid][q + "_max"], s)
                assert _close(env.min[q], js["envelopes"][mid][q + "_min"], s)
        for mid, lst in res.checks.items():
            jl = js["checks"][mid]
            assert [c.id for c in lst] == [c["id"] for c in jl], mid
            for c, jcheck in zip(lst, jl):
                assert c.status == jcheck["status"], (mid, c.id)
                # symmetric twins can tie; the governing value must agree, the combination when not tied
                for key in ("utilisation", "demand", "capacity"):
                    pv, jv = getattr(c, key), jcheck[key]
                    if pv is None or not math.isfinite(pv):
                        assert jv is None, (mid, c.id, key)
                    else:
                        assert jv == pytest.approx(pv, rel=1e-6, abs=1e-8), (mid, c.id, key)
                if c.combination != jcheck["combination"]:
                    continue
                assert [s.symbol for s in c.steps] == [s[0] for s in jcheck["steps"]], (mid, c.id)
                for s, (_, jv) in zip(c.steps, jcheck["steps"]):
                    if s.value is None or not math.isfinite(s.value):
                        assert jv is None
                    else:
                        assert jv == pytest.approx(s.value, rel=1e-6, abs=1e-8), (mid, c.id, s.symbol)
        assert js["summary"]["status"] == res.overall_status()
        gov = res.governing()
        if gov is not None and gov.utilisation is not None and math.isfinite(gov.utilisation):
            assert js["summary"]["max_utilisation"] == pytest.approx(gov.utilisation, rel=1e-6)
