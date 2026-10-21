"""The browser engine (web/engine.js) must reproduce the Python core.

Runs both on the example models and on random frames with variable and
grouped load cases, and compares every displacement, reaction, internal force,
combination and check result.
"""
import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tragwerk.model import parse_model
from tragwerk.project import run
from tragwerk.verification.frames import random_frame

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node.js not available")


def _variants():
    models = [json.loads(p.read_text()) for p in sorted((ROOT / "examples").glob("*.json"))]
    for seed in range(10):
        m = random_frame(seed)
        loads = m["load_cases"][0]["loads"]
        g = [ld for k, ld in enumerate(loads) if k % 3 == 0 or ld["type"] == "nodal" and False]
        q = [ld for k, ld in enumerate(loads) if k % 3 == 1 and ld["type"] != "nodal"]
        w = [ld for ld in loads if ld["type"] == "nodal"]
        rest = [ld for k, ld in enumerate(loads) if k % 3 == 2 and ld["type"] != "nodal"]
        wl = copy.deepcopy(w)
        for ld in wl:
            ld["fx"] = -ld["fx"]
        m["load_cases"] = [
            {"id": "G", "type": "permanent", "self_weight": True, "loads": g + rest},
            {"id": "Q", "type": "variable", "category": "B", "loads": q},
            {"id": "W+", "type": "variable", "category": "wind", "group": "w", "loads": w},
            {"id": "W-", "type": "variable", "category": "wind", "group": "w", "loads": wl},
        ]
        m["design"] = {"combination_rule": "6.10ab" if seed % 2 else "6.10"}
        if seed % 3 == 0:  # make one beam a cantilever-like overhang
            m["members"][-1]["design"] = {"type": "beam", "deflection_system": "end_span", "creep_coefficient": 2.5}
        models.append(m)
    return models


def _close(a, b, scale, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.allclose(a, b, rtol=0, atol=tol * max(scale, 1e-12))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    models = _variants()
    path = tmp_path_factory.mktemp("parity") / "models.json"
    path.write_text(json.dumps(models))
    out = subprocess.run([NODE, str(ROOT / "web" / "parity.js"), str(path)], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return list(zip(models, json.loads(out.stdout)))


def test_engine_runs_every_model(results):
    for _, js in results:
        assert js["ok"], js.get("error")


def test_parity_analysis_and_checks(results):
    for data, js in results:
        res = run(parse_model(data))
        # combinations
        py_combos = [(c.id, c.kind, tuple(round(f, 12) for f in c.factors), c.label) for c in res.combos.combos]
        js_combos = [(c["id"], c["kind"], tuple(round(f, 12) for f in c["factors"]), c["label"]) for c in js["combos"]]
        assert py_combos == js_combos
        # load cases
        for cid, cr in res.cases.items():
            jc = js["cases"][cid]
            sd = float(np.max(np.abs(cr.displacements))) or 1.0
            assert _close(cr.displacements, jc["displacements"], sd)
            sr = max(float(np.max(np.abs(v))) for v in cr.reactions.values()) or 1.0
            for nid, R in cr.reactions.items():
                assert _close(R, jc["reactions"][nid], sr)
            for mid, mr in cr.members.items():
                jm = jc["members"][mid]
                assert _close(mr.x, jm["x"], mr.x[-1])
                for q in ("N", "V", "M"):
                    s = max(float(np.max(np.abs(getattr(mr, q)))), sr * 1e-3, 1e-6)
                    assert _close(getattr(mr, q), jm[q], s), (cid, mid, q)
                assert _close(mr.v, jm["v"], sd)
        # checks
        for mid, lst in res.checks.items():
            jl = js["checks"][mid]
            assert [c.id for c in lst] == [c["id"] for c in jl], mid
            for c, jcheck in zip(lst, jl):
                assert c.status == jcheck["status"], (mid, c.id)
                assert c.combination == jcheck["combination"], (mid, c.id)
                for key in ("utilisation", "demand", "capacity"):
                    pv, jv = getattr(c, key), jcheck[key]
                    if pv is None or not math.isfinite(pv):
                        assert jv is None
                    else:
                        assert jv == pytest.approx(pv, rel=1e-7, abs=1e-9), (mid, c.id, key)
        assert js["summary"]["status"] == res.overall_status()
