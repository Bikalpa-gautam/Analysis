"""The browser dynamics (web/dynamics.js) must reproduce tragwerk.modal and tragwerk.seismic.

Frequencies, effective masses, mode shapes (for well-separated modes), the CQC
seismic action effects at every station, the seismic combinations and every
check result including the drift checks are compared.
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


def _models():
    out = [
        generate_building({"modes": 12}),
        generate_building({"spans_x": [6.0, 7.0], "spans_y": [5.0], "storeys": [3.5, 3.0, 3.0], "modes": 8,
                           "seismic": {"agR": 1.2, "ground": "C", "q": 1.5}}),
        generate_building({"spans_x": [5.0, 5.0, 5.0], "spans_y": [6.0, 4.0], "storeys": [4.0, 3.2], "pattern_by_storey": True,
                           "seismic": {"agR": 2.0, "ground": "D", "spectrum_type": 2, "importance": 1.2, "directions": ["Y"], "drift_limit": 0.0075}}),
    ]
    lumped = generate_building({"spans_x": [6.0], "spans_y": [6.0, 5.0], "storeys": [3.0, 3.0], "seismic": {"agR": 1.0}})
    lumped["dynamics"] = {"modes": 10, "mass": "lumped", "segments": 1, "masses": [{"node": "A1-L2", "m": 12.0}]}
    out.append(lumped)
    for seed in range(4):
        m = random_frame3d(seed)
        m["dynamics"] = {"modes": 6, "segments": 1 + seed % 3, "mass": "consistent" if seed % 2 == 0 else "lumped"}
        m["seismic"] = {"agR": 1.5, "ground": "B"}
        for mm in m["members"]:
            mm.setdefault("design", {})
        out.append(m)
    return out


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    models = _models()
    path = tmp_path_factory.mktemp("pdyn") / "m.json"
    path.write_text(json.dumps(models))
    out = subprocess.run([NODE, str(ROOT / "web" / "parity3d.js"), str(path)], capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, out.stderr
    return list(zip(models, json.loads(out.stdout)))


def test_runs(results):
    for _, js in results:
        assert js["ok"], js.get("error")


def test_modal_parity(results):
    for data, js in results:
        res = run3d(parse_model3d(copy.deepcopy(data)))
        mr, jm = res.modal, js["modal"]
        assert np.allclose(jm["omega"], mr.omega, rtol=1e-9)
        for d in "XYZ":
            assert math.isclose(jm["mtot"][d], mr.mtot[d], rel_tol=1e-12)
        assert jm["factors"] == pytest.approx(mr.factors)
        # mode shapes and mass ratios for modes separated from their neighbours by more than 1 %
        w = mr.omega
        for k in range(len(w)):
            gaps = [abs(w[k] - w[j]) / w[k] for j in range(len(w)) if j != k]
            if gaps and min(gaps) < 0.01:
                continue
            for d in "XYZ":
                assert jm["ratio"][d][k] == pytest.approx(mr.ratio(d)[k], abs=1e-9)
            for nid in res.model.nodes:
                i = mr.analysis.nidx[nid]
                assert np.allclose(jm["shapes"][nid][k], mr.phi[6 * i:6 * i + 6, k], atol=1e-8 * np.max(np.abs(mr.phi[:, k])))


def test_seismic_parity(results):
    for data, js in results:
        if "seismic" not in data:
            continue
        res = run3d(parse_model3d(copy.deepcopy(data)))
        sr, js_s = res.seismic, js["seismic"]
        for d, v in sr.base_shear.items():
            assert js_s["base_shear"][d] == pytest.approx(v, rel=1e-8)
        assert np.allclose(js_s["Sd"], sr.Sd, rtol=1e-12)
        for mid, e in sr.members.items():
            for q, arr in e.items():
                scale = max(float(np.max(np.abs(arr))), 1e-6)
                assert np.allclose(js_s["members"][mid][q], arr, rtol=0, atol=1e-7 * scale), (mid, q)
        for nid, v in sr.reactions.items():
            assert np.allclose(js_s["reactions"][nid], v, rtol=0, atol=1e-7 * max(float(np.max(v)), 1e-6))
        assert len(js_s["storeys"]) == len(sr.storeys)
        for a, b in zip(js_s["storeys"], sr.storeys):
            assert a["P"] == pytest.approx(b["P"], rel=1e-9) and a["theta"]["x"] == pytest.approx(b["theta"]["x"], rel=1e-7, abs=1e-12)
        # combinations (including the seismic ones) and checks
        py = [(c.id, c.label) for c in res.combos.combos]
        assert py == [(c["id"], c["label"]) for c in js["combos"]]
        for mid, lst in res.checks.items():
            jl = js["checks"][mid]
            assert [c.id for c in lst] == [c["id"] for c in jl], mid
            for c, jc in zip(lst, jl):
                assert c.status == jc["status"], (mid, c.id)
                if c.utilisation is not None and math.isfinite(c.utilisation):
                    assert jc["utilisation"] == pytest.approx(c.utilisation, rel=1e-6, abs=1e-8), (mid, c.id)
        assert js["summary"]["status"] == res.overall_status()
