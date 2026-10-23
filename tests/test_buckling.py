"""Linear buckling of 3D frames (tragwerk.buckling) and its browser port (web/dynamics.js)."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tragwerk.buckling import buckling, geometric_stiffness_local, to_dict_buckling
from tragwerk.model3d import parse_model3d

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
OFFICE = json.loads((ROOT / "examples" / "office_building_3d.json").read_text())


def test_geometric_stiffness_properties():
    g = geometric_stiffness_local(-100.0, 3.0, 0.02)
    assert np.allclose(g, g.T)
    # a rigid translation produces no geometric forces; a rigid rotation θ turns the axial force N into transverse
    # end forces ∓Nθ and no end moments
    t = np.zeros(12)
    t[[1, 7]] = 1.0
    assert np.allclose(g @ t, 0)
    r = np.zeros(12)                                      # rotation about z: v = θ x, θz = θ
    r[[5, 11]] = 1.0
    r[7] = 3.0
    f = g @ r
    assert f[1] == pytest.approx(100.0) and f[7] == pytest.approx(-100.0) and np.allclose(f[[5, 11]], 0, atol=1e-12)


def test_office_building_alpha_cr():
    m = parse_model3d(OFFICE)
    br = buckling(m, modes=3, segments=2)
    assert br.governing is not None and br.alpha_cr > 1
    assert all(c.alpha_cr is None or c.alpha_cr >= br.alpha_cr - 1e-9 for c in br.combos)
    d = to_dict_buckling(br, m, points_per_element=2)
    assert len(d["modes"]) == len(br.modes) and set(d["modes"][0]) == set(m.members)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_parity_buckling(tmp_path):
    from tragwerk.verification.benchmarks_dyn import _column
    FIX = {"ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True}
    col = _column(5.0, FIX, None).raw
    items = [{"model": col, "options": {"modes": 2, "segments": 4}}, {"model": OFFICE, "options": {"modes": 3, "segments": 2}}]
    p = tmp_path / "b.json"
    p.write_text(json.dumps(items))
    out = json.loads(subprocess.run([NODE, str(ROOT / "web" / "parity_buckling.js"), str(p)], capture_output=True, text=True, check=True).stdout)
    for it, js in zip(items, out):
        assert js["ok"], js.get("error")
        br = buckling(parse_model3d(it["model"]), **it["options"])
        assert js["governing"] == br.governing
        assert np.allclose(js["alpha"], br.alpha, rtol=1e-8)
        for jc, pc in zip(js["combos"], br.combos):
            assert jc["id"] == pc.id and np.allclose(jc["alpha"], pc.alpha, rtol=1e-8)
        # the first mode (distinct eigenvalue) agrees; repeated eigenvalues (symmetric columns) span the same space
        if len(br.alpha) < 2 or abs(br.alpha[1] - br.alpha[0]) > 1e-6 * br.alpha[0]:
            assert np.allclose(js["modes"][0], br.modes[0], atol=1e-6)
