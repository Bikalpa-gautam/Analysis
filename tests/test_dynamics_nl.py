"""Time-history analysis (tragwerk.dynamics_nl) and its browser port (web/dynamics_nl.js)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tragwerk.dynamics_nl import (BWParams, DynOptions, LCG, bouc_wen_sdof, elastic_spectrum, fe_dynamic, parse_record, record_from_values,
                                  response_spectrum, synthetic_accelerogram)
from tragwerk.fe2d import FEError, parse_fe
from tragwerk.verification.benchmarks_dyn_nl import _wall

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
REC = {"type": "synthetic", "ag_g": 0.3, "soil": "C", "spectrum_type": 1, "duration_s": 8.0, "dt_s": 0.01, "seed": 3, "rise_s": 1.0, "strong_s": 4.0}


def test_lcg_and_spectrum():
    g = LCG(1)
    x1 = g.next()
    assert x1 == ((1664525 + 1013904223) % 2 ** 32) / 2 ** 32
    assert 0 <= g.next() < 1
    ag = 2.0
    S, TB, TC, TD = 1.2, 0.15, 0.5, 2.0
    assert elastic_spectrum([0.0], ag, "B")[0] == pytest.approx(ag * S)
    assert elastic_spectrum([0.3], ag, "B")[0] == pytest.approx(2.5 * ag * S)
    assert elastic_spectrum([1.0], ag, "B")[0] == pytest.approx(2.5 * ag * S * TC / 1.0)
    assert elastic_spectrum([3.0], ag, "B")[0] == pytest.approx(2.5 * ag * S * TC * TD / 9.0)
    assert elastic_spectrum([0.3], ag, "B", xi=0.10)[0] == pytest.approx(2.5 * ag * S * np.sqrt(10 / 15))


def test_records():
    r = parse_record(REC)
    assert r.pga >= 0.3 * 1.15 * 9.81 - 1e-9 and len(r.a) == 801
    v = np.concatenate([[0.0], np.cumsum(0.5 * (r.a[1:] + r.a[:-1]) * r.dt)])
    assert abs(v[-1]) < 1e-9 * np.abs(v).max()                          # baseline-corrected: zero final velocity
    assert np.array_equal(parse_record(REC).a, r.a)                    # the seed makes it reproducible
    v = record_from_values([0, 0.1, -0.2, 0.05, 0], 0.01)
    assert v.pga == pytest.approx(0.2 * 9.81)
    with pytest.raises(FEError):
        record_from_values([0, 1], 0.01)
    with pytest.raises(FEError):
        synthetic_accelerogram(0.3, "B", 1, 10.0, 0.01, rise=2, strong=9)
    with pytest.raises(FEError):
        parse_record({"type": "csv"})


def test_bouc_wen_limits():
    rec = parse_record(REC)
    lin = bouc_wen_sdof(BWParams(alpha=1.0), rec)
    assert lin["E_hyst"][-1] == pytest.approx(0.0, abs=1e-6 * lin["E_in"].max())   # elastic: nothing dissipated but damping
    soft = bouc_wen_sdof(BWParams(uy=0.002, alpha=0.02), rec)
    assert soft["fs_max"] < 1.1 * (0.02 * 4e4 * soft["u_max"] + 0.98 * 4e4 * 0.002)          # bounded by the yield force
    assert soft["E_hyst"][-1] > 0 and soft["energy_error"] < 1e-9


def test_fe_dynamic_basics():
    m = parse_fe(_wall())
    r = fe_dynamic(m, parse_record(REC), DynOptions(snapshots=10))
    assert r.status == "complete" and len(r.frames) >= 10
    assert r.energy["error"] < 1e-9 and r.periods[0] > r.periods[1]
    with pytest.raises(FEError, match="no mass"):
        d = _wall()
        d["self_weight"] = False
        d["loads"] = []
        fe_dynamic(parse_fe(d), parse_record(REC))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_parity_dynamics(tmp_path):
    j2 = _wall({"E_MPa": 210000, "nu": 0.3, "thickness_m": 0.008, "density_kNm3": 78.5, "model": "j2", "fy_MPa": 235, "H_MPa": 1000}, Lx=2.0, Ly=6.0, h=1.0)
    rc = _wall({"E_MPa": 33000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25, "model": "rc", "fck_MPa": 30, "values": "mean"},
               {"layers": [{"dir": "x", "dia_mm": 10, "spacing_mm": 200}, {"dir": "y", "dia_mm": 10, "spacing_mm": 200}]}, Lx=2.0, Ly=6.0, h=1.0)
    tasks = [{"kind": "record", "record": REC}, {"kind": "bw", "record": REC, "params": {"m": 100, "k": 4e4, "uy": 0.01, "alpha": 0.05, "n": 2}},
             {"kind": "fe", "record": REC, "model": _wall()}, {"kind": "fe", "record": REC, "model": j2}, {"kind": "fe", "record": REC, "model": rc}]
    p = tmp_path / "dyn.json"
    p.write_text(json.dumps(tasks))
    out = json.loads(subprocess.run([NODE, str(ROOT / "web" / "parity_dyn_nl.js"), str(p)], capture_output=True, text=True, check=True).stdout)
    assert all(o["ok"] for o in out), [o.get("error") for o in out]
    rec = parse_record(REC)
    assert np.allclose(out[0]["a"], rec.a, rtol=0, atol=1e-10 * rec.pga)
    assert np.allclose(out[0]["sa"], response_spectrum(rec.a, rec.dt, [0.1, 0.5, 1.0, 2.0])["Sa"], rtol=1e-9)
    assert np.allclose(out[0]["match"]["achieved"], rec.match["achieved"], rtol=1e-9)
    bw = bouc_wen_sdof(BWParams(m=100, k=4e4, uy=0.01, alpha=0.05, n=2), rec)
    assert np.allclose(out[1]["u"], bw["u"], rtol=0, atol=1e-10 * bw["u_max"]) and out[1]["iterations"] == bw["iterations"]
    for js, tk in zip(out[2:], tasks[2:]):
        r = fe_dynamic(parse_fe(tk["model"]), rec, DynOptions())
        assert js["status"] == r.status and js["monitor_dof"] == r.monitor_dof and js["iterations"] == r.iterations
        assert np.allclose(js["periods"], r.periods, rtol=1e-9) and np.allclose(js["rayleigh"], r.rayleigh, rtol=1e-9)
        assert np.allclose(js["u_mon"], r.u_mon, rtol=0, atol=1e-6 * np.abs(r.u_mon).max())
        assert np.allclose(js["base_shear"], r.base_shear, rtol=0, atol=1e-6 * np.abs(r.base_shear).max())
        assert js["mass_total"] == pytest.approx(r.mass_total, rel=1e-12)
