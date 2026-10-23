"""Nonlinear plane FE (tragwerk.fe2d_nl), reinforcement design of FE regions (tragwerk.fe2d_design) and their browser ports."""
import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tragwerk.fe2d import SCHEMA_FE, FEError, parse_fe, solve_fe
from tragwerk.fe2d_design import design_reinforcement, propose_layers
from tragwerk.fe2d_nl import NLOptions, crack_widths, nodal_stress, rc_state0, rc_update, run_nonlinear, summary
from tragwerk.verification.benchmarks_fe_nl import DEEP_BEAM, _panel

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _cantilever(k=1.0, et="Q8"):
    L, hb, t, E = 10.0, 0.25, 0.1, 10000.0
    EI = E * 1000 * t * hb ** 3 / 12
    return {"schema": SCHEMA_FE, "element": et, "material": {"E_MPa": E, "nu": 0.0, "thickness_m": t, "density_kNm3": 0},
            "geometry": {"type": "rectangle", "Lx": L, "Ly": hb, "h": 0.125}, "supports": [{"edge": "left", "ux": True, "uy": True}],
            "loads": [{"edge": "right", "qy": -k * EI / L ** 2 / hb}]}


def _hill():
    sy = 240.0
    return {"schema": SCHEMA_FE, "element": "Q8R", "analysis": "plane_strain",
            "material": {"E_MPa": 210000, "nu": 0.3, "thickness_m": 1.0, "density_kNm3": 0, "model": "j2", "fy_MPa": sy},
            "geometry": {"type": "ring", "a": 1, "b": 2, "n_theta": 2, "n_r": 4}, "supports": [{"edge": "left", "ux": True}, {"edge": "bottom", "uy": True}],
            "loads": [{"edge": "inner", "p": 2 / math.sqrt(3) * sy * math.log(2.0) * 1000}]}


def _rc_wall():
    return {"schema": SCHEMA_FE, "element": "Q8", "analysis": "plane_stress",
            "material": {"E_MPa": 33000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25, "model": "rc", "fck_MPa": 30, "values": "mean"},
            "reinforcement": {"layers": [{"dir": "x", "dia_mm": 10, "spacing_mm": 200}, {"dir": "y", "dia_mm": 10, "spacing_mm": 200},
                                         {"dir": "x", "dia_mm": 16, "n": 4, "region": {"y0": 0, "y1": 0.3}}]},
            "geometry": {"type": "rectangle", "Lx": 4, "Ly": 2, "h": 0.5}, "self_weight": True,
            "supports": [{"edge": "bottom", "from": 0, "to": 0.5, "uy": True}, {"edge": "bottom", "from": 3.5, "to": 4, "uy": True}, {"at": [0, 0], "ux": True}],
            "loads": [{"edge": "top", "qy": -200}, {"edge": "top", "qy": -100, "case": "Q"}]}


def test_nonlinear_elastic_equals_linear():
    d = {"schema": SCHEMA_FE, "element": "Q8", "material": {"E_MPa": 30000, "nu": 0.2, "thickness_m": 0.3},
         "geometry": {"type": "rectangle", "Lx": 6, "Ly": 3, "h": 0.5, "openings": [{"x0": 2, "y0": 0, "x1": 3, "y1": 2}]},
         "supports": [{"edge": "bottom", "ux": True, "uy": True}], "loads": [{"edge": "top", "qy": -100}, {"edge": "left", "qx": 20, "case": "Q"}], "self_weight": True}
    m = parse_fe(d)
    r = run_nonlinear(m, NLOptions(steps=2))
    lin = solve_fe(m, combo="uls")
    assert r.status == "complete"
    assert np.allclose(r.steps[-1].u.reshape(-1, 2), lin.u, rtol=0, atol=1e-12 * np.abs(lin.u).max())
    gp = np.array(lin.gp)[:, :, 2:]                    # stresses at the same integration points
    assert np.allclose(r.steps[-1].gp_stress, gp, rtol=0, atol=1e-9 * np.abs(gp).max())
    assert nodal_stress(r, r.steps[-1]).shape == lin.nodal.shape


def test_input_checks():
    with pytest.raises(FEError, match="geometric nonlinearity"):
        run_nonlinear(parse_fe(_hill()), NLOptions(geometric=True))
    bad = copy.deepcopy(DEEP_BEAM)
    bad["supports"][0] = {"edge": "bottom", "x0": 0.0, "x1": 0.5, "uy": True}
    with pytest.raises(FEError, match="unknown key 'x0'"):
        parse_fe(bad)
    bad = copy.deepcopy(DEEP_BEAM)
    bad["material"]["tension"] = "some"
    with pytest.raises(FEError, match="material.tension"):
        parse_fe(bad)


def test_rc_memory_unloads_along_the_secant():
    """A cracked point unloads to the origin along the secant; reinforcement unloads elastically from its yield plateau."""
    m = parse_fe(_panel(150, 150))
    mat = m.material
    rho, hcb, plain = np.array([[0.0075, 0.0075]]), np.array([1.0]), np.array([False])
    st = rc_state0(1)
    s_peak, _, st1, ex1 = rc_update(np.array([[0.004, 0.0, 0.0]]), st, mat, m.E, rho, hcb, plain)
    s_back, _, _, ex2 = rc_update(np.array([[0.002, 0.0, 0.0]]), st1, mat, m.E, rho, hcb, plain)
    assert ex1["ssx"][0] == pytest.approx(mat["fy"], rel=1e-12)                     # yielded (k = 1)
    assert ex2["ssx"][0] == pytest.approx(mat["fy"] - mat["Es"] * 0.002, rel=1e-9)  # elastic unloading
    assert ex1["sc1"][0] == 0.0                                                     # crack check: no tension at yield
    assert ex2["sc1"][0] == 0.0                                                     # and none on unloading either
    s_mid, _, st2, ex3 = rc_update(np.array([[0.0015, 0.0, 0.0]]), st, mat, m.E, rho, hcb, plain)   # steel still elastic
    s_un, _, _, ex4 = rc_update(np.array([[0.0005, 0.0, 0.0]]), st2, mat, m.E, rho, hcb, plain)
    assert ex3["sc1"][0] > 0 and ex4["sc1"][0] == pytest.approx(ex3["sc1"][0] / 3, rel=1e-9)   # secant to the origin
    assert s_back[0, 0] < s_peak[0, 0] and s_un[0, 0] < s_mid[0, 0]


def test_reinforcement_proposal_satisfies_the_design():
    d = copy.deepcopy(DEEP_BEAM)
    m = parse_fe(d)
    des = design_reinforcement(m)
    layers, warns = propose_layers(m, des)
    assert any(L["label"].startswith("additional x") for L in layers) and any("l_bd" in w for w in warns)
    d["reinforcement"]["layers"] = layers
    des2 = design_reinforcement(parse_fe(d))
    assert float(np.max(des2["util_steel"])) <= 1.0 + 1e-12
    # the bottom tie runs over the supports (anchorage), not only where the stresses need it
    tie = [L for L in layers if L["label"].startswith("additional x") and L["region"]["y0"] == 0.0][0]
    assert tie["region"]["x0"] < 0.5 and tie["region"]["x1"] > 5.5


def test_rc_summary_and_crack_widths():
    r = run_nonlinear(parse_fe(_rc_wall()), NLOptions(combination="uls", lam_max=2.0, steps=10))
    s = summary(r)
    assert r.status == "complete" and s["rc"]["cracked_points"] > 0
    w = crack_widths(r, r.steps[-1])
    assert w.max() == pytest.approx(s["rc"]["max_crack_width_mm"]) and w.max() > 0
    assert np.all(w[~r.steps[-1].extra["cracked"]] == 0)


def _cases():
    q = 200
    panel = _panel(100, 250)
    panel["loads"] = [{"edge": "right", "qy": q}, {"edge": "left", "qy": -q}, {"edge": "top", "qx": q}, {"edge": "bottom", "qx": -q}]
    j2 = {"schema": SCHEMA_FE, "element": "Q4", "analysis": "plane_stress",
          "material": {"E_MPa": 210000, "nu": 0.3, "thickness_m": 0.01, "model": "j2", "fy_MPa": 355, "H_MPa": 2000},
          "geometry": {"type": "rectangle", "Lx": 1, "Ly": 0.5, "h": 0.25}, "supports": [{"edge": "left", "ux": True}, {"at": [0, 0], "uy": True}, {"edge": "right", "ux": 0.005}]}
    return [(_cantilever(), {"geometric": True, "steps": 5, "combination": "char"}),
            (j2, {"steps": 10, "combination": "char"}),
            (_hill(), {"steps": 8, "combination": "char", "control": "arc", "to_failure": True, "max_steps": 12}),
            (_hill(), {"steps": 8, "combination": "char", "control": "displacement", "to_failure": True, "max_steps": 12}),
            (panel, {"steps": 10, "combination": "char", "to_failure": True, "max_steps": 30}),
            (panel, {"steps": 10, "combination": "char", "to_failure": True, "control": "displacement", "max_steps": 30}),
            (_panel(150, 150), {"steps": 10, "combination": "char", "to_failure": True, "control": "arc", "max_steps": 30}),
            (_rc_wall(), {"steps": 10, "combination": "uls", "lam_max": 2.0, "max_steps": 30}),
            (_rc_wall(), {"steps": 10, "combination": "uls", "lam_max": 2.0, "max_steps": 30, "iteration": "secant"}),
            (_pushover(), {"steps": 8, "combination": "char", "load_pattern": "variable", "lam_max": 1.0, "max_steps": 30})]


def _pushover():
    d = _rc_wall()
    d["loads"][1]["case"] = "G"
    d["loads"].append({"edge": "left", "from": 1.5, "to": 2.0, "qx": 400, "case": "Q"})
    return d


def test_pushover_starts_from_the_gravity_state():
    d = _pushover()
    m = parse_fe(d)
    r = run_nonlinear(m, NLOptions(steps=8, combination="char", load_pattern="variable", lam_max=1.0))
    grav = copy.deepcopy(d)
    grav["loads"] = [L for L in d["loads"] if L.get("case") != "Q"]
    g = run_nonlinear(parse_fe(grav), NLOptions(steps=4, combination="char", lam_max=1.0))
    assert r.status == "complete" and r.steps[0].lam == 0.0
    assert np.allclose(r.steps[0].u, g.steps[-1].u, rtol=0, atol=1e-12 + 1e-9 * np.abs(g.steps[-1].u).max())
    # at λ = 1 the reactions balance the gravity load and the full lateral load
    assert r.steps[-1].reaction[0] == pytest.approx(-400 * 0.5, rel=1e-3)
    with pytest.raises(FEError, match="variable loads"):
        run_nonlinear(parse_fe(grav), NLOptions(load_pattern="variable"))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_parity_nonlinear(tmp_path):
    cases = _cases()
    p = tmp_path / "nl.json"
    p.write_text(json.dumps(cases))
    out = json.loads(subprocess.run([NODE, str(ROOT / "web" / "parity_fe_nl.js"), str(p)], capture_output=True, text=True, check=True).stdout)
    for (d, o), js in zip(cases, out):
        assert js["ok"], js.get("error")
        r = run_nonlinear(parse_fe(d), NLOptions(**o))
        lam = [s.lam for s in r.steps]
        # compare the path up to 95 % of the peak: near and beyond a limit point the path is ill-conditioned (the step
        # control and the acceptance of stalled increments react to round-off), and the peaks agree to 0.5 %
        peak = max(lam)
        k = next(i for i, x in enumerate(lam) if x >= 0.95 * peak)
        assert js["status"] == r.status
        assert abs(js["lam_peak"] - peak) <= 5e-3 * abs(peak)
        assert js["iterations"][:k + 1] == [s.iterations for s in r.steps[:k + 1]]
        assert np.allclose(js["lam"][:k + 1], lam[:k + 1], rtol=0, atol=1e-6 * max(abs(x) for x in lam))
        assert js["monitor_dof"] == r.monitor_dof
        js_at = json.loads(subprocess.run([NODE, str(ROOT / "web" / "parity_fe_nl.js"), str(_one(tmp_path, d, o, k))],
                                          capture_output=True, text=True, check=True).stdout)[0] if k < len(lam) - 1 else js
        last = r.steps[k]
        su, ss = np.abs(last.u).max(), np.abs(last.gp_stress).max()
        assert np.allclose(js_at["u"], last.u, rtol=0, atol=1e-6 * su)
        assert np.allclose(np.array(js_at["gp"]).reshape(-1, 3), last.gp_stress.reshape(-1, 3), rtol=0, atol=1e-6 * ss)
        assert np.allclose(js_at["nodal"], nodal_stress(r, last), rtol=0, atol=1e-6 * ss)
        if d["material"].get("model") == "rc":
            for key in ("e1", "e2", "sc1", "sc2", "ssx", "ssy"):
                v = last.extra[key].ravel()
                assert np.allclose(js_at["extra"][key], v, rtol=0, atol=1e-6 * max(np.abs(v).max(), 1e-12)), key
            for key in ("cracked", "yield_x", "yield_y", "crushed", "rupture"):
                assert np.array_equal(np.array(js_at["extra"][key], dtype=bool), last.extra[key].ravel()), key
            cw = crack_widths(r, last).ravel()
            assert np.allclose(js_at["crack"], cw, rtol=0, atol=1e-6 * max(cw.max(), 1e-9))


def _one(tmp_path, d, o, k):
    """A case cut off after step k (max_steps = k, so the last recorded step is the peak)."""
    p = tmp_path / f"one_{k}.json"
    p.write_text(json.dumps([(d, dict(o, max_steps=k))]))
    return p


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_parity_design(tmp_path):
    models = [copy.deepcopy(DEEP_BEAM)]
    wall = copy.deepcopy(DEEP_BEAM)
    wall["design"] = {"member": "wall"}
    wall["geometry"] = {"type": "rectangle", "Lx": 6.0, "Ly": 3.0, "h": 0.5, "openings": [{"x0": 2.5, "y0": 0.0, "x1": 3.5, "y1": 2.0}]}
    wall["supports"] = [{"edge": "bottom", "ux": True, "uy": True}]
    wall["loads"] = [{"edge": "top", "qy": -300}, {"edge": "left", "qx": 60, "case": "Q"}]
    models.append(wall)
    p = tmp_path / "des.json"
    p.write_text(json.dumps(models))
    script = ("const T=require(process.argv[1]+'/fe2d.js');require(process.argv[1]+'/fe2d_design.js');"
              "const ms=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));"
              "process.stdout.write(JSON.stringify(ms.map(d=>{const m=T.FE.parseFE(d),des=T.FEDesign.designReinforcement(m),p=T.FEDesign.proposeLayers(m,des);"
              "return {As_req:des.As_req,util_c:des.util_concrete,stress:des.stress,layers:p.layers,warns:p.warns};})))")
    out = json.loads(subprocess.run([NODE, "-e", script, str(ROOT / "web"), str(p)], capture_output=True, text=True, check=True).stdout)
    for d, js in zip(models, out):
        m = parse_fe(d)
        des = design_reinforcement(m)
        layers, warns = propose_layers(m, des)
        assert np.allclose(js["stress"], des["stress"], rtol=0, atol=1e-9 * np.abs(des["stress"]).max())
        assert np.allclose(js["As_req"], des["As_req"], rtol=1e-9, atol=1e-6)
        assert np.allclose(js["util_c"], des["util_concrete"], rtol=1e-9, atol=1e-12)
        assert js["layers"] == json.loads(json.dumps(layers))
        assert js["warns"] == warns
