"""CLI, HTTP API, Python API and report generation."""
import json
from pathlib import Path

import pytest

import tragwerk as tw
from tragwerk.cli import main

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "two_span_beam.json"


def test_python_api_and_report():
    res = tw.run(tw.load_model(EX))
    html = tw.report(res, project="Test")
    assert "Bending with axial force" in html
    assert "EN 1992-1-1 6.2" in html
    assert "all passed" in html
    assert "<svg" in html


def test_cli_run(tmp_path, capsys):
    out_json = tmp_path / "r.json"
    out_html = tmp_path / "r.html"
    code = main(["run", str(EX), "--json", str(out_json), "--report", str(out_html)])
    assert code == 0
    assert json.loads(out_json.read_text())["summary"]["status"] == "pass"
    assert out_html.read_text().startswith("<!doctype html>")
    assert "result: PASS" in capsys.readouterr().out


def test_cli_invalid_model(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"materials": {}}))
    assert main(["run", str(bad)]) == 3
    assert "error:" in capsys.readouterr().err


def test_cli_building_and_3d_run(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"spans_x": [6.0, 6.0], "spans_y": [5.0], "storeys": [3.5, 3.2]}))
    model = tmp_path / "b.json"
    assert main(["building", "--spec", str(spec), "--out", str(model)]) == 0
    data = json.loads(model.read_text())
    assert data["schema"] == "tragwerk3d/0.1" and len(data["nodes"]) == 3 * 2 * 3
    out_json, out_html = tmp_path / "r.json", tmp_path / "r.html"
    code = main(["run", str(model), "--json", str(out_json), "--report", str(out_html)])
    body = json.loads(out_json.read_text())
    assert code == {"pass": 0, "fail": 1}[body["summary"]["status"]]
    assert body["dimension"] == 3 and "C-A1-L1" in body["checks"]
    html = out_html.read_text()
    assert "Bending with axial force, both axes" in html and "<svg" in html


def test_cli_verify(tmp_path):
    out = tmp_path / "v.html"
    assert main(["verify", "--html", str(out)]) == 0
    assert "Comparisons passed" in out.read_text()


fastapi = pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from tragwerk.api import app
    return TestClient(app)


def test_api_analyze(client):
    r = client.post("/v1/analyze", json=json.loads(EX.read_text()))
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["status"] == "pass"
    assert "Span 1" in body["checks"]


def test_api_rejects_bad_model(client):
    m = json.loads(EX.read_text())
    m["members"][0]["section"] = "missing"
    r = client.post("/v1/analyze", json=m)
    assert r.status_code == 422
    assert "unknown section" in r.json()["detail"]


def test_api_unstable_structure(client):
    m = json.loads(EX.read_text())
    m["supports"] = [{"node": "A", "uy": True}, {"node": "C", "uy": True}]
    r = client.post("/v1/analyze", json=m)
    assert r.status_code == 422
    assert "mechanism" in r.json()["detail"]


def test_api_report_and_examples(client):
    m = client.get("/v1/examples/portal_frame").json()
    r = client.post("/v1/report?verify=false", json=m)
    assert r.status_code == 200 and "Col L" in r.text
    assert client.get("/v1/examples/../../etc").status_code == 404
    assert client.get("/v1/verification").json()["passed"] is True


def test_api_building_and_3d_analysis(client):
    r = client.post("/v1/building", json={"spans_x": [5.0], "spans_y": [5.0], "storeys": [3.0]})
    assert r.status_code == 200
    model = r.json()
    assert model["schema"] == "tragwerk3d/0.1" and len(model["members"]) == 4 + 4
    a = client.post("/v1/analyze", json=model)
    assert a.status_code == 200 and a.json()["dimension"] == 3
    assert client.post("/v1/building", json={"no_such_field": 1}).status_code == 422


def test_browser_app_builds(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build", ROOT / "web" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.build(tmp_path / "app.html")
    text = out.read_text()
    for marker in ("/*__ENGINE__*/", "/*__ENGINE3D__*/", "/*__EXAMPLES__*/", "/*__APP2D__*/", "/*__CSS2D__*/", "<!--__MARKUP2D__-->"):
        assert marker not in text
    assert "runProject" in text and "runProject3d" in text and "two_span_beam" in text
    assert 'id="mode-3d"' in text and 'id="mode-2d"' in text
    assert "three.js/r128/three.min.js" in text
    # every external script is pinned to an exact version on an allowed CDN
    import re
    for src in re.findall(r'<script src="([^"]+)"', text):
        assert src.startswith("https://cdnjs.cloudflare.com/ajax/libs/") and re.search(r"/r?\d+(\.\d+)*/", src)


# ---------------------------------------------------------------- nonlinear FE, design, time history, buckling
def _rc_panel(n=2):
    return {"schema": "tragwerk-fe/0.1", "name": "RC panel", "analysis": "plane_stress", "element": "Q4",
            "material": {"E_MPa": 30000, "nu": 0.0, "thickness_m": 0.2, "density_kNm3": 0, "model": "rc", "fck_MPa": 30, "values": "mean"},
            "design": {"member": "none"}, "reinforcement": {"k": 1.0, "layers": [{"dir": "x", "dia_mm": 12, "spacing_mm": 150, "faces": 2},
                                                                                 {"dir": "y", "dia_mm": 12, "spacing_mm": 150, "faces": 2}]},
            "geometry": {"type": "rectangle", "Lx": 1, "Ly": 1, "h": 1 / n},
            "supports": [{"at": [0, 0], "ux": True, "uy": True}, {"at": [1, 0], "uy": True}],
            "loads": [{"edge": "right", "qy": 200}, {"edge": "left", "qy": -200}, {"edge": "top", "qx": 200}, {"edge": "bottom", "qx": -200}],
            "self_weight": False, "nonlinear": {"control": "arc", "combination": "char", "steps": 10, "to_failure": True, "max_steps": 60, "tol": 1e-6}}


def test_cli_fe_design_and_nonlinear(tmp_path, capsys):
    p, out = tmp_path / "panel.json", tmp_path / "r.json"
    p.write_text(json.dumps(_rc_panel()))
    assert main(["run", str(p), "--design", "--nonlinear", "--json", str(out)]) == 0
    r = json.loads(out.read_text())
    rho = 2 * 3.14159265 * 36 / 150 / 200                  # 2 faces of Ø12 @ 150 in 200 mm
    assert r["nonlinear"]["lam_peak"] == pytest.approx(rho * 1.1 * 500, rel=1e-3)   # τ_u = ρ f_y, f_y = 1.1 f_yk (mean values)
    assert r["design"]["summary"]["max_As_req_x"] > 0 and r["design"]["proposal"]["layers"]
    text = capsys.readouterr().out
    assert "nonlinear:" in text and "design (Annex F" in text
    bad = _rc_panel()
    bad["nonlinear"]["no_such_option"] = 1
    p.write_text(json.dumps(bad))
    assert main(["run", str(p), "--nonlinear"]) == 3


def test_cli_buckling(tmp_path, capsys):
    b, out = tmp_path / "b.json", tmp_path / "r.json"
    assert main(["building", "--out", str(b)]) == 0
    data = json.loads(b.read_text())
    from tragwerk.building import generate_building
    small = generate_building({"spans_x": [6.0], "spans_y": [5.0], "storeys": [3.5, 3.2]})
    b.write_text(json.dumps(small))
    main(["run", str(b), "--buckling", "--buckling-modes", "2", "--json", str(out)])
    r = json.loads(out.read_text())
    assert r["buckling"]["governing"] and len(r["buckling"]["alpha"]) == 2 and r["buckling"]["alpha"][0] > 1
    assert "buckling: α_cr" in capsys.readouterr().out
    assert data["schema"] == "tragwerk3d/0.1"


def test_api_buckling_and_fe(client):
    from tragwerk.building import generate_building
    small = generate_building({"spans_x": [6.0], "spans_y": [5.0], "storeys": [3.5]})
    r = client.post("/v1/buckling", json={"model": small, "modes": 2, "segments": 2})
    assert r.status_code == 200 and r.json()["alpha"][0] > 1
    assert client.post("/v1/buckling", json={"model": json.loads(EX.read_text())}).status_code == 422
    panel = _rc_panel()
    opts = panel.pop("nonlinear")
    r = client.post("/v1/fe/nonlinear", json={"model": panel, "options": opts})
    assert r.status_code == 200 and r.json()["status"] in ("post-peak", "limit", "drop", "complete", "max-steps")
    assert client.post("/v1/fe/nonlinear", json={"model": panel, "options": {"bad": 1}}).status_code == 422
    r = client.post("/v1/fe/design", json=panel)
    assert r.status_code == 200 and "proposal" in r.json()
    wall = {"schema": "tragwerk-fe/0.1", "element": "Q4", "analysis": "plane_stress",
            "material": {"E_MPa": 30000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25},
            "geometry": {"type": "rectangle", "Lx": 2.0, "Ly": 4.0, "h": 1.0}, "self_weight": True,
            "supports": [{"edge": "bottom", "ux": True, "uy": True}], "loads": [{"edge": "top", "qy": -100}]}
    rec = {"type": "synthetic", "ag_g": 0.2, "soil": "B", "duration_s": 3.0, "dt_s": 0.01, "seed": 1, "rise_s": 0.5, "strong_s": 1.5}
    r = client.post("/v1/fe/time-history", json={"model": wall, "record": rec, "every": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete" and body["energy_balance_error"] < 1e-4 and len(body["t_s"]) == len(body["u_mon_m"])
