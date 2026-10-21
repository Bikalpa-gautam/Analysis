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
