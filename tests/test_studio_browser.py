"""Smoke test of the built browser studio in headless Chromium.

Loads dist-style output, serves three.js r128 from a local copy instead of the
CDN, and checks that the 3D workspace analyses the default building with the
same result as the Python core, that member selection and edits work, and that
no script errors occur. Skipped when Playwright, Chromium or the three.js file
are not available (CI installs them: see .github/workflows/ci.yml).
"""
import importlib.util
import math
import os
from pathlib import Path

import pytest

from tragwerk.building import generate_building
from tragwerk.model3d import parse_model3d
from tragwerk.project3d import run3d

ROOT = Path(__file__).resolve().parents[1]
THREE = Path(os.environ.get("TRAGWERK_THREE_JS", ROOT / "web" / "node_modules" / "three" / "build" / "three.min.js"))
playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.skipif(not THREE.exists(), reason="three.js r128 not available locally (npm install --prefix web three@0.128.0)")


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("build", ROOT / "web" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    html = mod.build(tmp_path_factory.mktemp("studio") / "studio.html")
    body = THREE.read_bytes()
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        except Exception as exc:  # pragma: no cover - environment without a browser
            pytest.skip(f"chromium not available: {exc}")
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.route("https://cdnjs.cloudflare.com/**", lambda r: r.fulfill(status=200, body=body, headers={"content-type": "application/javascript"}))
        pg.route("https://fonts.googleapis.com/**", lambda r: r.fulfill(status=200, body="", headers={"content-type": "text/css"}))
        pg.goto(html.as_uri())
        pg.wait_for_function("window.TragwerkStudio3D && window.TragwerkStudio3D.S.result !== null", timeout=60000)
        yield pg, errors
        browser.close()


def test_default_building_matches_python(page):
    pg, errors = page
    js = pg.evaluate("window.TragwerkStudio3D.S.result.summary")
    py = run3d(parse_model3d(generate_building()))
    gov = py.governing()
    assert js["status"] == py.overall_status()
    assert math.isclose(js["max_utilisation"], gov.utilisation, rel_tol=1e-6)
    assert not errors, errors


def test_viewer_selection_and_edit(page):
    pg, errors = page
    ok = pg.evaluate("window.TragwerkStudio3D.V.ok")
    if ok:  # WebGL available: click the member on screen
        pt = pg.evaluate("window.TragwerkStudio3D.V.screenOf('BX-B1-L3')")
        pg.mouse.click(pt["x"], pt["y"])
        assert pg.evaluate("window.TragwerkStudio3D.S.sel") == "BX-B1-L3"
    else:
        assert pg.locator("#b-canvas svg.fallback").count() == 1
    pg.evaluate("window.TragwerkStudio3D.select('C-B2-L1', true)")
    pg.click("#b-mrem")
    pg.wait_for_function("window.TragwerkStudio3D.S.edits.length === 1 && window.TragwerkStudio3D.S.result && window.TragwerkStudio3D.S.result.model.members['C-B2-L1'] === undefined", timeout=30000)
    assert pg.evaluate("window.TragwerkStudio3D.S.result.summary.status") == "fail"   # transfer beams fail without the column
    for show in ("My", "Mz", "N", "def", "model", "util"):
        pg.click(f"[data-show={show}]")
    pg.click("#mode-3d [data-mode='2d']")
    assert pg.locator("#mode-2d").is_visible() and not pg.locator("#mode-3d").is_visible()
    pg.click("#mode-2d [data-mode='3d']")
    assert not errors, errors
