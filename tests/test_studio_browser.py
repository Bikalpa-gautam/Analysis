"""Smoke test of the built browser studio in headless Chromium.

Loads dist-style output, serves three.js r128 from a local copy instead of the
CDN, and checks that the 3D workspace analyses the default building with the
same result as the Python core, that member selection and edits work, and that
no script errors occur. Skipped when Playwright, Chromium or the three.js file
are not available (CI installs them: see .github/workflows/ci.yml).
"""
import importlib.util
import json
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


def test_workspace_controls_preserve_the_model(page):
    """Layout, theme, guide and export must not change the engineering model."""
    pg, errors = page
    before = pg.evaluate("window.TragwerkStudio3D.S.data")
    pg.click("#studio-focus")
    assert not pg.locator("#mode-3d .editor").is_visible()
    assert not pg.locator("#mode-3d .results").is_visible()
    assert pg.locator("#mode-3d .draw").is_visible()
    pg.click("#studio-focus")
    assert pg.locator("#mode-3d .editor").is_visible()
    theme = pg.locator("html").get_attribute("data-theme")
    pg.click("#studio-theme")
    assert pg.locator("html").get_attribute("data-theme") != theme
    pg.click("#studio-theme")
    pg.click(".top-help")
    assert pg.locator("#studio-guide").is_visible()
    pg.keyboard.press("Escape")
    assert not pg.locator("#studio-guide").is_visible()
    pg.click("#studio-export")
    with pg.expect_download() as download:
        pg.locator("#b-drawer").get_by_role("button", name="Download JSON").click()
    assert json.loads(Path(download.value.path()).read_text()) == before
    pg.keyboard.press("Escape")
    assert pg.locator("#studio-export").evaluate("el => el === document.activeElement")
    assert pg.evaluate("window.TragwerkStudio3D.S.data") == before
    assert not errors, errors


@pytest.mark.parametrize("width", [360, 390, 768])
def test_compact_workspaces_keep_every_panel_accessible(page, width):
    pg, errors = page
    pg.set_viewport_size({"width": width, "height": 844})
    try:
        for mode in ("2d", "fe", "3d"):
            pg.click(f".studio-nav [data-mode='{mode}']")
            for panel in ("editor", "results", "draw"):
                pg.click(f".mobile-panels [data-panel='{panel}']")
                for candidate in ("editor", "results", "draw"):
                    assert pg.locator(f"#mode-{mode} .{candidate}").is_visible() == (candidate == panel)
                assert pg.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
    finally:
        pg.set_viewport_size({"width": 1400, "height": 900})


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
    pg.click(".studio-nav [data-mode='2d']")
    assert pg.locator("#mode-2d").is_visible() and not pg.locator("#mode-3d").is_visible()
    pg.click(".studio-nav [data-mode='3d']")
    assert not errors, errors


def _idle(pg):
    pg.wait_for_function("document.getElementById('b-busy').hidden", timeout=60000)


def test_modes_match_python(page):
    """The studio's modal analysis of the default building equals the Python core's."""
    pg, errors = page
    pg.select_option("#b-preset", "office")
    _idle(pg)
    data = pg.evaluate("window.TragwerkStudio3D.S.data")
    T_js = pg.evaluate("window.TragwerkStudio3D.S.result.modal.period")
    res = run3d(parse_model3d(data))
    assert len(T_js) == len(res.modal.period)
    for a, b in zip(T_js, res.modal.period):
        assert math.isclose(a, b, rel_tol=1e-6)
    pg.click("[data-show=modes]")
    pg.click("[data-rtab=modes]")
    pg.click("tr[data-mode='1']")
    assert pg.evaluate("window.TragwerkStudio3D.S.source") == "mode:1"
    # seismic on: EN 1998 pseudo-cases and SEI combinations join the checks, same as Python
    pg.click("[data-btab=dynamics]")
    pg.check("#b-seis-on")
    pg.wait_for_function("window.TragwerkStudio3D.S.result && window.TragwerkStudio3D.S.result.seismic", timeout=60000)
    _idle(pg)
    data = pg.evaluate("window.TragwerkStudio3D.S.data")
    js = pg.evaluate("({vb: window.TragwerkStudio3D.S.result.seismic.base_shear, u: window.TragwerkStudio3D.S.result.summary.max_utilisation})")
    res = run3d(parse_model3d(data))
    for d, v in res.seismic.base_shear.items():
        assert math.isclose(js["vb"][d], v, rel_tol=1e-6)
    assert math.isclose(js["u"], res.governing().utilisation, rel_tol=1e-6)
    pg.uncheck("#b-seis-on")
    _idle(pg)
    assert not errors, errors


def test_draw_a_frame_by_hand(page):
    """Blank model: draw a portal with the mouse, support it, load it, repeat it, compare with Python."""
    pg, errors = page
    J = "window.TragwerkStudio3D"
    pg.select_option("#b-preset", "blank")
    pg.evaluate(f"{J}.S.plane = {{kind: 'XZ', at: 0, step: 1}}; {J}.V.toolChanged(); {J}.V.setCamera('xz')")
    if pg.evaluate(f"{J}.V.ok"):
        for q in ([0, 0, 0], [0, 0, 3], [6, 0, 3], [6, 0, 0]):
            s = pg.evaluate(f"{J}.V.screenOfPoint({q})")
            pg.mouse.move(s["x"], s["y"])
            pg.mouse.click(s["x"], s["y"])
        pg.keyboard.press("Escape")
        pg.click("#b-tools [data-tool=support]")
        for q in ([0, 0, 0], [6, 0, 0]):
            s = pg.evaluate(f"{J}.V.screenOfPoint({q})")
            pg.mouse.click(s["x"], s["y"])
        pg.click("#b-tools [data-tool=select]")
    else:  # no WebGL: the same edits through the API
        pg.evaluate(f"""{J}.addEdit({{op: 'batch', ops: [
            {{op: 'addNode', id: 'N1', x: 0, y: 0, z: 0}}, {{op: 'addNode', id: 'N2', x: 0, y: 0, z: 3}},
            {{op: 'addNode', id: 'N3', x: 6, y: 0, z: 3}}, {{op: 'addNode', id: 'N4', x: 6, y: 0, z: 0}},
            {{op: 'addMember', id: 'M1', i: 'N1', j: 'N2', section: 'Beam 300×500'}}, {{op: 'addMember', id: 'M2', i: 'N2', j: 'N3', section: 'Beam 300×500'}},
            {{op: 'addMember', id: 'M3', i: 'N3', j: 'N4', section: 'Beam 300×500'}},
            {{op: 'support', node: 'N1', kind: 'fixed'}}, {{op: 'support', node: 'N4', kind: 'fixed'}}]}})""")
    _idle(pg)
    d = pg.evaluate(f"{J}.S.data")
    assert len(d["nodes"]) == 4 and len(d["members"]) == 3 and len(d["supports"]) == 2
    pg.evaluate(f"{J}.addEdit({{op: 'load', member: 'M2', w: -12, dir: 'global_z', caseId: 'Q'}})")
    pg.evaluate(f"{J}.addEdit({{op: 'repeat', members: ['M1', 'M2', 'M3'], nodes: [], d: [0, 5, 0], n: 2, link: true, supports: true}})")
    _idle(pg)
    data = pg.evaluate(f"{J}.S.data")
    assert len(data["nodes"]) == 12 and len(data["supports"]) == 6
    js = pg.evaluate(f"{J}.S.result.summary")
    res = run3d(parse_model3d(data))
    assert js["status"] == res.overall_status()
    assert math.isclose(js["max_utilisation"], res.governing().utilisation, rel_tol=1e-6)
    # undo and redo walk the edit history
    n = pg.evaluate(f"{J}.S.edits.length")
    pg.keyboard.press("Control+z")
    _idle(pg)
    assert pg.evaluate(f"{J}.S.edits.length") == n - 1 and len(pg.evaluate(f"{J}.S.data.nodes")) == 4
    pg.keyboard.press("Control+y")
    _idle(pg)
    assert len(pg.evaluate(f"{J}.S.data.nodes")) == 12
    for show in ("loads", "My", "def", "model"):
        pg.click(f"[data-show={show}]")
    assert not errors, errors


def test_element_matrices_panel(page):
    pg, errors = page
    pg.select_option("#b-preset", "office")
    _idle(pg)
    pg.evaluate("window.TragwerkStudio3D.select('BX-B1-L1', true)")
    pg.click("#b-mat > summary")
    pg.wait_for_selector("#b-mat-bd .matx")
    assert pg.locator("#b-mat-bd .matx").count() == 4
    assert not errors, errors


def test_fe_lab_matches_python(page):
    """The FE workspace solves the default wall like tragwerk.fe2d, and shows element matrices."""
    from tragwerk.fe2d import run_fe
    pg, errors = page
    pg.click(".studio-nav [data-mode='fe']")
    pg.wait_for_function("window.TragwerkFE && window.TragwerkFE.S.res", timeout=30000)
    for key in ("wall", "deep", "cantilever", "kirsch"):
        pg.select_option("#f-preset", key)
        pg.wait_for_function("document.getElementById('f-busy').hidden", timeout=30000)
        model = pg.evaluate("window.TragwerkFE.S.model")
        js = pg.evaluate("({u: Math.max(...window.TragwerkFE.S.res.field('umag')), vm: Math.max(...window.TragwerkFE.S.res.field('mises'))})")
        r = run_fe(model)
        assert math.isclose(js["u"], float(max((r.u ** 2).sum(axis=1) ** 0.5)), rel_tol=1e-9)
        assert math.isclose(js["vm"], float(r.field("mises").max()), rel_tol=1e-9)
    if pg.evaluate("!!window.TragwerkFE.S.view"):
        box = pg.locator("canvas.fe").bounding_box()
        pt = pg.evaluate("window.TragwerkFE.P.toS([2.0, 2.0])")
        pg.mouse.click(box["x"] + pt[0], box["y"] + pt[1])
        assert pg.evaluate("window.TragwerkFE.S.sel") is not None
        assert pg.locator("#f-results .mat").count() >= 5
    for f in ("mesh", "umag", "sxx", "s1", "mises"):
        pg.click(f"[data-field={f}]")
    pg.click(".studio-nav [data-mode='3d']")
    assert not errors, errors


def test_buckling_matches_python(page):
    """α_cr of every ULS combination and the governing buckling modes equal tragwerk.buckling; mode rows select the view."""
    import numpy as np

    from tragwerk.buckling import buckling
    pg, errors = page
    J = "window.TragwerkStudio3D"
    if pg.locator("#mode-fe").is_visible():
        pg.click(".studio-nav [data-mode='3d']")
    pg.select_option("#b-preset", "office")
    _idle(pg)
    pg.wait_for_function(f"{J}.S.buckRes && !{J}.S.buckProg", timeout=120000)
    js = pg.evaluate(f"({{g: {J}.S.buckRes.governing, a: {J}.S.buckRes.alpha, c: {J}.S.buckRes.combos.map((c) => [c.id, c.alpha_cr]), o: {J}.S.buck}})")
    br = buckling(parse_model3d(pg.evaluate(f"{J}.S.data")), modes=js["o"]["modes"], segments=js["o"]["segments"])
    assert js["g"] == br.governing and np.allclose(js["a"], br.alpha, rtol=1e-8)
    assert [c[0] for c in js["c"]] == [c.id for c in br.combos]
    assert np.allclose([c[1] for c in js["c"]], [c.alpha_cr for c in br.combos], rtol=1e-8)
    pg.click("[data-rtab=modes]")
    pg.click("tr[data-bmode='1']")
    assert pg.evaluate(f"{J}.S.source") == "buck:1" and pg.evaluate(f"{J}.S.show") == "modes"
    pg.wait_for_function("document.getElementById('b-legend').textContent.includes('Buckling mode 2')", timeout=10000)
    pg.click("[data-show=util]")
    assert not errors, errors


def test_fe_nonlinear_matches_python(page):
    """Nonlinear runs in the FE lab (worker) give the same load factors as tragwerk.fe2d_nl."""
    from dataclasses import fields

    from tragwerk.fe2d import parse_fe
    from tragwerk.fe2d_nl import NLOptions, run_nonlinear
    pg, errors = page
    if pg.locator("#mode-3d").is_visible():
        pg.click(".studio-nav [data-mode='fe']")
    pg.wait_for_function("window.TragwerkFE && window.TragwerkFE.S.res", timeout=30000)
    names = {f.name for f in fields(NLOptions)}
    for key in ("panel", "hill", "elastica"):
        pg.select_option("#f-preset", key)
        pg.wait_for_function("document.getElementById('f-busy').hidden", timeout=30000)
        pg.evaluate("window.TragwerkFE.runNonlinear()")
        pg.wait_for_function("window.TragwerkFE.S.nl && !window.TragwerkFE.S.nlRun", timeout=120000)
        js = pg.evaluate("({lam: window.TragwerkFE.S.nl.steps.map((s) => s.lam), o: window.TragwerkFE.S.nl.options, m: window.TragwerkFE.S.model})")
        res = run_nonlinear(parse_fe(js["m"]), NLOptions(**{k: v for k, v in js["o"].items() if k in names}))
        peak_py = max(s.lam for s in res.steps)
        assert abs(max(js["lam"]) - peak_py) <= 5e-3 * abs(peak_py), (key, max(js["lam"]), peak_py)
        assert pg.evaluate("window.TragwerkFE.S.src") == "nl"
        if key == "panel":                                 # mesh check: the panel's capacity does not depend on the mesh
            pg.click("[data-frtab=nl]")
            pg.click("#f-mesh2")
            pg.wait_for_function("window.TragwerkFE.S.nlMesh && window.TragwerkFE.S.nlMesh.rows.length === 2 && !window.TragwerkFE.S.nlMesh.running", timeout=60000)
            lams = pg.evaluate("window.TragwerkFE.S.nlMesh.rows.map((r) => r.lam)")
            assert abs(lams[1] - lams[0]) <= 1e-3 * lams[0]
            assert "The meshes agree" in pg.locator("#f-results").inner_text()
    for f in ("crack", "s2", "umag") if pg.locator("[data-field=crack]").count() else ("umag",):
        if pg.locator(f"[data-field={f}]").count():
            pg.click(f"[data-field={f}]")
    pg.click(".studio-nav [data-mode='3d']")
    assert not errors, errors


def test_bridge_matches_python(page):
    """The bridge presets in the studio give the same results as the Python core."""
    pg, errors = page
    J = "window.TragwerkStudio3D"
    pg.click(".studio-nav [data-mode='3d']") if pg.locator("#mode-fe").is_visible() else None
    for key in ("bridge", "bridge4"):
        pg.select_option("#b-preset", key)
        pg.wait_for_function(f"document.getElementById('b-busy').hidden && {J}.S.result", timeout=120000)
        data = pg.evaluate(f"{J}.S.data")
        assert data["meta"]["generator"] == "tragwerk.bridge"
        js = pg.evaluate(f"{J}.S.result.summary")
        res = run3d(parse_model3d(data))
        assert js["status"] == res.overall_status()
        assert math.isclose(js["max_utilisation"], res.governing().utilisation, rel_tol=1e-6)
    pg.click("[data-btab=grid]")
    pg.click("[data-step='spans:1']")
    pg.wait_for_function(f"document.getElementById('b-busy').hidden && {J}.S.data.meta.spec.spans.length === 5", timeout=120000)
    assert not errors, errors
