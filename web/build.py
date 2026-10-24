"""Build the single-file browser studio: the 3D building workspace and the 2D frame
workspace in one page, with both engines and the example models inlined."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parts_2d(html: str) -> tuple[str, str, str]:
    """(css, markup, app script) of the 2D workspace template."""
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    markup = html[html.index("</style>") + len("</style>"):html.index("<script>")].strip()
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    app = scripts[-1]
    return css, markup, app


def build(out: Path | None = None) -> Path:
    page = (ROOT / "web" / "app3d.html").read_text(encoding="utf-8")
    studio_css = (ROOT / "web" / "studio.css").read_text(encoding="utf-8")
    studio_js = (ROOT / "web" / "studio.js").read_text(encoding="utf-8")
    studio_shell = (ROOT / "web" / "studio-shell.html").read_text(encoding="utf-8")
    css2d, markup2d, app2d = _parts_2d((ROOT / "web" / "app.html").read_text(encoding="utf-8"))
    engine = (ROOT / "web" / "engine.js").read_text(encoding="utf-8")
    engine3d = (ROOT / "web" / "engine3d.js").read_text(encoding="utf-8")
    dynamics = (ROOT / "web" / "dynamics.js").read_text(encoding="utf-8")
    modeller = (ROOT / "web" / "modeller.js").read_text(encoding="utf-8")
    bridge = (ROOT / "web" / "bridge.js").read_text(encoding="utf-8")
    fe2d = (ROOT / "web" / "fe2d.js").read_text(encoding="utf-8")
    fe2d_nl = (ROOT / "web" / "fe2d_nl.js").read_text(encoding="utf-8")
    fe2d_design = (ROOT / "web" / "fe2d_design.js").read_text(encoding="utf-8")
    dyn_nl = (ROOT / "web" / "dynamics_nl.js").read_text(encoding="utf-8")
    fe_html = (ROOT / "web" / "app_fe.html").read_text(encoding="utf-8")
    fe_markup = fe_html[:fe_html.index("<script>")].strip()
    fe_app = re.findall(r"<script>(.*?)</script>", fe_html, re.S)[-1]
    ex2, ex3, exfe = {}, {}, {}
    for p in sorted((ROOT / "examples").glob("*.json")):
        data = json.loads(p.read_text())
        schema = data.get("schema", "tragwerk/0.1")
        (ex3 if schema == "tragwerk3d/0.1" else exfe if schema == "tragwerk-fe/0.1" else ex2)[p.stem] = data
    safe = lambda s: s.replace("</script", "<\\/script")  # noqa: E731
    page = page.replace("/*__CSS2D__*/", css2d)
    page = page.replace("<!--__MARKUP2D__-->", markup2d)
    page = page.replace("/*__ENGINE__*/", safe(engine))
    page = page.replace("/*__ENGINE3D__*/", safe(engine3d))
    page = page.replace("/*__DYNAMICS__*/", safe(dynamics))
    page = page.replace("/*__MODELLER__*/", safe(modeller))
    page = page.replace("/*__BRIDGE__*/", safe(bridge))
    page = page.replace("/*__FE2D__*/", safe(fe2d))
    page = page.replace("/*__FE2DNL__*/", safe(fe2d_nl))
    page = page.replace("/*__FE2DDESIGN__*/", safe(fe2d_design))
    page = page.replace("/*__DYNNL__*/", safe(dyn_nl))
    page = page.replace("<!--__MARKUPFE__-->", fe_markup)
    page = page.replace("/*__APPFE__*/", safe(fe_app))
    page = page.replace("/*__EXAMPLES__*/", safe("window.TRAGWERK_EXAMPLES = " + json.dumps(ex2) + ";\nwindow.TRAGWERK_EXAMPLES3D = " + json.dumps(ex3) + ";\nwindow.TRAGWERK_EXAMPLES_FE = " + json.dumps(exfe) + ";"))
    page = page.replace("/*__APP2D__*/", safe(app2d))
    page = page.replace("<!--__STUDIO_SHELL__-->", studio_shell)
    page = page.replace("/*__STUDIO_CSS__*/", studio_css)
    page = page.replace("/*__STUDIO_JS__*/", safe(studio_js))
    assert "__" not in re.sub(r"(__proto__|__new|__init__|__version__)", "", "".join(re.findall(r"/\*__\w+__\*/|<!--__\w+__-->", page))), "unfilled placeholder"
    out = out or ROOT / "dist" / "tragwerk-studio.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
