"""Build the single-file browser app: inline the engine and the example models."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(out: Path | None = None) -> Path:
    html = (ROOT / "web" / "app.html").read_text(encoding="utf-8")
    engine = (ROOT / "web" / "engine.js").read_text(encoding="utf-8")
    examples = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "examples").glob("*.json"))}
    html = html.replace("/*__ENGINE__*/", engine.replace("</script", "<\\/script"))
    html = html.replace("/*__EXAMPLES__*/", "window.TRAGWERK_EXAMPLES = " + json.dumps(examples) + ";")
    out = out or ROOT / "dist" / "tragwerk-studio.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
