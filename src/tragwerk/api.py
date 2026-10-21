"""HTTP API (FastAPI). Start with ``tragwerk serve`` or ``uvicorn tragwerk.api:app``.

POST /v1/analyze      model JSON -> full results JSON
POST /v1/report       model JSON -> HTML calculation report
POST /v1/report.pdf   model JSON -> PDF calculation report (needs playwright)
GET  /v1/verification -> benchmark results JSON
GET  /v1/examples/{name}
GET  /v1/health
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from . import __version__
from .model import ModelError, parse_model
from .project import run, to_dict
from .solver import StructureUnstable

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

app = FastAPI(title="Tragwerk API", version=__version__,
              description="Linear frame analysis and Eurocode 2 checks with traceable calculation steps.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _project(model: dict):
    try:
        return run(parse_model(model))
    except (ModelError, StructureUnstable, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.get("/v1/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/v1/analyze")
def analyze(model: dict = Body(...)):
    return to_dict(_project(model))


@app.post("/v1/report", response_class=HTMLResponse)
def report(model: dict = Body(...), verify: bool = True, author: str = "", project: str = "Project"):
    from .report.html import render
    from .verification.benchmarks import run_all
    res = _project(model)
    return render(res, verification=run_all() if verify else None, project_info={"author": author, "project": project})


@app.post("/v1/report.pdf")
def report_pdf(model: dict = Body(...), author: str = "", project: str = "Project"):
    from .report.html import render
    from .report.pdf import html_to_pdf
    from .verification.benchmarks import run_all
    res = _project(model)
    text = render(res, verification=run_all(), project_info={"author": author, "project": project})
    with tempfile.TemporaryDirectory() as d:
        try:
            path = html_to_pdf(text, Path(d) / "report.pdf", res.model.name)
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from None
        data = path.read_bytes()
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="calculation.pdf"'})


@app.get("/v1/verification")
def verification():
    from .verification.benchmarks import run_all
    out = []
    for bm, cs, err in run_all():
        out.append({"id": bm.id, "category": bm.category, "title": bm.title, "reference": bm.reference, "error": err,
                    "passed": bool(err is None and cs and all(c.passed for c in cs)),
                    "comparisons": [{"quantity": c.quantity, "computed": float(c.computed), "expected": float(c.expected),
                                     "unit": c.unit, "passed": bool(c.passed)} for c in cs]})
    return {"version": __version__, "benchmarks": out, "passed": all(b["passed"] for b in out)}


@app.get("/v1/examples/{name}")
def example(name: str):
    path = EXAMPLES / f"{name}.json"
    if not name.replace("_", "").isalnum() or not path.exists():
        raise HTTPException(status_code=404, detail="unknown example")
    return json.loads(path.read_text())
