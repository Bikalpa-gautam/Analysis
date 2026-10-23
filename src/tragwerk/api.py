"""HTTP API (FastAPI). Start with ``tragwerk serve`` or ``uvicorn tragwerk.api:app``.

POST /v1/analyze      model JSON -> full results JSON (frame models, or tragwerk-fe/0.1 continuum models)
POST /v1/report       model JSON -> HTML calculation report
POST /v1/report.pdf   model JSON -> PDF calculation report (needs playwright)
POST /v1/building     building spec -> 3D model JSON (2D and 3D models are accepted everywhere)
POST /v1/bridge       bridge spec -> 3D model JSON of a continuous girder bridge
POST /v1/buckling     {"model": 3D model, "modes": 4, "segments": 4} -> α_cr per ULS combination, buckling modes
POST /v1/fe/design    FE model -> EN 1992-1-1 Annex F reinforcement design and a bar proposal
POST /v1/fe/nonlinear {"model": FE model, "options": NLOptions fields} -> load path, capacity, crack and yield summary
POST /v1/fe/time-history {"model": FE model, "record": {...}, "options": DynOptions fields} -> histories and energies
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
              description="Frame, building, bridge and plane FE analysis (linear, nonlinear, dynamic, buckling) with Eurocode checks and traceable calculation steps.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _project(model: dict):
    from .fe2d import SCHEMA_FE, FEError, run_fe
    if isinstance(model, dict) and model.get("schema") == SCHEMA_FE:
        try:
            return run_fe(model)
        except FEError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
    from .model3d import load_any
    from .project3d import run3d
    try:
        m = load_any(model)
        return run3d(m) if getattr(m, "is3d", False) else run(m)
    except (ModelError, StructureUnstable, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _dict(res):
    from .fe2d import FEResult, to_dict_fe
    if isinstance(res, FEResult):
        return to_dict_fe(res)
    from .project3d import to_dict3d
    return to_dict3d(res) if getattr(res.model, "is3d", False) else to_dict(res)


def _render(res, verify, info):
    from .fe2d import FEResult
    if isinstance(res, FEResult):
        raise HTTPException(status_code=422, detail="calculation reports are produced for frame models; use /v1/analyze for FE results")
    from .report.html import render
    from .report.html3d import render3d
    from .verification.benchmarks import run_all
    ver = run_all() if verify else None
    return render3d(res, verification=ver, project_info=info) if getattr(res.model, "is3d", False) else render(res, verification=ver, project_info=info)


@app.get("/v1/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/v1/analyze")
def analyze(model: dict = Body(...)):
    return _dict(_project(model))


@app.post("/v1/building")
def building(spec: dict = Body(default={})):
    from .building import generate_building
    try:
        return generate_building(spec or None)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/v1/bridge")
def bridge(spec: dict = Body(default={})):
    from .bridge import generate_bridge
    try:
        return generate_bridge(spec or None)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/v1/report", response_class=HTMLResponse)
def report(model: dict = Body(...), verify: bool = True, author: str = "", project: str = "Project"):
    return _render(_project(model), verify, {"author": author, "project": project})


@app.post("/v1/report.pdf")
def report_pdf(model: dict = Body(...), author: str = "", project: str = "Project"):
    from .report.pdf import html_to_pdf
    res = _project(model)
    text = _render(res, True, {"author": author, "project": project})
    with tempfile.TemporaryDirectory() as d:
        try:
            path = html_to_pdf(text, Path(d) / "report.pdf", res.model.name)
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from None
        data = path.read_bytes()
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="calculation.pdf"'})


def _fe_model(model: dict):
    from .fe2d import FEError, parse_fe
    try:
        return parse_fe(model)
    except FEError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _options(cls, opts: dict, what: str):
    from dataclasses import fields
    names = {f.name for f in fields(cls)}
    bad = sorted(set(opts or {}) - names)
    if bad:
        raise HTTPException(status_code=422, detail=f"{what}: unknown option {bad[0]!r} (allowed: {', '.join(sorted(names))})")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in (opts or {}).items()})


@app.post("/v1/buckling")
def buckling_endpoint(body: dict = Body(...)):
    from .buckling import buckling, to_dict_buckling
    from .model3d import load_any
    try:
        m = load_any(body.get("model", {}))
        if not getattr(m, "is3d", False):
            raise ValueError("linear buckling needs a 3D model (tragwerk3d/0.1)")
        br = buckling(m, modes=int(body.get("modes", 4)), segments=int(body.get("segments", 4)))
    except (ModelError, StructureUnstable, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return to_dict_buckling(br, m)


@app.post("/v1/fe/design")
def fe_design(model: dict = Body(...)):
    from .fe2d import FEError
    from .fe2d_design import design_reinforcement, propose_layers, to_dict_design
    m = _fe_model(model)
    try:
        des = design_reinforcement(m)
    except FEError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    out = to_dict_design(m, des)
    layers, notes = propose_layers(m, des)
    out["proposal"] = {"layers": layers, "notes": notes}
    return out


@app.post("/v1/fe/nonlinear")
def fe_nonlinear(body: dict = Body(...)):
    from .fe2d import FEError
    from .fe2d_nl import NLOptions, run_nonlinear, summary
    m = _fe_model(body.get("model", {}))
    try:
        return summary(run_nonlinear(m, _options(NLOptions, body.get("options"), "options")))
    except (FEError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/v1/fe/time-history")
def fe_time_history(body: dict = Body(...)):
    from .dynamics_nl import DynOptions, fe_dynamic, parse_record, to_dict_dynamic
    from .fe2d import FEError
    m = _fe_model(body.get("model", {}))
    try:
        rec = parse_record(body.get("record", {}))
        return to_dict_dynamic(fe_dynamic(m, rec, _options(DynOptions, body.get("options"), "options")), rec, every=int(body.get("every", 1)))
    except (FEError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


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
