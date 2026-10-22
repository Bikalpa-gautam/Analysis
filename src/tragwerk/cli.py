"""Command-line interface.

  tragwerk run MODEL.json [--report R.html] [--pdf R.pdf] [--json R.json]
                                 (frame models tragwerk/0.1 and tragwerk3d/0.1, or continuum FE models tragwerk-fe/0.1)
  tragwerk building [--spec SPEC.json] --out MODEL.json
  tragwerk bridge [--spec SPEC.json] --out MODEL.json
  tragwerk verify [--html V.html] [--crosscheck]
  tragwerk serve [--host 127.0.0.1] [--port 8000]

Exit codes for ``run``: 0 all checks pass, 1 a check fails, 2 some checks
are not covered, 3 the model is invalid or unstable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import __version__
from .model import ModelError, load_model
from .solver import StructureUnstable


def _run_fe(args, data) -> int:
    from .fe2d import FEError, run_fe, to_dict_fe
    try:
        res = run_fe(data)
    except FEError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    out = to_dict_fe(res)
    s = out["summary"]
    print(f"{res.model.name or 'FE model'}: {res.model.etype} {res.model.analysis.replace('_', ' ')}, {s['elements']} elements, "
          f"{s['nodes']} nodes, {s['dofs']} DOFs")
    print(f"  max displacement {s['max_displacement_m'] * 1000:.3f} mm, max von Mises {s['max_mises_MPa']:.3f} MPa "
          f"at ({s['at'][0]:.3f}, {s['at'][1]:.3f}) m, strain energy {s['strain_energy_kNm']:.4g} kNm")
    print(f"  equilibrium: Σ loads ({s['load_sum_kN'][0]:.3f}, {s['load_sum_kN'][1]:.3f}) kN, "
          f"Σ reactions ({s['reaction_sum_kN'][0]:.3f}, {s['reaction_sum_kN'][1]:.3f}) kN")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, allow_nan=False))
        print(f"wrote {args.json}")
    if args.report or args.pdf:
        print("note: calculation reports are produced for frame models; FE results are exported with --json", file=sys.stderr)
    return 0


def _run(args) -> int:
    try:
        raw = json.loads(Path(args.model).read_text())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    from .fe2d import SCHEMA_FE
    if isinstance(raw, dict) and raw.get("schema") == SCHEMA_FE:
        return _run_fe(args, raw)
    from .model3d import load_any
    from .project import run, to_dict
    from .project3d import run3d, to_dict3d
    try:
        model = load_any(args.model)
        is3d = getattr(model, "is3d", False)
        res = run3d(model) if is3d else run(model)
    except (ModelError, StructureUnstable, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"{model.name}: {len(model.nodes)} nodes, {len(model.members)} members, "
          f"{len(res.combos.combos)} combinations, {res.runtime_s:.2f} s")
    for w in res.warnings:
        print(f"warning: {w}")
    width = max(len(m) for m in res.checks) if res.checks else 6
    for mid, lst in res.checks.items():
        for c in lst:
            u = "" if c.utilisation is None else ("inf" if not math.isfinite(c.utilisation) else f"{c.utilisation:6.3f}")
            print(f"  {mid:<{width}}  {c.title:<46} {u:>7}  {c.status}")
    gov = res.governing()
    status = res.overall_status()
    print(f"result: {status.upper()}" + (f" (max utilisation {gov.utilisation:.3f}: {gov.member}, {gov.title})"
                                          if gov and gov.utilisation is not None and math.isfinite(gov.utilisation) else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(to_dict3d(res) if is3d else to_dict(res), indent=1, allow_nan=False))
        print(f"wrote {args.json}")
    if args.report or args.pdf:
        from .report.html import render
        from .report.html3d import render3d
        from .verification.benchmarks import run_all
        info = {"author": args.author or "", "project": args.project or "Project"}
        text = (render3d(res, verification=run_all(), project_info=info, full=args.full) if is3d
                else render(res, verification=run_all(), project_info=info))
        if args.report:
            Path(args.report).write_text(text, encoding="utf-8")
            print(f"wrote {args.report}")
        if args.pdf:
            from .report.pdf import html_to_pdf
            html_to_pdf(text, args.pdf, model.name)
            print(f"wrote {args.pdf}")
    return {"pass": 0, "fail": 1}.get(status, 2)


def _building(args) -> int:
    from .building import generate_building
    spec = json.loads(Path(args.spec).read_text()) if args.spec else None
    model = generate_building(spec)
    Path(args.out).write_text(json.dumps(model, indent=1))
    print(f"wrote {args.out}: {len(model['nodes'])} nodes, {len(model['members'])} members, {len(model['load_cases'])} load cases")
    return 0


def _bridge(args) -> int:
    from .bridge import generate_bridge
    spec = json.loads(Path(args.spec).read_text()) if args.spec else None
    try:
        model = generate_bridge(spec)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    Path(args.out).write_text(json.dumps(model, indent=1))
    print(f"wrote {args.out}: {len(model['nodes'])} nodes, {len(model['members'])} members, {len(model['load_cases'])} load cases")
    return 0


def _verify(args) -> int:
    from .verification.benchmarks import run_all
    results = run_all()
    bad = 0
    for bm, cs, err in results:
        ok = err is None and cs and all(c.passed for c in cs)
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {bm.id} {bm.title}" + (f"  ({err})" if err else ""))
    cross = None
    if args.crosscheck:
        from .verification.crosscheck import run_crosschecks
        from .verification.crosscheck3d import run_crosschecks3d, run_modal_crosschecks
        cross = run_crosschecks() + run_crosschecks3d() + run_modal_crosschecks()
        for r in cross:
            if r.get("kind") == "modal":
                print(f"  {r['solver']:<8} frame {r['seed']:>2}: frequencies {r['frequency']:.1e}, min MAC {r['mac']:.12f}")
            else:
                print(f"  {r['solver']:<8} frame {r['seed']:>2}: displacements {r['displacement']:.1e}, reactions {r['reaction']:.1e}")
    if args.html:
        from .verification.report import render
        Path(args.html).write_text(render(results, cross), encoding="utf-8")
        print(f"wrote {args.html}")
    print("verification " + ("PASSED" if not bad else f"FAILED ({bad} benchmarks)"))
    return 0 if not bad else 1


def _serve(args) -> int:
    import uvicorn
    from .api import app
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tragwerk", description="Verifiable RC frame analysis and Eurocode 2 checks")
    ap.add_argument("--version", action="version", version=f"tragwerk {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse and check a model")
    r.add_argument("model")
    r.add_argument("--report", help="write the HTML calculation report")
    r.add_argument("--pdf", help="write the calculation report as PDF (needs playwright)")
    r.add_argument("--json", help="write all results as JSON")
    r.add_argument("--author")
    r.add_argument("--project")
    r.add_argument("--full", action="store_true", help="3D: detailed checks for every member")
    r.set_defaults(fn=_run)
    b = sub.add_parser("building", help="generate a 3D building model from a spec")
    b.add_argument("--spec", help="JSON file with BuildingSpec fields (defaults for the rest)")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=_building)
    br = sub.add_parser("bridge", help="generate a 3D girder bridge model (EN 1991-2 LM1) from a spec")
    br.add_argument("--spec", help="JSON file with BridgeSpec fields (defaults for the rest)")
    br.add_argument("--out", required=True)
    br.set_defaults(fn=_bridge)
    v = sub.add_parser("verify", help="run the verification suite")
    v.add_argument("--html", help="write the verification report")
    v.add_argument("--crosscheck", action="store_true", help="also compare with PyNite/OpenSees if installed")
    v.set_defaults(fn=_verify)
    s = sub.add_parser("serve", help="start the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_serve)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
