# Tragwerk

Verifiable structural analysis and Eurocode 2 checks for reinforced-concrete frames.
One analysis core runs behind a Python API, a command-line tool, an HTTP service and a
browser app, and every result comes with the clause, the formula and the substituted values
that produced it.

Status: working MVP (v0.1.0). Not yet for use on real projects without independent checking.

## What it does today

| Area | Scope |
| --- | --- |
| Analysis | Linear-elastic 2D frames, direct stiffness method, Euler–Bernoulli members with axial stiffness, moment releases (hinges), elastic spring supports, automatic pin nodes, mechanism detection with a plain-language message. Uniform, partial, trapezoidal, point and nodal loads in global or local directions; self weight. Internal forces and deflections between nodes are exact for these loads (no meshing error). |
| Combinations | EN 1990 Eq. 6.10 or 6.10a/b for ULS, characteristic, frequent and quasi-permanent SLS. Every variable case on or off (pattern loading), each takes a turn as leading action, exclusive groups (wind left or right), recommended ψ factors with overrides. |
| EN 1992-1-1 | Bending with axial force by strain compatibility (parabola-rectangle, Fig. 6.1 pivots, any bar layout), minimum eccentricity for columns, shear with and without links (6.2.2, 6.2.3, variable cot θ), minimum and maximum reinforcement, link ratio and spacing, clear bar spacing, column detailing, slenderness limit (5.8.3.1), span/depth ratio (7.4.2) and long-term deflection by curvature interpolation with creep (7.4.3). Recommended values of all nationally determined parameters, each overridable. |
| Output | Clause-by-clause calculation report (HTML and PDF) with model drawing, load listing, combinations, equilibrium self-check, reactions, M/V/N envelopes, deflected shape, check summary and detailed steps. Full results as JSON. |
| SIA 262 | Interface in place; every member reports "not covered" until the rules are written against the standard text. |

## Verification

Run `tragwerk verify --html verification.html --crosscheck`. The suite currently contains:

* 25 benchmarks (114 comparisons) against closed-form solutions and hand calculations:
  beam formulas, three-moment equation, Gerber beam, portal frames, springs, inclined members,
  EN 1990 combination sets, EC2 Table 3.1 properties, exact parabola-rectangle bending,
  rectangular-block comparison, axial resistance, shear hand calculations, Table 7.4N ratios,
  cracked-section properties, uncracked deflection.
* Cross-checks on 24 random multi-bay, multi-storey frames against PyNite (MIT) and OpenSees:
  displacements and reactions agree to about 1e-14, moments to about 1e-9.
* Physical property tests: stiffness symmetry, Maxwell–Betti reciprocity, rotation invariance,
  equilibrium of every load case, axial compatibility along members.
* Parity between the Python core and the browser engine (`web/engine.js`) on 13 models,
  including every check result, to 1e-7.

`pytest` runs all of the above (122 tests). A GitHub Actions workflow runs them and publishes the verification report on every push.

## Use it

```bash
pip install -e ".[dev,api,pdf]"
playwright install chromium            # only for PDF reports

tragwerk run examples/two_span_beam.json --report report.html --pdf report.pdf --json results.json
tragwerk verify --html verification.html --crosscheck
tragwerk serve                          # HTTP API on http://127.0.0.1:8000/docs
```

Python:

```python
import tragwerk as tw

model = tw.load_model("examples/portal_frame.json")
result = tw.run(model)
print(result.overall_status(), result.governing().utilisation)
for check in result.checks["Col L"]:
    print(check.title, check.status, check.utilisation)
html = tw.report(result, project="Demo", author="B. Gautam")
```

HTTP:

```
POST /v1/analyze      model JSON -> full results JSON
POST /v1/report       model JSON -> HTML calculation report
POST /v1/report.pdf   model JSON -> PDF calculation report
GET  /v1/verification -> benchmark results
GET  /v1/examples/{two_span_beam|portal_frame|cantilever}
```

Browser app: `python web/build.py` writes `dist/tragwerk-studio.html`, a single file with the
engine and examples inlined. It runs fully in the browser.

## Model format (`tragwerk/0.1`)

Spans in metres, forces in kN, moments in kNm, section dimensions in millimetres (keys end in
`_mm`), strengths in MPa. Global Y points up, so gravity loads are negative. Member local x
runs from node i to node j; "bottom" reinforcement is on the right-hand side when walking from
i to j. See `examples/` for complete models.

```json
{
  "schema": "tragwerk/0.1",
  "name": "Two-span beam",
  "design": {"design_code": "EC2", "combination_rule": "6.10", "params": {"alpha_cc": 0.85}},
  "materials": {"C30/37": {"type": "concrete", "fck": 30}, "B500B": {"type": "rebar", "fyk": 500}},
  "sections": {"B300x550": {"type": "rc_rect", "b_mm": 300, "h_mm": 550, "concrete": "C30/37",
               "rebar": "B500B", "cover_mm": 30, "bottom": {"n": 3, "dia_mm": 20},
               "top": {"n": 3, "dia_mm": 20}, "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200}}},
  "nodes": [{"id": "A", "x": 0, "y": 0}, {"id": "B", "x": 6, "y": 0}],
  "supports": [{"node": "A", "ux": true, "uy": true}, {"node": "B", "uy": true}],
  "members": [{"id": "M1", "i": "A", "j": "B", "section": "B300x550",
               "releases": {"i": false, "j": false},
               "design": {"type": "beam", "deflection_system": "end_span", "creep_coefficient": 2.0}}],
  "load_cases": [
    {"id": "G", "type": "permanent", "self_weight": true,
     "loads": [{"type": "udl", "member": "M1", "w": -12}]},
    {"id": "Q", "type": "variable", "category": "B",
     "loads": [{"type": "point", "member": "M1", "p": -40, "a": 2.5}]}
  ]
}
```

## Repository layout

```
src/tragwerk/
  model.py           model format, validation with field-level messages
  solver.py          frame analysis (stiffness, condensation, exact member results)
  combinations.py    EN 1990
  project.py         analysis + combinations + envelopes + checks, JSON export
  codes/base.py      traceable Step / CheckResult
  codes/ec2/         params (NDPs), materials, section (M-N, cracked properties), checks
  codes/sia262/      placeholder
  report/            HTML report, SVG drawings, PDF printing
  verification/      benchmarks, random frames, cross-checks, verification report
  cli.py, api.py
web/engine.js        browser engine, a port of the Python core
web/app.html         browser app template; web/build.py inlines engine + examples
tests/               pytest suite
examples/            example models
docs/                generated reports
```

## Known limitations

* Plane frames only; no torsion, out-of-plane or biaxial effects.
* Linear analysis: no second-order effects, imperfections or moment redistribution. Slender
  columns are flagged as not covered.
* Reinforcement is uniform along each member; no curtailment, anchorage or lap checks.
* Shear is checked at every station including the support axis (conservative).
* Deflection uses elastic moments with cracked curvatures and excludes shrinkage; for
  continuous members this can underestimate span deflection.
* Crack widths (7.3), stress limits (7.2), durability and fire are not checked yet.
* Implemented against EN 1992-1-1:2004 + AC:2010, not the second-generation EN 1992-1-1:2023.

## Next steps

1. SIA 262 rules, written and checked against the standard by an engineer who holds it.
2. Crack width (7.3.4) and stress limits (7.2).
3. Reinforcement zones along members, and a design mode that proposes bars.
4. Second-order analysis for slender columns (5.8.6 / 5.8.7).
5. Import and export of common analysis formats.
