# Tragwerk

Verifiable structural analysis and Eurocode 2 checks for reinforced-concrete frames and
buildings, in 2D and 3D. One analysis core runs behind a Python API, a command-line tool, an
HTTP service and a browser studio, and every result comes with the clause, the formula and the
substituted values that produced it.

Status: working MVP (v0.2.0). Not yet for use on real projects without independent checking.

## New in 0.2: 3D buildings

* **Space-frame solver**: 6 degrees of freedom per node, axial, St Venant torsion and biaxial
  bending stiffness, member roll angle, releases per axis (torsion, M_y, M_z), springs, loads in
  global or local directions, exact internal forces and deflections between nodes.
* **Building generator**: a complete reinforced-concrete frame from a few numbers (spans in X and
  Y, storey heights, sections, loads). Slab loads go to the beams by 45° yield lines or one-way;
  facade, snow and wind in four exclusive directions are generated. Grid lines A, B, C … / 1, 2 …,
  levels L0 … Ln, as on drawings.
* **3D EN 1992-1-1 checks**: biaxial bending with axial force (6.1 with Eq. 5.39), shear in both
  planes, torsion with shear (6.3), slenderness about both axes and slender columns by the
  nominal curvature method (5.8.8), plus all detailing and serviceability checks of 0.1.
* **3D studio in the browser**: build the building, orbit it in 3D (WebGL), click a member to see
  its forces and every check with its calculation steps, remove members, change sections, pin
  ends, add loads. Show M_y, M_z, N diagrams, the deformed shape and utilisation colours.
  The same file still contains the 2D frame workspace.
* **3D calculation report** (HTML, PDF) with axonometric drawings and per-member check steps.

## What it does today

| Area | Scope |
| --- | --- |
| Analysis | Linear-elastic 2D and 3D frames, direct stiffness method, Euler–Bernoulli members with axial stiffness, moment releases (hinges), elastic spring supports, automatic pin nodes, mechanism detection with a plain-language message. Uniform, partial, trapezoidal, point and nodal loads in global or local directions; self weight. Internal forces and deflections between nodes are exact for these loads (no meshing error). |
| Combinations | EN 1990 Eq. 6.10 or 6.10a/b for ULS, characteristic, frequent and quasi-permanent SLS. Every variable case on or off (pattern loading), each takes a turn as leading action, exclusive groups (wind left or right), recommended ψ factors with overrides. |
| EN 1992-1-1 | Bending with axial force by strain compatibility, biaxial interaction (5.8.9, Eq. 5.39), torsion (6.3), nominal curvature second-order check (5.8.8) (parabola-rectangle, Fig. 6.1 pivots, any bar layout), minimum eccentricity for columns, shear with and without links (6.2.2, 6.2.3, variable cot θ), minimum and maximum reinforcement, link ratio and spacing, clear bar spacing, column detailing, slenderness limit (5.8.3.1), span/depth ratio (7.4.2) and long-term deflection by curvature interpolation with creep (7.4.3). Recommended values of all nationally determined parameters, each overridable. |
| Output | Clause-by-clause calculation report (HTML and PDF) with model drawing, load listing, combinations, equilibrium self-check, reactions, M/V/N envelopes, deflected shape, check summary and detailed steps. Full results as JSON. |
| SIA 262 | Interface in place; every member reports "not covered" until the rules are written against the standard text. |

## Verification

Run `tragwerk verify --html verification.html --crosscheck`. The suite currently contains:

* 35 benchmarks (169 comparisons) against closed-form solutions and hand calculations:
  beam formulas, three-moment equation, Gerber beam, portal frames, springs, inclined members,
  EN 1990 combination sets, EC2 Table 3.1 properties, exact parabola-rectangle bending,
  rectangular-block comparison, axial resistance, shear hand calculations, Table 7.4N ratios,
  cracked-section properties, uncracked deflection; and in 3D: space cantilevers in both planes,
  torsion, biaxial base moments, grillages, wind on a building, generator load totals, biaxial
  interaction, torsion resistance and the nominal curvature method by hand.
* Cross-checks on 40 random frames (24 plane, 16 spatial with hinges, partial loads and rolled
  members) against PyNite (MIT) and OpenSees: displacements and reactions agree to about 1e-14.
* Physical property tests: stiffness symmetry, Maxwell–Betti reciprocity, rotation invariance,
  equilibrium of every load case, axial compatibility, and 2D-versus-3D consistency.
* Parity between the Python core and the browser engines (`web/engine.js`, `web/engine3d.js`) on
  24 models, including every check result and calculation step, to 1e-6 or better; the building
  generator produces byte-identical models in both languages.
* A headless-browser smoke test of the studio (WebGL through SwiftShader).

`pytest` runs all of the above (173 tests). A GitHub Actions workflow runs them and publishes the
verification report and the built studio on every push.

## Use it

```bash
pip install -e ".[dev,api,pdf]"
playwright install chromium            # only for PDF reports

tragwerk run examples/two_span_beam.json --report report.html --pdf report.pdf --json results.json
tragwerk building --spec my_building.json --out building.json    # spec fields: see BuildingSpec
tragwerk run building.json --report building.html --pdf building.pdf
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

# 3D: generate a building, analyse it, find the most utilised member
from tragwerk.building import generate_building
from tragwerk.model3d import parse_model3d
from tragwerk.project3d import run3d

data = generate_building({"spans_x": [7.5, 7.5], "spans_y": [6.0], "storeys": [4.0, 3.5, 3.5]})
res = run3d(parse_model3d(data))
gov = res.governing()
print(gov.member, gov.title, round(gov.utilisation, 3))
```

HTTP:

```
POST /v1/building     building spec -> tragwerk3d model JSON
POST /v1/analyze      model JSON (2D or 3D) -> full results JSON
POST /v1/report       model JSON -> HTML calculation report
POST /v1/report.pdf   model JSON -> PDF calculation report
GET  /v1/verification -> benchmark results
GET  /v1/examples/{two_span_beam|portal_frame|cantilever}
```

Browser studio: `python web/build.py` writes `dist/tragwerk-studio.html`, a single file with
both engines and the examples inlined (three.js r128 is loaded from cdnjs for the 3D view; without
WebGL it falls back to a 2D drawing). Analysis and checks run in the browser; the default
building (60 nodes, 116 members, 74 ULS combinations) takes about one second.

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

## 3D model format (`tragwerk3d/0.1`)

Same units and structure as the 2D format, with `z` on nodes (Z up), six restraints
`ux uy uz rx ry rz` and springs `kx … krz` on supports, `side` bars in sections (n per side
face), `roll_deg` and releases `{"t", "my", "mz"}` on members, loads with `fx … mz` (nodal) and
directions `global_x|y|z` or `local_x|y|z`. Member local axes: x from i to j; for non-vertical
members y = Z × x and z = x × y (up); for vertical members z = global X. Section width b lies
along local y, depth h along local z. `M_y` (x-z plane) is positive with compression on the +z
face, so it is sagging for beams. `tragwerk building --out b.json` writes a complete example.

## Repository layout

```
src/tragwerk/
  model.py           model format, validation with field-level messages
  model3d.py         3D format, local axes, 3D sections (bars in both directions)
  solver.py          plane frame analysis (stiffness, condensation, exact member results)
  solver3d.py        space frame analysis, same approach with 6 DOF per node
  building.py        parametric building generator
  combinations.py    EN 1990
  project.py         analysis + combinations + envelopes + checks, JSON export
  project3d.py       the same for 3D models
  codes/base.py      traceable Step / CheckResult
  codes/ec2/         params (NDPs), materials, section (M-N, cracked properties), checks, checks3d
  codes/sia262/      placeholder
  report/            HTML reports (2D, 3D), SVG drawings, PDF printing
  verification/      benchmarks, random frames, cross-checks, verification report
  cli.py, api.py
web/engine.js        browser engine (2D), a port of the Python core
web/engine3d.js      browser engine (3D) and building generator
web/app3d.html       studio page and 3D workspace; web/app.html the 2D workspace
web/build.py         composes both workspaces, engines and examples into one file
tests/               pytest suite
examples/            example models
docs/                generated reports
```

## Known limitations

* Frames only: slabs and walls are not modelled as elements (slab loads reach beams by
  tributary areas; no rigid-diaphragm constraint). Lateral stability by cores or walls is
  represented only through the column effective length factor.
* Linear first-order analysis. In 3D, slender columns are checked by the nominal curvature
  method (5.8.8) as isolated members; in 2D they are still flagged as not covered. No global
  second-order analysis, no moment redistribution.
* Torsion uses the gross section (with an optional stiffness factor); compatibility torsion is
  reported when it exceeds 0.1·T_Rd,c.
* Reinforcement is uniform along each member; no curtailment, anchorage or lap checks.
* Shear is checked at every station including the support axis (conservative).
* Deflection uses elastic moments with cracked curvatures and excludes shrinkage; for
  continuous members this can underestimate span deflection.
* Crack widths (7.3), stress limits (7.2), durability and fire are not checked yet.
* Implemented against EN 1992-1-1:2004 + AC:2010, not the second-generation EN 1992-1-1:2023.

## Next steps

1. Shell elements for slabs and walls, rigid diaphragms, cores.
2. SIA 262 rules, written and checked against the standard by an engineer who holds it.
3. Crack width (7.3.4) and stress limits (7.2).
4. Reinforcement zones along members, and a design mode that proposes bars.
5. Global second-order analysis (5.8.6 / 5.8.7) and seismic response spectrum (EN 1998).
6. Import and export of IFC and common analysis formats.
