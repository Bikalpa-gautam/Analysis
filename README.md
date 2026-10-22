# Tragwerk

Verifiable structural analysis and Eurocode checks for reinforced-concrete frames, buildings
and girder bridges, in 2D and 3D, with modal and response spectrum analysis and a plane
finite-element lab. One analysis core runs behind a Python API, a command-line tool, an HTTP
service and a browser studio, and every result comes with the clause, the formula and the
substituted values that produced it.

Status: working MVP (v0.3.0). Not yet for use on real projects without independent checking.

## New in 0.3: model anything, dynamics, FE, bridges

* **Free modelling in the 3D studio**, in the way of a general frame program: add nodes on a
  snapping working plane (XY, XZ or YZ at any level) or by coordinates, draw member chains,
  box-select, translational repeat with linking members (build storeys or frames by copying),
  move, split, merge coincident nodes, assign sections, member types and roll angles, supports
  (fixed, pinned, roller, free), member line and point loads, nodal loads and floor loads
  carried to the beams by tributary areas; a load case manager; node and member tables;
  undo and redo of every edit. Edits are layered on a generated building or bridge and survive
  a change of its grid, or start from a blank model. Nothing of 0.2 was removed: the generator,
  its ± steppers and all views are still there.
* **Dynamics**: consistent (or lumped) mass matrices, every member divided into elements,
  mass from the load cases with EN 1998 ψ<sub>E</sub> factors, natural frequencies by subspace
  iteration (browser) and a dense eigensolver (Python), participating mass ratios, and animated
  mode shapes in 3D.
* **Seismic, EN 1998-1** (low ductility, DCL): design spectrum type 1 or 2 for ground types A–E,
  modal response spectrum analysis with CQC, the 30 % directional rule, design seismic action
  effects A<sub>Ed</sub> on every member, the seismic design situation G + ψ<sub>2</sub>Q ± A<sub>Ed</sub>
  in the ULS checks, storey shears, the P-Δ coefficient θ and a damage limitation (drift) check
  for every column.
* **FE plate lab** (plane stress and plane strain) in the notation of the ETH Method of Finite
  Elements I course: T3, T6, Q4 and Q8 isoparametric elements with Gauss quadrature, walls and
  deep beams with openings, a plate with a hole on a curved isoparametric mesh, supports and loads
  on any part of an edge, banded contours of displacements, stresses, principal stresses and von
  Mises, principal stress directions (strut-and-tie hints), element-by-element or averaged
  stresses, and for any element its shape functions, Jacobian, B, C, k<sub>e</sub> (with its
  rigid-body eigenvalues), m<sub>e</sub> and f<sub>e</sub>. A convergence study compares the four
  element types against the exact Timoshenko & Goodier cantilever.
* **Element matrices of frame members**: local k (with condensed releases), rotation λ,
  global Tᵀ k T and the consistent mass matrix, next to the numbers they come from.
* **Girder bridges**: a generator for continuous reinforced-concrete road bridges (spans, deck
  width, girders, piers under the girders or with a pier cap) with EN 1991-2 Load Model 1
  (lanes by Table 4.1, UDL pattern per span, the tandem system moved along the bridge as an
  exclusive group), and EN 1990 Annex A2 factors (γ<sub>Q</sub> = 1.35, LM1 as one multi-component
  action).

## In 0.2: 3D buildings

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
| Dynamics | Consistent or lumped mass, Rayleigh–Ritz subspace iteration and dense eigensolvers, mass-normalised modes, participation factors and effective masses in X, Y, Z. |
| EN 1998-1 | Elastic and design spectra (Eqs 3.2–3.5, 3.13–3.16), CQC, directional combination, seismic action effects and combinations, damage limitation (4.4.3.2), θ (4.4.2.2). DCL only: no capacity design, no accidental torsion. |
| Continuum FE | Linear plane stress and plane strain, T3/T6/Q4/Q8, structured and mapped meshes, edge, point and body loads, stress recovery by extrapolation from Gauss points, reactions and strain energy (`tragwerk-fe/0.1`). |
| Bridges | Continuous RC girder road bridges, EN 1991-2 LM1 (UDL and tandem system), lever-rule transverse distribution, EN 1990 A2 traffic factors. |
| Modelling | Browser studio with nodes, members, supports, loads, load cases, repeat, split, merge and undo on top of generated or blank models. |
| SIA 262 | Interface in place; every member reports "not covered" until the rules are written against the standard text. |

## Verification

Run `tragwerk verify --html verification.html --crosscheck`. The suite currently contains:

* 49 benchmarks (232 comparisons) against closed-form solutions and hand calculations:
  beam formulas, three-moment equation, Gerber beam, portal frames, springs, inclined members,
  EN 1990 combination sets, EC2 Table 3.1 properties, exact parabola-rectangle bending,
  rectangular-block comparison, axial resistance, shear hand calculations, Table 7.4N ratios,
  cracked-section properties, uncracked deflection; in 3D: space cantilevers in both planes,
  torsion, biaxial base moments, grillages, wind on a building, generator load totals, biaxial
  interaction, torsion resistance and the nominal curvature method by hand; in dynamics: the
  consistent mass matrix, cantilever and simply supported beam frequencies, the axial bar mode,
  a two-storey shear frame, the EN 1998 design spectrum, the CQC coefficient and the base shear
  and drift of a single mass; in continuum FE: patch tests
  of T3, T6, Q4 and Q8 on a distorted mesh, the CST in closed form, rigid-body modes, the
  Timoshenko & Goodier cantilever, convergence rates, the Kirsch plate with a hole and global
  equilibrium of a wall with an opening.
* Cross-checks on 40 random frames (24 plane, 16 spatial with hinges, partial loads and rolled
  members) against PyNite (MIT) and OpenSees: displacements and reactions agree to about 1e-14;
  natural frequencies of 6 random space frames agree with OpenSees to about 1e-13 (MAC = 1).
* Physical property tests: stiffness symmetry, Maxwell–Betti reciprocity, rotation invariance,
  equilibrium of every load case, axial compatibility, and 2D-versus-3D consistency; mass and
  stiffness orthogonality of the modes and agreement with an independent sparse eigensolver;
  load totals of every bridge load case.
* Parity between the Python core and the browser engines (`web/engine.js`, `web/engine3d.js`,
  `web/dynamics.js`, `web/fe2d.js`) on frames, buildings, bridges, modal and seismic analyses and
  FE models, including every check result and calculation step, to 1e-6 or better; the building
  and bridge generators produce identical models in both languages.
* Headless-browser tests of the studio (WebGL through SwiftShader): drawing a frame with the
  mouse, supports, loads, repeat, undo and redo, modes and seismic results, bridges and the FE
  lab, each compared with the Python core.

`pytest` runs all of the above (247 tests). A GitHub Actions workflow runs them and publishes the
verification report and the built studio on every push.

## Use it

```bash
pip install -e ".[dev,api,pdf]"
playwright install chromium            # only for PDF reports

tragwerk run examples/two_span_beam.json --report report.html --pdf report.pdf --json results.json
tragwerk building --spec my_building.json --out building.json    # spec fields: see BuildingSpec
tragwerk bridge --spec my_bridge.json --out bridge.json          # spec fields: see BridgeSpec
tragwerk run examples/wall_opening_fe.json --json fe.json        # plane FE model (tragwerk-fe/0.1)
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
POST /v1/bridge       bridge spec -> tragwerk3d model JSON
POST /v1/analyze      model JSON (2D, 3D or FE) -> full results JSON
POST /v1/report       model JSON -> HTML calculation report
POST /v1/report.pdf   model JSON -> PDF calculation report
GET  /v1/verification -> benchmark results
GET  /v1/examples/{two_span_beam|portal_frame|cantilever}
```

Browser studio: `python web/build.py` writes `dist/tragwerk-studio.html`, a single file with
all engines and the examples inlined (three.js r128 is loaded from cdnjs for the 3D view; without
WebGL it falls back to a 2D drawing). It has three workspaces: 3D (buildings, bridges and free
models, with dynamics), 2D frame and FE plate. Analysis and checks run in the browser; the
default building (60 nodes, 116 members, 74 ULS combinations, 12 modes) takes about two seconds.

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

Dynamics and seismic action are optional blocks:

```json
"dynamics": {"modes": 12, "mass": "consistent", "segments": 2,
             "mass_cases": {"G1": 1.0, "Q": 0.3}, "masses": [{"node": "A1-L2", "m": 12.0}]},
"seismic": {"agR": 1.2, "importance": 1.0, "ground": "C", "spectrum_type": 1, "q": 1.5,
            "directions": ["X", "Y"], "damping": 0.05, "drift_limit": 0.005, "nu": 0.5}
```

Road traffic uses the categories `traffic_TS` and `traffic_UDL` (EN 1990 Table A2.1); a model
may override `gamma_Q_traffic` (default 1.35) in `design.params`.

## FE model format (`tragwerk-fe/0.1`)

```json
{
  "schema": "tragwerk-fe/0.1", "name": "Shear wall", "analysis": "plane_stress", "element": "Q8",
  "material": {"E_MPa": 33000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25},
  "geometry": {"type": "rectangle", "Lx": 6, "Ly": 3, "h": 0.25,
               "openings": [{"x0": 2.4, "y0": 0, "x1": 3.4, "y1": 2.2}]},
  "supports": [{"edge": "bottom", "ux": true, "uy": true}],
  "loads": [{"edge": "top", "qy": -120}, {"edge": "left", "from": 1.0, "to": 3.0, "qx": 25},
            {"at": [6, 3], "fx": -15}],
  "self_weight": true
}
```

Geometry is `rectangle` (with rectangular openings), `plate_hole` (quarter plate `W`, hole
radius `a`, `n_theta`, `n_r`, `grading`) or `mesh` (explicit `nodes` and `elements`). Edges are
`left`, `right`, `bottom`, `top` (and `hole`); `from`/`to` limit a support or load to part of an
edge; `"profile": "parabolic"` with `P` gives a parabolic shear traction. Edge loads are kN/m
through the thickness, stresses are reported in MPa.

## Repository layout

```
src/tragwerk/
  model.py           model format, validation with field-level messages
  model3d.py         3D format, local axes, 3D sections (bars in both directions)
  solver.py          plane frame analysis (stiffness, condensation, exact member results)
  solver3d.py        space frame analysis, same approach with 6 DOF per node
  building.py        parametric building generator
  bridge.py          parametric girder bridge generator (EN 1991-2 LM1)
  modal.py           mass matrices, eigenvalues, participation
  seismic.py         EN 1998-1 spectrum, CQC, seismic actions, drift
  fe2d.py            plane stress / plane strain finite elements
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
web/dynamics.js      browser modal and response spectrum analysis
web/modeller.js      editing operations of the 3D studio (add, repeat, split, loads …)
web/bridge.js        bridge generator; web/fe2d.js the FE engine
web/app3d.html       studio page and 3D workspace; web/app.html the 2D workspace;
                     web/app_fe.html the FE workspace
web/build.py         composes both workspaces, engines and examples into one file
tests/               pytest suite
examples/            example models
docs/                generated reports
```

## Known limitations

* The building and bridge models are frames: slabs and walls are not elements of the frame
  (slab loads reach beams by tributary areas, bridge decks by the lever rule; no rigid-diaphragm
  constraint). Lateral stability by cores or walls is represented only through the column
  effective length factor. Walls and deep beams can be studied separately in the FE lab, which
  is linear elastic and does not design reinforcement.
* Seismic design is DCL: no capacity design, no accidental torsional effects, no vertical
  component; θ is reported, not applied.
* Bridges: no braking, wind, temperature, fatigue (LM3), creep and shrinkage, bearings or
  EN 1992-2 specific rules; lanes are placed from the left kerb only.
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

1. Plate and shell elements in the frame model for slabs and walls, rigid diaphragms, cores.
2. SIA 262 rules, written and checked against the standard by an engineer who holds it.
3. Crack width (7.3.4) and stress limits (7.2); reinforcement design of FE regions.
4. Reinforcement zones along members, and a design mode that proposes bars.
5. Global second-order analysis (5.8.6 / 5.8.7), DCM capacity design and accidental torsion.
6. Moving-load influence lines and the remaining traffic actions for bridges.
7. Import and export of IFC and common analysis formats.
