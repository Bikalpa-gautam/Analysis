# Tragwerk

Verifiable structural analysis and Eurocode checks for reinforced-concrete frames, buildings
and girder bridges, in 2D and 3D, with modal, response spectrum and linear buckling analysis,
and a plane finite-element lab that goes from linear elasticity to nonlinear reinforced
concrete, plasticity, large deflections, reinforcement design and earthquake time histories.
One analysis core runs behind a Python API, a command-line tool, an HTTP service and a browser
studio, and every result comes with the clause, the formula and the substituted values that
produced it.

Status: working MVP (v0.4.0). Not yet for use on real projects without independent checking.

## New in 0.4: nonlinear FE, reinforcement design, time histories, buckling

The FE plate lab was linear elastic and did not design reinforcement, and at re-entrant corners
of openings and at short supports its elastic stresses kept growing as the mesh was refined.
0.4 adds the content of the ETH Method of Finite Elements II course (Chatzi) on top of it:

* **Reinforced concrete, nonlinear**: a total-strain rotating smeared crack model in the spirit
  of the Modified Compression Field Theory. Concrete in compression follows the EN 1992-1-1
  parabola with compression softening by the cracking strain (Vecchio & Collins 1986, or the
  EN 1992 ν), in tension a linear branch to f<sub>ct</sub> and then tension stiffening
  (Collins & Mitchell), or crack-band softening with G<sub>F</sub> for plain concrete; unloading
  is secant with memory of the largest strains. Reinforcement is smeared per direction and
  region (meshes, boundary elements, lintels, ties), elastic–plastic with kinematic hardening up
  to rupture at ε<sub>ud</sub>, with the MCFT crack check. Crack widths by EN 1992-1-1 7.3.4.
  The stresses at corners and short supports no longer grow without bound: the concrete cracks,
  the bars yield and the forces redistribute, and the result is a load-carrying capacity
  λ<sub>u</sub> with the crack pattern, yielded bars and crushed zones that lead to it.
* **J2 (von Mises) plasticity** with isotropic hardening, radial return and the consistent
  tangent, in plane strain and (by a local Newton iteration on ε<sub>zz</sub>) plane stress.
* **Geometric nonlinearity**: total Lagrangian formulation with Green–Lagrange strains, second
  Piola–Kirchhoff stresses and the geometric stiffness; Cauchy stresses in the output.
* **Solution methods** (MFE II): full Newton–Raphson, modified Newton, secant iterations with
  Anderson acceleration, energy line search; load control with automatic step bisection,
  displacement control of a monitored node (Batoz–Dhatt) and the cylindrical arc-length method
  (Crisfield) for limit points, snap-through and softening; adaptive step size from the
  iteration count; stop at rupture, after the peak, or when the load would collapse in one
  increment. Pushover with gravity first, then the variable loads increased to failure.
* **Reinforcement design of FE regions**: EN 1992-1-1 Annex F (F.2–F.7) for the in-plane
  stresses of the ULS combination, averaged over a patch of one thickness (the usual
  regularisation of singular peaks), the strut check against ν f<sub>cd</sub>, minimum
  reinforcement of walls (9.6) and deep beams (9.7), and a proposed bar layout: a base mesh plus
  strips of additional bars where they are needed, extended by the anchorage length (8.4.3).
  "Apply" puts the proposal into the model, and the nonlinear analysis then verifies it to
  failure.
* **Earthquake time histories**: EN 1998-1 elastic spectra, spectrum-compatible synthetic
  accelerograms (sum of sinusoids, Jennings envelope, iterative matching, baseline correction,
  reproducible seeds) or a pasted record, response spectra by the exact Nigam–Jennings
  recurrence, a Bouc–Wen hysteretic oscillator, and direct integration of any FE model,
  linear or nonlinear, by the Newmark average-acceleration method with Newton iterations,
  consistent mass plus the gravity loads of the seismic combination as mass, Rayleigh damping
  from modes 1 and 2, gravity applied statically first, substepping where an increment does not
  converge, and an energy balance (input = kinetic + damping + internal).
* **Linear buckling of 3D frames**: (K + α K<sub>G</sub>) φ = 0 with the geometric stiffness of
  the ULS axial forces (cubic shape functions, torsional term), members divided into elements;
  α<sub>cr</sub> of every ULS combination, the buckling modes of the governing one (animated in
  the 3D view), the α<sub>cr</sub> ≥ 10 criterion of EN 1993-1-1 5.2.1 / EN 1992-1-1 5.8 and
  the amplification 1/(1 − 1/α<sub>cr</sub>). Block Krylov–Rayleigh–Ritz solver in the browser,
  warm-started from combination to combination, running in the background.
* **FE lab in the browser, rebuilt**: tabs for geometry (rectangles with openings, plates with
  holes, rings), loads (G and Q cases, pressures, combinations), material (elastic, J2,
  reinforced concrete, with the equations), reinforcement (layers, regions, cover, crack-width
  limit, design and proposal) and analysis (linear, nonlinear static, earthquake); results as
  linear, design, nonlinear or dynamic fields with crack, yield/crushing and rebar overlays; a
  timeline to step through the load path or the earthquake; the λ–u curve, convergence log,
  material state at every Gauss point, the record and its spectrum against the target, the
  Bouc–Wen hysteresis and energies, and mesh convergence. Long analyses run in a web worker
  with live progress.
* **Presets**: RC shear wall with an opening pushed over to failure, a deep beam on short
  supports designed and then verified to failure, the Vecchio–Collins panel, Hill's thick
  cylinder, the elastica, and a slender RC wall under a synthetic EN 1998 record.
* **CLI and API**: `tragwerk run model.json --design --nonlinear --time-history` for FE models,
  `--buckling` for 3D frames; `POST /v1/fe/design`, `/v1/fe/nonlinear`, `/v1/fe/time-history`
  and `/v1/buckling`.

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
| Continuum FE | Plane stress and plane strain, T3/T6/Q4/Q8 (and Q8 with reduced integration), structured and mapped meshes, edge, point, pressure and body loads, stress recovery by extrapolation from Gauss points, reactions and strain energy (`tragwerk-fe/0.1`). Nonlinear: reinforced concrete (rotating smeared cracks, tension stiffening, compression softening, smeared elastic–plastic bars to rupture), J2 plasticity, large deflections (total Lagrangian); Newton, modified Newton and secant iterations; load, displacement and arc-length control. |
| FE design | EN 1992-1-1 Annex F reinforcement of in-plane stress fields, strut check, minimum reinforcement (9.6, 9.7), bar proposal with anchorage lengths (8.4), crack widths (7.3.4) from the nonlinear analysis. |
| Time history | EN 1998-1 spectra, synthetic spectrum-compatible and pasted records, Nigam–Jennings response spectra, Bouc–Wen SDOF, Newmark direct integration of linear and nonlinear FE models with Rayleigh damping and energy balance. |
| Buckling | Linear buckling of 3D frames: α<sub>cr</sub> per ULS combination, buckling modes, 1/(1 − 1/α<sub>cr</sub>). |
| Bridges | Continuous RC girder road bridges, EN 1991-2 LM1 (UDL and tandem system), lever-rule transverse distribution, EN 1990 A2 traffic factors. |
| Modelling | Browser studio with nodes, members, supports, loads, load cases, repeat, split, merge and undo on top of generated or blank models. |
| SIA 262 | Interface in place; every member reports "not covered" until the rules are written against the standard text. |

## Verification

Run `tragwerk verify --html verification.html --crosscheck`. The suite currently contains:

* 62 benchmarks (300 comparisons) against closed-form solutions and hand calculations:
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
  equilibrium of a wall with an opening; in nonlinear FE: the elastica (Mattiasson), J2
  uniaxial response and the quadratic convergence of Newton's method, Hill's thick cylinder
  (elastic limit and plastic collapse), reinforced concrete panels in pure shear (τ<sub>u</sub> =
  ρ f<sub>y</sub> and √(ρ<sub>x</sub>ρ<sub>y</sub>) f<sub>y</sub> with the compression field angle,
  Nielsen, Vecchio & Collins), a reinforced concrete tie (tension stiffening, yield plateau,
  EN 1992 crack width), a deep beam designed by Annex F that carries λ<sub>u</sub> ≥ 1 in the
  nonlinear analysis, and Annex F, minimum reinforcement and anchorage by hand; in time-history
  analysis: SDOF responses against SciPy's exact discretisation, resonance, the Bouc–Wen loading
  branch in closed form, EN 1998 spectrum compatibility of synthetic records, an FE wall by
  Newmark against modal superposition, and the energy balance of a yielding steel wall; linear
  buckling of columns against Euler for four support conditions and of a sway portal frame.
* Cross-checks on 40 random frames (24 plane, 16 spatial with hinges, partial loads and rolled
  members) against PyNite (MIT) and OpenSees: displacements and reactions agree to about 1e-14;
  natural frequencies of 6 random space frames agree with OpenSees to about 1e-13 (MAC = 1).
* Physical property tests: stiffness symmetry, Maxwell–Betti reciprocity, rotation invariance,
  equilibrium of every load case, axial compatibility, and 2D-versus-3D consistency; mass and
  stiffness orthogonality of the modes and agreement with an independent sparse eigensolver;
  load totals of every bridge load case.
* Parity between the Python core and the browser engines (`web/engine.js`, `web/engine3d.js`,
  `web/dynamics.js`, `web/fe2d.js`, `web/fe2d_nl.js`, `web/fe2d_design.js`, `web/dynamics_nl.js`)
  on frames, buildings, bridges, modal, seismic and buckling analyses, linear and nonlinear FE
  models, reinforcement design, records, spectra and time histories, including every check
  result and calculation step, to 1e-6 or better (nonlinear paths to the peak within 0.5 %);
  the building and bridge generators produce identical models in both languages.
* Headless-browser tests of the studio (WebGL through SwiftShader): drawing a frame with the
  mouse, supports, loads, repeat, undo and redo, modes and seismic results, buckling, bridges,
  the FE lab and its nonlinear analyses in the web worker, each compared with the Python core.

`pytest` runs all of the above (284 tests). A GitHub Actions workflow runs them and publishes the
verification report and the built studio on every push.

## Use it

```bash
pip install -e ".[dev,api,pdf]"
playwright install chromium            # only for PDF reports

tragwerk run examples/two_span_beam.json --report report.html --pdf report.pdf --json results.json
tragwerk building --spec my_building.json --out building.json    # spec fields: see BuildingSpec
tragwerk bridge --spec my_bridge.json --out bridge.json          # spec fields: see BridgeSpec
tragwerk run examples/wall_opening_fe.json --json fe.json        # plane FE model (tragwerk-fe/0.1)
tragwerk run examples/rc_deep_beam_fe.json --design --nonlinear --json beam.json   # design, then to failure
tragwerk run building.json --report building.html --pdf building.pdf
tragwerk run building.json --buckling --json building.json.out   # α_cr of every ULS combination
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

# Nonlinear FE: design a deep beam by Annex F, put the proposed bars in, push it to failure
import json
from tragwerk.fe2d import parse_fe
from tragwerk.fe2d_design import design_reinforcement, propose_layers
from tragwerk.fe2d_nl import NLOptions, run_nonlinear

beam = json.load(open("examples/rc_deep_beam_fe.json"))
m = parse_fe(beam)
layers, notes = propose_layers(m, design_reinforcement(m))
beam["reinforcement"]["layers"] = layers
res = run_nonlinear(parse_fe(beam), NLOptions(control="arc", combination="uls", to_failure=True, max_steps=80))
print(res.status, res.lam_peak)            # λ_u ≥ 1: the design carries the ULS loads

# Earthquake time history of an FE model with a synthetic EN 1998 record
from tragwerk.dynamics_nl import DynOptions, fe_dynamic, synthetic_accelerogram
rec = synthetic_accelerogram(ag_g=0.25, soil="B", duration=10.0, dt=0.01, seed=2)
th = fe_dynamic(parse_fe(beam), rec, DynOptions(combination="qp"))

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
POST /v1/buckling     {"model", "modes", "segments"} -> α_cr per ULS combination, buckling modes
POST /v1/fe/design    FE model -> Annex F reinforcement and a bar proposal
POST /v1/fe/nonlinear {"model", "options"} -> load path, capacity, cracks, yielding
POST /v1/fe/time-history {"model", "record", "options"} -> histories, energies
GET  /v1/verification -> benchmark results
GET  /v1/examples/{two_span_beam|portal_frame|cantilever}
```

Browser studio: `python web/build.py` writes `dist/tragwerk-studio.html`, a single file with
all engines and the examples inlined (three.js r128 is loaded from cdnjs for the 3D view; without
WebGL it falls back to a 2D drawing). It has three workspaces: 3D (buildings, bridges and free
models, with dynamics), 2D frame and FE plate. Analysis and checks run in the browser; the
default building (60 nodes, 116 members, 74 ULS combinations, 12 modes) takes about two seconds,
and its linear buckling analysis (α<sub>cr</sub> of all 74 combinations, 2 376 degrees of freedom)
follows in the background in about two more. The FE lab runs its nonlinear and time-history
analyses in a web worker: the RC panel in under a second, the designed deep beam to failure in a
few seconds, the RC wall pushover in under a minute, the 10 s earthquake on the slender wall in
about ten seconds.

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
radius `a`, `n_theta`, `n_r`, `grading`), `ring` (radii `a`, `b`) or `mesh` (explicit `nodes`
and `elements`). Edges are `left`, `right`, `bottom`, `top` (and `hole`, `inner`, `outer`);
`from`/`to` limit a support or load to part of an edge; `"profile": "parabolic"` with `P` gives
a parabolic shear traction, `p` a pressure. Edge loads are kN/m through the thickness, stresses
are reported in MPa. Unknown keys in supports and loads are rejected with the list of allowed
ones.

Nonlinear materials and reinforcement:

```json
"material": {"model": "rc", "E_MPa": 33000, "nu": 0.2, "thickness_m": 0.25, "density_kNm3": 25,
             "fck_MPa": 30, "values": "design", "softening": "mcft", "tension": "stiffening"},
"reinforcement": {"cover_mm": 30, "fyk_MPa": 500, "k": 1.08, "euk": 0.05,
                  "layers": [{"dir": "y", "dia_mm": 10, "spacing_mm": 200, "faces": 2},
                             {"dir": "y", "dia_mm": 16, "n": 6, "region": {"x0": 0, "x1": 0.5, "y0": 0, "y1": 3}}]},
"design": {"member": "wall", "w_max_mm": 0.3},
"nonlinear": {"control": "arc", "combination": "uls", "steps": 10, "to_failure": true},
"time_history": {"record": {"type": "synthetic", "ag_g": 0.25, "soil": "B", "duration_s": 10},
                 "options": {"combination": "qp", "xi": 0.05}}
```

`"model": "j2"` with `fy_MPa` and `H_MPa` gives von Mises plasticity; `"values": "mean"` uses
mean strengths (f<sub>cm</sub>, f<sub>ctm</sub>, 1.1 f<sub>yk</sub>) instead of design values.
The `nonlinear` block takes the fields of `NLOptions` (control `load | displacement | arc`,
iteration `newton | modified | secant`, `geometric`, `load_pattern` `all | variable`, `steps`,
`lam_max`, `to_failure`, `tol`, `max_steps`, `monitor` …), `time_history.options` those of
`DynOptions`.

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
  fe2d_nl.py         nonlinear FE: RC smeared cracks, J2, total Lagrangian, path following
  fe2d_design.py     EN 1992-1-1 Annex F design of FE regions, bar proposal
  dynamics_nl.py     spectra, synthetic records, Bouc–Wen, Newmark time history of FE models
  buckling.py        linear buckling of 3D frames
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
web/fe2d_nl.js, web/fe2d_design.js, web/dynamics_nl.js   browser ports of the nonlinear FE,
                     design and time-history modules (buckling is in web/dynamics.js)
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
  effective length factor. Walls and deep beams can be studied, designed and verified separately
  in the FE lab, but are not coupled to the frame model.
* Seismic design is DCL: no capacity design, no accidental torsional effects, no vertical
  component; θ is reported, not applied.
* Bridges: no braking, wind, temperature, fatigue (LM3), creep and shrinkage, bearings or
  EN 1992-2 specific rules; lanes are placed from the left kerb only.
* Frames: linear first-order analysis. Linear buckling gives α<sub>cr</sub> and the
  amplification 1/(1 − 1/α<sub>cr</sub>), but the amplified forces are not fed back into the
  checks. In 3D, slender columns are checked by the nominal curvature method (5.8.8) as isolated
  members; in 2D they are still flagged as not covered. No moment redistribution.
* Nonlinear FE: the smeared rotating crack model is a continuum model. Where cracks localise
  (shear failures, a few wide cracks), strains and crack widths depend on the mesh, and so does
  the capacity to a lesser degree (crack-band regularisation only for plain concrete). The RC
  wall preset, with the storey shear along its top edge, gives λ<sub>u</sub> = 2.36 with
  h = 0.5 m and 2.23 with h = 0.25 m, and its load–displacement curves on h = 0.5, 0.25 and
  0.125 m agree within 4 % up to 1 mm; the finest mesh then needs many small increments. The
  same lateral load concentrated on 0.5 m of the side edge gave 3.7 and 4.7: a load or support
  on a short length crushes the concrete locally, and that local mechanism is mesh-dependent.
  Introduce loads as they act, and use the lab's mesh check (rerun at 2h or h/2) before
  trusting a capacity. No bond slip, dowel action or
  aggregate interlock as separate mechanisms (they are inside the MCFT relations); cyclic
  behaviour is approximate (secant unloading, no pinching); no discrete cracks, interfaces or
  contact; geometric nonlinearity only with elastic material. Increments that stall below a
  1 % residual are accepted for reinforced concrete and flagged as such (≈ in the log).
* Linear elastic FE results still have singular stresses at re-entrant corners and point-like
  supports: that is the elastic solution, and no mesh removes it. Use the design (averaged over
  one thickness) or the nonlinear analysis there, not the elastic peak.
* Time histories: one horizontal component, rigid base, Rayleigh damping; synthetic records are
  spectrum-compatible on average, not site-specific hazard.
* Torsion uses the gross section (with an optional stiffness factor); compatibility torsion is
  reported when it exceeds 0.1·T_Rd,c.
* Reinforcement is uniform along each member; no curtailment, anchorage or lap checks.
* Shear is checked at every station including the support axis (conservative).
* Deflection uses elastic moments with cracked curvatures and excludes shrinkage; for
  continuous members this can underestimate span deflection.
* Crack widths (7.3) are computed in the nonlinear FE lab only; frame members get no crack-width
  or stress-limit (7.2) check yet; durability and fire are not checked.
* Implemented against EN 1992-1-1:2004 + AC:2010, not the second-generation EN 1992-1-1:2023.

## Next steps

1. Plate and shell elements in the frame model for slabs and walls, rigid diaphragms, cores.
2. SIA 262 rules, written and checked against the standard by an engineer who holds it.
3. Fracture-energy regularisation of reinforced concrete (crack band with tension stiffening)
   and a mesh-objective comparison in the lab; discrete cracks and bond slip.
4. Crack width and stress limits for frame members; reinforcement zones along members, and a
   design mode that proposes bars.
5. Global second-order analysis of frames (5.8.6 / 5.8.7) using the buckling modes for
   imperfections, nonlinear frame elements (fibre sections), DCM capacity design and
   accidental torsion.
6. Moving-load influence lines and the remaining traffic actions for bridges.
7. Import and export of IFC and common analysis formats.
