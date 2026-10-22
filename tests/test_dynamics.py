"""Modal analysis and EN 1998 response spectrum analysis."""
import copy
import json
import math

import numpy as np
import pytest

from tragwerk.building import generate_building
from tragwerk.model3d import ModelError, parse_model3d
from tragwerk.modal import modal, refine
from tragwerk.project3d import run3d, to_dict3d
from tragwerk.seismic import CASE_E, CASE_E2, cqc_rho, design_spectrum
from tragwerk.solver3d import FrameAnalysis3D
from tragwerk.verification.frames3d import random_frame3d


def _building(**kw):
    d = generate_building({"spans_x": [6.0, 5.0], "spans_y": [5.5], "storeys": [3.6, 3.2, 3.2], **kw})
    d["dynamics"] = {"modes": 10}
    return d


@pytest.mark.parametrize("seed", range(6))
def test_modal_against_opensees(seed):
    pytest.importorskip("openseespy.opensees")
    from tragwerk.verification.crosscheck3d import compare_opensees_modal
    r = compare_opensees_modal(seed)
    assert r["frequency"] < 1e-9 and r["mac"] > 1 - 1e-9


@pytest.mark.parametrize("mass", ["consistent", "lumped"])
def test_eigenpairs_are_exact(mass):
    """Residual, orthogonality and agreement with an independent sparse eigen-solver."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    d = _building()
    d["dynamics"]["mass"] = mass
    mr = modal(parse_model3d(d))
    fa, M = mr.analysis, mr.M
    f = fa.free
    K, Mf, phi = fa.K[np.ix_(f, f)], M[np.ix_(f, f)], mr.phi[f]
    assert np.allclose(phi.T @ Mf @ phi, np.eye(phi.shape[1]), atol=1e-10)
    assert np.allclose(phi.T @ K @ phi, np.diag(mr.omega ** 2), rtol=1e-9, atol=1e-6 * mr.omega[-1] ** 2)
    res = K @ phi - Mf @ phi * mr.omega ** 2
    assert np.max(np.abs(res)) < 1e-8 * np.max(np.abs(K @ phi))
    w2 = np.sort(eigsh(csr_matrix(K), k=len(mr.omega), M=csr_matrix(Mf), sigma=0, which="LM")[0])
    assert np.allclose(np.sqrt(w2), mr.omega, rtol=1e-9)


def test_effective_masses_complete():
    """With every mode, the effective masses add up to the mass that can move (lumped masses)."""
    d = _building(spans_x=[5.0], storeys=[3.0, 3.0])
    d["dynamics"] = {"modes": 60, "mass": "lumped", "segments": 1}
    m = parse_model3d(d)
    mr = modal(m)
    nmass = int(np.sum(np.diag(mr.M[np.ix_(mr.analysis.free, mr.analysis.free)]) > 0))
    assert len(mr.omega) == min(60, nmass)
    if nmass <= 60:
        for q, ax in enumerate("XYZ"):
            free_mass = sum(mr.M[d_, d_] for d_ in mr.analysis.free if d_ % 6 == q)
            assert math.isclose(float(np.sum(mr.meff[ax])), free_mass, rel_tol=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_refined_model_is_statically_identical(seed):
    """Splitting members (with load mapping) changes nothing in a static analysis: the elements are exact."""
    m = parse_model3d(random_frame3d(seed))
    ref, submap = refine(m, 3)
    a, b = FrameAnalysis3D(m), FrameAnalysis3D(ref)
    ra, rb = a.solve_case(m.load_cases["G"]), b.solve_case(ref.load_cases["G"])
    for nid in m.nodes:
        assert np.allclose(ra.displacements[a.nidx[nid]], rb.displacements[b.nidx[nid]], rtol=1e-8, atol=1e-12)
    for nid, R in ra.reactions.items():
        assert np.allclose(R, rb.reactions[nid], rtol=1e-8, atol=1e-8)


def test_mass_from_loads():
    """Total mass = self weight + permanent loads + psi2 x imposed, divided by g."""
    d = _building()
    m = parse_model3d(d)
    mr = modal(m)
    fa = FrameAnalysis3D(m)
    W = 0.0
    for cid, f in mr.factors.items():
        r = fa.solve_case(m.load_cases[cid])
        W += f * sum(R[2] for R in r.reactions.values())
    assert math.isclose(mr.mtot["Z"], W / 9.81, rel_tol=1e-9)
    assert mr.factors["Q"] == 0.3 and mr.factors["G1"] == 1.0 and mr.factors["W+X"] == 0.0


def test_spectrum_continuity_and_bounds():
    from tragwerk.model3d import SeismicSettings
    for st in (1, 2):
        for g in "ABCDE":
            s = SeismicSettings(agR=1.5, ground=g, spectrum_type=st, q=1.5)
            T = np.linspace(0, 4, 4001)
            Sd = design_spectrum(T, s)
            assert np.max(np.abs(np.diff(Sd))) < 0.05          # continuous
            assert np.all(Sd >= s.beta * s.agR - 1e-12)


def test_cqc_reduces_to_srss_for_separated_modes():
    rho = cqc_rho(np.array([1.0, 10.0, 100.0]), 0.05)
    assert np.allclose(rho, np.eye(3), atol=2e-3)


def test_seismic_project():
    d = _building()
    d["seismic"] = {"agR": 1.2, "ground": "C", "q": 1.5}
    res = run3d(parse_model3d(d))
    sr, mr = res.seismic, res.modal
    # pseudo cases and combinations
    assert CASE_E in res.cases and CASE_E2 in res.cases
    sei = [c for c in res.combos.combos if c.id.startswith("SEI")]
    assert len(sei) == 4 and all(c.kind == "ULS" for c in sei)
    mid = "C-A1-L1"
    assert np.allclose(res.cases[CASE_E].members[mid].N, -res.cases[CASE_E2].members[mid].N)
    assert np.allclose(res.cases[CASE_E].members[mid].My, res.cases[CASE_E2].members[mid].My)
    # drift check on every column, none on beams
    for m_id, lst in res.checks.items():
        has = any(c.id == "ec8.drift" for c in lst)
        assert has == (res.model.members[m_id].design.type == "column")
    # base shear never exceeds the absolute sum of modal base shears, and is at least the largest one
    for dname in ("X", "Y"):
        vb = np.abs(sr.modal_base_shear[dname])
        assert vb.max() - 1e-9 <= sr.base_shear[dname] <= vb.sum() + 1e-9
    # storey shears grow towards the base
    Vx = [r["V"]["x"] for r in sr.storeys]
    assert all(Vx[k] > Vx[k + 1] for k in range(len(Vx) - 1))
    out = to_dict3d(res)
    json.dumps(out, allow_nan=False)
    assert out["modal"]["modes"][0]["T_s"] == pytest.approx(mr.period[0])
    assert set(out["seismic"]["base_shear"]) == {"X", "Y"}


def test_seismic_symmetry():
    """A doubly symmetric building: E magnitudes are symmetric between mirror-image members."""
    d = generate_building({"spans_x": [6.0, 6.0], "spans_y": [6.0, 6.0], "storeys": [3.0, 3.0], "wind": 0.0})
    d["dynamics"] = {"modes": 12}
    d["seismic"] = {"agR": 1.0, "ground": "B"}
    res = run3d(parse_model3d(d))
    e = res.seismic.members
    assert np.allclose(e["C-A1-L1"]["N"], e["C-C3-L1"]["N"], rtol=1e-6)
    assert np.allclose(e["C-A1-L1"]["N"], e["C-A3-L1"]["N"], rtol=1e-6)


def test_dynamics_validation():
    d = _building()
    d["dynamics"] = {"modes": 100}
    with pytest.raises(ModelError):
        parse_model3d(d)
    d["dynamics"] = {"mass_cases": {"nope": 1.0}}
    with pytest.raises(ModelError):
        parse_model3d(d)
    d["dynamics"] = {}
    d["seismic"] = {"agR": 1.0, "ground": "Z"}
    with pytest.raises(ModelError):
        parse_model3d(d)
