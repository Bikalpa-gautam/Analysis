"""EC2 section and member checks: properties and hand-calculated cases."""
import math

import numpy as np
import pytest

from helpers import build
from tragwerk.codes.ec2 import checks as ec2c
from tragwerk.codes.ec2.params import EC2Params, make_params
from tragwerk.codes.ec2.section import RCSection
from tragwerk.model import Member, MemberDesign

P = EC2Params()


def section(**kw):
    sec = {"type": "rc_rect", "b_mm": 300, "h_mm": 500, "concrete": "C30/37", "rebar": "B500B", "cover_mm": 30,
           "bottom": {"n": 4, "dia_mm": 20}, "top": {"n": 2, "dia_mm": 12},
           "links": {"dia_mm": 8, "legs": 2, "spacing_mm": 200}}
    sec.update(kw)
    m = build({"A": (0, 0), "B": (5, 0)}, {"M": ("A", "B")}, [{"node": "A", "ux": True, "uy": True}, {"node": "B", "uy": True}],
              [{"id": "G", "type": "permanent", "loads": []}], sections={"S": sec})
    return m.sections["S"]


def forces(M, N=None, V=None, qp=None, L=5.0):
    M = np.atleast_2d(np.asarray(M, dtype=float))
    z = np.zeros_like(M)
    return ec2c.MemberForces(np.linspace(0, L, M.shape[1]), L, z if N is None else np.atleast_2d(N), z if V is None else np.atleast_2d(V),
                             M, ["U1"] * M.shape[0], ["u"] * M.shape[0], M if qp is None else np.atleast_2d(qp), ["Q1"], ["q"])


def test_symmetric_section_has_symmetric_diagram():
    rc = RCSection(section(top={"n": 4, "dia_mm": 20}), P)
    for N in (-300.0, 0.0, 800.0, 2500.0):
        hi, lo = rc.m_bounds(N)
        assert hi == pytest.approx(-lo, rel=1e-6)


def test_interpolated_diagram_matches_exact_solution():
    rc = RCSection(section(), P)
    for N in np.linspace(rc.N_min * 0.98, rc.N_max * 0.98, 15):
        hi, lo = rc.m_bounds(N)
        assert hi == pytest.approx(rc.solve_state(N).M, rel=2e-3, abs=0.5)
        assert lo == pytest.approx(rc.solve_state(N, flip=True).M, rel=2e-3, abs=0.5)


def test_axial_force_monotonic_along_boundary():
    rc = RCSection(section(), P)
    N, _ = rc.curve(False)
    assert np.all(np.diff(N) >= -1e-9)


def test_utilisation_is_one_on_the_boundary():
    rc = RCSection(section(), P)
    for N in (-200.0, 0.0, 1200.0):
        st = rc.solve_state(N)
        assert rc.utilisation(np.array([N]), np.array([st.M]))[0] == pytest.approx(1.0, abs=3e-3)


def test_high_strength_concrete_parameters():
    from tragwerk.codes.ec2.materials import concrete_props
    from tragwerk.model import Concrete
    c = concrete_props(Concrete("C90/105", 90), P)
    assert c.eps_cu2 == pytest.approx(2.6e-3)
    assert c.n == pytest.approx(1.4)
    assert c.lam == pytest.approx(0.7)
    assert c.eta == pytest.approx(0.8)


def test_national_parameter_override():
    p = make_params({"gamma_c": 1.5, "alpha_cc": 0.85})
    rc = RCSection(section(), p)
    assert rc.c.fcd == pytest.approx(0.85 * 30 / 1.5)
    with pytest.raises(ValueError, match="unknown design parameter"):
        make_params({"gamma_cc": 1.0})


def test_column_minimum_eccentricity():
    sec = section(h_mm=400, b_mm=400, bottom={"n": 3, "dia_mm": 20}, top={"n": 3, "dia_mm": 20})
    rc = RCSection(sec, P)
    member = Member("C", "A", "B", "S", design=MemberDesign(type="column"))
    f = forces([[0.0, 0.0]], N=[[-2000.0, -2000.0]])   # pure compression, no moment
    res = ec2c.check_bending_axial(member, rc, f, P)
    e0 = max(400 / 30, 20)
    assert res.demand == pytest.approx(2000 * e0 * 1e-3)


def test_shear_uses_tension_face_reinforcement():
    sec = section(top={"n": 2, "dia_mm": 12}, links=None)
    rc = RCSection(sec, P)
    r = ec2c.shear_resistance(rc, sec, P, np.array([[0.0, 0.0]]), np.array([[50.0, -50.0]]))
    assert r["Asl"][0, 0] == pytest.approx(4 * math.pi * 100)
    assert r["Asl"][0, 1] == pytest.approx(2 * math.pi * 36)
    assert r["VRdc"][0, 0] > r["VRdc"][0, 1]


def test_axial_tension_reduces_concrete_shear_resistance():
    sec = section(links=None)
    rc = RCSection(sec, P)
    r = ec2c.shear_resistance(rc, sec, P, np.array([[0.0, -200.0, 500.0]]), np.array([[1.0, 1.0, 1.0]]))
    v = r["VRdc"][0]
    assert v[1] < v[0] < v[2]


def test_beam_without_links_fails_minimum_shear_reinforcement():
    sec = section(links=None)
    rc = RCSection(sec, P)
    member = Member("B", "A", "B", "S", design=MemberDesign(type="beam"))
    out = {c.id: c for c in ec2c.check_detailing(member, rc, sec, forces([[10.0, 20.0]]), P)}
    assert out["ec2.links_min"].status == "fail"


def test_cantilever_deflection_is_tip_relative_to_root():
    sec = section(top={"n": 4, "dia_mm": 20}, bottom={"n": 2, "dia_mm": 12})
    rc = RCSection(sec, P)
    L, P_ = 2.0, 5.0
    x = np.linspace(0, L, 101)
    M = -(P_ * (L - x))[None, :]          # tip load, root at x = 0 (uncracked range)
    f = forces(M, qp=M, L=L)
    r = ec2c.deflection_curves(rc, sec, f, 2.0, P, cantilever=True)
    props = rc.transformed(rc.c.Ecm / 3.0, sagging=False)
    exact = P_ * L ** 3 / (3 * rc.c.Ecm / 3.0 * 1e3 * props["I_I"] * 1e-12)
    assert abs(r["w"][0, -1]) == pytest.approx(exact, rel=1e-3)


def test_slenderness_flags_second_order_effects():
    sec = section(h_mm=250, b_mm=250, bottom={"n": 2, "dia_mm": 16}, top={"n": 2, "dia_mm": 16})
    rc = RCSection(sec, P)
    member = Member("C", "A", "B", "S", design=MemberDesign(type="column", l0_factor=2.0))
    f = forces([[0.0, 0.0]], N=[[-600.0, -600.0]], L=4.0)
    assert ec2c.check_slenderness(member, rc, f, P).status == "not_covered"
