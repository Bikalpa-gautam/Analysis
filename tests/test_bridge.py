"""Girder bridge generator (EN 1991-2 Load Model 1) and the road traffic combination factors."""
import json
import math

import numpy as np
import pytest

from tragwerk.bridge import BridgeSpec, generate_bridge, lever, lever_strip, notional_lanes
from tragwerk.model3d import parse_model3d
from tragwerk.project3d import run3d


@pytest.mark.parametrize("ys", [[0.0], [-2.5, 2.5], [-4.0, 0.0, 4.0], [-4.5, -1.5, 1.5, 4.5]])
def test_lever_rule(ys):
    for y in np.linspace(-7, 7, 57):
        w = lever(float(y), ys)
        assert math.isclose(sum(w), 1.0, rel_tol=1e-12) and min(w) >= 0
        if ys[0] <= y <= ys[-1]:
            assert math.isclose(sum(wi * yi for wi, yi in zip(w, ys)), y, abs_tol=1e-12)   # moment balance about the axis
    sh = lever_strip(-5.0, 5.0, ys)
    assert math.isclose(sum(sh), 10.0, rel_tol=1e-12)


def test_notional_lanes():
    """EN 1991-2 Table 4.1."""
    assert notional_lanes(5.0) == (1, 3.0)
    assert notional_lanes(5.6) == (2, 2.8)
    assert notional_lanes(8.0) == (2, 3.0)
    assert notional_lanes(9.0) == (3, 3.0)
    assert notional_lanes(11.5) == (3, 3.0)


def _total(res, cid):
    return sum(r[2] for r in res.cases[cid].reactions.values())


@pytest.mark.parametrize("spec", [{}, {"girders": 3, "width": 12.0, "carriageway": 10.0}, {"pier_columns": "single"},
                                  {"girders": 1, "spans": [18.0], "width": 8.0, "carriageway": 6.5}])
def test_load_totals(spec):
    d = generate_bridge(spec)
    s = BridgeSpec(**{**BridgeSpec().__dict__, **spec})
    m = parse_model3d(d)
    res = run3d(m)
    L = sum(s.spans)
    # G2: surfacing on the carriageway and two parapets
    assert math.isclose(_total(res, "G2"), (s.surfacing * s.carriageway + 2 * s.parapet) * L, rel_tol=1e-9)
    # G1: deck slab plus the self weight of every member
    sw = sum(25.0 * m.sections[mm.section].area_m2 * res.analysis.members[mid].L for mid, mm in m.members.items())
    assert math.isclose(_total(res, "G1"), 25.0 * s.slab_mm / 1000 * s.width * L + sw, rel_tol=1e-9)
    # UDL on span 1: 9 kN/m² on lane 1, 2.5 kN/m² elsewhere on the carriageway
    nl, wl = notional_lanes(s.carriageway)
    q = sum((9.0 if k == 0 else 2.5) * wl for k in range(nl)) + 2.5 * (s.carriageway - nl * wl)
    assert math.isclose(_total(res, "UDL1"), q * s.spans[0], rel_tol=1e-9)
    # every tandem position carries all the axles of up to three lanes
    axles = 2 * sum((300.0, 200.0, 100.0)[:min(nl, 3)])
    for cid, lc in m.load_cases.items():
        if cid.startswith("TS"):
            assert math.isclose(_total(res, cid), axles, rel_tol=1e-9)
            assert lc.group == "TS" and lc.category == "traffic_TS"
    for cr in res.cases.values():
        assert max(abs(v) for v in cr.equilibrium.values()) < 1e-6 * max(1.0, abs(_total(res, "G1")))


def test_default_bridge_and_traffic_factors():
    d = generate_bridge()
    json.dumps(d)
    res = run3d(parse_model3d(d))
    assert res.overall_status() == "pass"
    ids = res.combos.case_ids
    traffic = [k for k, c in enumerate(ids) if c.startswith(("TS", "UDL"))]
    uls = [c for c in res.combos.combos if c.kind == "ULS"]
    # LM1 is one action (group gr1a): when it leads, tandem and UDL both take gamma_Q = 1.35 times the characteristic value
    assert all(math.isclose(c.factors[k], 1.35) for c in uls for k in traffic if c.factors[k])
    assert any(c.factors[ids.index("UDL1")] and c.factors[ids.index("TS1.2")] for c in uls)
    freq = [c for c in res.combos.combos if c.kind == "SLS_freq"]
    assert all(math.isclose(c.factors[k], 0.75 if ids[k].startswith("TS") else 0.40) for c in freq for k in traffic if c.factors[k])
    # the tandem positions never act together
    for c in res.combos.combos:
        assert sum(1 for cid, f in zip(ids, c.factors) if cid.startswith("TS") and f) <= 1
    # quasi-permanent: no traffic (psi_2 = 0)
    for c in res.combos.combos:
        if c.kind == "SLS_qp":
            assert all(f == 0 for cid, f in zip(ids, c.factors) if cid.startswith(("TS", "UDL")))


def test_traffic_with_another_action():
    """With wind leading, the traffic components take psi_0: 0.75 for TS and 0.40 for UDL (EN 1990 Table A2.1)."""
    d = generate_bridge({"spans": [20.0], "ts_stations": [0.5]})
    d["load_cases"].append({"id": "W", "name": "Wind", "type": "variable", "category": "wind",
                            "loads": [{"type": "udl", "member": "G1-S1-1", "w": 1.0, "direction": "global_y"}]})
    res = run3d(parse_model3d(d))
    ids = res.combos.case_ids
    wlead = [c for c in res.combos.combos if c.kind == "ULS" and c.leading == "W" and c.factors[ids.index("TS1.1")]]
    assert wlead
    for c in wlead:
        assert math.isclose(c.factors[ids.index("W")], 1.5)
        assert math.isclose(c.factors[ids.index("TS1.1")], 1.35 * 0.75)
        if c.factors[ids.index("UDL1")]:
            assert math.isclose(c.factors[ids.index("UDL1")], 1.35 * 0.40)


def test_validation():
    for bad in ({"spans": []}, {"girders": 0}, {"divisions": 7}, {"carriageway": 20.0}, {"pier_columns": "wall"}, {"ts_stations": [1.5]}):
        with pytest.raises(ValueError):
            generate_bridge(bad)


def test_bridge_modes():
    d = generate_bridge({"modes": 6})
    res = run3d(parse_model3d(d))
    assert len(res.modal.period) == 6 and res.modal.period[0] > 0
