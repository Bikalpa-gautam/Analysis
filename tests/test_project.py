import json
import math
from pathlib import Path

import pytest

from tragwerk.model import load_model
from tragwerk.project import run, to_dict

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["two_span_beam", "portal_frame", "cantilever"])
def test_examples_run_and_serialise(name):
    res = run(load_model(ROOT / "examples" / f"{name}.json"))
    d = to_dict(res)
    text = json.dumps(d, allow_nan=False)
    assert d["summary"]["status"] in ("pass", "fail", "not_covered")
    assert not res.warnings
    for lst in res.checks.values():
        for c in lst:
            assert c.steps or c.notes, f"{c.id} has no traceable steps"
    assert len(text) > 1000


def test_two_span_support_moment_matches_hand_value():
    res = run(load_model(ROOT / "examples" / "two_span_beam.json"))
    bending = next(c for c in res.checks["Span 1"] if c.id == "ec2.bending_axial")
    g = 12.0 + 25 * 0.3 * 0.55
    w1, w2 = 1.35 * g + 1.5 * 10, 1.35 * g + 1.05 * 10
    assert bending.demand == pytest.approx(-(w1 + w2) * 6 ** 2 / 16, rel=1e-9)


def test_sia_code_reports_not_covered():
    m = load_model(ROOT / "examples" / "two_span_beam.json")
    m.settings = type(m.settings)(design_code="SIA262")
    res = run(m)
    assert res.overall_status() == "not_covered"
