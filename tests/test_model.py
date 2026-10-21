import json
from pathlib import Path

import pytest

from helpers import model_dict, single_case
from tragwerk.model import ModelError, parse_model

ROOT = Path(__file__).resolve().parents[1]


def base(**over):
    d = model_dict({"A": (0, 0), "B": (5, 0)}, {"M": ("A", "B")},
                   [{"node": "A", "ux": True, "uy": True}, {"node": "B", "uy": True}],
                   single_case([{"type": "udl", "member": "M", "w": -10}]))
    d.update(over)
    return d


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["members"][0].update(j="Z"), "unknown node 'Z'"),
    (lambda d: d["members"][0].update(section="nope"), "unknown section"),
    (lambda d: d["nodes"].append({"id": "A", "x": 1, "y": 1}), "duplicate node id"),
    (lambda d: d["materials"]["C30/37"].update(fck=120), "12 to 90 MPa"),
    (lambda d: d["sections"]["S"].update(h_mm=-5), "must be > 0"),
    (lambda d: d["load_cases"][0]["loads"].append({"type": "point", "member": "M", "p": 1, "a": 9}), "beyond the member length"),
    (lambda d: d["load_cases"].append({"id": "Q", "type": "variable", "loads": []}), "category"),
    (lambda d: d.update(supports=[]), "no supports"),
    (lambda d: d["load_cases"][0]["loads"].append({"type": "udl", "member": "M", "w": 1, "direction": "up"}), "direction"),
    (lambda d: d["sections"]["S"].update(cover_mm=600), "outside the section"),
])
def test_validation_messages(mutate, message):
    d = base()
    mutate(d)
    with pytest.raises(ModelError, match=message):
        parse_model(d)


def test_member_type_inferred_from_inclination():
    d = base()
    d["nodes"][1] = {"id": "B", "x": 0.5, "y": 4}
    m = parse_model(d)
    assert m.members["M"].design.type == "column"


@pytest.mark.parametrize("name", ["two_span_beam", "portal_frame", "cantilever"])
def test_examples_parse(name):
    parse_model(json.loads((ROOT / "examples" / f"{name}.json").read_text()))
