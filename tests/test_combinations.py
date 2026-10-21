import pytest

from helpers import build
from tragwerk.combinations import generate


def model(cases, rule="6.10"):
    return build({"A": (0, 0), "B": (5, 0)}, {"M": ("A", "B")},
                 [{"node": "A", "ux": True, "uy": True}, {"node": "B", "uy": True}],
                 [{"id": "G", "type": "permanent", "loads": []}] + cases, combination_rule=rule)


def factors(cs, kind):
    return {tuple(round(f, 6) for f in c.factors) for c in cs.combos if c.kind == kind}


def test_grouped_cases_never_act_together():
    cs = generate(model([
        {"id": "W1", "type": "variable", "category": "wind", "group": "wind", "loads": []},
        {"id": "W2", "type": "variable", "category": "wind", "group": "wind", "loads": []},
    ]))
    for f in factors(cs, "ULS") | factors(cs, "SLS_char"):
        assert not (f[1] and f[2])


def test_610ab():
    cs = generate(model([{"id": "Q", "type": "variable", "category": "B", "loads": []}], rule="6.10ab"))
    uls = factors(cs, "ULS")
    assert (1.35, 1.05) in uls          # 6.10a
    assert (1.1475, 1.5) in uls         # 6.10b: 0.85 * 1.35
    assert (1.0, 1.5) in uls            # favourable permanent, xi not applied


def test_sls_factors():
    cs = generate(model([{"id": "Q", "type": "variable", "category": "C", "loads": []},
                         {"id": "S", "type": "variable", "category": "snow", "loads": []}]))
    assert (1.0, 1.0, 0.5) in factors(cs, "SLS_char")       # Q leading, snow psi0 = 0.5
    assert (1.0, 0.7, 0.0) in factors(cs, "SLS_freq")       # Q leading psi1 = 0.7, snow psi2 = 0
    assert (1.0, 0.6, 0.0) in factors(cs, "SLS_qp")


def test_roof_load_never_accompanies():
    cs = generate(model([{"id": "H", "type": "variable", "category": "H", "loads": []},
                         {"id": "Q", "type": "variable", "category": "B", "loads": []}]))
    for f in factors(cs, "ULS"):
        assert f[1] in (0.0, 1.5), "psi0 = 0 for roofs: H only ever acts as the leading action"


def test_psi_override():
    m = model([{"id": "Q", "type": "variable", "category": "B", "loads": []}])
    m.settings.params["psi"] = {"B": [0.5, 0.5, 0.2]}
    assert (1.0, 0.2) in factors(generate(m), "SLS_qp")


def test_too_many_variable_cases():
    cases = [{"id": f"Q{k}", "type": "variable", "category": "B", "loads": []} for k in range(13)]
    with pytest.raises(ValueError, match="at most 12"):
        generate(model(cases))
